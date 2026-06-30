#!/usr/bin/env python3
"""
Train patch-based pixel-space flow matching for HEALPix shell correction.

Key differences from train_flow_matching.py
--------------------------------------------
- Operates in pixel space, not Alm space → better small-scale correction
- Patch-based: each pixel's velocity predicted from its local neighborhood
- Patch indices precomputed and cached on disk → cheap to reuse across epochs
- Per-shell chunked processing to handle 50M+ pixel maps on GPU

Usage
-----
Single GPU:
  python train_patch_flow.py --data-dir /path/to/data --nside 2048 --epochs 20

DDP (torchrun):
  torchrun --nproc_per_node=4 train_patch_flow.py --data-dir /path/to/data
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

from MLP import PatchMLP, ShellPixelDataset, get_or_build_patch_idx, patch_flow_loss


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        # Set device BEFORE init_process_group so NCCL uses local_rank, not global rank.
        # Wrong order caused "Guessing device ID" warnings and multi-node hangs.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return local_rank, rank, world_size
    return 0, 0, 1


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train(args):
    local_rank, rank, world_size = setup_distributed()
    is_dist = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    ds = ShellPixelDataset(
        data_dir=args.data_dir,
        nside_target=args.nside,
        verbose=is_main(),
    )

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) \
        if is_dist else None
    dl = DataLoader(
        ds,
        batch_size=1,            # one shell at a time — each shell is ~200MB
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Patch indices (build once, cache to disk)
    # ------------------------------------------------------------------
    if is_main():
        patch_idx = get_or_build_patch_idx(
            nside=args.nside,
            depth=args.patch_depth,
            cache_dir=args.cache_dir,
        )
        # Broadcast path to other ranks in distributed setting
        patch_idx_path = str(
            Path(args.cache_dir) / f"patch_idx_nside{args.nside}_depth{args.patch_depth}.npy"
        )

    if is_dist:
        dist.barrier()  # wait for rank-0 to finish building
        if not is_main():
            patch_idx = np.load(
                Path(args.cache_dir) / f"patch_idx_nside{args.nside}_depth{args.patch_depth}.npy"
            )

    patch_size = patch_idx.shape[1]
    patch_idx_t = torch.from_numpy(patch_idx).long().to(device)

    if is_main():
        print(f"[Train] nside={args.nside} | patch_size={patch_size} | "
              f"shells={len(ds)} | device={device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    sample_x0, _, sample_cond = ds[0]
    cond_dim = sample_cond.shape[0]

    model = PatchMLP(
        patch_size=patch_size,
        cond_dim=cond_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
    ).to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Train] PatchMLP params: {n_params:,} | cond_dim={cond_dim} | "
              f"patch_size={patch_size} | hidden={args.hidden}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = len(dl) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = cosine_warmup_scheduler(opt, warmup_steps, total_steps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    loss_history = []

    for ep in range(args.epochs):
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(ep)

        model.train()
        epoch_loss = 0.0

        for step, (x0, x1, cond) in enumerate(dl):
            # x0, x1: (1, Npix) → (Npix,)
            x0 = x0.squeeze(0).to(device, non_blocking=True)
            x1 = x1.squeeze(0).to(device, non_blocking=True)
            cond = cond.squeeze(0).to(device, non_blocking=True)

            # Sample random flow time
            t = float(torch.rand(1).item())

            # Compute chunked loss (backward is called inside patch_flow_loss
            # per chunk for gradient accumulation — returns a plain float)
            opt.zero_grad()
            loss_val = patch_flow_loss(
                model=model,
                x0=x0,
                x1=x1,
                cond=cond,
                patch_idx_t=patch_idx_t,
                t=t,
                sigma=args.sigma,
                chunk_size=args.chunk_size,
                device=device,
                use_amp=torch.cuda.is_available(),
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            epoch_loss += loss_val
            loss_history.append(loss_val)

            if is_main() and (step + 1) % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Ep {ep+1}/{args.epochs} | step {step+1}/{len(dl)} | "
                      f"loss={loss_val:.6f} | lr={lr_now:.2e}")

        if is_main():
            avg = epoch_loss / max(len(dl), 1)
            print(f"Epoch {ep+1}/{args.epochs} | avg_loss={avg:.6f} | "
                  f"elapsed={time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if is_dist:
        dist.barrier()

    if is_main():
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        state = model.module.state_dict() if is_dist else model.state_dict()
        torch.save(state, out / "patch_flow_mlp.pth")

        metadata = {
            "nside": args.nside,
            "patch_depth": args.patch_depth,
            "patch_size": patch_size,
            "cond_dim": cond_dim,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "max_shell_idx": ds.max_shell_idx,
            "cond_mean": ds.cond_mean.tolist(),
            "cond_std": ds.cond_std.tolist(),
        }
        torch.save(metadata, out / "patch_metadata.pth")
        print(f"[Train] Saved model + metadata to {out}")

        # Loss plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(loss_history, lw=0.5, alpha=0.7)
            axes[0].set_yscale("log")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("MSE")
            axes[0].set_title("Per-step loss")
            axes[0].grid(True, alpha=0.3)

            spe = max(len(loss_history) // max(args.epochs, 1), 1)
            epoch_losses = [
                np.mean(loss_history[i * spe:(i + 1) * spe])
                for i in range(args.epochs)
            ]
            axes[1].plot(range(1, args.epochs + 1), epoch_losses, "o-", markersize=3)
            axes[1].set_yscale("log")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Avg MSE")
            axes[1].set_title("Per-epoch loss")
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(out / "patch_loss.png", dpi=150)
            plt.close(fig)
            np.save(out / "patch_loss.npy", np.array(loss_history))
        except Exception as e:
            print(f"Loss plot error: {e}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--patch-depth", type=int, default=1,
                   help="Neighborhood depth. 1→9 pixels, 2→25 pixels")
    p.add_argument("--cache-dir", type=str, default="/capstor/scratch/cscs/damrein/healpy_patch_cache",
                   help="Directory to cache precomputed patch index arrays")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--chunk-size", type=int, default=1_000,
                   help="Pixels per GPU batch during chunked forward pass")
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="./patch_model")
    args = p.parse_args()

    try:
        train(args)
    except Exception as e:
        print(f"Training crashed: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        os._exit(0)