#!/usr/bin/env python3
"""Conditional Flow-Matching trainer for HEALPix shells in Spherical Harmonic (Alm) space."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

from MLP import MLP
from ShellAlmDataset import ShellAlmDataset


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        if is_main_process():
            print(f"[DDP] Initialized: world_size={world_size}, backend=nccl")
        return local_rank, rank, world_size
    else:
        print("[DDP] Not running distributed.")
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    local_rank, rank, world_size = setup_distributed()
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ds = ShellAlmDataset(
        args.data_dir,
        lmax=args.lmax,
        max_shells=args.max_shells,
        verbose=is_main_process(),
    )

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    dl = DataLoader(
        ds,
        batch_size=min(args.batch_size, len(ds)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    sample_x0, _, sample_cond = ds[0]
    dim_in = sample_x0.shape[-1]
    cond_dim = sample_cond.shape[-1] if sample_cond.ndim > 0 else 1

    model = MLP(dim_in=dim_in, cond_dim=cond_dim, hidden=args.hidden).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    mse = nn.MSELoss()

    if is_main_process():
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"N_alms (dim_in): {dim_in} | Cond dim: {cond_dim} | Hidden: {args.hidden}")

    loss_history = []
    for ep in range(args.epochs):
        t0 = time.time()
        running = 0.0
        step_count = 0

        if sampler is not None:
            sampler.set_epoch(ep)

        model.train()
        for x0, x1, cond in dl:
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)

            B = x0.shape[0]

            x0_exp = x0.repeat_interleave(1, dim=0)
            x1_exp = x1.repeat_interleave(1, dim=0)
            cond_exp = cond.repeat_interleave(1, dim=0)

            t = torch.rand(B, 1, device=device)
            mu_t = t * x1_exp + (1 - t) * x0_exp
            eps = torch.randn_like(x0_exp) * args.sigma
            xt = mu_t + eps
            ut = x1_exp - x0_exp

            pred = model(xt, t.squeeze(1), cond=cond_exp)
            loss = mse(pred, ut)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running += loss.item()
            step_count += 1
            loss_history.append(loss.item())

        scheduler.step()
        avg_loss = running / max(step_count, 1)

        if is_main_process() and (ep + 1) % args.log_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {ep+1}/{args.epochs} | loss: {avg_loss:.6f} | lr: {lr_now:.2e} | {time.time()-t0:.1f}s")


    # ===== SYNC ALL RANKS before rank-0 I/O =====
    if is_distributed:
        dist.barrier()

    if is_main_process():
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        state_dict = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save(state_dict, out / "flow_mlp.pth")

        metadata = {
            "lmax": args.lmax,
            "sample_dim": dim_in,
            "cond_dim": cond_dim,
            "hidden": args.hidden,
        }
        torch.save(metadata, out / "metadata.pth")
        print(f"Saved model + metadata to {out}")

        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(loss_history, linewidth=0.5, alpha=0.7)
            axes[0].set_yscale("log")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("MSE")
            axes[0].set_title("Per-step loss")
            axes[0].grid(True, alpha=0.3)

            spe = max(len(loss_history) // args.epochs, 1)
            epoch_losses = [
                np.mean(loss_history[i * spe : (i + 1) * spe]) for i in range(args.epochs)
            ]
            axes[1].plot(range(1, args.epochs + 1), epoch_losses, "o-", markersize=2)
            axes[1].set_yscale("log")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Avg MSE")
            axes[1].set_title("Per-epoch loss")
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(out / "loss.png", dpi=150)
            plt.close(fig)
            np.save(out / "loss.npy", np.array(loss_history))
        except Exception as e:
            print(f"Loss plot error: {e}")

    cleanup_distributed()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/Users/david/testData")
    parser.add_argument("--lmax", type=int, default=1024)
    parser.add_argument("--max-shells", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="./models")
    parser.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()
    try:
        train(args)
    except Exception as e:
        print(f"Training crashed with error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        # Force-exit to kill any lingering torchrun agent threads
        os._exit(0)
