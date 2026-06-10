#!/usr/bin/env python3
"""Minimal conditional Flow-Matching trainer for HEALPix shells (proof-of-concept).

- Loads `params.yml` from a user-specified data directory as conditioning vector.
- Loads paired maps from two NPZ files (low-res and high-res).
- Trains a small MLP to regress the conditional vector field u_t(x|z) = x1 - x0
  following the flow-matching tutorial (toy-style training loop).
- Supports multi-node / multi-GPU training via PyTorch DDP.

This is intentionally small/safe for local smoke tests. Use bigger nside/batches for real runs.
"""

import argparse
import os
from pathlib import Path
import yaml
import time

import numpy as np
import healpy as hp
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from mlp import SmallMLP


def is_main_process():
    """Returns True if this is rank 0 (or if not running distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_distributed():
    """Initialize distributed process group if environment variables are set."""
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
        # Not running distributed — single GPU or CPU fallback
        print("[DDP] Not running distributed (no RANK/WORLD_SIZE env vars found).")
        return 0, 0, 1


def cleanup_distributed():
    """Destroy the process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


class ShellPairsDataset(Dataset):
    """Loads paired shells from either a single data directory or multiple `cosmo_*` subdirectories.

    If `data_dir` contains subfolders named like `cosmo_0000001` each must contain:
      - `params.yml`
      - low-res npz (default `disco_shells.npz`)
      - high-res npz (default `compressed_shells.npz`)

    The dataset aggregates all shells across subfolders and attaches the corresponding
    cosmology vector to each shell.
    """

    def __init__(
        self,
        data_dir: Path,
        low_name: str = "disco_shells.npz",
        high_name: str = "compressed_shells.npz",
        max_shells: int = 0,
        verbose: bool = True,
    ):
        data_dir = Path(data_dir)
        assert data_dir.exists(), f"data_dir not found: {data_dir}"

        # collect per-shell lists
        low_maps = []
        high_maps = []
        cosmo_vecs = []

        # detect cosmo_* subdirs
        subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name.startswith("cosmo_")]

        if len(subdirs) == 0:
            # fallback: treat data_dir itself as containing the NPZ files
            subdirs = [data_dir]

        total_collected = 0
        shell_total = int(max_shells) if max_shells and max_shells > 0 else None
        pbar_shells = tqdm(total=shell_total, desc="Loading shells", unit="shell", disable=not verbose)
        for sd in subdirs:
            # stop early to avoid opening/reading more NPZs once we've collected enough shells
            if max_shells and total_collected >= max_shells:
                break
            params_yml = sd / "params.yml"
            low_npz = sd / low_name
            high_npz = sd / high_name
            if not (params_yml.exists() and low_npz.exists() and high_npz.exists()):
                # skip incomplete directories
                continue

            params = yaml.safe_load(params_yml.read_text())
            keys = sorted(params.keys())
            cosmo_vec = np.array([float(params[k]) for k in keys], dtype=np.float32)

            low = np.load(low_npz, allow_pickle=False)
            high = np.load(high_npz, allow_pickle=False)
            assert "shells" in low and "shells" in high, f"NPZ must contain 'shells' in {sd}"

            low_shells = low["shells"]
            high_shells = high["shells"]
            assert low_shells.shape[0] == high_shells.shape[0], f"Mismatched shell counts in {sd}"

            for i in range(low_shells.shape[0]):
                if max_shells and total_collected >= max_shells:
                    break
                low_maps.append(np.asarray(low_shells[i], dtype=np.float64))
                high_maps.append(np.asarray(high_shells[i], dtype=np.float64))
                cosmo_vecs.append(cosmo_vec)
                total_collected += 1
                pbar_shells.update(1)

        pbar_shells.close()

        assert len(low_maps) > 0, f"No valid shell pairs found under {data_dir}"



def train(args):
    # ----------------------------------------------------------------
    # 1. Setup distributed
    # ----------------------------------------------------------------
    local_rank, rank, world_size = setup_distributed()
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Device: {device} | World size: {world_size}")

    # ----------------------------------------------------------------
    # 2. Dataset & DataLoader
    # ----------------------------------------------------------------
    data_dir = Path(args.data_dir)
    ds = ShellPairsDataset(
        data_dir,
        low_name=args.low_npz,
        high_name=args.high_npz,
        max_shells=args.max_shells,
        verbose=is_main_process(),
    )

    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),  # only shuffle if not using DistributedSampler
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    # 3. Model, optimizer, loss
    # ----------------------------------------------------------------
    cond_dim = ds.cosmo_mat.shape[1]
    dim_in = ds.sample_dim
    model = SmallMLP(dim_in=dim_in, cond_dim=cond_dim, hidden=args.hidden).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    sigma = args.sigma
    epochs = args.epochs

    if is_main_process():
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Effective batch size: {args.batch_size * world_size}")

    # ----------------------------------------------------------------
    # 4. Training loop
    # ----------------------------------------------------------------
    loss_history = []
    for ep in range(epochs):
        t0 = time.time()
        running = 0.0
        step_count = 0

        # Set epoch on sampler for proper shuffling across epochs
        if sampler is not None:
            sampler.set_epoch(ep)

        # Only show progress bar on rank 0
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", unit="step", disable=not is_main_process())
        for i, (x0_np, x1_np, cosmo_np) in enumerate(pbar):
            x0 = x0_np.to(device, non_blocking=True)   # low-res
            x1 = x1_np.to(device, non_blocking=True)   # high-res
            cosmo = cosmo_np.to(device, non_blocking=True)

            # If patches were returned, shapes are [B, P, D]
            is_patches = x0.dim() == 3
            if is_patches:
                B, P, Dp = x0.shape
                x0 = x0.view(B * P, Dp)
                x1 = x1.view(B * P, Dp)
                cosmo = cosmo.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)

            # Sample t and construct conditional vector field
            B = x0.shape[0]
            t = torch.rand(B, device=device)
            mu_t = t.view(-1, 1) * x1 + (1 - t).view(-1, 1) * x0
            eps = torch.randn_like(x0) * sigma
            xt = mu_t + eps   # input point
            ut = x1 - x0     # conditional vector field (target)

            # Forward pass
            pred = model(xt, t, cond=cosmo)
            loss = mse(pred, ut)

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Logging
            loss_val = loss.item()
            running += loss_val
            step_count += 1
            loss_history.append(loss_val)

            if is_main_process() and step_count % args.log_interval == 0:
                avg_loss = running / step_count
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        epoch_time = time.time() - t0
        if is_main_process():
            avg_epoch_loss = running / max(step_count, 1)
            print(f"Epoch {ep+1}/{epochs} done | avg loss: {avg_epoch_loss:.6f} | time: {epoch_time:.1f}s")

    # ----------------------------------------------------------------
    # 5. Save model and loss (only on rank 0)
    # ----------------------------------------------------------------
    if is_main_process():
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save state dict (unwrap DDP module)
        state_dict = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save(state_dict, out / "flow_mlp.pth")
        print("Saved model to", out / "flow_mlp.pth")

        # Save loss history
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(len(loss_history)), loss_history, marker='.', linewidth=1)
            ax.set_yscale('log')
            ax.set_xlabel('Training step')
            ax.set_ylabel('MSE loss')
            ax.set_title('Training loss')
            fig.tight_layout()
            loss_png = out / 'loss.png'
            fig.savefig(loss_png, dpi=150)
            plt.close(fig)
            np.save(out / 'loss.npy', np.array(loss_history))
            print('Saved loss plot to', loss_png)
        except Exception as e:
            print('Could not save loss plot:', e)

    # ----------------------------------------------------------------
    # 6. Cleanup
    # ----------------------------------------------------------------
    cleanup_distributed()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/Users/david/testData')
    parser.add_argument('--low-npz', type=str, default='disco_shells.npz')
    parser.add_argument('--high-npz', type=str, default='compressed_shells.npz')
    parser.add_argument('--max-shells', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader num_workers')
    parser.add_argument('--out-dir', type=str,
                        default='/Users/david/Library/CloudStorage/OneDrive-ETHZurich/ETH-Material/Master Project/github/models')
    parser.add_argument('--log-interval', type=int, default=10)
    args = parser.parse_args()

    train(args)
