#!/usr/bin/env python3
"""Minimal conditional Flow-Matching trainer for HEALPix shells (proof-of-concept).

- Loads `params.yml` from a user-specified data directory as conditioning vector.
- Loads paired maps from two NPZ files (low-res and high-res).
- Splits each shell into HEALPix patches for memory-efficient training.
- Trains a small MLP to regress the conditional vector field u_t(x|z) = x1 - x0
  following the flow-matching tutorial (toy-style training loop).
- Supports multi-node / multi-GPU training via PyTorch DDP.
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
        print("[DDP] Not running distributed (no RANK/WORLD_SIZE env vars found).")
        return 0, 0, 1


def cleanup_distributed():
    """Destroy the process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def split_into_patches(shell_map, nside_patch=16):
    """Split a full-resolution HEALPix map into patches defined by lower-res pixels.

    Each patch corresponds to one pixel at nside_patch, containing all the
    sub-pixels from the full-resolution map that fall within it.

    Args:
        shell_map: 1D array of shape [npix_full]
        nside_patch: nside defining the patch grid

    Returns:
        patches: array of shape [n_patches, pixels_per_patch]
    """
    nside_full = hp.npix2nside(len(shell_map))
    assert nside_full >= nside_patch, (
        f"Full nside ({nside_full}) must be >= patch nside ({nside_patch})"
    )
    npix_patch = hp.nside2npix(nside_patch)
    pixels_per_patch = (nside_full // nside_patch) ** 2

    # HEALPix NESTED ordering guarantees contiguous sub-pixels
    # Convert to nested ordering if needed, then reshape
    shell_nested = hp.reorder(shell_map, r2n=True)
    patches = shell_nested.reshape(npix_patch, pixels_per_patch)
    return patches


class ShellPairsDataset(Dataset):
    """Loads paired shells and splits them into patches for memory-efficient training.

    Directory structure supported:
      data_dir/
      ├── cosmo_0000001/
      │   ├── params.yml
      │   ├── run_0/
      │   │   ├── shells_nside=2048.npz
      │   │   └── compressed_shells.npz
      │   └── run_1/ ...
      ├── cosmo_0000002/ ...

    Each sample returned is a single patch (not an entire shell).
    """

    def __init__(
        self,
        data_dir: Path,
        low_name: str = "shells_nside=2048.npz",
        high_name: str = "compressed_shells.npz",
        nside_patch: int = 248,
        max_shells: int = 0,
        verbose: bool = True,
    ):
        data_dir = Path(data_dir)
        assert data_dir.exists(), f"data_dir not found: {data_dir}"

        self.nside_patch = nside_patch

        # collect per-shell lists
        low_patches_list = []
        high_patches_list = []
        cosmo_vecs = []

        # detect cosmo_* subdirs
        subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name.startswith("cosmo_")]

        if len(subdirs) == 0:
            subdirs = [data_dir]

        # Expand each cosmo dir into its run_* subdirs (if they exist)
        leaf_dirs = []
        for sd in subdirs:
            run_dirs = [r for r in sorted(sd.iterdir()) if r.is_dir() and r.name.startswith("run_")]
            if run_dirs:
                leaf_dirs.extend(run_dirs)
            else:
                leaf_dirs.append(sd)

        total_collected = 0
        shell_total = int(max_shells) if max_shells and max_shells > 0 else None
        pbar_shells = tqdm(total=shell_total, desc="Loading shells", unit="shell", disable=not verbose)

        for ld in leaf_dirs:
            if max_shells and total_collected >= max_shells:
                break

            # params.yml can live in the run dir or the parent cosmo dir
            params_yml = ld / "params.yml"
            if not params_yml.exists():
                params_yml = ld.parent / "params.yml"

            low_npz = ld / low_name
            high_npz = ld / high_name

            if not (params_yml.exists() and low_npz.exists() and high_npz.exists()):
                continue

            params = yaml.safe_load(params_yml.read_text())
            keys = sorted(params.keys())
            cosmo_vec = np.array([float(params[k]) for k in keys], dtype=np.float32)

            low = np.load(low_npz, allow_pickle=False)
            high = np.load(high_npz, allow_pickle=False)
            assert "shells" in low and "shells" in high, f"NPZ must contain 'shells' in {ld}"

            low_shells = low["shells"]
            high_shells = high["shells"]
            assert low_shells.shape[0] == high_shells.shape[0], f"Mismatched shell counts in {ld}"

            for i in range(low_shells.shape[0]):
                if max_shells and total_collected >= max_shells:
                    break

                # Split each shell into patches: [n_patches, pixels_per_patch]
                low_p = split_into_patches(low_shells[i], nside_patch=nside_patch)
                high_p = split_into_patches(high_shells[i], nside_patch=nside_patch)

                low_patches_list.append(low_p.astype(np.float32))
                high_patches_list.append(high_p.astype(np.float32))
                cosmo_vecs.append(cosmo_vec)
                total_collected += 1
                pbar_shells.update(1)

        pbar_shells.close()

        assert len(low_patches_list) > 0, f"No valid shell pairs found under {data_dir}"

        # Stack: [N_shells, N_patches, D_patch]
        low_all = np.stack(low_patches_list)   # [N, P, D]
        high_all = np.stack(high_patches_list)  # [N, P, D]
        cosmo_all = np.stack(cosmo_vecs)        # [N, C]

        N, P, D = low_all.shape
        C = cosmo_all.shape[1]

        # Flatten shells × patches into individual samples: [N*P, D]
        self.low_mat = torch.from_numpy(low_all.reshape(N * P, D))
        self.high_mat = torch.from_numpy(high_all.reshape(N * P, D))

        # Calculate statistics over the input dataset
        self.data_mean = self.low_mat.mean()
        self.data_std = self.low_mat.std()
        
        # Standardize both inputs and targets to N(0, 1)
        self.low_mat = (self.low_mat - self.data_mean) / self.data_std
        self.high_mat = (self.high_mat - self.data_mean) / self.data_std
        # -------------------------

        # Repeat cosmo vector for each patch in a shell: [N*P, C]
        self.cosmo_mat = torch.from_numpy(
            np.repeat(cosmo_all, P, axis=0)  # [N*P, C]
        )

        self.sample_dim = D
        self.n_shells = N
        self.n_patches_per_shell = P

        if verbose:
            print(
                f"Dataset ready: {N} shells × {P} patches = {len(self)} samples | "
                f"patch_dim={D} | cosmo_dim={C} | nside_patch={nside_patch}"
            )

    def __len__(self):
        return self.low_mat.shape[0]

    def __getitem__(self, idx):
        return self.low_mat[idx], self.high_mat[idx], self.cosmo_mat[idx]


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
        nside_patch=args.nside_patch,
        max_shells=args.max_shells,
        verbose=is_main_process(),
    )

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
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
        print(f"dim_in (patch size): {dim_in}")
        print(f"Effective batch size: {args.batch_size * world_size}")

    # ----------------------------------------------------------------
    # 4. Training loop
    # ----------------------------------------------------------------
    loss_history = []
    for ep in range(epochs):
        t0 = time.time()
        running = 0.0
        step_count = 0

        if sampler is not None:
            sampler.set_epoch(ep)

        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", unit="step", disable=not is_main_process())
        for i, (x0, x1, cosmo) in enumerate(pbar):
            x0 = x0.to(device, non_blocking=True)       # [B, D]
            x1 = x1.to(device, non_blocking=True)       # [B, D]
            cosmo = cosmo.to(device, non_blocking=True)  # [B, C]

            B = x0.shape[0]
            t = torch.rand(B, device=device)
            mu_t = t.view(-1, 1) * x1 + (1 - t).view(-1, 1) * x0
            eps = torch.randn_like(x0) * sigma
            xt = mu_t + eps
            ut = x1 - x0  # conditional vector field target

            pred = model(xt, t, cond=cosmo)
            loss = mse(pred, ut)

            opt.zero_grad()
            loss.backward()
            opt.step()

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

        state_dict = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save(state_dict, out / "flow_mlp.pth")
        print("Saved model to", out / "flow_mlp.pth")

        # Save metadata for inference (needed to reconstruct patches → full map)
        metadata = {
            "nside_patch": args.nside_patch,
            "sample_dim": dim_in,
            "cond_dim": cond_dim,
            "hidden": args.hidden,
            "data_mean": ds.data_mean.item(),
            "data_std": ds.data_std.item(),
        }
        torch.save(metadata, out / "metadata.pth")

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
    parser.add_argument('--nside-patch', type=int, default=16,
                        help='Nside for patch grid. With nside_full=2048 and nside_patch=16, '
                             'each patch has (2048/16)^2 = 16384 pixels. '
                             'Use 8 for larger patches (65536 px) or 32 for smaller (4096 px).')
    parser.add_argument('--max-shells', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size in patches (not shells). With 3072 patches/shell, '
                             'batch_size=64 means ~0.02 shells per step.')
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
