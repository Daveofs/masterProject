#!/usr/bin/env python3
"""Conditional Flow-Matching trainer for HEALPix shells in Spherical Harmonic (Alm) space.

- Converts maps to complex Alm coefficients up to a chosen lmax.
- Stacks [real, imag] into a real 1D vector of length 2 * N_alm.
- Uses feature-wise Z-score normalization to naturally whiten the power spectrum.
- Integrates a clean YAML parser to whitelist true cosmological parameters.
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


def load_clean_params(params_path: Path):
    """Safely extracts cosmological floats, ignoring SLURM IDs, paths, and seeds."""
    params = yaml.safe_load(params_path.read_text())
    bad_subphrases = ["seed", "job", "part", "box", "step", "nside", "path", "dir", "file", "rank", "node", "gpu", "time"]
    
    valid_keys = []
    for k, v in sorted(params.items()):
        if any(b in k.lower() for b in bad_subphrases):
            continue
        try:
            float(v)
            valid_keys.append(k)
        except (ValueError, TypeError):
            continue

    vec = np.array([float(params[k]) for k in valid_keys], dtype=np.float32)
    return vec, valid_keys, params


class ShellAlmDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        low_name: str = "shells_nside=2048.npz",
        high_name: str = "compressed_shells.npz",
        lmax: int = 1024,
        max_shells: int = 0,
        verbose: bool = True,
    ):
        data_dir = Path(data_dir)
        assert data_dir.exists(), f"data_dir not found: {data_dir}"
        self.lmax = lmax
        self.N_alm = hp.Alm.getsize(lmax)

        low_list = []
        high_list = []
        cosmo_list = []

        subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name.startswith("cosmo_")]
        if len(subdirs) == 0:
            subdirs = [data_dir]

        leaf_dirs = []
        for sd in subdirs:
            run_dirs = [r for r in sorted(sd.iterdir()) if r.is_dir() and r.name.startswith("run_")]
            if run_dirs:
                leaf_dirs.extend(run_dirs)
            else:
                leaf_dirs.append(sd)

        total_collected = 0
        pbar = tqdm(total=max_shells if max_shells > 0 else None, desc="Transforming maps -> Alms", disable=not verbose)

        param_names = None
        for ld in leaf_dirs:
            if max_shells and total_collected >= max_shells:
                break

            params_yml = ld / "params.yml"
            if not params_yml.exists():
                params_yml = ld.parent / "params.yml"

            low_npz = ld / low_name
            high_npz = ld / high_name

            if not (params_yml.exists() and low_npz.exists() and high_npz.exists()):
                continue

            cosmo_vec, p_names, raw_dict = load_clean_params(params_yml)
            if param_names is None and verbose:
                param_names = p_names
                print(f"\n[YAML Parser] Active Conditioning Vector ({len(p_names)} params):")
                for k in p_names:
                    print(f"   {k}: {raw_dict[k]}")

            low = np.load(low_npz, allow_pickle=False)["shells"]
            high = np.load(high_npz, allow_pickle=False)["shells"]

            for i in range(low.shape[0]):
                if max_shells and total_collected >= max_shells:
                    break

                alm_low = hp.map2alm(low[i], lmax=lmax, iter=1)
                alm_high = hp.map2alm(high[i], lmax=lmax, iter=1)

                vec_low = np.concatenate([alm_low.real, alm_low.imag]).astype(np.float32)
                vec_high = np.concatenate([alm_high.real, alm_high.imag]).astype(np.float32)

                low_list.append(vec_low)
                high_list.append(vec_high)
                cosmo_list.append(cosmo_vec)

                total_collected += 1
                pbar.update(1)

        pbar.close()
        assert len(low_list) > 0, "No valid shell pairs found."

        self.low_mat = torch.from_numpy(np.stack(low_list))     # [N_shells, 2 * N_alm]
        self.high_mat = torch.from_numpy(np.stack(high_list))
        self.cosmo_mat = torch.from_numpy(np.stack(cosmo_list)) # [N_shells, cond_dim]

        # 1. Feature-wise Z-score Whitening across the Harmonic spectrum
        self.data_mean = self.low_mat.mean(dim=0)
        self.data_std = self.low_mat.std(dim=0).clamp(min=1e-8)

        self.low_mat = (self.low_mat - self.data_mean) / self.data_std
        self.high_mat = (self.high_mat - self.data_mean) / self.data_std

        # 2. Standardize Conditioning Vector
        self.cosmo_mean = self.cosmo_mat.mean(dim=0)
        self.cosmo_std = self.cosmo_mat.std(dim=0).clamp(min=1e-8)
        self.cosmo_mat = (self.cosmo_mat - self.cosmo_mean) / self.cosmo_std

        if verbose:
            print(f"Dataset ready: {len(self)} complete shells | Vector dim: {self.low_mat.shape[1]} | Cond dim: {self.cosmo_mat.shape[1]}")

    def __len__(self):
        return self.low_mat.shape[0]

    def __getitem__(self, idx):
        return self.low_mat[idx], self.high_mat[idx], self.cosmo_mat[idx]


def train(args):
    local_rank, rank, world_size = setup_distributed()
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ds = ShellAlmDataset(
        args.data_dir,
        low_name=args.low_npz,
        high_name=args.high_npz,
        lmax=args.lmax,
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
        drop_last=False,
    )

    cond_dim = ds.cosmo_mat.shape[1]
    dim_in = ds.low_mat.shape[1]
    model = SmallMLP(dim_in=dim_in, cond_dim=cond_dim, hidden=args.hidden).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    if is_main_process():
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    loss_history = []
    for ep in range(args.epochs):
        t0 = time.time()
        running = 0.0
        step_count = 0

        if sampler is not None:
            sampler.set_epoch(ep)

        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{args.epochs}", unit="step", disable=not is_main_process())
        for x0, x1, cosmo in pbar:
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            cosmo = cosmo.to(device, non_blocking=True)

            B = x0.shape[0]
            t = torch.rand(B, 1, device=device)
            mu_t = t * x1 + (1 - t) * x0
            eps = torch.randn_like(x0) * args.sigma
            xt = mu_t + eps
            ut = x1 - x0

            pred = model(xt, t.squeeze(1), cond=cosmo)
            loss = mse(pred, ut)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            step_count += 1
            loss_history.append(loss.item())

            if is_main_process() and step_count % args.log_interval == 0:
                pbar.set_postfix(loss=f"{running / step_count:.4f}")

        if is_main_process():
            print(f"Epoch {ep+1}/{args.epochs} done | avg loss: {running / step_count:.6f} | time: {time.time() - t0:.1f}s")

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
            "data_mean": ds.data_mean.cpu(),
            "data_std": ds.data_std.cpu(),
            "cosmo_mean": ds.cosmo_mean.cpu(),
            "cosmo_std": ds.cosmo_std.cpu(),
        }
        torch.save(metadata, out / "metadata.pth")
        print(f"Saved model and whitening metadata to {out}")
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(np.arange(len(loss_history)), loss_history, marker='.', linewidth=0.5)
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

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/Users/david/testData")
    parser.add_argument("--low-npz", type=str, default="shells_nside=2048.npz")
    parser.add_argument("--high-npz", type=str, default="compressed_shells.npz")
    parser.add_argument("--lmax", type=int, default=1024, help="Maximum spherical harmonic multipole degree.")
    parser.add_argument("--max-shells", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size in full shells.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="./models")
    parser.add_argument("--log-interval", type=int, default=5)
    train(parser.parse_args())
