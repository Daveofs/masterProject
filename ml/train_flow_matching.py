#!/usr/bin/env python3
"""Minimal conditional Flow-Matching trainer for HEALPix shells (proof-of-concept).

- Loads `params.yml` from a user-specified data directory as conditioning vector.
- Loads paired maps from two NPZ files (low-res and high-res).
- Downsamples both to a small `nside_small` for fast experiments.
- Trains a small MLP to regress the conditional vector field u_t(x|z) = x1 - x0
  following the flow-matching tutorial (toy-style training loop).

This is intentionally small/safe for local smoke tests. Use bigger nside/batches for real runs.
"""

import argparse
from pathlib import Path
import yaml
import time

import numpy as np
import healpy as hp
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from mlp import SmallMLP


class ShellPairsDataset(Dataset):
    """Loads paired shells from either a single data directory or multiple `cosmo_*` subdirectories.

    If `data_dir` contains subfolders named like `cosmo_0000001` each must contain:
      - `params.yml`
      - low-res npz (default `shells_nside=512.npz`)
      - high-res npz (default `compressed_shells.npz`)

    The dataset aggregates all shells across subfolders and attaches the corresponding
    cosmology vector to each shell.
    """

    def __init__(
        self,
        data_dir: Path,
        low_name: str = "shells_nside=512.npz",
        high_name: str = "shells_nside=512_noisy_shuffle.npz",
        nside_small: int = 32,
        max_shells: int = 0,
        n_patches: int = 1,
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
        pbar_shells = tqdm(total=shell_total, desc="Loading shells", unit="shell")
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

        self.nshells = len(low_maps)
        self.nside_small = int(nside_small)
        assert hp.isnsideok(self.nside_small), "nside_small invalid"
        self.npix_small = hp.nside2npix(self.nside_small)
        self.n_patches = int(n_patches)
        if self.n_patches < 1:
            raise ValueError("n_patches must be >= 1")
        if self.npix_small % self.n_patches != 0:
            raise ValueError(
                f"n_patches={self.n_patches} does not evenly divide npix_small={self.npix_small}"
            )
        self.patch_npix = self.npix_small // self.n_patches
        self.sample_dim = self.patch_npix if self.n_patches > 1 else self.npix_small

        # TODO: Remove, when using on cluster
        # Precompute downsampled maps into RAM (small for smoke tests)
        self.low_down = np.empty((self.nshells, self.npix_small), dtype=np.float32)
        self.high_down = np.empty((self.nshells, self.npix_small), dtype=np.float32)
        self.cosmo_mat = np.empty((self.nshells, len(cosmo_vecs[0])), dtype=np.float32)

        for i in range(self.nshells):
            m_low = low_maps[i]
            m_high = high_maps[i]

            # downsample: for low map use averaging (density-like), for high map use count-preserving
            self.low_down[i] = hp.ud_grade(m_low, nside_out=self.nside_small, power=0).astype(np.float32)
            self.high_down[i] = hp.ud_grade(m_high, nside_out=self.nside_small, power=-2).astype(np.float32)
            self.cosmo_mat[i] = cosmo_vecs[i]

    def __len__(self):
        return int(self.nshells)

    def __getitem__(self, idx):
        # return low, high and cosmo vector
        if self.n_patches and self.n_patches > 1:
            # split each shell into `n_patches` equal contiguous parts
            patches_low = self.low_down[idx].reshape(self.n_patches, self.patch_npix)
            patches_high = self.high_down[idx].reshape(self.n_patches, self.patch_npix)
            return patches_low, patches_high, self.cosmo_mat[idx]
        return self.low_down[idx], self.high_down[idx], self.cosmo_mat[idx]


def train(args):
    data_dir = Path(args.data_dir)
    # Build dataset by scanning `data_dir` for cosmo_* subfolders (or use data_dir directly)
    ds = ShellPairsDataset(
        data_dir,
        low_name=args.low_npz,
        high_name=args.high_npz,
        nside_small=args.nside_small,
        max_shells=args.max_shells,
        n_patches=args.n_patches,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_dim = ds.cosmo_mat.shape[1]
    dim_in = ds.sample_dim
    model = SmallMLP(dim_in=dim_in, cond_dim=cond_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    sigma = args.sigma
    epochs = args.epochs

    loss_history = []
    for ep in range(epochs):
        t0 = time.time()
        running = 0.0
        # use tqdm to show progress per-epoch
        for i, (x0_np, x1_np, cosmo_np) in enumerate(tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", unit='step')):
            # x0,x1: [B, D], cosmo: [B, C]
            x0 = x0_np.to(device) # low-res
            x1 = x1_np.to(device) # high-res
            cosmo = cosmo_np.to(device)

            # If patches were returned, shapes are [B, P, D]
            is_patches = x0.dim() == 3
            if is_patches:
                B, P, Dp = x0.shape
                x0 = x0.view(B * P, Dp)
                x1 = x1.view(B * P, Dp)
                cosmo = cosmo.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)

            # sample t and conditional vector field
            B = x0.shape[0]
            t = torch.rand(B, device=device)
            mu_t = t.view(-1, 1) * x1 + (1 - t).view(-1, 1) * x0
            eps = torch.randn_like(x0) * sigma # some noise which regula regularizes training slightly so the model doesn't overfit to the exact straight line
            xt = mu_t + eps # input point
            ut = x1 - x0 # conditional vector field

            # prepare conditioning (per-sample cosmo vector)
            pred = model(xt, t, cond=cosmo)
            # By minimizing loss, the velocity is predicted and optimized from x0 to x1
            loss = mse(pred, ut)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            loss_history.append(float(loss.item()))
            if (i + 1) % args.log_interval == 0:
                print(f"ep {ep+1}/{epochs} step {i+1}/{len(dl)} loss {running / args.log_interval:.4f}")
                running = 0.0

        print(f"Epoch {ep+1} done, time {time.time()-t0:.1f}s")

    # save a small model
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "flow_mlp.pth")
    print("Saved model to", out / "flow_mlp.pth")

    # save loss history plot
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/Users/david/testData')
    parser.add_argument('--low-npz', type=str, default='shells_nside=512_noisy_shuffle.npz')
    parser.add_argument('--high-npz', type=str, default='compressed_shells.npz')
    parser.add_argument('--nside-small', type=int, default=128)
    parser.add_argument('--max-shells', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-patches', dest='n_patches', type=int, default=4, help='Number of patches to sample per map (same size)')
    parser.add_argument('--out-dir', type=str, default='/Users/david/Library/CloudStorage/OneDrive-ETHZurich/ETH-Material/Master Project/github/models')
    parser.add_argument('--log-interval', type=int, default=10)
    args = parser.parse_args()


    train(args)
