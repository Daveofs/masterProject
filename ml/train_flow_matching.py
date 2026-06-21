#!/usr/bin/env python3
"""Conditional Flow-Matching trainer for HEALPix shells in Spherical Harmonic (Alm) space."""

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
        n_total_shells: int = 69,
        verbose: bool = True,
    ):
        data_dir = Path(data_dir)
        assert data_dir.exists(), f"data_dir not found: {data_dir}"
        self.lmax = lmax
        self.N_alm = hp.Alm.getsize(lmax)
        self.n_total_shells = n_total_shells

        low_list = []
        high_list = []
        cosmo_list = []
        shell_idx_list = []  # NEW: track which shell index each sample came from

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

            n_available = min(low.shape[0], high.shape[0])

            for i in range(n_available):
                if max_shells and total_collected >= max_shells:
                    break

                alm_low = hp.map2alm(low[i], lmax=lmax, iter=1)
                alm_high = hp.map2alm(high[i], lmax=lmax, iter=1)

                vec_low = np.concatenate([alm_low.real, alm_low.imag]).astype(np.float32)
                vec_high = np.concatenate([alm_high.real, alm_high.imag]).astype(np.float32)

                low_list.append(vec_low)
                high_list.append(vec_high)
                cosmo_list.append(cosmo_vec)
                shell_idx_list.append(i)  # store shell index

                total_collected += 1
                pbar.update(1)

        pbar.close()
        assert len(low_list) > 0, "No valid shell pairs found."

        self.low_mat = torch.from_numpy(np.stack(low_list))
        self.high_mat = torch.from_numpy(np.stack(high_list))
        self.cosmo_mat = torch.from_numpy(np.stack(cosmo_list))
        self.shell_indices = torch.tensor(shell_idx_list, dtype=torch.float32)

        # --- Per-shell whitening ---
        # Group by shell index and compute per-shell mean/std
        unique_shells = sorted(set(shell_idx_list))
        self.per_shell_mean = {}
        self.per_shell_std = {}

        # For simplicity, use global whitening but normalize the VELOCITY (x1-x0) instead
        # This avoids the issue of different shells having wildly different scales

        # Global statistics for the DIFFERENCE (velocity target)
        diff_mat = self.high_mat - self.low_mat
        self.vel_mean = diff_mat.mean(dim=0)
        self.vel_std = diff_mat.std(dim=0).clamp(min=1e-8)

        # Per-sample normalization: normalize each sample by its own L2 norm
        # This makes the model predict a DIRECTION + SCALE rather than raw values
        self.low_norms = self.low_mat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.high_norms = self.high_mat.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Normalize to unit-ish scale per sample
        self.low_normed = self.low_mat / self.low_norms
        self.high_normed = self.high_mat / self.high_norms

        # Store the norm ratio for reconstruction
        self.norm_ratios = (self.high_norms / self.low_norms).squeeze(1)

        # Global stats on the normalized data
        self.data_mean = self.low_normed.mean(dim=0)
        self.data_std = self.low_normed.std(dim=0).clamp(min=1e-8)

        self.low_white = (self.low_normed - self.data_mean) / self.data_std
        self.high_white = (self.high_normed - self.data_mean) / self.data_std

        # Shell index normalized to [0, 1]
        self.shell_norm = self.shell_indices / max(self.n_total_shells - 1, 1)

        # Cosmo conditioning (append shell index)
        self.cosmo_mean = self.cosmo_mat.mean(dim=0)
        self.cosmo_std = self.cosmo_mat.std(dim=0).clamp(min=1e-8)
        self.cosmo_normed = (self.cosmo_mat - self.cosmo_mean) / self.cosmo_std

        # Norm ratio stats for conditioning
        self.norm_ratio_mean = self.norm_ratios.mean()
        self.norm_ratio_std = self.norm_ratios.std().clamp(min=1e-8)

        if verbose:
            print(f"Dataset ready: {len(self)} shells | Vec dim: {self.low_white.shape[1]} | Cond dim: {self.cosmo_normed.shape[1] + 1}")
            print(f"Shell indices seen: {unique_shells}")
            print(f"Norm ratios: mean={self.norm_ratios.mean():.4f}, std={self.norm_ratios.std():.4f}, range=[{self.norm_ratios.min():.4f}, {self.norm_ratios.max():.4f}]")

    def __len__(self):
        return self.low_white.shape[0]

    def __getitem__(self, idx):
        # Conditioning = [cosmo_params..., shell_index_normalized]
        cond = torch.cat([self.cosmo_normed[idx], self.shell_norm[idx:idx+1]])
        return self.low_white[idx], self.high_white[idx], cond, self.norm_ratios[idx]


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
        n_total_shells=args.n_total_shells,
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

    # +1 for shell index in conditioning
    cond_dim = ds.cosmo_normed.shape[1] + 1
    dim_in = ds.low_white.shape[1]
    model = SmallMLP(dim_in=dim_in, cond_dim=cond_dim, hidden=args.hidden).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    mse = nn.MSELoss()

    if is_main_process():
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Dim in: {dim_in} | Cond dim: {cond_dim} | Hidden: {args.hidden}")
        print(f"Time samples per pair: {args.time_samples}")

    loss_history = []
    best_loss = float("inf")

    for ep in range(args.epochs):
        t0 = time.time()
        running = 0.0
        step_count = 0

        if sampler is not None:
            sampler.set_epoch(ep)

        model.train()
        for x0, x1, cond, norm_ratio in dl:
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)

            B = x0.shape[0]
            K = args.time_samples

            x0_exp = x0.repeat_interleave(K, dim=0)
            x1_exp = x1.repeat_interleave(K, dim=0)
            cond_exp = cond.repeat_interleave(K, dim=0)

            t = torch.rand(B * K, 1, device=device)
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

        if avg_loss < best_loss and is_main_process():
            best_loss = avg_loss
            state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, out / "flow_mlp_best.pth")

        if is_main_process() and (ep + 1) % args.log_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {ep+1}/{args.epochs} | loss: {avg_loss:.6f} | best: {best_loss:.6f} | lr: {lr_now:.2e} | {time.time()-t0:.1f}s")

    # Save
    if is_main_process():
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Use best checkpoint
        best_path = out / "flow_mlp_best.pth"
        if best_path.exists():
            import shutil
            shutil.copy2(best_path, out / "flow_mlp.pth")
            print(f"Using best checkpoint (loss={best_loss:.6f})")
        else:
            state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            torch.save(state_dict, out / "flow_mlp.pth")

        metadata = {
            "lmax": args.lmax,
            "sample_dim": dim_in,
            "cond_dim": cond_dim,
            "hidden": args.hidden,
            "n_total_shells": args.n_total_shells,
            "data_mean": ds.data_mean.cpu(),
            "data_std": ds.data_std.cpu(),
            "cosmo_mean": ds.cosmo_mean.cpu(),
            "cosmo_std": ds.cosmo_std.cpu(),
            "norm_ratio_mean": ds.norm_ratio_mean.cpu(),
            "norm_ratio_std": ds.norm_ratio_std.cpu(),
        }
        torch.save(metadata, out / "metadata.pth")
        print(f"Saved model + metadata to {out}")

        # Loss plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(loss_history, linewidth=0.5, alpha=0.7)
            axes[0].set_yscale('log')
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('MSE')
            axes[0].set_title('Per-step loss')
            axes[0].grid(True, alpha=0.3)

            spe = max(len(loss_history) // args.epochs, 1)
            epoch_losses = [np.mean(loss_history[i*spe:(i+1)*spe]) for i in range(args.epochs)]
            axes[1].plot(range(1, args.epochs+1), epoch_losses, 'o-', markersize=2)
            axes[1].set_yscale('log')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Avg MSE')
            axes[1].set_title('Per-epoch loss')
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(out / 'loss.png', dpi=150)
            plt.close(fig)
            np.save(out / 'loss.npy', np.array(loss_history))
        except Exception as e:
            print(f'Loss plot error: {e}')

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/Users/david/testData")
    parser.add_argument("--low-npz", type=str, default="shells_nside=2048.npz")
    parser.add_argument("--high-npz", type=str, default="compressed_shells.npz")
    parser.add_argument("--lmax", type=int, default=1024)
    parser.add_argument("--max-shells", type=int, default=20)
    parser.add_argument("--n-total-shells", type=int, default=69, help="Total shells in a full simulation (for index normalization).")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--time-samples", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="./models")
    parser.add_argument("--log-interval", type=int, default=5)
    train(parser.parse_args())
