#!/usr/bin/env python3
"""Train the EDM conditional diffusion model (see model.py) on patches from
make_patch_dataset.py. Same DDP/checkpoint infra as unet/train_flow.py (deliberate
local duplicate of its structure, see feedback-decoupled-pipeline-modules memory) --
only the loss (edm_loss instead of flow-matching MSE) and the model (DenoiserUNet+
EDMPrecond instead of FlowUNet) differ.

Single GPU:
    python train_diffusion.py --patch-dir <patch dir> --out-dir <run dir>

Multi-GPU (single node, DDP via torchrun - see run_diffusion.sh):
    python -m torch.distributed.run --standalone --nproc_per_node=4 train_diffusion.py ...
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PatchDataset, split_by_cosmo, transform_pair, cosmo_z_vector
from model import DenoiserUNet, EDMPrecond, edm_loss, residual_target, cutoff_from_chi


def is_distributed():
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def reduce_mean(value: float, device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


@torch.no_grad()

def _hp_cut(batch, dtype, device, hp_scale_mpc_h):
    """Scalar angular cutoff, or one per patch from its shell's comoving distance
    when --hp-scale-mpc-h is set (see model.cutoff_from_chi)."""
    return cutoff_from_chi(batch["shell_com"].to(device).to(dtype),
                           batch["reso_arcmin"].to(device).to(dtype), hp_scale_mpc_h)


def estimate_sigma_data(loader, device, hp_scale_mpc_h: float,
                        space: str, n_batches: int = 8) -> float:
    """Std of the diffusion TARGET (the high-pass residual, residual_target) over a
    few real batches -- EDM's own sigma_data=0.5 default is tuned for 8-bit CIFAR
    pixel statistics and has no reason to fit this. The target here is the SMALL-SCALE
    residual (highpass(high_f-low_f)), which has far less variance than the full
    field, so measuring sigma_data on it (not on high_f) is what keeps the EDM
    preconditioning c_skip/c_out/c_in correctly scaled."""
    vals = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        low_raw = batch["low"].to(device)
        high_raw = batch["high"].to(device)
        low_f, high_f = transform_pair(low_raw, high_raw, space)
        cut = _hp_cut(batch, low_f.dtype, device, hp_scale_mpc_h)
        x1 = residual_target(low_f, high_f, cut)
        vals.append(x1.flatten())
    return torch.cat(vals).std().item()


def run_epoch(precond, loader, optimizer, device, train: bool, epoch: int,
              grad_clip: float, is_main: bool, p_mean: float, p_std: float,
              sigma_data: float, hp_scale_mpc_h: float,
              space: str, log_every: int = 100):
    precond.train(train)
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for step, batch in enumerate(loader):
            low_raw = batch["low"].to(device, non_blocking=True)
            high_raw = batch["high"].to(device, non_blocking=True)
            low_f, high_f = transform_pair(low_raw, high_raw, space)
            # cond = the FULL low map (all scales -- the denoiser needs large-scale
            # context); x1 = the high-pass residual it must generate (small scales).
            cond = low_f
            cut = _hp_cut(batch, low_f.dtype, device, hp_scale_mpc_h)
            x1 = residual_target(low_f, high_f, cut)

            cosmo = batch["cosmo"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True).to(x1.dtype)
            cosmo_z = cosmo_z_vector(cosmo, z).to(x1.dtype)

            loss = edm_loss(precond, x1, cond, sigma_data, cosmo_z=cosmo_z, p_mean=p_mean, p_std=p_std)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(precond.parameters(), grad_clip)
                optimizer.step()

            bs = x1.shape[0]
            total_loss += loss.item() * bs
            n += bs

            if train and is_main and (step + 1) % log_every == 0:
                dt = time.time() - t0
                rate = (step + 1) / dt
                eta = (n_batches - step - 1) / rate
                print(f"  [step {step + 1}/{n_batches}] loss={loss.item():.5f} "
                      f"grad_norm={grad_norm:.3f} ({rate:.2f} it/s/rank, eta {eta / 60:.1f} min)", flush=True)

    mean_loss = total_loss / n
    if is_distributed():
        mean_loss = reduce_mean(mean_loss, device)
    return mean_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--noise-emb-dim", type=int, default=128)
    p.add_argument("--use-cosmo-cond", action=argparse.BooleanOptionalAction, default=True,
                   help="condition DenoiserUNet on cosmology + shell redshift at the "
                        "bottleneck (see model.DenoiserUNet). Default: on.")
    # High-pass residual formulation (see model.py docstring): the model diffuses only
    # highpass(high_f-low_f); large scales below the cutoff are supplied by the low
    # map at compose time. MUST be saved into the checkpoint (it is, via
    # args.json/ckpt["args"]) so apply_diffusion.py composes with the exact same
    # cutoff the target was built with -- it reads the scale from there and refuses a
    # checkpoint that lacks it, rather than taking its own flag.
    # THE space the model works in. 'delta' (linear overdensity n/<n>-1) is the
    # space analysis.full_sky.od_cl actually measures, and is the default after the
    # 2026-07-18 finding that training on 'log1p' optimizes a statistic that is
    # already ~correct while leaving the evaluated one broken -- see
    # dataset.raw_to_delta_pair for the measured numbers. 'log1p' reproduces the old
    # (broken) formulation for comparison only.
    p.add_argument("--space", choices=["delta", "log1p"], default="delta",
                   help="field the residual is modelled in (default: delta, the space "
                        "od_cl measures)")
    # The ONLY high-pass knob. The fixed angular --hp-cutoff/--hp-transition pair
    # this replaces is gone: an angular cutoff can only be correct for one shell,
    # and the raised-cosine ramp starved exactly the band (l ~ 100-400) the maps
    # were most deficient in. See model.cutoff_from_chi.
    p.add_argument("--hp-scale-mpc-h", type=float, default=17.0,
                   help="comoving scale (Mpc/h) below which the model corrects. The\n"
                        "cutoff multipole is 2*pi*chi/L, so it tracks each shell. "
                        "17.0 is the measured onset of the particle-mesh deficit "
                        "(17.0 +- 1.1 Mpc/h over 25 shells).")
    p.add_argument("--sigma-data", type=float, default=None,
                   help="EDM preconditioning scale (see model.EDMPrecond). Default: "
                        "measure it from --sigma-data-batches real training batches "
                        "instead of guessing (EDM's CIFAR-tuned 0.5 default has no "
                        "reason to fit log1p(overdensity) patches).")
    p.add_argument("--sigma-data-batches", type=int, default=8)
    p.add_argument("--p-mean", type=float, default=-1.2,
                   help="EDM training noise schedule: ln(sigma) ~ N(p_mean, p_std^2)")
    p.add_argument("--p-std", type=float, default=1.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-lr-scaling", action="store_true")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    distributed = is_distributed()
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)
    is_main = rank == 0

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    train_idx, val_idx, val_cosmos = split_by_cosmo(args.patch_dir, args.val_frac, args.seed)
    if is_main:
        print(f"[train_diffusion] {len(train_idx)} train patches, {len(val_idx)} val patches "
              f"({len(val_cosmos)} held-out cosmologies: {val_cosmos})")
        if distributed:
            print(f"[train_diffusion] distributed: world_size={world_size}, "
                  f"global batch={args.batch_size * world_size}")

    train_ds = PatchDataset(args.patch_dir, train_idx)
    val_ds = PatchDataset(args.patch_dir, val_idx)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                            shuffle=True, seed=args.seed, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                   drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    sigma_data = args.sigma_data
    if sigma_data is None:
        sigma_data = estimate_sigma_data(train_loader, device, args.hp_scale_mpc_h, args.space,
                                          args.sigma_data_batches)
        if distributed:
            sigma_data = reduce_mean(sigma_data, device)  # average the per-rank estimates
        if is_main:
            print(f"[train_diffusion] measured sigma_data={sigma_data:.4f} from "
                  f"{args.sigma_data_batches} training batches", flush=True)
    args.sigma_data = sigma_data  # so it lands in args.json / the checkpoint for apply_diffusion.py

    if is_main:
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    lr = args.lr if args.no_lr_scaling else args.lr * world_size
    net = DenoiserUNet(in_channels=2, out_channels=1, base_channels=args.base_channels,
                       noise_emb_dim=args.noise_emb_dim,
                       use_cosmo_cond=args.use_cosmo_cond).to(device)
    precond = EDMPrecond(net, sigma_data=sigma_data).to(device)
    if distributed:
        precond = DDP(precond, device_ids=[local_rank])
    optimizer = torch.optim.Adam(precond.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        (precond.module if distributed else precond).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        if is_main:
            print(f"[train_diffusion] resumed from {args.resume} at epoch {start_epoch}")

    log_path = out_dir / "train_log.jsonl"
    # see train_flow.py's identical guard: a fresh (non-resumed) run must not APPEND
    # onto a stale log left by an earlier run that reused this --out-dir.
    if not args.resume and is_main and log_path.exists():
        log_path.unlink()
    n_params = sum(p.numel() for p in (precond.module if distributed else precond).parameters())
    if is_main:
        print(f"[train_diffusion] model has {n_params:,} parameters, device={device}, lr={lr:.2e}, "
              f"use_cosmo_cond={args.use_cosmo_cond}, sigma_data={sigma_data:.4f}, "
              f"space={args.space}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = run_epoch(precond, train_loader, optimizer, device, True, epoch,
                                args.grad_clip, is_main, args.p_mean, args.p_std, sigma_data,
                                args.hp_scale_mpc_h, args.space)
        val_loss = run_epoch(precond, val_loader, optimizer, device, False, epoch,
                              args.grad_clip, is_main, args.p_mean, args.p_std, sigma_data,
                              args.hp_scale_mpc_h, args.space)
        scheduler.step()
        dt = time.time() - t0

        if not is_main:
            continue

        row = {"epoch": epoch, "time_s": dt, "lr": scheduler.get_last_lr()[0],
               "train_loss": train_loss, "val_loss": val_loss}
        print(f"[epoch {epoch}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} ({dt:.1f}s)", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        ckpt = {
            "model": (precond.module if distributed else precond).state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_val": best_val, "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, out_dir / "best.pt")

    if is_main:
        print(f"[train_diffusion] done. best val_loss={best_val:.5f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
