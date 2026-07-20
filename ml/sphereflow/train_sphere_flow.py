#!/usr/bin/env python3
"""Train the DeepSphere-conv flow-matching model (sphere_flow.SphereFlowNet) on
HEALPix-superpixel patches from make_patch_dataset.py.

Formulation (direct): condition on the raw DISCO map, generate the CosmoGrid
high-res signal.
    cond = arcsinh-signal( delta_low )      delta = rho/mean_shell - 1
    x1   = arcsinh-signal( delta_high )
    x0   ~ N(0, I);  xt = (1-t) x0 + t x1;  target velocity v* = x1 - x0
    loss = MSE( v_theta(xt, t, cond, cosmo), v* )
At inference (apply_sphere_flow.py) x0 is drawn fresh and the ODE integrated to
t=1, so different noise -> different small-scale detail with learned statistics.

REBUILT 2026-07-14 on unet/train_flow.py's structure, which does not suffer the
crashes this trainer kept hitting. What changed and why:

  * SINGLE NODE, torchrun --standalone (was: 4 nodes, c10d rendezvous). The
    repeated failures (PTLTE_NOT_FOUND, "NET/OFI ... Device or resource busy",
    SIGABRT at ~step 13.2k, twice deterministically) were all CROSS-NODE
    Slingshot/libfabric errors. One node x 4 GPUs uses only intra-node NVLink
    for NCCL and never touches the fabric, removing that entire failure class
    rather than papering over it with FI_CXI_* tuning. unet trains this way and
    is stable.
  * Dataset + DistributedSampler + DataLoader (was: a hand-rolled per-rank
    infinite producer thread, each rank streaming its OWN shard of runs with
    its own RNG). With drop_last=True every rank runs an IDENTICAL number of
    batches and hits every collective in lockstep -- the rank-desync that made
    one straggler hang the whole job is now structurally impossible, so the
    hand-rolled collective non-finite-loss guard is gone too.
  * Reads come off a compact patch memmap instead of random-seeking 14 GB
    per-run shell stacks (that Lustre thrash was the other half of the problem).
  * Real train/VAL split by cosmology, per-epoch last.pt/best.pt + train_log.jsonl
    -- so a crash costs one epoch, and val loss is actually visible.

  torchrun --standalone --nproc_per_node=4 train_sphere_flow.py \\
      --patch-dir <patches> --out-dir <run>
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sphere_flow as sf
from dataset import (SpherePatchDataset, split_by_cosmo, raw_to_signal_pair,
                     estimate_sig_scale)


def is_distributed():
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def reduce_mean(value: float, device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              sig_scale: float, softening: float, grad_clip: float, amp: bool,
              cmean: torch.Tensor, cstd: torch.Tensor,
              is_main: bool, log_every: int = 100):
    model.train(train)
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    total_loss, n = 0.0, 0
    n_batches = len(loader)
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for step, batch in enumerate(loader):
            low = batch["low"].to(device, non_blocking=True)            # (B, M) raw
            high = batch["high"].to(device, non_blocking=True)
            lo_m = batch["low_shell_mean"].to(device, non_blocking=True)
            hi_m = batch["high_shell_mean"].to(device, non_blocking=True)
            cosmo = batch["cosmo"].to(device, non_blocking=True)        # (B, P) raw
            shell_norm = batch["shell_norm"].to(device, non_blocking=True)
            # Cosmology standardized HERE on GPU (not in the Dataset) using the TRAIN
            # split's (mean, std) -- the same normalization apply_sphere_flow.cosmo_vector
            # reapplies at inference from meta.npz's cosmo_mean/cosmo_std. Doing it in
            # the loop keeps the Dataset a plain picklable class (a closure over
            # cmean/cstd inside it would break a spawn-based DataLoader).
            cosmo = (cosmo - cmean) / cstd
            # conditioning vector = [cosmo params, normalized shell index]
            cond_vec = torch.cat([cosmo, shell_norm[:, None]], dim=-1)

            cond, x1 = raw_to_signal_pair(low, high, lo_m, hi_m, sig_scale, softening)

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                loss = sf.flow_matching_loss(model, x1, cond, cond_vec)

            bs = x1.shape[0]
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * bs
            n += bs

            if train and is_main and (step + 1) % log_every == 0:
                dt = time.time() - t0
                rate = (step + 1) / dt
                eta = (n_batches - step - 1) / rate
                print(f"  [step {step + 1}/{n_batches}] loss={loss.item():.5f} "
                      f"grad_norm={grad_norm:.3f} ({rate:.2f} it/s/rank, "
                      f"eta {eta / 60:.1f} min)", flush=True)

    mean_loss = total_loss / max(n, 1)
    if is_distributed():
        mean_loss = reduce_mean(mean_loss, device)
    return mean_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True,
                   help="make_patch_dataset.py output (low.npy/high.npy/metadata.npy/cosmo.npy)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32,
                   help="per-GPU batch. Patches are 16,384 px at order=16 (much bigger "
                        "than unet's 256x256 images), so this is far smaller than unet's.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-cosmos", nargs="*", default=None,
                   help="Pin the held-out cosmologies explicitly instead of drawing them "
                        "via --val-frac/--seed (e.g. to match another pipeline's split).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--softening", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-lr-scaling", action="store_true",
                   help="by default lr is scaled by world_size (same as unet/train_flow.py)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast (the gather-based ChebConv supports it).")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    torch.manual_seed(args.seed)

    train_idx, val_idx, val_cosmos = split_by_cosmo(
        args.patch_dir, args.val_frac, args.seed, val_cosmos=args.test_cosmos)

    # Patch/model geometry comes from the DATASET, not from flags -- the graph
    # Laplacian must be built for exactly the patch size the data was cut at, so
    # letting them be set independently is just a way to silently mismatch them.
    meta_arr = np.load(Path(args.patch_dir) / "metadata.npy")
    nside = int(meta_arr[0]["nside"])
    order = int(meta_arr[0]["order"])
    npix_patch = int(np.load(Path(args.patch_dir) / "low.npy", mmap_mode="r").shape[1])
    if npix_patch != sf.patch_npix(nside, order):
        raise SystemExit(f"patch dataset has {npix_patch} px/patch but nside={nside}, "
                         f"order={order} implies {sf.patch_npix(nside, order)}")

    # Cosmology normalization from the TRAIN split only (validation cosmologies must
    # not influence the model's input scaling).
    cosmo_all = np.load(Path(args.patch_dir) / "cosmo.npy")
    ctrain = cosmo_all[train_idx].astype(np.float64)
    cmean = ctrain.mean(0)
    cstd = np.where(ctrain.std(0) < 1e-8, 1.0, ctrain.std(0))
    cond_dim = cosmo_all.shape[1] + 1                      # + normalized shell index

    sig_scale = estimate_sig_scale(args.patch_dir, train_idx, softening=args.softening,
                                   seed=args.seed)

    if is_main:
        print(f"[train] {len(train_idx):,} train / {len(val_idx):,} val patches "
              f"({len(val_cosmos)} held-out cosmologies: {val_cosmos})", flush=True)
        print(f"[train] nside={nside} order={order} | {npix_patch} px/patch | "
              f"cond_dim={cond_dim} | sig_scale={sig_scale:.4g} | world={world_size}",
              flush=True)

    # Applied on GPU inside run_epoch (see there) -- the Dataset stays a plain,
    # picklable class returning raw values.
    cmean_t = torch.from_numpy(cmean.astype(np.float32)).to(device)
    cstd_t = torch.from_numpy(cstd.astype(np.float32)).to(device)

    train_ds = SpherePatchDataset(args.patch_dir, train_idx)
    val_ds = SpherePatchDataset(args.patch_dir, val_idx)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                           shuffle=True, seed=args.seed, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                  drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                                drop_last=True, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                drop_last=True, **loader_kwargs)

    lr = args.lr if args.no_lr_scaling else args.lr * world_size
    L = sf.healpix_laplacian(nside, order=order)
    model = sf.SphereFlowNet(L, cond_dim=cond_dim, hidden=args.hidden,
                             n_layers=args.n_layers, K=args.K).to(device)
    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_mod = model.module if distributed else model
    raw_mod = getattr(raw_mod, "_orig_mod", raw_mod)       # unwrap torch.compile

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_mod.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        if is_main:
            print(f"[train] resumed from {args.resume} at epoch {start_epoch}", flush=True)

    if is_main:
        n_params = sum(p.numel() for p in raw_mod.parameters())
        print(f"[train] model has {n_params:,} parameters, device={device}, lr={lr:.2e}",
              flush=True)

    def save(path, epoch, best):
        torch.save({"model": raw_mod.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_val": best, "args": vars(args)}, path)
        # meta.npz: everything apply_sphere_flow.py needs to rebuild + condition the
        # net, written next to every checkpoint so a saved model is never separated
        # from the normalization/geometry it was trained with.
        np.savez(out_dir / "meta.npz", nside=nside, order=order, K=args.K,
                 hidden=args.hidden, n_layers=args.n_layers, cond_dim=cond_dim,
                 sig_scale=sig_scale, resid_scale=1.0, softening=args.softening,
                 formulation="direct", cosmo_mean=cmean, cosmo_std=cstd,
                 test_cosmos=np.array(sorted(val_cosmos)),
                 # 2026-07-20: patches are now drawn at random (lon,lat,psi) via
                 # sphere_flow.rotated_patch_ids, not the old disjoint quad-tree
                 # blocks -- apply_sphere_flow.py uses this marker to pick the
                 # overlap-blend reconstruction path (a checkpoint trained this
                 # way has never seen the old fixed alignment, and vice versa;
                 # see sphere_flow.py's "OVERLAPPING patch geometry" section).
                 patch_mode="overlap")

    log_path = out_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, True, epoch,
                               sig_scale, args.softening, args.grad_clip, args.amp,
                               cmean_t, cstd_t, is_main)
        val_loss = run_epoch(model, val_loader, optimizer, device, False, epoch,
                             sig_scale, args.softening, args.grad_clip, args.amp,
                             cmean_t, cstd_t, is_main)
        scheduler.step()
        dt = time.time() - t0

        if not is_main:
            continue
        row = {"epoch": epoch, "time_s": dt, "lr": scheduler.get_last_lr()[0],
               "train_loss": train_loss, "val_loss": val_loss}
        print(f"[epoch {epoch}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
              f"({dt:.1f}s)", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        save(out_dir / "last.pt", epoch, best_val)
        if val_loss < best_val:
            best_val = val_loss
            save(out_dir / "best.pt", epoch, best_val)
            # apply_sphere_flow.py loads sphere_flow.pth -- keep it pointing at the
            # BEST epoch, not merely the last one.
            torch.save(raw_mod.state_dict(), out_dir / "sphere_flow.pth")

    if is_main:
        print(f"[train] done. best val_loss={best_val:.5f} -> {out_dir}", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
