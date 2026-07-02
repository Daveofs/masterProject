#!/usr/bin/env python3
"""Train the Spherical-Harmonic (alm) flow-matching model (flow_matching_alm.MLP).

Learns to transport low-res alms -> high-res alms via rectified flow, in PER-ELL
WHITENED space (each alm divided by s(ell)=sqrt(<Cl_low(ell)>)) so the MSE loss
weights all angular scales equally — the fix for "small + large scales don't match"
(raw alms are dominated by the monopole/low-ell). Data-parallel via torchrun.

  torchrun --nproc_per_node=4 train_flow_matching.py --data-dir ... --lmax 3000
"""

from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import healpy as hp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ShellAlmDataset import ShellAlmDataset
import flow_matching_alm as fm


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"))
        return local, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


def compute_whiten_scale(ds, lmax, n_sample=16):
    """Per-ell whitening scale from the mean Cl of a sample of HIGH alms.

    Whitening by the TARGET (high) power makes the whitened target ~unit variance
    at every ell, so the generative transport noise->target is variance-preserving
    (no per-ell amplification) — this prevents high-ell overshoot in the samples.
    """
    N_alm = (lmax + 1) * (lmax + 2) // 2
    idxs = np.linspace(0, len(ds) - 1, min(n_sample, len(ds))).astype(int)
    cls = []
    for i in idxs:
        _, x1, _ = ds[i]                       # HIGH alms
        v = x1.numpy()
        a = (v[:N_alm] + 1j * v[N_alm:]).astype(np.complex128)
        cls.append(hp.alm2cl(a, lmax=lmax))
    cl_ref = np.mean(cls, axis=0)
    return fm.whiten_scale_vector(cl_ref, lmax)          # (2*N_alm,) float32


def train(args):
    local, rank, world = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    ds = ShellAlmDataset(args.data_dir, lmax=args.lmax, verbose=is_main())
    x0_0, _, cond_0 = ds[0]
    dim_in, cond_dim = x0_0.shape[0], cond_0.shape[0]

    scale = torch.from_numpy(compute_whiten_scale(ds, args.lmax)).to(dev)  # (2N,)
    if is_main():
        print(f"[data] {len(ds)} shells | lmax={args.lmax} | dim_in={dim_in:,} | "
              f"cond_dim={cond_dim} | whiten scale [{float(scale.min()):.2e},"
              f"{float(scale.max()):.2e}]", flush=True)

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) \
        if world > 1 else None
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                    shuffle=(sampler is None), num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    model = fm.MLP(dim_in, cond_dim=cond_dim, hidden=args.hidden, lmax=args.lmax).to(dev)
    if world > 1:
        model = DDP(model, device_ids=[local])
    raw = model.module if world > 1 else model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = max(len(dl) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))

    if is_main():
        print(f"[train] {len(dl)} steps/epoch x {args.epochs} = {total_steps} | "
              f"batch/gpu={args.batch_size}", flush=True)

    step = 0
    loss_hist = []
    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        for x0, x1, cond in dl:
            x0 = (x0.to(dev, non_blocking=True) / scale)     # whiten
            x1 = (x1.to(dev, non_blocking=True) / scale)
            cond = cond.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = fm.flow_matching_loss(raw, x0, x1, cond, sigma=args.sigma)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if is_main():
                    print(f"  step {step}: non-finite, skipped", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if is_main() and step % args.log_interval == 0:
                print(f"  step {step}/{total_steps} | loss={loss.item():.5f} | "
                      f"{step/(time.time()-t0):.1f} it/s | lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
                loss_hist.append(loss.item())

    if is_main():
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save(raw.state_dict(), out / "flow_mlp.pth")
        np.savez(out / "flow_meta.npz", lmax=args.lmax, dim_in=dim_in, cond_dim=cond_dim,
                 hidden=args.hidden, whiten_scale=scale.cpu().numpy(),
                 cond_mean=ds.cond_mean.numpy(), cond_std=ds.cond_std.numpy(),
                 max_shell_idx=ds.max_shell_idx, loss_hist=np.array(loss_hist))
        print(f"[train] saved model + meta to {out}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--out-dir", default="./flow_model")
    args = p.parse_args()
    try:
        train(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
