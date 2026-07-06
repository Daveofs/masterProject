"""DDP trainer for the simple 2D-UNet flow-matching generator (unet_flow.py).

Reuses the EXACT data path of the sphere-flow model: build_runs + ResidualStreamer
(run-major streaming, background producer, per-shell cosmology+redshift conditioning,
arcsinh signal transform) from train_sphere_flow.py. The only difference is that each
1D HEALPix patch (B, M) is reshaped to a 2D image (B, 1, L, L) via the Morton map so a
plain 2D UNet can consume it.

  torchrun --nnodes N --nproc_per_node 4 train_unet_flow.py \
      --data-root <grid> --test-cosmo cosmo_000122 --include-test \
      --nside 2048 --order 16 --base 64 --epochs 8 --batch-size 128 --out-dir <dir>
"""

from __future__ import annotations
import argparse
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sphere_flow as sf
import unet_flow as uf
from train_sphere_flow import build_runs, estimate_scales, ResidualStreamer


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"),
                                timeout=timedelta(minutes=10))
        return local, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


def train(args):
    local, rank, world = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    runs = build_runs(args.data_root, args.test_cosmo, args.nside,
                      args.include_test, prefix="low")
    if not runs:
        raise RuntimeError("no prepared runs found (need low_/high_shells npy)")
    my_runs = runs[rank::world] or [runs[rank % len(runs)]]
    cond_dim = len(runs[0][2]) + 1          # cosmo params + normalized shell index
    L = args.nside // args.order            # patch side (128 at nside2048/order16)

    # signal scale (shared convention with sphere_flow); direct formulation:
    # x1 = signal(delta_high), cond = signal(delta_low).
    sig_scale, resid_scale = estimate_scales(my_runs, args.nside, args.order,
                                             args.softening, seed=rank)
    if world > 1:                            # average the scale across ranks
        t = torch.tensor([sig_scale], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.AVG); sig_scale = float(t.item())

    streamer = ResidualStreamer(my_runs, args.nside, args.order, sig_scale,
                                resid_scale, softening=args.softening,
                                patch_frac=args.patch_frac,
                                formulation="direct", seed=rank)

    # Morton reindex (nested patch -> 2D image) as a device buffer.
    to_img, _ = uf.nested_grid_perms(L)
    to_img = torch.from_numpy(to_img).to(dev)

    net = uf.SimpleUNet(in_ch=2, out_ch=1, base=args.base,
                        ch_mult=tuple(int(m) for m in args.ch_mult.split(",")),
                        cond_dim=cond_dim, time_dim=args.time_dim).to(dev)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_dir) / "checkpoint.pt"
    ckpt = None
    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ckpt["model"])
        if is_main():
            print(f"[resume] loaded {ckpt_path} at step {ckpt['step']:,}", flush=True)

    if world > 1:
        net = DDP(net, device_ids=[local])
    raw = net.module if world > 1 else net

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    ppshell = int(sf.n_patches(args.order) * args.patch_frac) if args.order > 1 else 1
    steps_per_epoch = max(streamer.n_shells * ppshell // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))

    start_step, ema = 0, None
    hist_path = Path(args.out_dir) / "loss_history.npy"
    loss_hist = list(map(tuple, np.load(hist_path))) if hist_path.exists() else []
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"])
        start_step = int(ckpt["step"]); ema = ckpt.get("ema")

    def save_ckpt(step):
        tmp = ckpt_path.with_suffix(f".tmp{os.getpid()}")
        torch.save({"step": step, "model": raw.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ema": ema,
                    "cond_dim": cond_dim, "L": L, "sig_scale": sig_scale,
                    "base": args.base, "ch_mult": args.ch_mult,
                    "time_dim": args.time_dim, "order": args.order,
                    "nside": args.nside, "softening": args.softening}, tmp)
        os.replace(tmp, ckpt_path)

    if is_main():
        nparam = sum(p.numel() for p in raw.parameters())
        print(f"[unet-flow] {len(my_runs)} runs/rank | {world} ranks | cond_dim={cond_dim} "
              f"| patch {L}x{L} | {nparam/1e6:.2f}M params | sig_scale={sig_scale:.4f}",
              flush=True)
        print(f"[train] ~{steps_per_epoch:,} steps/epoch x {args.epochs} = {total_steps:,} "
              f"| batch/gpu={args.batch_size} | start_step={start_step:,}", flush=True)

    net.train()
    it = streamer.batches(args.batch_size, dev)
    t0 = time.time(); last_t, last_step = t0, start_step

    for step in range(start_step + 1, total_steps + 1):
        x1, cond, cosmo = next(it)                      # (B,M),(B,M),(B,cond_dim)
        B = x1.shape[0]
        x1 = x1[:, to_img].view(B, 1, L, L)             # nested patch -> image
        cond = cond[:, to_img].view(B, 1, L, L)

        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = uf.flow_matching_loss(net, x1, cond, cosmo)

        # Collective non-finite guard: all ranks skip together if ANY loss is
        # non-finite (a lone rank skipping backward would desync the grad all-reduce).
        finite = torch.isfinite(loss).to(torch.float32).detach()
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() < 1.0:
            opt.zero_grad(set_to_none=True); sched.step()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if is_main():
            lv = loss.item()
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            if step % args.log_every == 0:
                now = time.time()
                inst = (step - last_step) / max(now - last_t, 1e-9)
                eta = (total_steps - step) / max(inst, 1e-9) / 3600.0
                print(f"  step {step:,}/{total_steps:,} | loss={lv:.4f} ema={ema:.4f} "
                      f"| {inst:.2f} steps/s | ETA {eta:.1f}h | lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
                last_t, last_step = now, step
                loss_hist.append((step, lv, ema))       # flow-matching loss curve
            if step % args.ckpt_every == 0:
                save_ckpt(step)
                np.save(hist_path, np.array(loss_hist, dtype=np.float64))

    if is_main():
        save_ckpt(total_steps)
        np.save(hist_path, np.array(loss_hist, dtype=np.float64))
        torch.save(raw.state_dict(), Path(args.out_dir) / "unet_flow.pth")
        print(f"[train] done -> {args.out_dir}", flush=True)
    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000122")
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--ch-mult", default="1,2,2")
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--softening", type=float, default=1.0)
    p.add_argument("--patch-frac", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out-dir", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
