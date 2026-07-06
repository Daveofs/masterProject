"""DDP trainer for the deterministic residual-correction UNet (unet_diff.py).

Reuses the sphere-flow data path (build_runs + ResidualStreamer, formulation="direct":
cond = signal(DISCO), x1 = signal(high)). The per-shell PREPROCESSING DIFFERENCE is
    diff = x1 - cond = signal(high) - signal(DISCO)
and the model regresses it; corrected = cond + pred. Loss = pixel MSE (== MSE(pred, diff)).

VALIDATION CURVE: a few runs are HELD OUT from training (never contribute gradients).
Every --val-every steps we evaluate the same MSE loss on fixed validation batches. The
validation curve is that held-out loss vs step: if it tracks the training loss the model
generalizes; if it flattens/rises while training loss keeps dropping, it is overfitting.

  torchrun --nnodes N --nproc_per_node 4 train_unet_diff.py \
      --data-root <grid> --test-cosmo cosmo_000122 --nside 2048 --order 16 \
      --epochs 8 --batch-size 128 --out-dir <dir>
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# robust imports regardless of cwd: this dir (unet/) + the sphereflow/ helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sphereflow"))
import sphere_flow as sf
import unet_diff as ud
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


class DownscaledStreamer(ResidualStreamer):
    """ResidualStreamer that block-mean downscales each shell BEFORE overdensity/signal
    processing -- for fast dev iteration (smaller patches, far fewer FLOPs). factor=1
    is a no-op identical to the base class. See unet_diff.downscale_nested."""

    def __init__(self, *a, factor=1, **kw):
        super().__init__(*a, **kw)
        self.factor = factor

    def _process_shell(self, tc, hi, si, n_run, cosmo):
        if self.factor > 1:
            tc = ud.downscale_nested(tc[None], self.factor)[0]
            hi = ud.downscale_nested(hi[None], self.factor)[0]
        return super()._process_shell(tc, hi, si, n_run, cosmo)


def to_images(x1, cond, to_img, L):
    """(B, M) nested patches -> (B, 1, L, L) images; returns (cond_img, diff_target, x1_img)."""
    B = x1.shape[0]
    x1i = x1[:, to_img].view(B, 1, L, L)
    ci = cond[:, to_img].view(B, 1, L, L)
    return ci, (x1i - ci), x1i               # cond image, diff target, truth image


def train(args):
    local, rank, world = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    # Test cosmo is fully held out (never built). Of the remaining runs, the last
    # --n-val are the VALIDATION holdout; the rest are training runs.
    runs = build_runs(args.data_root, args.test_cosmo, args.nside,
                      include_test=False, prefix="low")
    if len(runs) <= args.n_val + 1:
        raise RuntimeError(f"not enough runs ({len(runs)}) for a val split")
    val_runs = runs[-args.n_val:]
    train_runs = runs[:-args.n_val]
    my_runs = train_runs[rank::world] or [train_runs[rank % len(train_runs)]]
    cond_dim = len(runs[0][2]) + 1
    L = (args.nside // args.downscale) // args.order

    # NOTE: estimated on full-res shells regardless of --downscale (arcsinh scale is
    # only mildly resolution-dependent; fine for a fast dev tool, not exact).
    sig_scale, resid_scale = estimate_scales(my_runs, args.nside, args.order,
                                             args.softening, seed=rank)
    if world > 1:
        t = torch.tensor([sig_scale], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.AVG); sig_scale = float(t.item())

    mk = lambda rr, seed: DownscaledStreamer(rr, args.nside, args.order, sig_scale,
                                             resid_scale, softening=args.softening,
                                             patch_frac=args.patch_frac,
                                             formulation="direct", seed=seed,
                                             factor=args.downscale)
    streamer = mk(my_runs, rank)

    to_img, _ = ud.nested_grid_perms(L)
    to_img = torch.from_numpy(to_img).to(dev)
    sbins, scounts, sn_bins = ud.radial_bins(L, dev)

    # FIXED validation set: pull val_batches once from the held-out runs and keep the
    # tensors. (Calling streamer.batches() repeatedly would spawn a new ~28 GB producer
    # thread each time -> host OOM. A fixed set is also the correct validation scheme:
    # the exact same batches are scored at every checkpoint, so the curve is comparable.)
    val_data = []
    if is_main():
        vit = mk(val_runs, 7000).batches(args.batch_size, dev)
        for _ in range(args.val_batches):
            x1, cond, cosmo = next(vit)
            ci, dt, x1i = to_images(x1, cond, to_img, L)
            val_data.append((ci.detach(), dt.detach(), x1i.detach(), cosmo.detach()))
        del vit                          # abandon the single producer (bounded RAM)

    net = ud.DiffUNet(in_ch=1, out_ch=1, base=args.base,
                      ch_mult=tuple(int(m) for m in args.ch_mult.split(",")),
                      bottleneck=args.bottleneck, cond_dim=cond_dim).to(dev)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_dir) / "checkpoint.pt"
    ckpt = None
    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ckpt["model"])
        if is_main():
            print(f"[resume] {ckpt_path} at step {ckpt['step']:,}", flush=True)

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
    tr_path = Path(args.out_dir) / "train_history.npy"
    va_path = Path(args.out_dir) / "val_history.npy"
    train_hist = list(map(tuple, np.load(tr_path))) if tr_path.exists() else []
    val_hist = list(map(tuple, np.load(va_path))) if va_path.exists() else []
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"])
        start_step = int(ckpt["step"]); ema = ckpt.get("ema")

    def save_ckpt(step):
        tmp = ckpt_path.with_suffix(f".tmp{os.getpid()}")
        torch.save({"step": step, "model": raw.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ema": ema, "cond_dim": cond_dim, "L": L,
                    "sig_scale": sig_scale, "base": args.base, "ch_mult": args.ch_mult,
                    "bottleneck": args.bottleneck, "order": args.order, "nside": args.nside,
                    "softening": args.softening, "val_every": args.val_every,
                    "val_batches": args.val_batches, "n_val": args.n_val,
                    "lambda_spec": args.lambda_spec, "downscale": args.downscale,
                    "ema_decay": 0.99}, tmp)
        os.replace(tmp, ckpt_path)
        np.save(tr_path, np.array(train_hist, dtype=np.float64))
        np.save(va_path, np.array(val_hist, dtype=np.float64) if val_hist else np.zeros((0, 4)))

    @torch.no_grad()
    def val_loss():
        # (combined, pixel, spectral) mean over the FIXED held-out batches.
        net.eval()
        tp = ts = 0.0
        for ci, dt, x1i, cosmo in val_data:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                pred = raw(ci, cosmo)
                pl = ud.correction_loss(pred, dt)
                sl = ud.spectral_loss(ci + pred, x1i, sbins, scounts, sn_bins)
            tp += pl.item(); ts += sl.item()
        net.train()
        n = max(len(val_data), 1)
        tp, ts = tp / n, ts / n
        return tp + args.lambda_spec * ts, tp, ts

    if is_main():
        nparam = sum(p.numel() for p in raw.parameters())
        print(f"[unet-diff] train={len(train_runs)} val={len(val_runs)} runs | {world} ranks "
              f"| bottleneck={raw.bottleneck} | patch {L}x{L} (downscale={args.downscale}) "
              f"| {nparam/1e6:.2f}M params | sig_scale={sig_scale:.4f}", flush=True)
        print(f"[train] ~{steps_per_epoch:,} steps/epoch x {args.epochs} = {total_steps:,} "
              f"| start_step={start_step:,}", flush=True)

    net.train()
    it = streamer.batches(args.batch_size, dev)
    t0 = time.time(); last_t, last_step = t0, start_step

    for step in range(start_step + 1, total_steps + 1):
        x1, cond, cosmo = next(it)
        ci, dt, x1i = to_images(x1, cond, to_img, L)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            pred = net(ci, cosmo)
            pixel_l = ud.correction_loss(pred, dt)
            spec_l = ud.spectral_loss(ci + pred, x1i, sbins, scounts, sn_bins)
            loss = pixel_l + args.lambda_spec * spec_l
        finite = torch.isfinite(loss).to(torch.float32).detach()
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() < 1.0:
            opt.zero_grad(set_to_none=True); sched.step(); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if is_main():
            lv = loss.item()
            ema = lv if ema is None else 0.99 * ema + 0.01 * lv
            if step % args.log_every == 0:
                now = time.time()
                inst = (step - last_step) / max(now - last_t, 1e-9)
                eta = (total_steps - step) / max(inst, 1e-9) / 3600.0
                print(f"  step {step:,}/{total_steps:,} | loss={lv:.4f} (pixel={pixel_l.item():.4f} "
                      f"spec={spec_l.item():.4f}) ema={ema:.4f} | {inst:.2f} steps/s "
                      f"| ETA {eta:.1f}h | lr={sched.get_last_lr()[0]:.2e}", flush=True)
                last_t, last_step = now, step
                train_hist.append((step, lv, ema))
            if step % args.val_every == 0:
                vl, vp, vs = val_loss()
                val_hist.append((step, vl, vp, vs))
                print(f"    [val] step {step:,} | val_loss={vl:.4f} (pixel={vp:.4f} "
                      f"spec={vs:.4f}) (train_ema={ema:.4f})", flush=True)
            if step % args.ckpt_every == 0:
                save_ckpt(step)

    if is_main():
        save_ckpt(total_steps)
        torch.save(raw.state_dict(), Path(args.out_dir) / "unet_diff.pth")
        print(f"[train] done -> {args.out_dir}", flush=True)
    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000122")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--downscale", type=int, default=1,
                   help="block-mean downscale factor (nside -> nside/factor) for fast "
                        "dev iterations; 1 = full resolution")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--ch-mult", default="1,2,4,8")
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--lambda-spec", type=float, default=0.5,
                   help="weight of the radial-power-spectrum loss term added to pixel MSE")
    p.add_argument("--softening", type=float, default=1.0)
    p.add_argument("--patch-frac", type=float, default=0.5)
    p.add_argument("--n-val", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out-dir", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
