"""DDP trainer: conditional flow matching on the DISCO->CosmoGrid RESIDUAL.

Why this and not the deterministic unet_diff:
    A deterministic MSE regressor is provably unable to fix small-scale power. Its
    optimum is the conditional mean, i.e. std(pred) = corr * std(target). Measured on
    the trained diff model: corr ~ 0.25-0.30 and std(pred)/std(target) ~ 0.37-0.41,
    exactly as theory says. Shrunk amplitude IS the small-scale Cl deficit, and the
    fully-converged relative loss (0.7348) barely beat "predict zero" (0.750) -- only
    ~2% of the residual is deterministically predictable from DISCO. Forcing the
    amplitude back up with a spectral penalty just fabricated uncorrelated power
    (Cl ratio overshoot to 1.2 + visible speckle).

    A conditional GENERATIVE model instead samples r ~ p(residual | DISCO, cosmo, z).
    A sample carries the full residual variance by construction -- no shrinkage -- so
    the corrected map has the right small-scale power and realistic (non-Gaussian)
    structure. It is NOT pixel-exact, and cannot be: that information is not in DISCO.

Formulation (reuses sphereflow's ResidualStreamer, formulation="residual"):
    cond = signal(delta_disco)                        (conditioning map, 1 channel)
    x1   = (signal(delta_high) - cond) / resid_scale  (the normalized residual = target)
    rectified flow:  x_t = (1-t) x0 + t x1,  x0 ~ N(0,I),  learn v = x1 - x0
    sampling: integrate noise -> r, then corrected_signal = cond + resid_scale * r
    The shell index z is part of the cosmo vector, so the model conditions the residual
    amplitude on redshift (that amplitude varies ~20x across shells).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sphereflow"))
import sphere_flow as sf
import unet_flow as uf
import unet_diff as ud                      # nested_grid_perms, downscale_nested
from train_sphere_flow import build_runs, estimate_scales
from train_unet_diff import DownscaledStreamer


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"),
                                timeout=timedelta(minutes=30))
        return local, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


def to_images(x1, cond, to_img, L):
    """(B, M) nested patches -> (B,1,L,L) images. Returns (cond_img, residual_target)."""
    B = x1.shape[0]
    return cond[:, to_img].view(B, 1, L, L), x1[:, to_img].view(B, 1, L, L)


def collect_val_batches(streamer, n_batches, batch_size, to_img, L, dev, seed=12345,
                        max_per_shell=2):
    """Fixed validation batches spanning MANY shells (cap per shell: shells differ ~20x
    in residual amplitude, so a single-shell validation set measures nothing)."""
    rng = np.random.RandomState(seed)
    order = list(range(len(streamer.runs)))
    rng.shuffle(order)
    out = []
    for ri in order:
        if len(out) >= n_batches:
            break
        tc_path, hi_path, cosmo = streamer.runs[ri]
        low_mm = np.load(tc_path, mmap_mode="r")
        high_mm = np.load(hi_path, mmap_mode="r")
        n = min(low_mm.shape[0], high_mm.shape[0])
        for si in rng.permutation(n):
            if len(out) >= n_batches:
                break
            tc = np.asarray(low_mm[si], dtype=np.float32)
            hi = np.asarray(high_mm[si], dtype=np.float32)
            x1p, cp, cvec = streamer._process_shell(tc, hi, int(si), n, cosmo)
            k = min(x1p.shape[0] // batch_size, max_per_shell)
            if k < 1:
                continue
            idx = rng.permutation(x1p.shape[0])
            cosmo_t = torch.from_numpy(np.repeat(cvec[None], batch_size, 0)).to(dev)
            for b in range(k):
                if len(out) >= n_batches:
                    break
                bi = idx[b * batch_size:(b + 1) * batch_size]
                x1 = torch.from_numpy(x1p[bi]).to(dev)
                cond = torch.from_numpy(cp[bi]).to(dev)
                ci, x1i = to_images(x1, cond, to_img, L)
                out.append((ci.detach(), x1i.detach(), cosmo_t.detach()))
    return out


def train(args):
    local, rank, world = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    runs = build_runs(args.data_root, args.test_cosmo, args.nside,
                      include_test=False, prefix="low")
    if len(runs) <= args.n_val + 1:
        raise RuntimeError(f"not enough runs ({len(runs)}) for a val split")
    val_runs, train_runs = runs[-args.n_val:], runs[:-args.n_val]
    my_runs = train_runs[rank::world] or [train_runs[rank % len(train_runs)]]
    cond_dim = len(runs[0][2]) + 1
    L = (args.nside // args.downscale) // args.order

    # No cross-rank all_reduce here on purpose: each rank samples different shells and
    # a slow disk read on one rank would stall the collective (this killed a real job).
    sig_scale, resid_scale = estimate_scales(my_runs, args.nside, args.order,
                                             args.softening, seed=rank)

    mk = lambda rr, seed: DownscaledStreamer(rr, args.nside, args.order, sig_scale,
                                             resid_scale, softening=args.softening,
                                             patch_frac=args.patch_frac,
                                             formulation="residual", seed=seed,
                                             factor=args.downscale)
    streamer = mk(my_runs, rank)

    to_img, _ = ud.nested_grid_perms(L)
    to_img = torch.from_numpy(to_img).to(dev)

    val_data = []
    if is_main():
        val_data = collect_val_batches(mk(val_runs, 7000), args.val_batches,
                                       args.batch_size, to_img, L, dev)
    if world > 1:
        dist.barrier()      # rank0's val collection can be slow; keep ranks in lockstep

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
                    "sig_scale": sig_scale, "resid_scale": resid_scale,
                    "base": args.base, "ch_mult": args.ch_mult, "time_dim": args.time_dim,
                    "order": args.order, "nside": args.nside, "softening": args.softening,
                    "downscale": args.downscale, "val_every": args.val_every,
                    "val_batches": args.val_batches, "n_val": args.n_val,
                    "ema_decay": 0.99}, tmp)
        os.replace(tmp, ckpt_path)
        np.save(tr_path, np.array(train_hist, dtype=np.float64))
        np.save(va_path, np.array(val_hist, dtype=np.float64) if val_hist else np.zeros((0, 2)))

    @torch.no_grad()
    def val_loss():
        net.eval()
        tot = 0.0
        for ci, x1i, cosmo in val_data:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                tot += uf.flow_matching_loss(raw, x1i, ci, cosmo).item()
        net.train()
        return tot / max(len(val_data), 1)

    if is_main():
        nparam = sum(p.numel() for p in raw.parameters())
        print(f"[unet-resflow] train={len(train_runs)} val={len(val_runs)} runs | {world} ranks "
              f"| patch {L}x{L} (downscale={args.downscale}, order={args.order}) "
              f"| {nparam/1e6:.2f}M params | sig_scale={sig_scale:.4f} "
              f"resid_scale={resid_scale:.4f}", flush=True)
        print(f"[train] ~{steps_per_epoch:,} steps/epoch x {args.epochs} = {total_steps:,} "
              f"| start_step={start_step:,}", flush=True)

    net.train()
    it = streamer.batches(args.batch_size, dev)
    last_t, last_step = time.time(), start_step

    for step in range(start_step + 1, total_steps + 1):
        x1, cond, cosmo = next(it)
        ci, x1i = to_images(x1, cond, to_img, L)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = uf.flow_matching_loss(net, x1i, ci, cosmo)
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
                print(f"  step {step:,}/{total_steps:,} | loss={lv:.4f} ema={ema:.4f} "
                      f"| {inst:.2f} steps/s | ETA {eta:.1f}h | lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
                last_t, last_step = now, step
                train_hist.append((step, lv, ema))
            if step % args.val_every == 0:
                vl = val_loss()
                val_hist.append((step, vl))
                print(f"    [val] step {step:,} | val_loss={vl:.4f} (train_ema={ema:.4f})",
                      flush=True)
            if step % args.ckpt_every == 0:
                save_ckpt(step)

    if is_main():
        save_ckpt(total_steps)
        torch.save(raw.state_dict(), Path(args.out_dir) / "unet_resflow.pth")
        print(f"[train] done -> {args.out_dir}", flush=True)
    if world > 1:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000122")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--downscale", type=int, default=1)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--ch-mult", default="1,2,2")
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--softening", type=float, default=1.0)
    p.add_argument("--patch-frac", type=float, default=0.5)
    p.add_argument("--n-val", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
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
