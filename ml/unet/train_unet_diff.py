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
        # 30 min (not 10): rank0-only setup work (val-batch collection) can hit a slow
        # Lustre day and stall well past 10 min while other ranks idle -- previously
        # that skew made ranks 1-3 race ahead into DDP()'s internal ALLGATHER before
        # rank0 arrived, which then hit the (old, shorter) NCCL timeout and crashed the
        # whole job. The barrier below is the real fix (keeps ranks in lockstep); this
        # timeout is just headroom so a genuinely slow read doesn't trip NCCL first.
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"),
                                timeout=timedelta(minutes=30))
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


def collect_val_batches(streamer, n_batches, batch_size, to_img, L, dev, seed=12345,
                        max_per_shell=2):
    """Build n_batches FIXED validation batches via light mmap shell reads.

    streamer.batches() (used for training) does a full ~14-28 GB SEQUENTIAL np.load of
    an entire run pair before yielding anything -- fine for training (amortized over
    ~100k patches per run) but wasteful here: it stalled every job's startup (DDP init/
    header print) by tens of seconds to minutes just to gather a handful of validation
    batches. Instead, mmap each run and slice out only as few individual shells as
    possible (~200 MB each at full res) via streamer._process_shell, which accepts
    plain numpy shell arrays regardless of how they were read.

    A single shell yields far more patches than one batch needs (3072 at order=16 --
    _process_shell does NOT apply patch_frac), so we pull a few disjoint batches per
    shell to limit how many shells must be read (each read is I/O, occasionally slow
    under filesystem load; observed 100+s tail latency for a single 200MB shell).

    But max_per_shell CAPS that: shells differ enormously (std of the diff target spans
    ~20x from shell 3 to shell 50), so a validation set drawn from one shell is not
    representative of the model at all. Without the cap, 3072//128 = 24 >= n_batches
    meant EVERY validation batch came from the first shell of the first run -- the
    validation curve was flat and meaninglessly low while real eval was poor.
    """
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
            n_from_shell = min(x1p.shape[0] // batch_size, max_per_shell)
            if n_from_shell < 1:
                continue
            idx = rng.permutation(x1p.shape[0])
            cosmo_t = torch.from_numpy(np.repeat(cvec[None], batch_size, 0)).to(dev)
            for b in range(n_from_shell):
                if len(out) >= n_batches:
                    break
                bi = idx[b * batch_size:(b + 1) * batch_size]
                x1 = torch.from_numpy(x1p[bi]).to(dev)
                cond = torch.from_numpy(cp[bi]).to(dev)
                ci, dt, x1i = to_images(x1, cond, to_img, L)
                out.append((ci.detach(), dt.detach(), x1i.detach(), cosmo_t.detach()))
    return out


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
    #
    # Deliberately NOT cross-rank-averaged (no all_reduce here): each rank reads a
    # DIFFERENT random sample of shells via disk I/O whose latency can vary wildly
    # under filesystem load (observed: a single 200MB shell read occasionally taking
    # 100+ seconds instead of ~1s). A collective at this point makes every rank wait
    # for the SLOWEST rank's I/O -- and it killed a real job (SeqNum=2 ALLREDUCE
    # timeout at the 30-min NCCL default, before any training step had even run).
    # Per-rank estimates are already close (same underlying signal statistics), so
    # the average was a nice-to-have, not a correctness requirement -- skip it.
    sig_scale, resid_scale = estimate_scales(my_runs, args.nside, args.order,
                                             args.softening, seed=rank)

    mk = lambda rr, seed: DownscaledStreamer(rr, args.nside, args.order, sig_scale,
                                             resid_scale, softening=args.softening,
                                             patch_frac=args.patch_frac,
                                             formulation="direct", seed=seed,
                                             factor=args.downscale)
    streamer = mk(my_runs, rank)

    to_img, _ = ud.nested_grid_perms(L)
    to_img = torch.from_numpy(to_img).to(dev)
    sbins, scounts, sn_bins = ud.radial_bins(L, dev)

    # FIXED validation set, built once via light mmap shell reads (see
    # collect_val_batches) -- avoids the multi-GB-per-run full sequential load that
    # otherwise stalls every job's startup, and gives the correct validation scheme:
    # the exact same batches are scored at every checkpoint, so the curve is comparable.
    val_data = []
    if is_main():
        val_data = collect_val_batches(mk(val_runs, 7000), args.val_batches,
                                       args.batch_size, to_img, L, dev)
    if world > 1:
        # Rank 0 alone can hit a slow-Lustre day here while ranks 1..N-1 have nothing
        # to do -- without this barrier they'd race ahead into DDP()'s internal
        # ALLGATHER and time out waiting for rank 0 (this crashed a real run). Block
        # everyone here instead, where the (now 30 min) process-group timeout applies
        # to a single well-understood wait rather than an opaque collective mismatch.
        dist.barrier()

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
                    "huber_delta": args.huber_delta, "ema_decay": 0.99}, tmp)
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
                pl = ud.correction_loss(pred, dt, args.huber_delta, args.relative_loss)
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
            pixel_l = ud.correction_loss(pred, dt, args.huber_delta, args.relative_loss)
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
    p.add_argument("--relative-loss", action="store_true", default=True,
                   help="scale the pixel loss by the batch's target RMS so shells with "
                        "wildly different diff amplitudes contribute equally (default on)")
    p.add_argument("--absolute-loss", dest="relative_loss", action="store_false",
                   help="use the raw (unnormalized) pixel loss")
    p.add_argument("--huber-delta", type=float, default=0.1,
                   help="Huber transition point for the pixel term (robust to outlier "
                        "pixels/batches); large delta recovers plain MSE")
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
