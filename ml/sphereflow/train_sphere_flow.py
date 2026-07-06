#!/usr/bin/env python3
"""Train the DeepSphere-conv flow-matching model on TRANSFER-CORRECTED residuals.

Formulation (v2 — residual on top of the transfer-function baseline):
    baseline  : tcorr = alm2map( low_alm * T(ell, shell) )   [prepare_tcorr_dataset]
    condition : c  = arcsinh-signal( delta_tcorr )
    target    : x1 = ( arcsinh-signal(delta_high) - c ) / resid_scale
    corrected : signal = c + resid_scale * sampled_residual  ->  invert -> map

The transfer function already fixes the Cl (first order, phase-preserving); the
flow learns ONLY the small non-Gaussian / stochastic remainder. This removes the
variance-calibration burden that made pure generative models overshoot, and the
worst case degenerates to the (already validated) transfer-function result.

Speed (vs the 8h v1 run):
  * gather-based ChebConv + torch.compile + bf16 autocast (~1.8x / GPU)
  * DDP FIX: v1 bypassed gradient sync by calling the unwrapped module — ranks
    trained independently. Now the DDP-wrapped model is used (true 4x/node).
  * raw .npy mmap per-shell reads (no 14 GB npz decompression in the loop)
  * --patch-frac subsampling (patches are redundant; 0.5 halves the epoch)

  torchrun --nproc_per_node=4 train_sphere_flow.py --data-root ... [--include-test]
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

import sphere_flow as sf


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        from datetime import timedelta
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        # Collective timeout. Data is now RUN-MAJOR (one big sequential read per
        # run feeds ~1600 steps), so per-step Lustre stragglers are gone -> a long
        # timeout just wastes 30 min per hang while debugging. 10 min still tolerates
        # the one-time 14 GB run load; a genuinely hung/desynced rank fails 3x faster.
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"),
                                timeout=timedelta(minutes=10))
        return local, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


# ---------------------------------------------------------------------------
# Data: mmap per-shell streaming of (tcorr, high) pairs
# ---------------------------------------------------------------------------

def build_runs(data_root, test_cosmo, nside, include_test, prefix="low"):
    """(input_npy, high_npy, cosmo_vec) per run that has the prepared dataset.

    prefix='low'   -> raw DISCO input   (single-model 'direct' formulation)
    prefix='tcorr' -> T-corrected input ('residual' formulation)
    """
    data_root = Path(data_root)
    runs = []
    for c in sorted(d for d in data_root.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_")):
        if (not include_test) and c.name == test_cosmo:
            continue
        for ld in sorted(r for r in c.iterdir()
                         if r.is_dir() and r.name.startswith("run_")) or [c]:
            tc = ld / f"{prefix}_shells_nside={nside}.npy"
            hi = ld / f"high_shells_nside={nside}.npy"
            if not (tc.exists() and hi.exists()):
                continue
            pf = ld / "params.yml"
            vec = _cosmo_vector(pf) if pf.exists() else np.zeros(1, np.float32)
            runs.append((tc, hi, vec))
    return runs


def _cosmo_vector(params_yml):
    import yaml
    p = yaml.safe_load(Path(params_yml).read_text())
    keys = sorted(k for k, v in p.items() if _is_num(v))
    return np.array([float(p[k]) for k in keys], dtype=np.float32)


def _is_num(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False


class ResidualStreamer:
    """Streams (x1, cond, cosmo) patch batches from mmap'd shell files.

    Random-accesses one shell at a time (mmap: ~200 MB read, no decompression) and
    prepares it on a BACKGROUND THREAD so CPU prep overlaps GPU compute. The
    conditioning vector = [cosmo params, normalized shell index] — the shell index
    tells the model which redshift/noise regime it is in (dropping it in v2 made
    one model serve faint and dense shells blindly).

    formulation='direct'  : x1 = signal(delta_high)                (single model)
    formulation='residual': x1 = (signal(delta_high) - cond)/resid_scale
    """

    def __init__(self, runs, nside, order, sig_scale, resid_scale, softening=1.0,
                 patch_frac=1.0, formulation="direct", seed=0):
        self.runs = runs
        self.nside, self.order = nside, order
        self.npix = hp.nside2npix(nside)
        self.sig_scale, self.resid_scale, self.soft = sig_scale, resid_scale, softening
        self.patch_frac = patch_frac
        self.formulation = formulation
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        # Per-run shell counts (peek headers only — no data read).
        self._n = [min(np.load(tc, mmap_mode="r").shape[0],
                       np.load(hi, mmap_mode="r").shape[0])
                   for tc, hi, _v in runs]

    @property
    def n_shells(self):
        return int(sum(self._n))

    def _process_shell(self, tc, hi, si, n_run, cosmo):
        """Build (x1_patches, cond_patches, cvec) from one shell already in RAM."""
        d_tc, _ = sf.to_overdensity(tc[None]); d_hi, _ = sf.to_overdensity(hi[None])
        cond = sf.signal_forward(d_tc, self.sig_scale, self.soft)
        sig_hi = sf.signal_forward(d_hi, self.sig_scale, self.soft)
        x1 = sig_hi if self.formulation == "direct" \
            else (sig_hi - cond) / self.resid_scale
        shell_norm = np.float32(si / max(n_run - 1, 1))
        cvec = np.concatenate([cosmo, [shell_norm]]).astype(np.float32)
        return (sf.map_to_patches(x1, self.order),
                sf.map_to_patches(cond, self.order), cvec)

    def batches(self, batch_size, device, prefetch=4):
        import threading
        import queue as _q
        import gc
        # RUN-MAJOR streaming: load ONE run's file pair fully into RAM with a
        # SEQUENTIAL np.load (fast), then serve all its shells from RAM before
        # moving to the next run. The previous global random shuffle of shells
        # across all 44 runs made the mmap thrash between 14 GB files -> Lustre
        # random-seek contention dominated (~90% of wall). Each run's ~69 shells
        # feed ~1600 steps of compute, so the one-time 14 GB sequential read per
        # run is negligible and overlaps the queue drain. RAM: ~one run pair
        # (~28 GB) + a few processed shells (bounded queue).
        q = _q.Queue(maxsize=prefetch)
        prod_rng = np.random.RandomState(self.seed + 991)

        def producer():
            # A daemon thread that raises is SILENT: its exception is swallowed and
            # the consumer's q.get() blocks forever -> at scale that shows up only as
            # a 30-min NCCL collective timeout on every OTHER rank (impossible to
            # diagnose). So: skip a shell/run that fails to process (logged), and if
            # something unrecoverable happens, push the exception to the queue so the
            # consumer RAISES it with a real traceback instead of hanging.
            run_order = np.arange(len(self.runs))
            try:
                while True:
                    prod_rng.shuffle(run_order)
                    for ri in run_order:
                        tc_path, hi_path, cosmo = self.runs[ri]
                        try:
                            low = np.load(tc_path)      # full SEQUENTIAL read into RAM
                            high = np.load(hi_path)
                        except Exception as e:
                            print(f"[streamer WARN] seed={self.seed} skipping run "
                                  f"{tc_path}: load failed: {e}", flush=True)
                            continue
                        n = min(low.shape[0], high.shape[0])
                        shells = prod_rng.permutation(n)
                        for si in shells:
                            try:
                                item = self._process_shell(
                                    np.asarray(low[si], np.float32),
                                    np.asarray(high[si], np.float32), int(si), n, cosmo)
                            except Exception as e:
                                print(f"[streamer WARN] seed={self.seed} skipping shell "
                                      f"{si} of {tc_path}: {e}", flush=True)
                                continue
                            q.put(item)
                        del low, high
                        gc.collect()
            except Exception as e:            # unrecoverable: hand the error to consumer
                import traceback as _tb
                q.put(("__PRODUCER_ERROR__", f"{e}\n{_tb.format_exc()}"))

        threading.Thread(target=producer, daemon=True).start()
        while True:
            item = q.get()
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__PRODUCER_ERROR__":
                raise RuntimeError(f"data producer thread died:\n{item[1]}")
            x1p, cp, cvec = item
            idx = self.rng.permutation(x1p.shape[0])
            if self.patch_frac < 1.0:
                idx = idx[: max(int(len(idx) * self.patch_frac), batch_size)]
            cosmo_t = torch.from_numpy(
                np.repeat(cvec[None], batch_size, 0)).to(device)
            for b in range(0, len(idx) - batch_size + 1, batch_size):
                bi = idx[b:b + batch_size]
                yield (torch.from_numpy(x1p[bi]).to(device),
                       torch.from_numpy(cp[bi]).to(device), cosmo_t)


def estimate_scales(runs, nside, order, softening, n_shells=8, seed=0):
    """sig_scale = std(arcsinh(delta_tcorr)); resid_scale = std(residual signal)."""
    rng = np.random.RandomState(seed)
    sigs, resids = [], []
    picks = [(ri, si) for ri, (tc, hi, _v) in enumerate(runs)
             for si in rng.choice(np.load(tc, mmap_mode="r").shape[0],
                                  size=max(n_shells // len(runs), 2), replace=False)]
    for ri, si in picks:
        tc_path, hi_path, _ = runs[ri]
        tc = np.asarray(np.load(tc_path, mmap_mode="r")[si], dtype=np.float32)
        hi = np.asarray(np.load(hi_path, mmap_mode="r")[si], dtype=np.float32)
        d_tc, _ = sf.to_overdensity(tc[None]); d_hi, _ = sf.to_overdensity(hi[None])
        s = np.arcsinh(d_tc / softening)
        sigs.append(s.std())
        sig_scale_i = s.std()
        c = sf.signal_forward(d_tc, sig_scale_i, softening)
        h = sf.signal_forward(d_hi, sig_scale_i, softening)
        resids.append((h - c).std())
    return float(np.mean(sigs) + 1e-12), float(np.mean(resids) + 1e-12)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    local, rank, world = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    prefix = "tcorr" if args.formulation == "residual" else "low"
    runs = build_runs(args.data_root, args.test_cosmo, args.nside, args.include_test,
                      prefix=prefix)
    if not runs:
        raise RuntimeError(f"no prepared runs found ({prefix}_shells_nside={args.nside}.npy"
                           " missing — run prepare_tcorr_dataset.py)")
    clen = max(len(v) for _t, _h, v in runs)
    carr = np.stack([np.pad(v, (0, clen - len(v))) for _t, _h, v in runs]).astype(np.float64)
    cmean, cstd = carr.mean(0), np.where(carr.std(0) < 1e-8, 1.0, carr.std(0))
    runs = [(t, h, ((carr[i] - cmean) / cstd).astype(np.float32))
            for i, (t, h, _v) in enumerate(runs)]
    cond_dim = clen + 1                     # + normalized shell index

    sig_scale, resid_scale = estimate_scales(runs, args.nside, args.order, args.softening)
    if args.formulation == "direct":
        resid_scale = 1.0
    my_runs = runs[rank::world] or [runs[rank % len(runs)]]
    if is_main():
        mode = "INCLUDE-TEST (sanity gate)" if args.include_test else f"LOO ({args.test_cosmo} out)"
        print(f"[data] {len(runs)} runs [{mode}, {args.formulation}, input={prefix}] | "
              f"{world} ranks | cond_dim={cond_dim} | "
              f"sig_scale={sig_scale:.4g} resid_scale={resid_scale:.4g}", flush=True)

    L = sf.healpix_laplacian(args.nside, order=args.order)
    net = sf.SphereFlowNet(L, cond_dim=cond_dim, hidden=args.hidden,
                           n_layers=args.n_layers, K=args.K).to(dev)

    # ---- resume from checkpoint (transient fabric/NCCL failures at scale keep
    # killing multi-hour jobs; with checkpoints a crash costs minutes, not the
    # run). Model state is loaded BEFORE compile/DDP so the keys match the raw
    # module; optimizer/scheduler state is loaded after they are created below.
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_dir) / "checkpoint.pt"
    ckpt = None
    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ckpt["model"])
        if is_main():
            print(f"[resume] loaded {ckpt_path} at step {ckpt['step']:,}", flush=True)

    if args.compile:
        net = torch.compile(net)
    if world > 1:
        net = DDP(net, device_ids=[local], broadcast_buffers=False)
    # NOTE: the loss must go through `net` (the DDP wrapper) so gradient allreduce
    # fires — v1 passed the unwrapped module and silently trained per-rank models.
    raw_mod = net.module if world > 1 else net
    raw_mod = getattr(raw_mod, "_orig_mod", raw_mod)   # unwrap torch.compile

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    streamer = ResidualStreamer(my_runs, args.nside, args.order, sig_scale,
                                resid_scale, softening=args.softening,
                                patch_frac=args.patch_frac,
                                formulation=args.formulation, seed=rank)
    ppshell = int(sf.n_patches(args.order) * args.patch_frac) if args.order > 1 else 1
    steps_per_epoch = max(streamer.n_shells * ppshell // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))

    start_step, loss_hist, ema = 0, [], None
    if ckpt is not None:
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = int(ckpt["step"])
        ema = ckpt.get("ema", None)
        loss_hist = list(ckpt.get("loss_hist", []))

    if is_main():
        print(f"[train] {streamer.n_shells} shells/rank | ~{steps_per_epoch:,} steps/epoch "
              f"x {args.epochs} = {total_steps:,} | batch/gpu={args.batch_size} | "
              f"patch_frac={args.patch_frac} | compile={args.compile} | "
              f"start_step={start_step:,}", flush=True)

    def save_ckpt(step):
        # atomic: torch.save to tmp then rename (a killed job must never leave a
        # truncated checkpoint at the final name — same lesson as prepare_maps).
        tmp = ckpt_path.with_suffix(f".tmp{os.getpid()}")
        torch.save({"step": step, "model": raw_mod.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ema": ema, "loss_hist": loss_hist}, tmp)
        os.replace(tmp, ckpt_path)

    net.train()
    it = streamer.batches(args.batch_size, dev)
    t0 = time.time()
    last_t, last_step = t0, start_step   # for WINDOWED (instantaneous) steps/s
    # DIAGNOSTIC heartbeat: a whole node (highest ranks) deterministically stops
    # participating in the collective at ~step 13,254 regardless of data/compile/
    # checkpoint. Print per-RANK, per-PHASE around that window so the hang's exact
    # location (data load / forward / finite all-reduce / backward) is visible.
    HB_LO = int(os.environ.get("HB_LO", "0"))
    HB_HI = int(os.environ.get("HB_HI", "0"))
    def hb(phase):
        if HB_LO <= step <= HB_HI:
            print(f"[hb] rank{rank} step{step} {phase}", flush=True)
    for step in range(start_step + 1, total_steps + 1):
        hb("A:pre-data")
        x1, cond, cosmo = next(it)
        hb("B:got-data")
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = sf.flow_matching_loss(net, x1, cond, cosmo)
        hb("C:loss-done")
        # Non-finite guard MUST be collective: each rank trains on its OWN data
        # stream (seed=rank), so at a given step one rank can get a non-finite
        # loss while the others are fine. If that single rank `continue`s and
        # skips loss.backward(), it never enters the DDP gradient all-reduce that
        # the other ranks DO enter -> they block forever on the missing peer ->
        # ncclSystemError (this was the deterministic "step 13,254" crash that the
        # PTLTE_NOT_FOUND flood was merely a symptom of). Agree across ALL ranks:
        # if ANY rank is non-finite, EVERY rank skips backward this step together.
        finite = torch.isfinite(loss).to(torch.float32).detach()   # 1.0 / 0.0, on GPU
        hb("D:pre-finite-allreduce")
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)   # 0 if any rank non-finite
        hb("E:post-finite-allreduce")
        if finite.item() < 1.0:
            opt.zero_grad(set_to_none=True); sched.step()
            if is_main():
                print(f"  step {step}: non-finite loss on >=1 rank, skipped (all ranks)",
                      flush=True)
            continue
        loss.backward()
        hb("F:backward-done")
        if HB_LO <= step <= HB_HI:
            # per-rank loss + pre-clip grad norm. Accessing p.grad forces a sync on
            # the (async) gradient all-reduce, so: if a rank PRINTS this, its grads
            # arrived and we can see if values exploded (inf/huge => numerical bug);
            # if a rank does NOT print this, it is stuck IN the grad all-reduce
            # itself (pure comms hang, grads never arrived).
            with torch.no_grad():
                gn = 0.0
                for p in raw_mod.parameters():
                    if p.grad is not None:
                        gn += float((p.grad.detach().float() ** 2).sum())
                gn = gn ** 0.5
            print(f"[hb] rank{rank} step{step} G:loss={float(loss):.3e} gradnorm={gn:.3e}",
                  flush=True)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if is_main():
            lv = loss.item()
            # EMA smooths the large per-shell loss variance (faint vs dense) so the
            # actual convergence trend is readable in the log.
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            if step % args.log_every == 0:
                now = time.time()
                # WINDOWED rate over the last log interval (NOT cumulative since
                # start): cumulative is dragged down for a long time by the one-off
                # startup (first 14 GB run load + torch.compile) and misleadingly
                # reads ~0 even when steady-state throughput is fine.
                inst = (step - last_step) / max(now - last_t, 1e-9)
                eta_h = (total_steps - step) / max(inst, 1e-9) / 3600.0
                avg = (step - start_step) / max(now - t0, 1e-9)
                print(f"  step {step:,}/{total_steps:,} | loss={lv:.4f} ema={ema:.4f} | "
                      f"{inst:.2f} steps/s (avg {avg:.2f}) | ETA {eta_h:.1f}h | "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                last_t, last_step = now, step
                loss_hist.append(ema)
            if step % args.ckpt_every == 0:
                save_ckpt(step)

    if is_main():
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        save_ckpt(total_steps)                    # final checkpoint too
        torch.save(raw_mod.state_dict(), out / "sphere_flow.pth")
        np.savez(out / "meta.npz", nside=args.nside, order=args.order, K=args.K,
                 hidden=args.hidden, n_layers=args.n_layers, cond_dim=cond_dim,
                 sig_scale=sig_scale, resid_scale=resid_scale,
                 softening=args.softening, formulation=args.formulation,
                 cosmo_mean=cmean, cosmo_std=cstd, loss_hist=np.array(loss_hist))
        print(f"[train] saved model + meta to {out}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000122")
    p.add_argument("--include-test", action="store_true",
                   help="SANITY GATE: include the test cosmology in training.")
    p.add_argument("--formulation", choices=["direct", "residual"], default="direct",
                   help="direct: SINGLE MODEL, condition on raw DISCO, generate the "
                        "high signal. residual: generate high-tcorr on top of the "
                        "transfer-function baseline.")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--softening", type=float, default=1.0)
    p.add_argument("--patch-frac", type=float, default=0.5,
                   help="Fraction of each shell's patches used per epoch.")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500,
                   help="Save an atomic checkpoint every N steps; on restart the run "
                        "auto-resumes from out-dir/checkpoint.pt (fabric/NCCL crashes "
                        "then cost minutes, not the whole run).")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore an existing checkpoint and start from scratch.")
    p.add_argument("--out-dir", default="./sphere_flow_model")
    args = p.parse_args()
    try:
        train(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
