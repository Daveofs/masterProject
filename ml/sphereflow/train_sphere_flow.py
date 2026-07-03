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

import masterProject.ml.sphereflow.sphere_flow as sf


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"))
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
        self._mm = {}
        # (run_idx, shell_idx, n_shells_in_run) sample list
        self.samples = []
        for ri, (tc, hi, _v) in enumerate(runs):
            n = min(np.load(tc, mmap_mode="r").shape[0],
                    np.load(hi, mmap_mode="r").shape[0])
            self.samples += [(ri, si, n) for si in range(n)]

    @property
    def n_shells(self):
        return len(self.samples)

    def _mmap(self, path):
        k = str(path)
        if k not in self._mm:
            self._mm[k] = np.load(k, mmap_mode="r")
        return self._mm[k]

    def _shell_xy(self, ri, si, n_run):
        tc_path, hi_path, cosmo = self.runs[ri]
        tc = np.asarray(self._mmap(tc_path)[si], dtype=np.float32)
        hi = np.asarray(self._mmap(hi_path)[si], dtype=np.float32)
        d_tc, _ = sf.to_overdensity(tc[None]); d_hi, _ = sf.to_overdensity(hi[None])
        cond = sf.signal_forward(d_tc, self.sig_scale, self.soft)
        sig_hi = sf.signal_forward(d_hi, self.sig_scale, self.soft)
        if self.formulation == "direct":
            x1 = sig_hi
        else:
            x1 = (sig_hi - cond) / self.resid_scale
        shell_norm = np.float32(si / max(n_run - 1, 1))
        cvec = np.concatenate([cosmo, [shell_norm]]).astype(np.float32)
        return (sf.map_to_patches(x1, self.order),
                sf.map_to_patches(cond, self.order), cvec)

    def batches(self, batch_size, device):
        import threading
        import queue as _q
        q = _q.Queue(maxsize=3)
        prod_rng = np.random.RandomState(self.seed + 991)

        def producer():
            order = np.arange(len(self.samples))
            while True:
                prod_rng.shuffle(order)
                for k in order:
                    q.put(self._shell_xy(*self.samples[k]))

        threading.Thread(target=producer, daemon=True).start()
        while True:
            x1p, cp, cvec = q.get()
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
    if args.compile:
        net = torch.compile(net)
    if world > 1:
        net = DDP(net, device_ids=[local], broadcast_buffers=False)
    # NOTE: the loss must go through `net` (the DDP wrapper) so gradient allreduce
    # fires — v1 passed the unwrapped module and silently trained per-rank models.

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
    if is_main():
        print(f"[train] {streamer.n_shells} shells/rank | ~{steps_per_epoch:,} steps/epoch "
              f"x {args.epochs} = {total_steps:,} | batch/gpu={args.batch_size} | "
              f"patch_frac={args.patch_frac} | compile={args.compile}", flush=True)

    net.train()
    it = streamer.batches(args.batch_size, dev)
    t0, loss_hist, ema = time.time(), [], None
    for step in range(1, total_steps + 1):
        x1, cond, cosmo = next(it)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = sf.flow_matching_loss(net, x1, cond, cosmo)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); sched.step()
            if is_main():
                print(f"  step {step}: non-finite, skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if is_main():
            lv = loss.item()
            # EMA smooths the large per-shell loss variance (faint vs dense) so the
            # actual convergence trend is readable in the log.
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            if step % args.log_every == 0:
                print(f"  step {step:,}/{total_steps:,} | loss={lv:.4f} ema={ema:.4f} | "
                      f"{step/(time.time()-t0):.2f} steps/s | lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)
                loss_hist.append(ema)

    if is_main():
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        raw = net.module if world > 1 else net
        raw = getattr(raw, "_orig_mod", raw)      # unwrap torch.compile
        torch.save(raw.state_dict(), out / "sphere_flow.pth")
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
    p.add_argument("--out-dir", default="./sphere_flow_model")
    args = p.parse_args()
    try:
        train(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
