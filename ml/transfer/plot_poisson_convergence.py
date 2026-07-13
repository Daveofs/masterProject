#!/usr/bin/env python3
"""Convergence ("loss") curve for the Poisson-resampler's per-ell Cl calibration,
in the same visual style as unet_flow_jbucko/plot_flow_loss.py (train vs val curve,
log-y, best-iteration marker), so the two pipelines' convergence can be viewed side
by side. Only a plotting/instrumentation wrapper -- reuses poisson_resample.py's
own cl2xi/xi2cl/_taper/_one_draw, nothing reimplemented.

There is no gradient-descent "training" here (resample_shell's calibration is a
damped fixed-point iteration on a per-ell correction, see its docstring), but the
same train/val DISTINCTION jbucko's loss plot makes is meaningful and reproduced
here for comparability:
  * "train" = misfit of the n_avg-draw-averaged Cl used to COMPUTE this iteration's
    update (i.e. what the correction is being fit to).
  * "val"   = misfit of ONE extra held-out Poisson draw, generated with the
    correction as it stood BEFORE this iteration's update (i.e. never used to fit
    anything) -- the direct analogue of jbucko's held-out-cosmology validation loss.
Metric = RMS(log(Cl_avg/Cl_high)) smoothed over ell 50-2500 (log-space, since Cl
spans decades) -- the flow-loss analogue of "how far is the reconstruction from the
target", on the same footing (lower is better, log-y axis).

Usage
-----
  python plot_poisson_convergence.py --run-dir <grid>/cosmo_X/run_0 \
      --transfer <noclip transfer.npz output run-dir> --shells 3 10 30 \
      --out convergence.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from poisson_resample import cl2xi, xi2cl, smooth_cl, _taper, _one_draw, _alm  # noqa: E402


def misfit(cl_a: np.ndarray, cl_h: np.ndarray, lo: int = 50, hi: int = 2500) -> float:
    """RMS(log(Cl_a/Cl_h)) over [lo,hi), smoothed first (see poisson_resample.smooth_cl
    -- an unsmoothed single-draw ratio is dominated by per-ell shot noise, see the
    oscillation discussion this plot exists to make legible)."""
    r = smooth_cl(cl_a)[lo:hi] / smooth_cl(cl_h)[lo:hi]
    return float(np.sqrt(np.mean(np.log(np.clip(r, 1e-6, None)) ** 2)))


def run_shell(rho, cl_high, source_map, lmax, nside, mu, w, rng,
             ell_c, taper, n_avg, n_iter, damp):
    """Reproduces resample_shell's iteration loop (poisson_resample.py) but records
    the train/val misfit history instead of only returning the final draw."""
    omega = 4 * np.pi / hp.nside2npix(nside)
    ells = np.arange(lmax + 1)
    lbar = float(rho.mean())

    cl_lam = np.clip(cl_high - lbar * omega, 0.0, None); cl_lam[0] = 0.0
    xi_g = np.log(np.clip(1.0 + cl2xi(cl_lam, mu) / lbar ** 2, 1e-12, None))
    cl_g = np.clip(xi2cl(xi_g, mu, w, lmax), 0.0, None); cl_g[0] = 0.0
    win = _taper(ells, ell_c, taper)

    corr = np.ones(lmax + 1)
    train_hist, val_hist = [], []
    for it in range(n_iter):
        # held-out validation draw FIRST, with corr as it stood before this update
        n_val, _ = _one_draw(rho, source_map, lbar, cl_g * corr, lmax, nside, ells, win, rng)
        val_hist.append(misfit(hp.anafast(n_val, lmax=lmax), cl_high))

        cl_n_sum = None
        for _k in range(n_avg):
            n_it, _ = _one_draw(rho, source_map, lbar, cl_g * corr, lmax, nside, ells, win, rng)
            cl_it = hp.anafast(n_it, lmax=lmax)
            cl_n_sum = cl_it.copy() if cl_n_sum is None else cl_n_sum + cl_it
        cl_n = cl_n_sum / n_avg
        train_hist.append(misfit(cl_n, cl_high))

        ratio = np.clip(smooth_cl(cl_high) / np.clip(smooth_cl(cl_n), 1e-30, None), 0.3, 3.0)
        corr = np.clip(corr * ratio ** damp, 0.02, 50.0)

    return train_hist, val_hist


def main():
    from numpy.polynomial.legendre import leggauss
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--corrected", required=True, help="--no-clip apply() output npz")
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--shells", type=int, nargs="+", default=[3, 10, 30])
    p.add_argument("--ell-c", type=int, default=300)
    p.add_argument("--taper", type=int, default=100)
    p.add_argument("--n-avg", type=int, default=4)
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--damp", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    n_alm = (lmax + 1) * (lmax + 2) // 2
    run = Path(args.run_dir)
    mu, w = leggauss(2 * lmax + 64)

    low_full = np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r")
    high_alms = np.load(run / f"high_alms_lmax{lmax}.npy", mmap_mode="r")
    corrected = np.load(args.corrected, mmap_mode="r")["shells"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(args.shells)))
    for s, color in zip(args.shells, colors):
        rng = np.random.default_rng(args.seed + s)
        rho = np.asarray(low_full[s], dtype=np.float64)
        cl_h = hp.alm2cl(_alm(np.asarray(high_alms[s]), n_alm), lmax=lmax)
        src = np.asarray(corrected[s], dtype=np.float64)
        train_hist, val_hist = run_shell(rho, cl_h, src, lmax, nside, mu, w, rng,
                                         args.ell_c, args.taper, args.n_avg,
                                         args.n_iter, args.damp)
        it = np.arange(len(train_hist))
        ax.plot(it, train_hist, "-o", ms=4, color=color, label=f"shell {s} train (n_avg draws)")
        ax.plot(it, val_hist, "--s", ms=4, color=color, alpha=0.6, label=f"shell {s} val (held-out draw)")
        best = int(np.argmin(val_hist))
        print(f"shell {s}: train {train_hist} val {val_hist} best_iter={best}", flush=True)

    ax.set_xlabel("iteration"); ax.set_ylabel("RMS(log(Cl/Cl_high)), ell 50-2500")
    ax.set_yscale("log"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_title("Poisson-resampler per-ell calibration: train vs held-out convergence\n"
                 r"metric $=\sqrt{\langle\log(C_\ell^{smooth}/C_\ell^{high,smooth})^2\rangle}$"
                 " (lower is better)", fontsize=10)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[plot] -> {out}", flush=True)


if __name__ == "__main__":
    main()
