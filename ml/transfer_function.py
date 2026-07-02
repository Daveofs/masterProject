#!/usr/bin/env python3
"""Per-ell transfer-function correction for DISCO-DJ -> CosmoGrid small scales.

Physics (measured via the per-ell cross-correlation r(ell)): for the shells that
carry signal, low and high SHARE PHASES (r~0.9-1.0 well into small scales); DISCO
just gets the small-scale AMPLITUDE wrong. So the Cl correction is deterministic:

    corrected_alm(ell, m) = low_alm(ell, m) * T(ell, shell),
    T(ell, shell) = sqrt( <Cl_high(ell, shell)> / <Cl_low(ell, shell)> )   (train avg)

This matches Cl_high exactly (by construction), preserves the (correct) phases, is
completely stable, and touches only the scales where T != 1 (small scales; T~1 at
large ell). The transfer function is ~a property of the sim-code pair, so a per-
shell (redshift) average over training cosmologies transfers to the test cosmology.

Usage
-----
  # 1. Fit T from training runs (excludes the test cosmology):
  python transfer_function.py fit --data-dir <grid> --lmax 3000 \
      --test-cosmo cosmo_000001 --out transfer.npz
  # 2. Apply to a test run's low alms -> corrected shells:
  python transfer_function.py apply --transfer transfer.npz \
      --run-dir <grid>/cosmo_000001/run_0 --nside 2048 --ell-min 0 --out corrected.npz
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import healpy as hp


def _alm(vec, N_alm):
    return (vec[:N_alm] + 1j * vec[N_alm:]).astype(np.complex128)


def ell_of_flat_index(lmax: int) -> np.ndarray:
    ell = np.empty((lmax + 1) * (lmax + 2) // 2, dtype=np.int64)
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            ell[m * (2 * lmax + 1 - m) // 2 + l] = l
    return ell


# ---------------------------------------------------------------------------
# Fit: accumulate mean Cl_low / Cl_high per (shell, ell) over training runs
# ---------------------------------------------------------------------------

def fit(args):
    data_dir = Path(args.data_dir)
    lmax = args.lmax
    N_alm = (lmax + 1) * (lmax + 2) // 2

    cosmos = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    runs = []
    for c in cosmos:
        # Leave the test cosmology out (proper generalization test) unless
        # --include-test is set (SANITY CHECK: fit on it too, expect ~perfect).
        if args.test_cosmo and c.name == args.test_cosmo and not args.include_test:
            continue
        rs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        for ld in (rs or [c]):
            lo = ld / f"low_alms_lmax{lmax}.npy"
            hi = ld / f"high_alms_lmax{lmax}.npy"
            if lo.exists() and hi.exists():
                runs.append((lo, hi))
    if not runs:
        raise RuntimeError(f"No low/high alm files (lmax={lmax}) found under {data_dir}")
    mode = "INCLUDING test (sanity check)" if args.include_test \
        else f"excluding {args.test_cosmo}"
    print(f"[fit] {len(runs)} training runs ({mode})", flush=True)

    sum_low = sum_high = None
    counts = None
    for lo, hi in runs:
        low = np.load(lo, mmap_mode="r")
        high = np.load(hi, mmap_mode="r")
        n = min(low.shape[0], high.shape[0])
        if sum_low is None:
            sum_low = np.zeros((n, lmax + 1))
            sum_high = np.zeros((n, lmax + 1))
            counts = np.zeros(n)
        for i in range(min(n, sum_low.shape[0])):
            sum_low[i] += hp.alm2cl(_alm(np.asarray(low[i]), N_alm), lmax=lmax)
            sum_high[i] += hp.alm2cl(_alm(np.asarray(high[i]), N_alm), lmax=lmax)
            counts[i] += 1
        print(f"  processed {lo.parent}", flush=True)

    mean_low = sum_low / counts[:, None]
    mean_high = sum_high / counts[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.sqrt(np.where(mean_low > 0, mean_high / mean_low, 1.0))
    T = np.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T=T, lmax=lmax, mean_low=mean_low.astype(np.float32),
             mean_high=mean_high.astype(np.float32))
    print(f"[fit] saved transfer function T{T.shape} to {args.out}", flush=True)


# ---------------------------------------------------------------------------
# Apply: corrected_alm = low_alm * T(ell, shell); alm2map
# ---------------------------------------------------------------------------

def apply(args):
    tf = np.load(args.transfer)
    T = tf["T"]                          # (n_shells_train, lmax+1)
    lmax = int(tf["lmax"])
    N_alm = (lmax + 1) * (lmax + 2) // 2
    ell = ell_of_flat_index(lmax)

    run = Path(args.run_dir)
    low = np.load(run / f"low_alms_lmax{lmax}.npy")
    n_shells = low.shape[0]
    npix = hp.nside2npix(args.nside)
    corrected = np.zeros((n_shells, npix), dtype=np.float32)

    for i in range(n_shells):
        Ti = T[min(i, T.shape[0] - 1)].copy()      # per-shell transfer
        if args.ell_min > 0:                         # leave large scales untouched
            Ti[: args.ell_min] = 1.0
        tvec = Ti[ell]                               # per-mode scale (N_alm,)
        v = low[i].astype(np.float64)
        alm = (v[:N_alm] * tvec + 1j * v[N_alm:] * tvec)
        corrected[i] = hp.alm2map(alm, nside=args.nside, lmax=lmax).astype(np.float32)
        if i % 10 == 0:
            print(f"  corrected shell {i}/{n_shells}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    info_npz = run / args.info_npz              # metadata (shell_info) from high npz
    extra = {}
    if info_npz.exists():
        d = np.load(info_npz, allow_pickle=True)
        extra = {k: d[k] for k in d.files if k != "shells"}
    np.savez(out, shells=corrected, **extra)
    print(f"[apply] saved corrected shells to {out}", flush=True)

    if args.plot_shells:
        _plot_cl(run, corrected, lmax, args.plot_shells, Path(args.plot_dir or out.parent))


def _plot_cl(run, corrected, lmax, shells, out_dir):
    """Cl-ratio plots: low/high and corrected/high (low,high from preprocessed alms)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(run / f"low_alms_lmax{lmax}.npy", mmap_mode="r")
    high = np.load(run / f"high_alms_lmax{lmax}.npy", mmap_mode="r")
    out_dir.mkdir(parents=True, exist_ok=True)
    ells = np.arange(lmax + 1)

    def cl_alm(v):
        a = (v[:N_alm] + 1j * v[N_alm:]).astype(np.complex128)
        return hp.alm2cl(a, lmax=lmax)

    for s in shells:
        if s >= corrected.shape[0]:
            continue
        cl_l = cl_alm(np.asarray(low[s])); cl_h = cl_alm(np.asarray(high[s]))
        cl_c = hp.anafast((corrected[s] / corrected[s].mean() - 1).astype(np.float64), lmax=lmax)
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax[0].loglog(ells, cl_l, label="DISCO (low)", color="seagreen")
        ax[0].loglog(ells, cl_c, label="corrected", color="steelblue")
        ax[0].loglog(ells, cl_h, "--", label="CosmoGrid (high)", color="tomato")
        ax[0].set_ylabel(r"$C_\ell$"); ax[0].legend(); ax[0].set_title(f"shell {s}")
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[1].semilogx(ells, cl_l / cl_h, ":", color="seagreen", label="low/high")
            ax[1].semilogx(ells, cl_c / cl_h, color="steelblue", label="corrected/high")
        ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0.5, 1.5)
        ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel("ratio"); ax[1].legend()
        fig.tight_layout(); fig.savefig(out_dir / f"cl_shell{s:03d}.png", dpi=150); plt.close(fig)
        print(f"  [plot] {out_dir / f'cl_shell{s:03d}.png'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit")
    pf.add_argument("--data-dir", required=True)
    pf.add_argument("--lmax", type=int, default=3000)
    pf.add_argument("--test-cosmo", default="")
    pf.add_argument("--include-test", action="store_true",
                    help="SANITY CHECK: include the test cosmology in the fit (expect "
                         "corrected/high ~ 1 by construction). Default is leave-one-out.")
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=fit)

    pa = sub.add_parser("apply")
    pa.add_argument("--transfer", required=True)
    pa.add_argument("--run-dir", required=True,
                    help="Test run dir with low_alms_lmax{lmax}.npy (+ high_alms for plots).")
    pa.add_argument("--info-npz", default="compressed_shells.npz",
                    help="npz to copy non-shell metadata (shell_info) from.")
    pa.add_argument("--nside", type=int, default=2048)
    pa.add_argument("--ell-min", type=int, default=0,
                    help="Leave ell<ell_min untouched (T=1) — correct only small scales.")
    pa.add_argument("--plot-shells", type=int, nargs="*", default=[3, 30, 50])
    pa.add_argument("--plot-dir", default="")
    pa.add_argument("--out", required=True)
    pa.set_defaults(func=apply)

    args = p.parse_args()
    args.func(args)
