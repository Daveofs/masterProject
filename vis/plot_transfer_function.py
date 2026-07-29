#!/usr/bin/env python3
"""Plot the fitted/emulated transfer function T(ell, shell) and its companion
phase correlation R(ell, shell), from a `transfer_function.py emulate` output.

These are the two objects the harmonic-space correction pipeline is built on
(report Sec. "Harmonic-Space Transfer-Function Pipeline"):

  T(ell, s) = sqrt( <Cl_high(s)> / <Cl_low(s)> )   the per-mode amplitude gain
  R(ell, s) = <Cl_lowxhigh> / sqrt(<Cl_low><Cl_high>)   the phase correlation

Both are stored per held-out cosmology as transfer_cosmo_*.npz by the `emulate`
step. NOTE what is plotted is the RAW emulator prediction: the per-shell
raised-cosine ramp that forces T -> 1 below ell_min(s) is applied later, at
apply time (apply_transfer.py), not here -- so the low-ell behaviour shown is
what the emulator itself predicts, which is the honest thing to show when the
point is to characterise the emulator.

Two panels:
  left  -- T(ell) for a selection of shells, coloured by shell redshift
  right -- R(ell) for the same shells

Usage
-----
    /users/damrein/miniforge3/bin/python plot_transfer_function.py \
        --run-dir /capstor/scratch/cscs/damrein/outputs/transfer/2884508_final

    # a different cosmology / other shells / spread across cosmologies
    /users/damrein/miniforge3/bin/python plot_transfer_function.py \
        --run-dir <dir> --cosmo cosmo_074758 --shells 5 20 40 60 --spread
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===========================================================================
# CONFIG
# ===========================================================================

# where the shell tables AND the low/high alm files live (for --raw-r-spread)
GRID_DIR = "/capstor/scratch/cscs/damrein/grid"
SHELL_NPZ = "compressed_shells.npz"

# 21-point log-space boxcar, matching the production run (report Sec. 3.3.1,
# "Configuration and training") -- used only by --raw-r-spread, to smooth a
# SINGLE cosmology's own (cl_low, cl_high, cl_cross) before dividing, the same
# reason transfer_function.py's smooth_cl() smooths before building T's own
# per-run training target: a single realisation's per-ell Cl is noisy, but the
# physical r(ell) is expected to vary smoothly with ell.
RAW_R_SMOOTH_WINDOW = 21

# shells to draw, spanning the lightcone
SHELLS = (5, 15, 25, 35, 45, 60)

# --- typography, matching vis/plot_cl_ratio.py (thesis figures at ~\textwidth) ---
_AXIS_FONTSIZE = 17
_TICK_FONTSIZE = 14
_LEGEND_FONTSIZE = 12

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _style(ax):
    #ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=_TICK_FONTSIZE)


def shell_redshifts(cosmo: str, n_shells: int) -> np.ndarray:
    """Mid-shell redshift from the cosmology's own shell table."""
    run = Path(GRID_DIR) / cosmo / "run_0" / SHELL_NPZ
    if not run.exists():
        print(f"  [warn] {run} missing; labelling shells by index")
        return np.arange(n_shells, dtype=float)
    with np.load(run, allow_pickle=False) as f:
        si = f["shell_info"]
    z = 0.5 * (si["lower_z"].astype(float) + si["upper_z"].astype(float))
    return z[:n_shells]


def _alm_fname(kind: str, lmax: int) -> str:
    """Matches transfer_function.py's alm_fname (log_density=False in this run,
    confirmed from the transfer_cosmo_*.npz 'log_density' field)."""
    return f"{kind}_alms_lmax{lmax}.npy"


def _to_alm(vec: np.ndarray, n_alm: int) -> np.ndarray:
    return (vec[:n_alm] + 1j * vec[n_alm:]).astype(np.complex128)


def _smooth_log(cl: np.ndarray, window: int) -> np.ndarray:
    """Boxcar-smooth a positive spectrum in log10-ell-bin space -- identical
    construction to transfer_function.py's smooth_cl()."""
    from scipy.ndimage import uniform_filter1d
    log_cl = np.log10(np.clip(cl, 1e-30, None))
    return 10.0 ** uniform_filter1d(log_cl, size=window, mode="nearest")


def _smooth_signed(cl: np.ndarray, window: int) -> np.ndarray:
    """Same boxcar, applied to a spectrum that can be negative (the cross
    spectrum): smooth the magnitude in log space, then restore the sign. This
    is a choice made FOR THIS DIAGNOSTIC SCRIPT ONLY -- the production pipeline
    never computes a per-run r(ell) at all (R is only ever an ensemble average
    over the whole training set, see transfer_function.py's _finalize_train),
    so there is no "reference" per-run smoothing to reproduce exactly. Smoothing
    the cross spectrum the same way cl_low/cl_high are already smoothed for T's
    own per-run training target is the natural analogue."""
    from scipy.ndimage import uniform_filter1d
    sign = np.sign(cl)
    log_mag = np.log10(np.clip(np.abs(cl), 1e-30, None))
    return sign * 10.0 ** uniform_filter1d(log_mag, size=window, mode="nearest")


def raw_r_per_cosmology(cosmo: str, shells: list[int], lmax: int,
                        window: int = RAW_R_SMOOTH_WINDOW) -> dict[int, np.ndarray]:
    """r(ell) computed directly from ONE cosmology's own low/high alm files, for
    each requested shell -- the genuine per-cosmology phase correlation, as
    opposed to the single ensemble-averaged R the emulator pipeline stores.
    Requires access to grid/<cosmo>/run_0/{low,high}_alms_lmax{lmax}.npy."""
    run = Path(GRID_DIR) / cosmo / "run_0"
    low_p, high_p = run / _alm_fname("low", lmax), run / _alm_fname("high", lmax)
    if not (low_p.exists() and high_p.exists()):
        print(f"  [warn] alm files missing for {cosmo} under {run}; skipping")
        return {}
    n_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(low_p, mmap_mode="r")
    high = np.load(high_p, mmap_mode="r")
    out = {}
    for s in shells:
        if s >= low.shape[0] or s >= high.shape[0]:
            continue
        al = _to_alm(np.asarray(low[s]), n_alm)
        ah = _to_alm(np.asarray(high[s]), n_alm)
        cl_l = _smooth_log(hp.alm2cl(al, lmax=lmax), window)
        cl_h = _smooth_log(hp.alm2cl(ah, lmax=lmax), window)
        cl_x = _smooth_signed(hp.alm2cl(al, ah, lmax=lmax), window)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cl_x / np.sqrt(cl_l * cl_h)
        out[s] = np.clip(np.nan_to_num(r, nan=0.0), 0.0, 1.0)
    return out


def load_transfer(run_dir: Path, cosmo: str | None):
    """(T, R, lmax, cosmo_name) from one transfer_cosmo_*.npz."""
    files = sorted(run_dir.glob("transfer_cosmo_*.npz"))
    if not files:
        raise SystemExit(f"no transfer_cosmo_*.npz under {run_dir}")
    if cosmo:
        match = [f for f in files if cosmo in f.name]
        if not match:
            raise SystemExit(f"{cosmo} not among: {[f.name for f in files]}")
        path = match[0]
    else:
        path = files[0]
    name = re.search(r"(cosmo_\d+)", path.name).group(1)
    with np.load(path) as d:
        return d["T"], d["R"], int(d["lmax"]), name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="a transfer run directory holding transfer_cosmo_*.npz")
    p.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/transfer")
    p.add_argument("--cosmo", default=None,
                   help="which held-out cosmology to draw (default: first found)")
    p.add_argument("--shells", type=int, nargs="+", default=list(SHELLS))
    p.add_argument("--spread", action="store_true",
                   help="overlay every cosmology in the run dir as thin lines, to "
                        "show how far the emulator moves T between cosmologies")
    p.add_argument("--raw-r-spread", action="store_true",
                   help="ALSO overlay the genuine per-cosmology r(ell,s), computed "
                        "directly from each held-out cosmology's own alm files "
                        "(needs grid/<cosmo>/run_0/{low,high}_alms_lmax<lmax>.npy). "
                        "Slower (an alm2cl pass per shell per cosmology) but shows "
                        "what a per-cosmology r emulator would actually be "
                        "learning from, unlike --spread's right panel (which only "
                        "ever replots the same shared, non-cosmology-dependent R "
                        "the pipeline currently stores).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T, R, lmax, cosmo = load_transfer(run_dir, args.cosmo)
    n_shells = T.shape[0]
    z = shell_redshifts(cosmo, n_shells)
    ell = np.arange(lmax + 1)
    shells = [s for s in args.shells if 0 <= s < n_shells]
    print(f"[transfer] {cosmo}: T{T.shape}, lmax={lmax}, drawing shells {shells}")

    cmap = plt.get_cmap("viridis")
    zs = np.array([z[s] for s in shells])
    norm = plt.Normalize(vmin=zs.min(), vmax=zs.max())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ---- optional: every other cosmology, thin and grey, as context ----
    # r(ell,s) is not actually emulated per cosmology: `train` computes it once as
    # an average over the whole training set and stores that same array in the
    # emulator, which `emulate` then copies verbatim into every cosmology's npz
    # (verified: the R arrays of all 10 held-out files are bit-identical). Drawn
    # anyway, on request -- with ten identical curves stacked on top of each
    # other the panel is then itself the demonstration that R carries no
    # cosmology dependence, rather than an assertion made only in this comment.
    if args.spread:
        for f in sorted(run_dir.glob("transfer_cosmo_*.npz")):
            with np.load(f) as d:
                To, Ro = d["T"], d["R"]
            for s in shells:
                axes[0].plot(ell, To[s], color="0.75", lw=0.6, alpha=0.7, zorder=1)
                axes[1].plot(ell, Ro[s], color="0.75", lw=0.6, alpha=0.7, zorder=1)

    # ---- optional: the GENUINE per-cosmology r(ell,s), from raw alm files ----
    if args.raw_r_spread:
        held_out = sorted(f.name for f in run_dir.glob("transfer_cosmo_*.npz"))
        held_out = [re.search(r"(cosmo_\d+)", n).group(1) for n in held_out]
        print(f"[transfer] computing raw per-cosmology r(ell,s) for "
              f"{len(held_out)} held-out cosmologies x {len(shells)} shells "
              f"(alm2cl, this is the slow part) ...")
        for c in held_out:
            r_by_shell = raw_r_per_cosmology(c, shells, lmax)
            for s, r in r_by_shell.items():
                axes[1].plot(ell, r, color="0.75", lw=0.6, alpha=0.7, zorder=1)

    for s in shells:
        c = cmap(norm(z[s]))
        axes[0].plot(ell, T[s], color=c, lw=1.8, zorder=2)
        axes[1].plot(ell, R[s], color=c, lw=1.8, zorder=2)

    axes[0].axhline(1.0, color=MUTED, lw=0.9, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_xlim(2, lmax)
    axes[0].set_xlabel(r"Multipole $\ell$", fontsize=_AXIS_FONTSIZE)
    axes[0].set_ylabel(r"$T(\ell, s)$", fontsize=_AXIS_FONTSIZE)
    _style(axes[0])

    axes[1].axhline(1.0, color=MUTED, lw=0.9, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlim(2, lmax)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel(r"Multipole $\ell$", fontsize=_AXIS_FONTSIZE)
    axes[1].set_ylabel(r"$r(\ell, s)$", fontsize=_AXIS_FONTSIZE)
    _style(axes[1])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("shell redshift $z$", fontsize=_AXIS_FONTSIZE)
    cb.ax.tick_params(labelsize=_TICK_FONTSIZE)

    if args.spread:
        axes[0].legend(handles=[Line2D([], [], color="0.75", lw=0.6,
                                       label="other held-out cosmologies")],
                       fontsize=_LEGEND_FONTSIZE, loc="upper left")
    # The label is already in the panel next to it
    # if args.raw_r_spread:
    #    axes[1].legend(handles=[Line2D([], [], color="0.75", lw=0.6,
    #                                   label="other held-out cosmologies (raw, per-cosmology)")],
    #                   fontsize=_LEGEND_FONTSIZE, loc="lower left")

    tag = ("_spread" if args.spread else "") + ("_rawR" if args.raw_r_spread else "")
    out = out_dir / f"transfer_function_{cosmo}{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[transfer] wrote {out}")


if __name__ == "__main__":
    main()
