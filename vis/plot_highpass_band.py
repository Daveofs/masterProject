#!/usr/bin/env python3
"""Where the generative correction is allowed to act, against where it is needed.

The flow and diffusion pipelines learn a HIGH-PASS residual: the target is
highpass(high - low), the corrected patch is low + highpass(sample). The filter is a
radial raised cosine on the patch's 2D FFT, set by two fractions of the patch Nyquist
frequency -- hp-cutoff (where it starts) and hp-transition (how wide the ramp is).
Both are ANGULAR: for 256-pixel patches cut at the map's own pixel scale,

    l_Nyquist = pi / hp.nside2resol(nside)          (1572 at nside=512)

so the mask is zero below hpc*l_Nyq and unity above (hpc+hpt)*l_Nyq, for every shell
regardless of redshift -- unlike the transfer pipeline, whose onset follows a fixed
COMOVING scale and therefore moves with the shell.

The mask multiplies the field, so the corrective POWER admitted at each multipole goes
as mask^2. That is the top panel. The bottom panel is what has to be repaired: the
measured per-shell deficit.

The point of the figure: with the production setting (hpc=0.05, hpt=0.20) the cutoff
is already low -- l = 79 -- but the RAMP is 0.20 of Nyquist wide, so full transmission
only arrives at l = 393. Almost none of the correction reaches l ~ 100-400, which is
exactly where the nearest shells are deficient. Removing the ramp (hpt = 0) fixes that
without moving the cutoff at all.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_highpass_band.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/users/damrein/masterProject/ml")
from analysis.full_sky import od_cl                                  # noqa: E402

FS_AXIS, FS_TICK, FS_LEGEND = 15, 13, 12
C_NOW, C_NEW = "#B85F34", "#28866A"
SHELL_COLS = ["#2B2B2B", "#5C6480", "#8E96AC", "#B9BFD0"]


def mask_power(ell, l_nyq, hpc, hpt):
    """Fraction of the corrective POWER admitted at each multipole (mask squared)."""
    r = np.asarray(ell, float) / l_nyq
    t = np.clip((r - hpc) / max(hpt, 1e-9), 0.0, 1.0)
    return (0.5 * (1.0 - np.cos(np.pi * t))) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/capstor/scratch/cscs/damrein/grid/cosmo_000176/run_0")
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--shells", type=int, nargs="+", default=[2, 5, 10, 20])
    ap.add_argument("--now", type=float, nargs=2, default=[0.05, 0.20], metavar=("HPC", "HPT"))
    ap.add_argument("--new", type=float, nargs=2, default=[0.05, 0.0], metavar=("HPC", "HPT"))
    ap.add_argument("--lmax", type=int, default=1500)
    ap.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/ic")
    a = ap.parse_args()
    run = Path(a.run_dir); out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    l_nyq = math.pi / hp.nside2resol(a.nside)
    ell = np.arange(2, a.lmax + 1)
    print(f"[hpf] nside={a.nside}   l_Nyquist = pi/hp.nside2resol = {l_nyq:.0f}")
    for tag, (hpc, hpt) in (("current ", a.now), ("proposed", a.new)):
        lo_l, hi_l = hpc * l_nyq, (hpc + hpt) * l_nyq
        print(f"  {tag}  hpc={hpc}, hpt={hpt}  ->  zero below l={lo_l:.0f}, "
              f"full above l={hi_l:.0f}" + ("   (step)" if hpt == 0 else ""))
    print(f"\n  {'l':>6s} {'power now':>11s} {'power proposed':>15s}")
    for L in (50, 100, 200, 300, 400, 800):
        print(f"  {L:6d} {mask_power(L, l_nyq, *a.now):11.3f} "
              f"{mask_power(L, l_nyq, *a.new):15.3f}")

    lo = np.load(run / f"low_shells_nside={a.nside}.npy", mmap_mode="r")
    hi = np.load(run / f"high_shells_nside={a.nside}.npy", mmap_mode="r")
    info = np.load(run / "compressed_shells.npz", allow_pickle=True)["shell_info"]
    z = 0.5 * (info["lower_z"].astype(float) + info["upper_z"].astype(float))

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.08})

    # ---- top: what the filter lets through -------------------------------------
    for (hpc, hpt), col, lab, ls in ((a.now, C_NOW, "current", "-"),
                                     (a.new, C_NEW, "proposed", "--")):
        lab_full = (f"{lab}:  hpc$=${hpc}, hpt$=${hpt}"
                    + (f"  (step at $\\ell={hpc*l_nyq:.0f}$)" if hpt == 0
                       else f"  ($\\ell\\,{hpc*l_nyq:.0f}\\to{(hpc+hpt)*l_nyq:.0f}$)"))
        ax.semilogx(ell, mask_power(ell, l_nyq, hpc, hpt), ls, color=col, lw=2.6,
                    label=lab_full)
    ax.set_ylabel("corrective power\nadmitted", fontsize=FS_AXIS)
    ax.set_ylim(-0.04, 1.08); ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, which="both", lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEGEND, loc="lower right", framealpha=1.0, borderpad=0.4)

    # ---- bottom: what needs repairing ------------------------------------------
    print("\n[shells] low/high Cl ratio")
    for i, s in enumerate(a.shells):
        cl_l = od_cl(np.asarray(lo[s], np.float32), a.lmax)
        cl_h = od_cl(np.asarray(hi[s], np.float32), a.lmax)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (cl_l / cl_h)[2:a.lmax + 1]
        k = 25
        rs = np.convolve(r, np.ones(k) / k, mode="same")
        axd.semilogx(ell, rs, "-", color=SHELL_COLS[i % len(SHELL_COLS)], lw=1.8,
                     label=f"shell {s}  ($z={z[s]:.2f}$)")
        print(f"  shell {s:2d} (z={z[s]:.2f}):  l=100 {np.mean(r[80:120]):.3f}   "
              f"l=200 {np.mean(r[180:220]):.3f}   l=400 {np.mean(r[380:420]):.3f}")
    axd.axhline(1.0, color="k", ls="--", lw=0.9)
    axd.set_xlabel(r"multipole $\ell$", fontsize=FS_AXIS)
    axd.set_ylabel(r"$C_\ell^{\rm low}/C_\ell^{\rm high}$", fontsize=FS_AXIS)
    axd.tick_params(labelsize=FS_TICK)
    axd.grid(alpha=0.25, which="both", lw=0.5); axd.set_axisbelow(True)
    axd.set_ylim(0.60, 1.04)
    axd.legend(fontsize=FS_LEGEND, loc="lower left", framealpha=1.0, borderpad=0.4)

    # the band the whole question is about
    for _a in (ax, axd):
        _a.axvspan(100, 400, color="#C2874B", alpha=0.10, lw=0, zorder=0)
    ax.text(200, 0.5, "the band in question", fontsize=FS_LEGEND - 1, color="#8a5f30",
            ha="center", va="center", rotation=0)
    ax.set_xlim(30, a.lmax)

    out = out_dir / "highpass_band.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\n[hpf] -> {out}")


if __name__ == "__main__":
    main()
