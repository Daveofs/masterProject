#!/usr/bin/env python3
"""How the 17 Mpc/h correction onset is obtained.

All three correction pipelines switch on at a per-shell multipole ell_min(s) that is
NOT chosen per shell: it is one fixed COMOVING length L, projected onto each shell
through the flat-sky (Limber) relation, ell = 2*pi*chi(z_s)/L. This script is where
that L comes from.

The measurement, per (cosmology, shell):

  1. Take the ratio of the two angular power spectra, R(l) = Cl_low / Cl_high, from
     the precomputed alms at lmax=3000 (nside=2048, the native resolution -- the
     nside=512 evaluation footing would cap the reach at l~1500 and truncate the
     onset on every distant shell).
  2. Smooth R with a boxcar, and define the ONSET l_onset as the first multipole at
     which the smoothed ratio falls below 1 - eps (eps = 1% by default). A 1% level
     is used rather than a deeper one deliberately: it is where the deficit BEGINS,
     which is the quantity that should be a fixed physical scale. Deeper thresholds
     measure how fast the deficit grows once started, which is shape, not onset.
  3. Convert to a comoving length, L = 2*pi*chi(s)/l_onset.

If the deficit really is set by the PM force resolution -- a fixed physical length --
then L is the same number on every shell of every cosmology, and l_onset vs chi is a
straight line through the origin whose slope is 2*pi/L. That line is the fit.

Shells whose onset falls outside [l_lo, l_hi] are dropped, not fitted: on the most
distant shells the onset lies beyond the measured band entirely (the field is still
close to linear there, so no deficit is resolvable), and near lmax the smoothed ratio
is dominated by the common shot-noise floor of the two runs rather than by clustering.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_onset_scale.py --n-cosmo 20
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_AXIS, FS_TICK, FS_LEGEND = 14, 12, 11
C_FIT, C_PTS, C_TH = "#B85F34", "#3F63A6", "#5C6480"
SHELL_COLS = ["#161D33", "#3F63A6", "#28866A", "#B85F34"]


def onset(cl_lo, cl_hi, eps, smooth, l_lo, l_hi, win, floor):
    """First multipole at which the smoothed low/high ratio drops below 1-eps AND
    stays there for a window of `win` multipoles.

    Three guards, each removing a way the naive "first crossing" lies:

      * `floor`  -- the shell must actually develop a deficit somewhere in the band
        (min ratio < floor). Distant shells are still near-linear and have none; a
        bare threshold test would nonetheless latch onto a noise dip and report a
        spurious onset (measured: 68 of 69 shells "usable" without this guard).
      * `win`    -- the departure must persist. A single smoothed excursion below
        1-eps is sample variance, not the onset of a deficit.
      * `l_lo`   -- the ratio must still be ABOVE the threshold at l_lo, otherwise
        the onset lies below the measured band and cannot be located from it. The
        innermost shells fail this and are excluded rather than pinned to l_lo.

    Returns (l_onset, smoothed_ratio, status)."""
    edge = smooth // 2 + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.nan_to_num(cl_lo / cl_hi, nan=1.0, posinf=1.0, neginf=1.0)
    Rs = np.convolve(R, np.ones(smooth) / smooth, mode="same")
    start = max(l_lo, edge)
    if Rs[start:len(Rs) - edge].min() > floor:
        return np.nan, Rs, "no-deficit"
    if Rs[start] < 1.0 - eps:
        return np.nan, Rs, "onset-below-band"
    below = Rs < (1.0 - eps)
    for l in range(start, min(l_hi, len(Rs) - edge - win)):
        if below[l:l + win].all():
            return float(l), Rs, "ok"
    return np.nan, Rs, "no-persistent-crossing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="/capstor/scratch/cscs/damrein/grid")
    ap.add_argument("--lmax", type=int, default=3000)
    ap.add_argument("--n-cosmo", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.01, help="deficit level defining the onset")
    ap.add_argument("--smooth", type=int, default=41)
    ap.add_argument("--l-lo", type=int, default=60)
    ap.add_argument("--win", type=int, default=150,
                   help="multipoles the departure must persist for")
    ap.add_argument("--floor", type=float, default=0.90,
                   help="shell must reach this ratio somewhere, else it has no onset")
    ap.add_argument("--l-hi", type=int, default=1800)
    ap.add_argument("--demo-shells", type=int, nargs="+", default=[5, 15, 25, 33])
    ap.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/ic")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    nalm = hp.Alm.getsize(a.lmax)
    cx = lambda v: (v[:nalm] + 1j * v[nalm:]).astype(np.complex128)

    runs = [Path(p) for p in sorted(glob.glob(f"{a.grid}/cosmo_*/run_0"))
            if Path(p, f"low_alms_lmax{a.lmax}.npy").exists()][:a.n_cosmo]
    print(f"[onset] {len(runs)} cosmologies, lmax={a.lmax}, eps={a.eps:.3f}, "
          f"accept l in [{a.l_lo},{a.l_hi}]")

    chi_all, l_all, z_all, cos_all, demo, stat = [], [], [], [], {}, {}
    for n, r in enumerate(runs):
        lo = np.load(r / f"low_alms_lmax{a.lmax}.npy", mmap_mode="r")
        hi = np.load(r / f"high_alms_lmax{a.lmax}.npy", mmap_mode="r")
        info = np.load(r / "compressed_shells.npz", allow_pickle=True)["shell_info"]
        chi = info["shell_com"].astype(float)
        z = 0.5 * (info["lower_z"].astype(float) + info["upper_z"].astype(float))
        keep = 0
        for s in range(len(chi)):
            cl_l = hp.alm2cl(cx(np.asarray(lo[s])), lmax=a.lmax)
            cl_h = hp.alm2cl(cx(np.asarray(hi[s])), lmax=a.lmax)
            lon, Rs, st = onset(cl_l, cl_h, a.eps, a.smooth, a.l_lo, a.l_hi, a.win, a.floor)
            stat[st] = stat.get(st, 0) + 1
            if n == 0 and s in a.demo_shells:
                demo[s] = (Rs, lon, z[s], chi[s])
            if np.isfinite(lon):
                chi_all.append(chi[s]); l_all.append(lon); z_all.append(z[s])
                cos_all.append(r.parent.name); keep += 1
        print(f"  [{n+1:2d}/{len(runs)}] {r.parent.name}: {keep}/{len(chi)} shells usable",
              flush=True)

    chi_all = np.asarray(chi_all); l_all = np.asarray(l_all); z_all = np.asarray(z_all)
    cos_all = np.asarray(cos_all)
    np.savez(out_dir / "onset_measurements.npz", chi=chi_all, ell=l_all, z=z_all,
             cosmo=cos_all)
    print("\n[onset] per-cosmology fitted L [Mpc/h]:")
    per = []
    for c in np.unique(cos_all):
        m = cos_all == c
        Lc = 2 * np.pi * (chi_all[m] ** 2).sum() / (chi_all[m] * l_all[m]).sum()
        per.append(Lc)
        print(f"    {c}: n={m.sum():3d}  L={Lc:5.2f}")
    per = np.asarray(per)
    print(f"  across cosmologies: median {np.median(per):.2f}  "
          f"min {per.min():.2f}  max {per.max():.2f}  std {per.std():.2f}")
    # least-squares slope through the origin: l = m*chi, L = 2 pi / m
    m = float((chi_all * l_all).sum() / (chi_all ** 2).sum())
    L = 2 * np.pi / m
    Lper = 2 * np.pi * chi_all / l_all                      # per-point implied length
    print(f"\n[onset] fit over {len(l_all)} (cosmology, shell) points")
    print(f"  slope 2pi/L = {m:.5f}  ->  L = {L:.2f} Mpc/h")
    print(f"  per-point L: median {np.median(Lper):.2f}  16-84% "
          f"[{np.percentile(Lper,16):.1f}, {np.percentile(Lper,84):.1f}]  Mpc/h")
    print(f"  production setting is L = 17 Mpc/h")
    print(f"  shell status counts: {stat}")

    # ---- figure ----------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # (a) the measurement
    ell = np.arange(a.lmax + 1)
    for i, s in enumerate(sorted(demo)):
        Rs, lon, zz, cc = demo[s]
        col = SHELL_COLS[i % len(SHELL_COLS)]
        axL.semilogx(ell[2:], Rs[2:], "-", color=col, lw=1.9,
                     label=f"shell {s}  ($z={zz:.2f}$)")
        if np.isfinite(lon):
            axL.plot([lon], [1.0 - a.eps], "o", color=col, ms=7, mec="white", mew=1.2, zorder=5)
    axL.axhline(1.0, color="k", ls=":", lw=0.9)
    axL.set_xlabel(r"multipole $\ell$", fontsize=FS_AXIS)
    axL.set_ylabel(r"$C_\ell^{\rm low}/C_\ell^{\rm high}$", fontsize=FS_AXIS)
    axL.set_xlim(30, a.lmax); axL.set_ylim(0.55, 1.06)
    axL.tick_params(labelsize=FS_TICK); axL.grid(alpha=0.25, which="both", lw=0.5)
    axL.set_axisbelow(True)
    axL.legend(fontsize=FS_LEGEND, loc="lower left", framealpha=1.0, borderpad=0.4)
    axL.set_title("(a)  measure where the deficit starts", fontsize=FS_AXIS, loc="left")

    # (b) the fit
    axR.plot(chi_all, l_all, "o", color=C_PTS, ms=4.0, alpha=0.35, mec="none",
             label=(f"{len(l_all)} shells" if len(np.unique(cos_all)) == 1
                    else f"{len(l_all)} (cosmology, shell) pairs"))
    xg = np.linspace(0, chi_all.max() * 1.06, 50)
    axR.plot(xg, m * xg, "-", color=C_FIT, lw=2.4,
             label=f"fit: $\\ell = 2\\pi\\chi/L$,  $L = {L:.1f}$ Mpc$/h$")
    axR.plot(xg, 2 * np.pi * xg / 17.0, "--", color="k", lw=1.3, alpha=0.75,
             label=r"production: $L = 17$ Mpc$/h$")
    axR.set_xlabel(r"comoving distance to shell  $\chi$  [Mpc$/h$]", fontsize=FS_AXIS)
    axR.set_ylabel(r"measured onset  $\ell_{\rm onset}$", fontsize=FS_AXIS)
    axR.set_xlim(0, chi_all.max() * 1.06); axR.set_ylim(0, l_all.max() * 1.12)
    axR.tick_params(labelsize=FS_TICK); axR.grid(alpha=0.25, lw=0.5)
    axR.set_axisbelow(True)
    axR.legend(fontsize=FS_LEGEND, loc="upper left", framealpha=1.0, borderpad=0.4)
    axR.set_title("(b)  one fixed comoving length fits every shell", fontsize=FS_AXIS, loc="left")

    out = out_dir / "onset_scale.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[onset] -> {out}")


if __name__ == "__main__":
    main()
