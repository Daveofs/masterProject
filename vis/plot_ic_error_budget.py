#!/usr/bin/env python3
"""Getting Disco-DJ onto PkdGrav3, one fix at a time.

Four simulations of the same cosmology, each divided by the PkdGrav3 run started from
the same initial conditions. Nothing here is linear theory: every curve is a run that
was actually performed, so the stages are the history of the setup rather than an
illustration of it.

  (1) no Omega_rad        Disco-DJ's background integrated without radiation at all
                          (early versions offered no input for it)          +5.7%
  (2) + Omega_rad         radiation supplied, but Disco-DJ still started from
                          CONCEPT's massive-neutrino linear input           +3.1%
  (3) + backscaled ICs    started instead from the particle load PkdGrav3
                          generates by backscaling                          -0.8%
  (4) + CDM+baryon T(k)   the transfer table (and its sigma_8) describing the
                          field the particles actually carry                -0.2%

Percentages are medians over k in [0.01, 0.02] h/Mpc -- the largest scales the box
represents, where all four offsets are flat and where the residual of stage (3) is
visible. Over a wider band stage (3) averages to +0.1% and its remaining percent-level
tilt is hidden.

Each ratio uses its OWN matched PkdGrav3 reference: the quantity being compared is
"Disco-DJ against the reference started from the same ICs". The references are not
interchangeable between stages.

The steep decline of every curve beyond k ~ 0.2 h/Mpc is Disco-DJ's particle-mesh force
resolution, not an initial-condition effect.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_ic_error_budget.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_AXIS, FS_TICK, FS_LEGEND = 16, 14, 12.5
S = Path("/capstor/scratch/cscs/damrein")
PK, CUS, BS = S / "pk", S / "outputs/pk_custom", S / "outputs/pk_backscaling"
# largest scales the 900 Mpc/h box represents: where the offsets are flat and where
# stage (3)'s residual percent is still visible
BAND = (1e-2, 2e-2)

STAGES = [
    (PK / "pk_disco_bullfrog_fiducial_david.txt", PK / "pk_pkd_fiducial_david.txt",
     r"(1) no $\Omega_{\rm rad}$", "#B85F34", 2.0),
    (CUS / "pk_disco_concept_nu_omega_r_5.58e-5.txt", CUS / "pk_pkd_standard_fiducial.txt",
     r"(2) $+\ \Omega_{\rm rad}$, \textsc{concept} input", "#C2874B", 2.0),
    (CUS / "pk_disco_backscaling_omega_r_5.58e-5.txt", CUS / "pk_pkd_standard_fiducial.txt",
     r"(3) $+$ backscaled ICs", "#5AA9C9", 2.2),
    (BS / "pk_disco_transfer_cdm+b.txt", BS / "pk_pkd_fiducial.txt",
     r"(4) $+$ CDM$+$baryon $T(k)$", "#28866A", 3.0),
]


def load(p: Path):
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(S / "outputs/plots/ic"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=2)
    for y in (0.99, 1.01):
        ax.axhline(y, color="0.65", ls=":", lw=0.8, zorder=1)

    print(f"[budget] median over k in {BAND} h/Mpc, each vs its own matched reference")
    for num, den, label, colour, lw in STAGES:
        k, p = load(num); kr, pr = load(den)
        n = min(len(k), len(kr))
        assert np.allclose(k[:n], kr[:n]), f"k grid differs: {num.name} vs {den.name}"
        r = p[:n] / pr[:n]; kk = k[:n]
        m = (kk >= BAND[0]) & (kk <= BAND[1])
        off = 100.0 * (np.median(r[m]) - 1.0)
        plain = label.replace(r"\textsc{concept}", "CONCEPT").replace("$", "")
        print(f"  {plain:38s} {off:+6.2f}%   (scatter {np.std(r[m]):.4f})   "
              f"{num.name}")
        ax.semilogx(kk, r, "-", color=colour, lw=lw, zorder=3 if lw > 2.5 else 2,
                    label=f"{label}  ({off:+.1f}\\%)" if plt.rcParams["text.usetex"]
                          else f"{label.replace(chr(92)+'textsc{concept}','CONCEPT')}  ({off:+.1f}%)")

    ax.set_xlabel(r"$k\ \ [h/\mathrm{Mpc}]$", fontsize=FS_AXIS)
    ax.set_ylabel("$P(k)$ Disco-DJ / PkdGrav3", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, which="both", lw=0.5); ax.set_axisbelow(True)
    # The data begin at k = 0.0099 h/Mpc, essentially the box fundamental
    # (2*pi/900 = 0.0070), so there is nothing further left to show. Instead give the
    # large scales most of the axis: the flat offsets that this figure is about all live
    # at k < 0.1, while everything beyond ~0.2 is the force-resolution roll-off, which is
    # a different story and only needs to be visible, not resolved.
    ax.set_xlim(9.3e-3, 0.4)
    ax.set_ylim(0.974, 1.068)
    # Legend ABOVE the axes: with the x-range narrowed onto the large scales there is no
    # in-panel corner that stays clear of all four curves, and every in-panel placement
    # tried covered either stage 1/2 or -- worse -- stages 3 and 4, which are the ones
    # the figure exists to show. Outside costs nothing but a little height.
    ax.legend(fontsize=FS_LEGEND, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=2, framealpha=1.0, borderpad=0.45, labelspacing=0.4,
              columnspacing=1.6, handlelength=2.2, handletextpad=0.6)

    out = out_dir / "ic_error_budget.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[budget] -> {out}")


if __name__ == "__main__":
    main()
