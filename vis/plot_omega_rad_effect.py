#!/usr/bin/env python3
"""The radiation density in Disco-DJ's background, and what omitting it costs.

Early Disco-DJ versions had no Omega_rad input at all: the background was integrated
with matter and dark energy only. Radiation is negligible at z=0 but not at the
starting redshift of these runs, so leaving it out mis-normalises the growth factor
that carries the initial conditions forward -- and, exactly like the backscaling
failure in plot_backscaling_vs_concept.py, it shows up as a FLAT offset across the
whole linear regime rather than as a distortion.

Two views, both against PkdGrav3:

  left   a run from before the input existed -- 5.9% too much large-scale power, flat
         to <0.1% (the symplectic run gives 4.8%, i.e. the offset is not an artefact of
         the integrator; it is loaded and printed but not drawn)
  right  the controlled pair once the input was available: the same configuration with
         Omega_r left unset vs set, which is the clean one-variable test

Usage
-----
    /users/damrein/miniforge3/bin/python plot_omega_rad_effect.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_AXIS, FS_TICK, FS_LEGEND = 15, 13, 12
SCRATCH = Path("/capstor/scratch/cscs/damrein")
C_OFF, C_OFF2, C_ON = "#B85F34", "#C98A5E", "#28866A"


def load(p: Path):
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1]


def ratio(num: Path, den: Path):
    k, p = load(num)
    kr, pr = load(den)
    n = min(len(k), len(kr))
    assert np.allclose(k[:n], kr[:n]), f"k grids differ: {num.name} vs {den.name}"
    return k[:n], p[:n] / pr[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(SCRATCH / "outputs/plots/ic"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pk, bs = SCRATCH / "pk", SCRATCH / "outputs/pk_backscaling"
    k_s, r_s = ratio(pk / "pk_disco_symplectic_fiducial_david.txt", pk / "pk_pkd_fiducial_david.txt")
    k_b, r_b = ratio(pk / "pk_disco_bullfrog_fiducial_david.txt", pk / "pk_pkd_fiducial_david.txt")
    k_n, r_n = ratio(bs / "pk_disco_fiducial_omega_r_not_fixed.txt",
                     bs / "pk_pkd_fiducial_omega_r_not_fixed.txt")
    k_f, r_f = ratio(bs / "pk_disco_fiducial.txt", bs / "pk_pkd_fiducial.txt")

    def med(k, r):
        m = k < 0.1
        return np.median(r[m]), np.std(r[m])
    print("[omega_r] median ratio to PkdGrav3 over k < 0.1 h/Mpc (scatter in brackets)")
    for nm, (k, r) in (("no Omega_rad, symplectic", (k_s, r_s)),
                       ("no Omega_rad, bullfrog", (k_b, r_b)),
                       ("Omega_r unset", (k_n, r_n)),
                       ("Omega_r set", (k_f, r_f))):
        m, s = med(k, r)
        print(f"  {nm:26s} {m:.4f}  ({s:.4f})")
    m_n, m_f = med(k_n, r_n)[0], med(k_f, r_f)[0]
    print(f"  -> setting Omega_r moves the large scales by "
          f"{100*(m_f - m_n):+.1f} percentage points")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax in axes:
        ax.axhline(1.0, color="k", ls="--", lw=0.9)
        for y in (0.99, 1.01):
            ax.axhline(y, color="0.6", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel(r"$k\ \ [h/\mathrm{Mpc}]$", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK); ax.grid(alpha=0.25, which="both", lw=0.5)
        ax.set_axisbelow(True); ax.set_xlim(1e-2, 1.0)

    # Only the bullfrog run is drawn -- it is the integrator the thesis uses, and the
    # figure is about Omega_rad, not about the integrator. The symplectic run is still
    # loaded and printed above as a cross-check that the offset is not integrator-specific.
    axes[0].plot(k_b, r_b, "-", color=C_OFF, lw=2.0,
                 label=f"Disco-DJ ({100*(med(k_b,r_b)[0]-1):+.1f}%)")
    axes[0].set_ylabel("$P(k)$ / PkdGrav3", fontsize=FS_AXIS)
    axes[0].legend(fontsize=FS_LEGEND, loc="lower left", framealpha=1.0)

    axes[1].plot(k_n, r_n, "-", color=C_OFF, lw=2.0,
                 label=f"$\\Omega_r$ unset ({100*(m_n-1):+.1f}%)")
    axes[1].plot(k_f, r_f, "--", color=C_ON, lw=2.0,
                 label=f"$\\Omega_r$ set ({100*(m_f-1):+.2f}%)")
    axes[1].legend(fontsize=FS_LEGEND, loc="upper right", framealpha=1.0)
    axes[0].set_ylim(0.94, 1.09)

    # No panel titles -- they belong in the caption (thesis convention). The
    # legends already distinguish the two panels.
    fig.tight_layout()
    out = out_dir / "omega_rad_effect.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[omega_r] -> {out}")


if __name__ == "__main__":
    main()
