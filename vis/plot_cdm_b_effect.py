#!/usr/bin/env python3
"""What the CDM+baryon transfer function buys, and what its sigma_8 partner buys.

get_transfer_function.py writes TWO tables for every cosmology:

    transfer_fiducial.dat       built from the total matter contrast  delta_m
    transfer_fiducial_cb.dat    built from the CDM+baryon contrast    delta_cb

PkdGrav3's IC generator reads one of them (achTfFile) and normalises the resulting
particle load to the dSigma8 value in the same configuration file. In both codes the
particles carry cold dark matter and baryons only -- neutrinos never become particles --
so the cb table is the correct one, and dSigma8 must be the cb sigma_8 to match it.

The two halves of that choice are independent and are drawn separately here:

  shape         (T_m / T_cb)^2 -- what changes in P(k) if the wrong TABLE is used.
                Below the neutrino free-streaming scale delta_nu << delta_cb, so
                delta_m < delta_cb and the particle load starts too smooth. The offset
                grows from zero on large scales to a plateau in the nonlinear range.
  normalisation (sigma_8,tot / sigma_8,cb)^2 -- what changes if the right table is
                paired with the wrong sigma_8. Exactly flat in k, because it is a pure
                rescaling of the whole spectrum.

Getting one right and the other wrong still leaves the initial conditions off by
roughly the same amount, which is why the two are quoted as a single fix.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_cdm_b_effect.py --tf-dir <dir with both .dat>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_AXIS, FS_TICK, FS_LEGEND = 16, 14, 13
C_SHAPE, C_NORM = "#7A51C6", "#B85F34"
# sigma_8 of the fiducial cosmology from the same CLASS solve that writes the tables
# (printed by get_transfer_function.py as "sigma8 from classy" / "sigma8_cb from classy")
S8_TOT_DEFAULT, S8_CB_DEFAULT = 0.8991614356982358, 0.9027749891344243


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-dir", required=True,
                    help="directory holding transfer_fiducial.dat and transfer_fiducial_cb.dat")
    ap.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/ic")
    ap.add_argument("--sigma8-tot", type=float, default=S8_TOT_DEFAULT)
    ap.add_argument("--sigma8-cb", type=float, default=S8_CB_DEFAULT)
    a = ap.parse_args()
    tf_dir = Path(a.tf_dir); out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    k_m, t_m = np.loadtxt(tf_dir / "transfer_fiducial.dat", unpack=True)
    k_cb, t_cb = np.loadtxt(tf_dir / "transfer_fiducial_cb.dat", unpack=True)
    assert np.allclose(k_m, k_cb), "the two tables are on different k grids"
    # the tables store -T/k^2; P(k) goes as T^2, so the ratio of the resulting spectra
    # is the square of the ratio of the tabulated columns
    r_shape = (t_m / t_cb) ** 2
    r_norm = (a.sigma8_tot / a.sigma8_cb) ** 2

    print("[cb] using the WRONG table (delta_m instead of delta_cb):")
    for lo, hi in [(1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, 10.0)]:
        m = (k_m >= lo) & (k_m <= hi)
        if m.any():
            print(f"   k {lo:8.1e}-{hi:8.1e}: P ratio {np.median(r_shape[m]):.4f} "
                  f"({100*(np.median(r_shape[m])-1):+.2f}%)")
    print(f"[cb] plateau at k > 1: {100*(np.median(r_shape[k_m > 1.0])-1):+.2f}%")
    print(f"[cb] using the WRONG sigma_8 (total instead of cb): "
          f"{100*(r_norm-1):+.2f}% at every k")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=2)
    for y in (0.99, 1.01):
        ax.axhline(y, color="0.65", ls=":", lw=0.8, zorder=1)

    plateau = 100 * (np.median(r_shape[k_m > 1.0]) - 1)
    ax.semilogx(k_m, r_shape, "-", color=C_SHAPE, lw=2.4,
                label=f"wrong table: $\\delta_m$ instead of $\\delta_{{cb}}$  "
                      f"({plateau:+.1f}% at small scales)")
    ax.axhline(r_norm, color=C_NORM, ls=(0, (1, 1.6)), lw=2.4,
               label=f"wrong $\\sigma_8$: full-field instead of cb  "
                     f"({100*(r_norm-1):+.1f}% at every $k$)")

    ax.set_xlabel(r"$k\ \ [h/\mathrm{Mpc}]$", fontsize=FS_AXIS)
    ax.set_ylabel("$P(k)$ wrong choice / correct", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, which="both", lw=0.5); ax.set_axisbelow(True)
    ax.set_xlim(k_m.min(), k_m.max())
    lo = min(r_shape.min(), r_norm)
    ax.set_ylim(lo - 0.004, 1.006)
    ax.legend(fontsize=FS_LEGEND, loc="lower left", framealpha=1.0,
              borderpad=0.45, labelspacing=0.4, handlelength=2.4, handletextpad=0.6)

    out = out_dir / "cdm+b_effect.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[cb] -> {out}")


if __name__ == "__main__":
    main()
