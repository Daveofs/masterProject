#!/usr/bin/env python3
"""The measurement that motivates the whole tabulated-transfer-function route:
handing Disco-DJ CONCEPT's massive-neutrino linear input directly leaves a flat
~3% excess on large scales, while going through PkdGrav3's backscaled ICs removes it.

Three z=0 snapshot spectra of the SAME fiducial cosmology, all at the same
Omega_r = 5.58e-5 so the background is not what separates them:

  PK_C  PkdGrav3 fiducial (bClass=1, CONCEPT linear species)     -- the target
  PK_A  Disco-DJ started from CONCEPT's massive-nu linear input  -- +3% at large k^-1
  PK_B  Disco-DJ started from PkdGrav3's backscaled ICs          -- matches PK_C

The offset in PK_A is scale-INDEPENDENT across the linear regime, which is the
signature of a growth-normalisation error rather than a shape error: backscaling
assumes one scale-independent growth factor D(a), but with massive neutrinos the
growth of the CDM+baryon field is scale-dependent below the free-streaming scale, so
no single D(a) can carry a z=0 massive-nu spectrum back to z_ini consistently.

The workaround PK_B is what the thesis pipeline implements: take the linear transfer
function at z=0 from the massive-nu CONCEPT solution, hand THAT to PkdGrav3's own
backscaling IC generator (bClass=0 + achTfFile), and start Disco-DJ from the resulting
particle load. Both codes then begin from the same backscaled ICs and the large-scale
offset disappears.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_backscaling_vs_concept.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_AXIS, FS_TICK, FS_LEGEND = 16, 14, 14
PK = Path("/capstor/scratch/cscs/damrein/outputs/pk_custom")
C_TARGET, C_CONCEPT, C_BACK = "#52514e", "#B85F34", "#28866A"


def load(path: Path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pk-dir", default=str(PK))
    ap.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/ic")
    ap.add_argument("--kmax", type=float, default=1.0,
                    help="upper k for the ratio panel; beyond this both Disco-DJ runs "
                         "are dominated by the PM force-resolution cutoff, not by ICs")
    a = ap.parse_args()
    pk_dir = Path(a.pk_dir); out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    k_c, p_c = load(pk_dir / "pk_pkd_standard_fiducial.txt")
    k_a, p_a = load(pk_dir / "pk_disco_concept_nu_omega_r_5.58e-5.txt")
    k_b, p_b = load(pk_dir / "pk_disco_backscaling_omega_r_5.58e-5.txt")
    assert np.allclose(k_a, k_c) and np.allclose(k_b, k_c), "spectra are on different k grids"
    r_a, r_b = p_a / p_c, p_b / p_c

    lin = k_c < 0.1                      # linear regime, where the IC amplitude shows
    print(f"[pk] median ratio for k < 0.1 h/Mpc (linear):")
    print(f"  CONCEPT massive-nu input / PkdGrav3 : {np.median(r_a[lin]):.4f}")
    print(f"  backscaled ICs          / PkdGrav3 : {np.median(r_b[lin]):.4f}")
    print(f"[pk] scatter of the CONCEPT offset over that range: "
          f"{np.std(r_a[lin]):.4f} (flat => normalisation, not shape)")

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 7.8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.07})
    ax.loglog(k_c, p_c, "-", color=C_TARGET, lw=2.4, label="PkdGrav3 fiducial (target)")
    ax.loglog(k_a, p_a, ":", color=C_CONCEPT, lw=2.2, label="Disco-DJ $\\leftarrow$ CONCEPT massive-$\\nu$ input")
    ax.loglog(k_b, p_b, "--", color=C_BACK, lw=2.0, label="Disco-DJ $\\leftarrow$ backscaled ICs")
    ax.set_ylabel(r"$P(k)\ \ [(\mathrm{Mpc}/h)^3]$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="lower left", framealpha=1.0,
              borderpad=0.4, labelspacing=0.35, handlelength=2.2, handletextpad=0.5)
    ax.grid(alpha=0.25, which="both", lw=0.5); ax.set_axisbelow(True)

    axr.semilogx(k_a, r_a, ":", color=C_CONCEPT, lw=2.2)
    axr.semilogx(k_b, r_b, "--", color=C_BACK, lw=2.0)
    axr.axhline(1.0, color="k", ls="--", lw=0.9)
    for y in (0.99, 1.01):
        axr.axhline(y, color="0.6", ls=":", lw=0.8)
    axr.annotate(f"$+{100*(np.median(r_a[lin])-1):.1f}\\%$, flat in $k$",
                 xy=(1.3e-2, np.median(r_a[lin])), xytext=(1.3e-2, np.median(r_a[lin]) + 0.012),
                 fontsize=FS_LEGEND, color=C_CONCEPT)
    axr.set_xlabel(r"$k\ \ [h/\mathrm{Mpc}]$", fontsize=FS_AXIS)
    axr.set_ylabel("ratio to PkdGrav3", fontsize=FS_AXIS)
    axr.tick_params(labelsize=FS_TICK)
    axr.grid(alpha=0.25, which="both", lw=0.5); axr.set_axisbelow(True)
    axr.set_ylim(0.95, 1.06)
    axr.set_xlim(k_c.min(), a.kmax)

    out = out_dir / "backscaling_vs_concept_nu.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[pk] -> {out}")


if __name__ == "__main__":
    main()
