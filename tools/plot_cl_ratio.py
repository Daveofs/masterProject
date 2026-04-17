"""
plot_cl_ratio.py
================
Compute angular power spectra Cl from paired DISCO-DJ and CosmoGridV1 HEALPix
shells, then plot the ratio Cl_disco / Cl_cosmogrid as a function of multipole l.

For each matched shell (matched by shell_id / z-bin index), the count map is
converted to an overdensity map:
    delta = n / mean(n) - 1
and then healpy.sphtfunc.anafast is used to compute the Cl.

Usage
-----
python plot_cl_ratio.py \\
    --disco      /path/to/shells_nside=2048.npz \\
    --cosmogrid  /path/to/compressed_shells.npz \\
    --out-dir    /path/to/output_dir \\
    [--shells    5 20 40]   # shell indices to plot (default: 5 evenly spaced)
    [--lmax      3000]      # maximum multipole (default: 3*nside-1)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_shells(path: str):
    """Load a shells NPZ file. Returns (shells array, shell_info array)."""
    d = np.load(path)
    return d["shells"], d["shell_info"]


def to_overdensity(count_map: np.ndarray) -> np.ndarray:
    """Convert a raw particle-count HEALPix map to an overdensity map."""
    mean = count_map.mean()
    if mean == 0:
        return np.zeros_like(count_map, dtype=np.float64)
    return count_map.astype(np.float64) / mean - 1.0


def compute_cl(count_map: np.ndarray, lmax: int) -> np.ndarray:
    """Return Cl array for a single HEALPix count map."""
    delta = to_overdensity(count_map)
    cl = hp.anafast(delta, lmax=lmax)
    return cl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Cl_disco / Cl_cosmogrid ratio")
    parser.add_argument("--disco",     required=True, help="Path to DISCO shells NPZ")
    parser.add_argument("--cosmogrid", required=True, help="Path to CosmoGridV1 shells NPZ")
    parser.add_argument("--out-dir",   default=".", help="Directory for output plots")
    parser.add_argument("--shells",    nargs="+", type=int, default=None,
                        help="Shell indices to plot (0-based). Default: 5 evenly spaced.")
    parser.add_argument("--lmax",      type=int, default=None,
                        help="Maximum multipole. Default: 3*nside-1.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading DISCO shells from:      {args.disco}")
    shells_d, info_d = load_shells(args.disco)
    print(f"Loading CosmoGridV1 shells from: {args.cosmogrid}")
    shells_c, info_c = load_shells(args.cosmogrid)

    n_shells = shells_d.shape[0]
    npix     = shells_d.shape[1]
    nside    = hp.npix2nside(npix)
    lmax     = args.lmax if args.lmax is not None else 3 * nside - 1

    print(f"nside={nside}, npix={npix}, lmax={lmax}, n_shells={n_shells}")

    # Shell selection
    if args.shells is not None:
        shell_indices = args.shells
    else:
        shell_indices = list(np.linspace(0, n_shells - 1, 5, dtype=int))

    # Validate CosmoGrid has same number of shells
    if shells_c.shape[0] != n_shells:
        raise ValueError(
            f"Shell count mismatch: DISCO has {n_shells}, CosmoGridV1 has {shells_c.shape[0]}"
        )

    ells = np.arange(lmax + 1)

    # ------------------------------------------------------------------
    # One summary figure: all selected shells overlaid
    # ------------------------------------------------------------------
    colors = cm.viridis(np.linspace(0.1, 0.9, len(shell_indices)))

    fig_ratio, ax_ratio = plt.subplots(figsize=(10, 5))
    fig_cl,    ax_cl    = plt.subplots(figsize=(10, 5))

    for color, idx in zip(colors, shell_indices):
        if idx < 0 or idx >= n_shells:
            print(f"  [WARN] shell index {idx} out of range [0, {n_shells-1}], skipping.")
            continue

        z_lo = float(info_d[idx]["lower_z"])
        z_hi = float(info_d[idx]["upper_z"])
        label = f"shell {idx}  z=[{z_lo:.3f}, {z_hi:.3f}]"

        print(f"  Computing Cl for {label} ...")

        cl_d = compute_cl(shells_d[idx], lmax)
        cl_c = compute_cl(shells_c[idx], lmax)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(cl_c != 0, cl_d / cl_c, np.nan)

        ax_cl.plot(ells, cl_d, color=color, lw=1.0, label=f"DISCO  {label}")
        ax_cl.plot(ells, cl_c, color=color, lw=1.0, linestyle="--", alpha=0.6)

        ax_ratio.plot(ells, ratio, color=color, lw=1.0, label=label)

    # Ratio plot formatting
    ax_ratio.axhline(1.0, color="k", lw=0.8, linestyle="--", label="ratio = 1")
    ax_ratio.set_xlabel(r"Multipole $\ell$", fontsize=13)
    ax_ratio.set_ylabel(r"$C_\ell^{\rm DISCO}\,/\,C_\ell^{\rm CosmoGrid}$", fontsize=13)
    ax_ratio.set_title("Angular power spectrum ratio: DISCO / CosmoGridV1", fontsize=13)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(2, lmax)
    ax_ratio.legend(fontsize=8, loc="upper right")
    ax_ratio.grid(True, which="both", alpha=0.3)
    fig_ratio.tight_layout()

    out_ratio = Path(args.out_dir) / "cl_ratio.png"
    fig_ratio.savefig(out_ratio, dpi=150)
    print(f"Saved ratio plot -> {out_ratio}")

    # Cl comparison plot formatting
    ax_cl.set_xlabel(r"Multipole $\ell$", fontsize=13)
    ax_cl.set_ylabel(r"$C_\ell$", fontsize=13)
    ax_cl.set_title("Angular power spectra: DISCO (solid) vs CosmoGridV1 (dashed)", fontsize=13)
    ax_cl.set_xscale("log")
    ax_cl.set_yscale("log")
    ax_cl.set_xlim(2, lmax)
    ax_cl.legend(fontsize=7, loc="upper right")
    ax_cl.grid(True, which="both", alpha=0.3)
    fig_cl.tight_layout()

    out_cl = Path(args.out_dir) / "cl_comparison.png"
    fig_cl.savefig(out_cl, dpi=150)
    print(f"Saved Cl comparison plot -> {out_cl}")

    # ------------------------------------------------------------------
    # Per-shell individual plots
    # ------------------------------------------------------------------
    for idx in shell_indices:
        if idx < 0 or idx >= n_shells:
            continue

        z_lo = float(info_d[idx]["lower_z"])
        z_hi = float(info_d[idx]["upper_z"])

        cl_d = compute_cl(shells_d[idx], lmax)
        cl_c = compute_cl(shells_c[idx], lmax)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(cl_c != 0, cl_d / cl_c, np.nan)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(ells, cl_d, label="DISCO",        lw=1.2, color="steelblue")
        axes[0].plot(ells, cl_c, label="CosmoGridV1",  lw=1.2, color="tomato", linestyle="--")
        axes[0].set_ylabel(r"$C_\ell$", fontsize=12)
        axes[0].set_yscale("log")
        axes[0].set_title(
            f"Shell {idx}  |  z = [{z_lo:.4f}, {z_hi:.4f}]  |  nside={nside}", fontsize=12
        )
        axes[0].legend(fontsize=10)
        axes[0].grid(True, which="both", alpha=0.3)

        axes[1].plot(ells, ratio, lw=1.2, color="darkorchid")
        axes[1].axhline(1.0, color="k", lw=0.8, linestyle="--")
        axes[1].set_xlabel(r"Multipole $\ell$", fontsize=12)
        axes[1].set_ylabel(r"$C_\ell^{\rm DISCO}\,/\,C_\ell^{\rm CosmoGrid}$", fontsize=12)
        axes[1].grid(True, which="both", alpha=0.3)

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(2, lmax)

        fig.tight_layout()
        out_single = Path(args.out_dir) / f"cl_ratio_shell{idx:03d}_z{z_lo:.3f}-{z_hi:.3f}.png"
        fig.savefig(out_single, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_single}")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
