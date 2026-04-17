"""
plot_pixel_histogram.py
=======================
Plot histograms of raw HEALPix pixel counts for paired DISCO-DJ and
CosmoGridV1 shells, to compare the one-point distribution of the two datasets.

Raw particle counts (not overdensity) are plotted so that differences in
mean counts n̄ between the two simulations are immediately visible on the
x-axis.  Both histograms use histtype='step' for maximum overlap visibility.

Usage
-----
python plot_pixel_histogram.py \\
    --disco      /path/to/shells_nside=2048.npz \\
    --cosmogrid  /path/to/compressed_shells.npz \\
    --out-dir    /path/to/output_dir \\
    [--shells    0 17 34 51 68]   # shell indices (default: 5 evenly spaced)
    [--nbins     200]             # histogram bins (default: 200)
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
    """Load a shells NPZ file.  Returns (shells array, shell_info array)."""
    d = np.load(path)
    return d["shells"], d["shell_info"]


def plot_shell_histogram(
    ax: plt.Axes,
    counts_d: np.ndarray,
    counts_c: np.ndarray,
    nbins: int,
    label: str,
):
    """Plot raw pixel-count histograms for one shell on *ax*."""
    counts_d = counts_d.astype(np.float64)
    counts_c = counts_c.astype(np.float64)

    nbar_d = counts_d.mean()
    nbar_c = counts_c.mean()

    # Shared bin edges clipped to 0.5–99.5 percentile of both datasets
    lo = min(np.percentile(counts_d, 0.5), np.percentile(counts_c, 0.5))
    hi = max(np.percentile(counts_d, 99.5), np.percentile(counts_c, 99.5))
    bins = np.linspace(lo, hi, nbins + 1)

    ax.hist(counts_d, bins=bins, density=True, histtype="step",
            lw=1.8, color="#2979ff", label=fr"DISCO-DJ  ($\bar{{n}}$={nbar_d:.2f})")
    ax.hist(counts_c, bins=bins, density=True, histtype="step",
            lw=1.8, color="#e53935", linestyle="--",
            label=fr"CosmoGridV1  ($\bar{{n}}$={nbar_c:.2f})")

    # Vertical lines at each mean
    ax.axvline(nbar_d, color="#2979ff", lw=1.0, linestyle=":", alpha=0.8)
    ax.axvline(nbar_c, color="#e53935", lw=1.0, linestyle=":", alpha=0.8)

    ax.set_xlabel("Pixel particle count $n$", fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot pixel count histograms: DISCO vs CosmoGridV1"
    )
    parser.add_argument("--disco",     required=True, help="Path to DISCO shells NPZ")
    parser.add_argument("--cosmogrid", required=True, help="Path to CosmoGridV1 shells NPZ")
    parser.add_argument("--out-dir",   default=".", help="Directory for output plots")
    parser.add_argument("--shells",    nargs="+", type=int, default=None,
                        help="Shell indices to plot (0-based). Default: 5 evenly spaced.")
    parser.add_argument("--nbins",     type=int, default=200,
                        help="Number of histogram bins (default: 200).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading DISCO shells from:       {args.disco}")
    shells_d, info_d = load_shells(args.disco)
    print(f"Loading CosmoGridV1 shells from: {args.cosmogrid}")
    shells_c, info_c = load_shells(args.cosmogrid)

    n_shells = shells_d.shape[0]
    npix     = shells_d.shape[1]
    nside    = hp.npix2nside(npix)

    print(f"nside={nside}, npix={npix}, n_shells={n_shells}")

    if shells_c.shape[0] != n_shells:
        raise ValueError(
            f"Shell count mismatch: DISCO has {n_shells}, CosmoGridV1 has {shells_c.shape[0]}"
        )

    # Shell selection
    if args.shells is not None:
        shell_indices = args.shells
    else:
        shell_indices = list(np.linspace(0, n_shells - 1, 5, dtype=int))

    valid_indices = [i for i in shell_indices if 0 <= i < n_shells]
    if len(valid_indices) < len(shell_indices):
        skipped = set(shell_indices) - set(valid_indices)
        print(f"  [WARN] Shell indices out of range [0, {n_shells-1}], skipped: {skipped}")
    shell_indices = valid_indices

    # ------------------------------------------------------------------
    # Summary grid figure: all selected shells in sub-panels
    # ------------------------------------------------------------------
    n = len(shell_indices)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig_grid, axes = plt.subplots(nrows, ncols,
                                  figsize=(5.5 * ncols, 4.5 * nrows),
                                  squeeze=False)

    # Flatten axes for easy iteration; hide unused panels
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax, idx in zip(axes_flat, shell_indices):
        z_lo  = float(info_d[idx]["lower_z"])
        z_hi  = float(info_d[idx]["upper_z"])
        label = f"shell {idx}  z=[{z_lo:.3f}, {z_hi:.3f}]"
        print(f"  Building histogram for {label} ...")
        nbar_d = shells_d[idx].mean()
        nbar_c = shells_c[idx].mean()
        print(f"    n̄  DISCO={nbar_d:.4f}  CosmoGrid={nbar_c:.4f}  ratio={nbar_d/nbar_c:.4f}")

        plot_shell_histogram(ax, shells_d[idx], shells_c[idx], args.nbins, label)

    fig_grid.suptitle(
        "Pixel count histograms: DISCO-DJ vs CosmoGridV1  (dotted lines = mean $\\bar{n}$)",
        fontsize=13, y=1.01
    )
    fig_grid.tight_layout()

    out_grid = Path(args.out_dir) / "pixel_histogram_grid.png"
    fig_grid.savefig(out_grid, dpi=150, bbox_inches="tight")
    print(f"Saved grid plot -> {out_grid}")
    plt.close(fig_grid)

    # ------------------------------------------------------------------
    # Summary overlay figure: all shells on one axes per dataset,
    # coloured by shell index (useful for seeing redshift evolution)
    # ------------------------------------------------------------------
    colors = cm.viridis(np.linspace(0.1, 0.9, n))
    fig_ov, axes_ov = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    ax_d, ax_c = axes_ov

    for color, idx in zip(colors, shell_indices):
        z_lo = float(info_d[idx]["lower_z"])
        z_hi = float(info_d[idx]["upper_z"])
        lbl  = f"shell {idx}  z=[{z_lo:.3f}, {z_hi:.3f}]"

        counts_d = shells_d[idx].astype(np.float64)
        counts_c = shells_c[idx].astype(np.float64)

        lo = min(np.percentile(counts_d, 0.5), np.percentile(counts_c, 0.5))
        hi = max(np.percentile(counts_d, 99.5), np.percentile(counts_c, 99.5))
        bins = np.linspace(lo, hi, args.nbins + 1)

        ax_d.hist(counts_d, bins=bins, density=True, histtype="step",
                  lw=1.2, color=color, label=lbl)
        ax_c.hist(counts_c, bins=bins, density=True, histtype="step",
                  lw=1.2, color=color, label=lbl)

    for ax, title in ((ax_d, "DISCO-DJ"), (ax_c, "CosmoGridV1")):
        ax.set_xlabel("Pixel particle count $n$", fontsize=12)
        ax.set_ylabel("Probability density", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig_ov.suptitle(
        "Pixel count distributions across shells: DISCO-DJ vs CosmoGridV1",
        fontsize=13
    )
    fig_ov.tight_layout()

    out_ov = Path(args.out_dir) / "pixel_histogram_overlay.png"
    fig_ov.savefig(out_ov, dpi=150, bbox_inches="tight")
    print(f"Saved overlay plot -> {out_ov}")
    plt.close(fig_ov)

    # ------------------------------------------------------------------
    # Per-shell individual comparison plots
    # ------------------------------------------------------------------
    for idx in shell_indices:
        z_lo  = float(info_d[idx]["lower_z"])
        z_hi  = float(info_d[idx]["upper_z"])
        label = f"shell {idx}  z=[{z_lo:.3f}, {z_hi:.3f}]"

        counts_d = shells_d[idx].astype(np.float64)
        counts_c = shells_c[idx].astype(np.float64)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_shell_histogram(ax, counts_d, counts_c, args.nbins, label)

        # Statistics annotation (raw counts)
        nbar_d = counts_d.mean()
        nbar_c = counts_c.mean()
        stats_txt = (
            f"DISCO:      n̄={nbar_d:.4f}  std={counts_d.std():.4f}  "
            f"min={counts_d.min():.0f}  max={counts_d.max():.0f}\n"
            f"CosmoGrid: n̄={nbar_c:.4f}  std={counts_c.std():.4f}  "
            f"min={counts_c.min():.0f}  max={counts_c.max():.0f}  "
            f"  ratio n̄_D/n̄_C={nbar_d/nbar_c:.4f}"
        )
        ax.set_title(f"Pixel count histogram – {label}\n{stats_txt}", fontsize=8)

        fig.tight_layout()
        out_f = Path(args.out_dir) / f"pixel_histogram_shell{idx:03d}.png"
        fig.savefig(out_f, dpi=150, bbox_inches="tight")
        print(f"  Saved per-shell plot -> {out_f}")
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
