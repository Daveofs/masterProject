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
import yaml
import pyccl as ccl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ccl_cosmo(params_yml: str) -> ccl.Cosmology:
    """Build a pyccl Cosmology from a CosmoGridV1 params.yml file."""
    with open(params_yml) as fh:
        p = yaml.safe_load(fh)

    h      = p["H0"] / 100.0
    Omega_c = p["O_cdm"]
    Omega_b = p["Ob"]
    n_s    = p["ns"]
    sigma8 = p["s8"]
    w0     = p.get("w0", -1.0)
    wa     = p.get("wa",  0.0)

    return ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        sigma8=sigma8,
        n_s=n_s,
        w0=w0,
        wa=wa,
    )


def compute_theory_cl(
    cosmo: ccl.Cosmology,
    z_lo: float,
    z_hi: float,
    ells: np.ndarray,
    n_z: int = 200,
) -> np.ndarray:
    """Compute theoretical angular power spectrum for a top-hat shell [z_lo, z_hi].

    Uses a NumberCountsTracer with bias=1 (matter power spectrum) and the
    Limber approximation via pyccl.
    """
    z = np.linspace(z_lo, z_hi, n_z)
    nz = np.ones_like(z)
    nz /= np.trapz(nz, z)

    tracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, nz),
        bias=(z, np.ones_like(z)),
    )
    return ccl.angular_cl(cosmo, tracer, tracer, ells)


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


# Physical scale -> multipole (Limber flat-sky: ell ~ chi / r)
_SCALE_LINES = [
    (1.0,   "1 cMpc/h",    "#e07b39"),  # 1 comoving Mpc/h
    (5.0,   "5 cMpc/h",    "#3a9e6f"),  # 5 comoving Mpc/h
    (900.0, "900 cMpc/h\n(box size)", "#2979ff"),  # simulation box size
]


def ell_from_scale(r_cmpch: float, chi_cmpch: float) -> float:
    """Multipole corresponding to comoving scale r [cMpc/h] at distance chi [cMpc/h].

    Uses the Limber flat-sky approximation: ell ~ chi / r.
    """
    if chi_cmpch <= 0 or r_cmpch <= 0:
        return np.nan
    return chi_cmpch / r_cmpch


def add_scale_vlines(ax, chi: float, nside: int, lmax: int,
                     alpha: float = 0.8, label_ypos: float = 0.97,
                     colors_override: dict | None = None,
                     grid_size: float | None = None):
    """Draw vertical lines at 1 cMpc/h, 5 cMpc/h scales, at ell=2*nside,
    and optionally at the PM grid cell size Lbox/res_pm.

    Parameters
    ----------
    ax        : matplotlib Axes
    chi       : comoving distance of shell [cMpc/h]
    nside     : HEALPix nside
    lmax      : maximum multipole plotted
    alpha     : line opacity
    label_ypos: y-position of text label in axes fraction
    colors_override : optional dict {r_cmpch: color} to override default colours
    grid_size : Lbox/res_pm in cMpc/h; draws an extra vline when provided
    """
    trans = ax.get_xaxis_transform()  # x in data, y in axes fraction

    # ---- comoving scale lines ----
    for r, txt, col in _SCALE_LINES:
        if colors_override and r in colors_override:
            col = colors_override[r]
        ell = ell_from_scale(r, chi)
        if np.isnan(ell) or ell < 2 or ell > lmax:
            continue
        ax.axvline(ell, color=col, lw=0.9, linestyle=":", alpha=alpha)
        ax.text(ell, label_ypos, txt, transform=trans,
                fontsize=6.5, color=col, ha="center", va="top",
                rotation=90, clip_on=True)

    # ---- PM grid cell size line ----
    if grid_size is not None and grid_size > 0:
        ell_grid = ell_from_scale(grid_size, chi)
        if not np.isnan(ell_grid) and 2 <= ell_grid <= lmax:
            _col_grid = "#c62828"
            ax.axvline(ell_grid, color=_col_grid, lw=0.9, linestyle="--", alpha=alpha)
            ax.text(ell_grid, label_ypos,
                    f"$L_{{\\rm box}}/N_{{\\rm pm}}$\n({grid_size:.2f} cMpc/h)",
                    transform=trans, fontsize=6.5, color=_col_grid,
                    ha="center", va="top", rotation=90, clip_on=True)

    # ---- 2*nside line ----
    ell_ns = 2 * nside
    if 2 <= ell_ns <= lmax:
        ax.axvline(ell_ns, color="#555555", lw=0.9, linestyle="-.", alpha=alpha)
        ax.text(ell_ns, label_ypos, r"$2\times N_{{\rm side}}$", transform=trans,
                fontsize=6.5, color="#555555", ha="center", va="top",
                rotation=90, clip_on=True)


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
    parser.add_argument("--lbox",      type=float, default=900.0,
                        help="Simulation box size in cMpc/h (default: 900).")
    parser.add_argument("--res-pm",    type=int, default=None,
                        help="PM grid resolution (number of cells per side). "
                             "Grid cell size = Lbox/res_pm is shown on Cl plots.")
    parser.add_argument("--disco-1664", dest="disco_1664", default=None,
                        help="Optional second DISCO shells NPZ to compare against (default: None).")
    parser.add_argument("--params-yml", dest="params_yml",
                        required=True,
                        help="Path to CosmoGridV1 params.yml used to build the CCL cosmology "
                             "for the theory Cl curve.")
    parser.add_argument("--show-theory", dest="show_theory",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Show CCL theory curves in plots (default: true).")
    parser.add_argument("--show-resid", dest="show_resid",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Show residual curves (DISCO - CosmoGrid) in plots (default: true).")
    parser.add_argument("--label-disco", default="DISCO",
                        help="Legend label base for DISCO curves.")
    parser.add_argument("--label-disco-1664", dest="label_disco_1664", default="DISCO_1664",
                        help="Legend label base for second DISCO curves.")
    parser.add_argument("--label-cosmogrid", default="CosmoGridV1",
                        help="Legend label base for CosmoGrid curves.")
    parser.add_argument("--label-theory", default="CCL theory",
                        help="Legend label for theory curves.")
    parser.add_argument("--label-resid", default="DISCO - CosmoGrid (resid)",
                        help="Legend label for DISCO residual curves.")
    parser.add_argument("--label-resid-1664", dest="label_resid_1664",
                        default="DISCO_1664 - CosmoGrid (resid)",
                        help="Legend label for DISCO_1664 residual curves.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading CCL cosmology from:     {args.params_yml}")
    ccl_cosmo = load_ccl_cosmo(args.params_yml)

    print(f"Loading DISCO shells from:      {args.disco}")
    shells_d, info_d = load_shells(args.disco)
    print(f"Loading CosmoGridV1 shells from: {args.cosmogrid}")
    shells_c, info_c = load_shells(args.cosmogrid)
    shells_d1664 = None
    info_d1664 = None
    if args.disco_1664:
        print(f"Loading DISCO 1664 shells from:  {args.disco_1664}")
        shells_d1664, info_d1664 = load_shells(args.disco_1664)

    n_shells = shells_d.shape[0]
    npix     = shells_d.shape[1]
    nside    = hp.npix2nside(npix)
    lmax     = args.lmax if args.lmax is not None else 3 * nside - 1

    grid_size: float | None = None
    if args.res_pm is not None:
        grid_size = args.lbox / args.res_pm
        print(f"Grid cell size: {args.lbox:.1f} / {args.res_pm} = {grid_size:.4f} cMpc/h")

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
        cl_d1664 = compute_cl(shells_d1664[idx], lmax) if shells_d1664 is not None else None

        cl_th = compute_theory_cl(ccl_cosmo, z_lo, z_hi, ells) if args.show_theory else None

        # Also compute Cls of the residual maps: (delta_d - delta_c) and
        # (delta_d1664 - delta_c). This requires making overdensity maps
        # and running anafast on their difference.
        delta_d = to_overdensity(shells_d[idx])
        delta_c = to_overdensity(shells_c[idx])
        delta_d1664 = to_overdensity(shells_d1664[idx]) if shells_d1664 is not None else None

        cl_resid = hp.anafast(delta_d - delta_c, lmax=lmax) if args.show_resid else None
        cl_resid1664 = hp.anafast(delta_d1664 - delta_c, lmax=lmax) if (args.show_resid and delta_d1664 is not None) else None

        # Ratios against CosmoGridV1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio     = np.where(cl_c != 0, cl_d    / cl_c, np.nan)
            ratio1664 = np.where(cl_c != 0, cl_d1664 / cl_c, np.nan) if cl_d1664 is not None else None

        ax_cl.plot(ells, cl_d, color=color, lw=1.0, label=f"{args.label_disco}  {label}")
        if cl_d1664 is not None:
            ax_cl.plot(ells, cl_d1664, color=color, lw=1.0, linestyle=":", alpha=0.9,
                       label=f"{args.label_disco_1664} {label}")
        ax_cl.plot(ells, cl_c, color=color, lw=1.0, linestyle="--", alpha=0.6)
        if cl_th is not None:
            ax_cl.plot(ells, cl_th, color=color, lw=1.2, linestyle="-.", alpha=0.8,
                       label=f"{args.label_theory} {label}")

        ax_ratio.plot(ells, ratio,     color=color, lw=1.0,
                      label=f"{args.label_disco} / {args.label_cosmogrid}  {label}")
        if ratio1664 is not None:
            ax_ratio.plot(ells, ratio1664, color=color, lw=1.0, linestyle=":", alpha=0.9,
                          label=f"{args.label_disco_1664} / {args.label_cosmogrid}  {label}")

        # Per-shell scale lines on summary plots (subtle dotted, shell colour)
        chi = float(info_d[idx]["shell_com"])
        scale_vals = [r for r, _, _ in _SCALE_LINES]
        if grid_size is not None:
            scale_vals.append(grid_size)
        for r in scale_vals:
            ell_s = ell_from_scale(r, chi)
            if not np.isnan(ell_s) and 2 <= ell_s <= lmax:
                for _ax in (ax_cl, ax_ratio):
                    _ax.axvline(ell_s, color=color, lw=0.6, linestyle=":", alpha=0.5)

    # ell = 2*nside line on summary plots (same for all shells)
    ell_ns = 2 * nside
    if 2 <= ell_ns <= lmax:
        for _ax, _yp in ((ax_ratio, 0.97), (ax_cl, 0.97)):
            _ax.axvline(ell_ns, color="#555555", lw=0.9, linestyle="-.", alpha=0.8)
            _ax.text(ell_ns, _yp, r"$2\times N_{{\rm side}}$",
                     transform=_ax.get_xaxis_transform(),
                     fontsize=6.5, color="#555555", ha="center", va="top",
                     rotation=90, clip_on=True)

    # Ratio plot formatting
    ax_ratio.axhline(1.0, color="k", lw=0.8, linestyle="--", label="ratio = 1")
    ax_ratio.set_xlabel(r"Multipole $\ell$", fontsize=13)
    ax_ratio.set_ylabel(
        rf"$C_\ell^{{\rm {args.label_disco}}}\,/\,C_\ell^{{\rm {args.label_cosmogrid}}}$",
        fontsize=13,
    )
    ax_ratio.set_title(
        f"Angular power spectrum ratio: {args.label_disco} / {args.label_cosmogrid}",
        fontsize=13,
    )
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
    ax_cl.set_title(
        f"Angular power spectra: {args.label_disco} (solid) vs {args.label_cosmogrid} (dashed)",
        fontsize=13,
    )
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
        cl_d1664 = compute_cl(shells_d1664[idx], lmax) if shells_d1664 is not None else None
        cl_th = compute_theory_cl(ccl_cosmo, z_lo, z_hi, ells) if args.show_theory else None

        delta_d = to_overdensity(shells_d[idx])
        delta_c = to_overdensity(shells_c[idx])
        delta_d1664 = to_overdensity(shells_d1664[idx]) if shells_d1664 is not None else None
        cl_resid = hp.anafast(delta_d - delta_c, lmax=lmax) if args.show_resid else None
        cl_resid1664 = hp.anafast(delta_d1664 - delta_c, lmax=lmax) if (args.show_resid and delta_d1664 is not None) else None

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio     = np.where(cl_c != 0, cl_d / cl_c, np.nan)
            ratio1664 = np.where(cl_c != 0, cl_d1664 / cl_c, np.nan) if cl_d1664 is not None else None
            ratio_d_th = np.where(cl_th != 0, cl_d / cl_th, np.nan) if cl_th is not None else None
            ratio_c_th = np.where(cl_th != 0, cl_c / cl_th, np.nan) if cl_th is not None else None

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(ells, cl_d, label=args.label_disco, lw=1.2, color="steelblue")
        if cl_d1664 is not None:
            axes[0].plot(ells, cl_d1664, label=args.label_disco_1664, lw=1.2, color="seagreen", linestyle=":")
        axes[0].plot(ells, cl_c, label=args.label_cosmogrid, lw=1.2, color="tomato", linestyle="--")
        if cl_th is not None:
            axes[0].plot(ells, cl_th, label=args.label_theory, lw=1.4, color="darkorange", linestyle="-.", alpha=0.9)

        # Plot residual Cls (DISCO - CosmoGrid) and (DISCO_1664 - CosmoGrid)
        if cl_resid is not None:
            axes[0].plot(ells, cl_resid, label=args.label_resid, lw=1.0,
                     color="purple", linestyle="-.")
            if cl_resid1664 is not None:
                axes[0].plot(ells, cl_resid1664, label=args.label_resid_1664, lw=1.0,
                         color="navy", linestyle=(0, (3, 1, 1, 1)))
        axes[0].set_ylabel(r"$C_\ell$", fontsize=12)
        axes[0].set_yscale("log")
        _grid_str = f"  |  grid={grid_size:.3f} cMpc/h" if grid_size is not None else ""
        axes[0].set_title(
            f"Shell {idx}  |  z = [{z_lo:.4f}, {z_hi:.4f}]  |  nside={nside}{_grid_str}",
            fontsize=12
        )
        axes[0].legend(fontsize=9)
        axes[0].grid(True, which="both", alpha=0.3)

        # Ratio panel: DISCO / CosmoGrid (and DISCO_1664 / CosmoGrid if present)
        axes[1].plot(ells, ratio, lw=1.2, color="darkorchid",
                     label=f"{args.label_disco} / {args.label_cosmogrid}")
        if ratio1664 is not None:
            axes[1].plot(ells, ratio1664, lw=1.2, color="midnightblue", linestyle=":",
                         label=f"{args.label_disco_1664} / {args.label_cosmogrid}")

        if ratio_d_th is not None and ratio_c_th is not None:
            axes[1].plot(ells, ratio_d_th, lw=1.0, color="orange", linestyle="--",
                         label=f"{args.label_disco} / {args.label_theory}")
            axes[1].plot(ells, ratio_c_th, lw=1.0, color="red", linestyle="--",
                         label=f"{args.label_cosmogrid} / {args.label_theory}")
        axes[1].axhline(1.0, color="k", lw=0.8, linestyle="--")
        axes[1].set_xlabel(r"Multipole $\ell$", fontsize=12)
        #axes[1].set_ylabel(r"$C_\ell^{\rm DISCO}\,/\,C_\ell^{\rm CosmoGrid}$", fontsize=12)
        #axes[1].set_ylim(0.7, 1.3)  # fixed y-limits for ratio plot
        axes[1].legend(fontsize=9)
        axes[1].grid(True, which="both", alpha=0.3)

        # Scale vertical lines on both panels
        chi = float(info_d[idx]["shell_com"])
        for ax in axes:
            add_scale_vlines(ax, chi, nside, lmax, grid_size=grid_size)

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
