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

    ax.hist(counts_d, bins=bins, density=False, histtype="step",
            lw=1.8, color="#2979ff", label=fr"DISCO-DJ  ($\bar{{n}}$={nbar_d:.2f})")
    ax.hist(counts_c, bins=bins, density=False, histtype="step",
            lw=1.8, color="#e53935", linestyle="--",
            label=fr"CosmoGridV1  ($\bar{{n}}$={nbar_c:.2f})")

    # Vertical lines at each mean
    ax.axvline(nbar_d, color="#2979ff", lw=1.0, linestyle=":", alpha=0.8)
    ax.axvline(nbar_c, color="#e53935", lw=1.0, linestyle=":", alpha=0.8)

    ax.set_xlabel("Pixel particle count $n$", fontsize=10)
    ax.set_ylabel("Pixel count", fontsize=10)
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Cosmology & mass helpers
# ---------------------------------------------------------------------------
def comoving_distance_mpc(z, omega_m: float, h: float, n_samples: int = 2000) -> float:
    """Return comoving radial distance in Mpc for redshift z (scalar) assuming flat LCDM."""
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=omega_m)
    # astropy handles array/scalar cases; ensure scalar out
    return float(cosmo.comoving_distance(z).value)


def rho_crit0_msun_per_mpc3(h: float) -> float:
    """Return critical density today in Msun / Mpc^3 for given h."""
    # Physical constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    Mpc_m = 3.085677581491367e22  # meters in 1 Mpc
    M_sun = 1.98847e30  # kg
    H0_km_s_Mpc = 100.0 * h
    H0_SI = H0_km_s_Mpc * 1000.0 / Mpc_m
    rho_crit_si = 3.0 * H0_SI ** 2 / (8.0 * np.pi * G)  # kg / m^3
    # convert to Msun / Mpc^3
    return float(rho_crit_si / M_sun * Mpc_m ** 3)



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
    parser.add_argument("--omega-m",  type=float, default=0.3,
                        help="Matter density parameter Omega_m (default: 0.3)")
    parser.add_argument("--h",        type=float, default=0.73,
                        help="Dimensionless Hubble parameter h (default: 0.73)")
    parser.add_argument("--lbox-disco", dest="lbox_disco", type=float, required=True,
                        help="Box side length for DISCO (comoving), in Mpc. REQUIRED.")
    parser.add_argument("--res-disco", dest="res_disco", type=int, required=True,
                        help="Particle resolution per axis for DISCO (i.e. res^3 = total number of particles). REQUIRED.")
    parser.add_argument("--lbox-cosmogrid", dest="lbox_cosmogrid", type=float, required=True,
                        help="Box side length for CosmoGridV1 (comoving), in Mpc. REQUIRED.")
    parser.add_argument("--res-cosmogrid", dest="res_cosmogrid", type=int, required=True,
                        help="Particle resolution per axis for CosmoGridV1 (i.e. res^3 = total number of particles). REQUIRED.")
    parser.add_argument("--lbox-units", dest="lbox_units", choices=["Mpc","Mpc/h"], default="Mpc/h",
                        help="Units for --lbox-* (default: 'Mpc/h' = comoving h^-1 Mpc).")
    parser.add_argument("--fsky",     type=float, default=1.0,
                        help="Sky fraction (0-1) for computing comoving shell volume (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading DISCO shells from:       {args.disco}")
    shells_d, info_d = load_shells(args.disco)
    print(f"Loading CosmoGridV1 shells from: {args.cosmogrid}")
    shells_c, info_c = load_shells(args.cosmogrid)

    # Cosmology and particle mass handling
    omega_m = args.omega_m
    h = args.h
    fsky = args.fsky

    # Compute per-particle mass from provided box size and resolution
    rho_crit0 = rho_crit0_msun_per_mpc3(h)

    if args.lbox_disco is None or args.res_disco is None or args.lbox_cosmogrid is None or args.res_cosmogrid is None:
        raise ValueError(
            "Please provide --lbox-disco, --res-disco, --lbox-cosmogrid and --res-cosmogrid; "
            "per-particle mass is computed as Omega_m * rho_crit0 * (Lbox^3/res^3)."
        )

    # Convert Lbox to physical Mpc if necessary. Default input units are 'Mpc/h'.
    if args.lbox_units == "Mpc/h":
        Lbox_d_mpc = float(args.lbox_disco) / h
        Lbox_c_mpc = float(args.lbox_cosmogrid) / h
    else:
        Lbox_d_mpc = float(args.lbox_disco)
        Lbox_c_mpc = float(args.lbox_cosmogrid)

    cell_vol_d = (Lbox_d_mpc ** 3) / float(args.res_disco) ** 3
    cell_vol_c = (Lbox_c_mpc ** 3) / float(args.res_cosmogrid) ** 3
    mpart_d = omega_m * rho_crit0 * cell_vol_d
    mpart_c = omega_m * rho_crit0 * cell_vol_c
    print(f"  Computed per-particle mass (Msun): DISCO={mpart_d:.6g}  CosmoGridV1={mpart_c:.6g}  (Lbox units: {args.lbox_units})")

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

        # Mass comparison (if particle masses available)
        r_lo = comoving_distance_mpc(z_lo, omega_m, h)
        r_hi = comoving_distance_mpc(z_hi, omega_m, h)
        V_shell = (4.0 * np.pi / 3.0) * (r_hi ** 3 - r_lo ** 3) * fsky
        rho_crit0 = rho_crit0_msun_per_mpc3(h)
        M_theory = omega_m * rho_crit0 * V_shell

        Ntot_d = counts_d.sum()
        Ntot_c = counts_c.sum()

        # per-particle mass from provided Lbox/res (same for all shells)
        mpart_d_shell = float(mpart_d)
        mpart_c_shell = float(mpart_c)

        mass_line = ""
        if mpart_d_shell is not None and mpart_c_shell is not None:
            Mpart_d = Ntot_d * mpart_d_shell
            Mpart_c = Ntot_c * mpart_c_shell
            note_d = "(from Lbox/res)"
            note_c = "(from Lbox/res)"
            mass_line = (
                f"\nTheory: M={M_theory:.3e} Msun; "
                f"DISCO M={Mpart_d:.3e} Msun (ratio={Mpart_d/M_theory:.3f}); "
                f"CosmoGrid M={Mpart_c:.3e} Msun (ratio={Mpart_c/M_theory:.3f})"
            )
            print(f"  Shell {idx}: M_theory={M_theory:.6g} Msun  DISCO M={Mpart_d:.6g} Msun  CosmoGrid M={Mpart_c:.6g} Msun")
            # Expected particle counts from uniform sampling of box
            if cell_vol_d > 0:
                expected_N_d = V_shell / cell_vol_d
            else:
                expected_N_d = float('nan')
            if cell_vol_c > 0:
                expected_N_c = V_shell / cell_vol_c
            else:
                expected_N_c = float('nan')
            print(f"    Expected N (DISCO) = {expected_N_d:.0f}, observed N = {Ntot_d:.0f}, obs/exp = {Ntot_d/expected_N_d if expected_N_d>0 else float('nan'):.6g}")
            print(f"    Expected N (CosmoGrid) = {expected_N_c:.0f}, observed N = {Ntot_c:.0f}, obs/exp = {Ntot_c/expected_N_c if expected_N_c>0 else float('nan'):.6g}")
        else:
            print(f"  Shell {idx}: Mass comparison skipped (no particle mass available and zero counts)")

        ax.set_title(f"Pixel count histogram – {label}\n{stats_txt}{mass_line}", fontsize=8)

        fig.tight_layout()
        out_f = Path(args.out_dir) / f"pixel_histogram_shell{idx:03d}.png"
        fig.savefig(out_f, dpi=150, bbox_inches="tight")
        print(f"  Saved per-shell plot -> {out_f}")
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
