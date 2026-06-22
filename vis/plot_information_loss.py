import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import tqdm

# ---------------------------------------------------------------------------
# Cl ratio utilities
# ---------------------------------------------------------------------------
def compute_cl(shell, lmax):
    """Compute angular power spectrum of a HEALPix shell (overdensity)."""
    mean = shell.mean()
    if mean == 0:
        delta = shell
    else:
        delta = shell / mean - 1.0
    return hp.anafast(delta, lmax=lmax)


def to_overdensity(shell):
    """Convert a density shell to overdensity delta = rho/mean(rho) - 1."""
    mean = shell.mean()
    if mean == 0:
        return shell
    return shell / mean - 1.0


def add_scale_vlines(ax, chi, nside, lmax, grid_size=None):
    """Add vertical lines indicating characteristic angular scales."""
    # Pixel scale
    pix_rad = hp.nside2resol(nside)
    ell_pix = np.pi / pix_rad
    if ell_pix <= lmax:
        ax.axvline(ell_pix, color="gray", ls="--", lw=0.7, alpha=0.6)
        ax.text(ell_pix, ax.get_ylim()[1] * 0.95, r"$\ell_{\rm pix}$",
                fontsize=8, color="gray", ha="left", va="top")

    # Grid scale
    if grid_size is not None and chi > 0:
        ell_grid = chi / grid_size * 2 * np.pi
        if 2 < ell_grid <= lmax:
            ax.axvline(ell_grid, color="green", ls="-.", lw=0.7, alpha=0.6)
            ax.text(ell_grid, ax.get_ylim()[1] * 0.90, r"$\ell_{\rm grid}$",
                    fontsize=8, color="green", ha="left", va="top")


def plot_cl_ratio_and_with_information_loss(
    test_npz,
    cosmogrid_npz,
    out_dir,
    shell_indices=[3, 65],
    lmax_power_spectrum=3000,
    lmax_information_loss=2048,
    lbox=900.0,
    res_pm=1664,
    label_test="DRF - Low-res input",
    label_corrected="DRF - Flow Corrected",
    label_cosmogrid="CosmoGridV1",
):
    """
    Compute and plot Cl ratio between test input, corrected (flow-matched),
    and CosmoGrid reference shells.
    """
    print("\n" + "=" * 80)
    print("Computing Cl ratios")
    print("=" * 80)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_test = np.load(test_npz, allow_pickle=False)
    data_cosmo = np.load(cosmogrid_npz, allow_pickle=False)

    shells_test = np.asarray(data_test["shells"], dtype=np.float64)
    shells_cosmo = np.asarray(data_cosmo["shells"], dtype=np.float64)

    n_available = shells_test.shape[0]
        

    for i in tqdm.tqdm(range(n_available), desc="Preprocessing shells for information loss analysis"):
        # Transform to Alm and back for information loss analysis
        alms = hp.map2alm(shells_test[i], lmax=lmax_information_loss, iter=1)
        shells_test[i] = hp.alm2map(alms, nside=hp.npix2nside(shells_test.shape[1]))

    n_shells = min(shells_test.shape[0], shells_cosmo.shape[0])
    nside = hp.npix2nside(shells_cosmo.shape[1])

    # Shell info (redshift bounds)
    has_info = "info" in data_cosmo
    info = data_cosmo["info"] if has_info else None

    # Grid size for vertical lines
    grid_size = lbox / res_pm  # cMpc/h per grid cell

    ells = np.arange(lmax_power_spectrum + 1)

    for idx in shell_indices:
        if idx < 0 or idx >= n_shells:
            print(f"  [WARN] Shell index {idx} out of range [0, {n_shells}), skipping.")
            continue

        # Redshift bounds
        if info is not None:
            z_lo = float(info[idx]["lower_z"]) if "lower_z" in info.dtype.names else 0.0
            z_hi = float(info[idx]["upper_z"]) if "upper_z" in info.dtype.names else 0.0
            chi = float(info[idx]["shell_com"]) if "shell_com" in info.dtype.names else 0.0
        else:
            z_lo, z_hi, chi = 0.0, 0.0, 0.0

        # Compute power spectra
        cl_test = compute_cl(shells_test[idx], lmax_power_spectrum)
        cl_cosmo = compute_cl(shells_cosmo[idx], lmax_power_spectrum)

        # Ratios
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_test = np.where(cl_cosmo != 0, cl_test / cl_cosmo, np.nan)

        # --- Plot ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top panel: Cl spectra
        axes[0].plot(ells, cl_test, label=label_test, lw=1.2, color="seagreen")
        axes[0].plot(ells, cl_cosmo, label=label_cosmogrid, lw=1.2, color="tomato", linestyle="--")

        axes[0].set_ylabel(r"$C_\ell$", fontsize=12)
        axes[0].set_yscale("log")
        _grid_str = f"  |  grid={grid_size:.3f} cMpc/h"
        axes[0].set_title(
            f"Shell {idx}  |  nside={nside}{_grid_str} | lmax_info_loss={lmax_information_loss}",
            fontsize=12,
        )
        axes[0].legend(fontsize=9)
        axes[0].grid(True, which="both", alpha=0.3)

        # Bottom panel: ratio
        axes[1].plot(ells, ratio_test, lw=1.2, color="seagreen", linestyle=":",
                     label=f"{label_test} / {label_cosmogrid}")
        axes[1].axhline(1.0, color="k", lw=0.8, linestyle="--")
        axes[1].set_xlabel(r"Multipole $\ell$", fontsize=12)
        axes[1].set_ylabel(r"$C_\ell\,/\,C_\ell^{\rm CosmoGrid}$", fontsize=12)
        axes[1].set_ylim(0.7, 1.3)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, which="both", alpha=0.3)

        # Vertical scale lines
        for ax in axes:
            add_scale_vlines(ax, chi, nside, lmax_power_spectrum, grid_size=grid_size)

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(2, lmax_power_spectrum)

        fig.tight_layout()
        out_path = out_dir / f"cl_ratio_shell{idx:03d}_z{z_lo:.3f}-{z_hi:.3f}_lmax{lmax_information_loss}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    plt.close("all")
    print("Cl ratio plots done.")


if __name__ == "__main__":
    from pathlib import Path

    test_npz = Path("/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/compressed_shells.npz")
    cosmogrid_npz = Path("/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/compressed_shells.npz")
    out_dir = Path("/capstor/scratch/cscs/damrein/outputs/plots/information_loss")
    lmax_power_spectrum = 3000
    lmax_information_loss = 3000
    lbox = 900.0
    res_pm = 1664  
    shell_indices = [3, 65]

    plot_cl_ratio_and_with_information_loss(
        test_npz=test_npz,
        cosmogrid_npz=cosmogrid_npz,
        out_dir=out_dir,
        shell_indices=shell_indices,
        lmax_power_spectrum=lmax_power_spectrum,
        lmax_information_loss=lmax_information_loss,
        lbox=lbox,
        res_pm=res_pm,
    )

