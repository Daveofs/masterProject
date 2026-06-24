import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path


def compute_cl(shell, lmax):
    """Compute angular power spectrum of a HEALPix shell (overdensity)."""
    mean = shell.mean()
    if mean == 0:
        delta = shell
    else:
        delta = shell / mean - 1.0
    return hp.anafast(delta, lmax=lmax)

def plot_cl_ratio_with_alms(
    alms_npy,
    cosmogrid_npz,
    out_dir,
    shell_indices=[3, 65],
    lmax_power_spectrum=3000,
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
    alms = np.load(alms_npy, mmap_mode="r")
    data_cosmo = np.load(cosmogrid_npz, allow_pickle=False)
    shells_cosmo = np.asarray(data_cosmo["shells"], dtype=np.float64)
    nside = hp.npix2nside(shells_cosmo.shape[1])
    shells_test = np.zeros((alms.shape[0], hp.nside2npix(nside)), dtype=np.float32)

    n_shells = min(shells_test.shape[0], shells_cosmo.shape[0])

    # Transfrom alms back to overdensity maps
    for i in range(alms.shape[0]):
        print(f"Transforming Alm to overdensity map for shell {i+1}/{alms.shape[0]}")
        vec = alms[i]
        nalm = vec.size // 2
        alm_complex = (
            vec[:nalm] +
            1j * vec[nalm:]
        )
        shells_test[i] = hp.alm2map(
            alm_complex,
            nside=nside,
            lmax=lmax_power_spectrum,
        )


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
            f"Shell {idx}  |  nside={nside}{_grid_str}",
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

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(2, lmax_power_spectrum)

        fig.tight_layout()
        out_path = out_dir / f"cl_ratio_shell{idx:03d}_z{z_lo:.3f}-{z_hi:.3f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    plt.close("all")
    print("Cl ratio plots done.")


if __name__ == "__main__":
    
    alms_npy = Path("/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/high_alms_lmax3000.npy")
    cosmogrid_npz = Path("/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/compressed_shells.npz")
    out_dir = Path("/capstor/scratch/cscs/damrein/outputs/plots/check_alms")
    lmax_power_spectrum = 3000
    lbox = 900.0
    res_pm = 1664  
    shell_indices = [3, 65]

    plot_cl_ratio_with_alms(
        alms_npy=alms_npy,
        cosmogrid_npz=cosmogrid_npz,
        out_dir=out_dir,
        shell_indices=shell_indices,
        lmax_power_spectrum=lmax_power_spectrum,
        lbox=lbox,
        res_pm=res_pm,
    )

