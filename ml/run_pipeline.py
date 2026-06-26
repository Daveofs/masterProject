#!/usr/bin/env python3
"""Leave-one-out pipeline orchestration for Harmonic Flow Matching."""

from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
import os
from pathlib import Path
import torch
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

vis_root = Path("/users/damrein/masterProject/vis")
if str(vis_root) not in sys.path:
    sys.path.insert(0, str(vis_root))
from visualize import plot_shells

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


def plot_cl_ratio(
    test_npz,
    corrected_npz,
    cosmogrid_npz,
    out_dir,
    shell_indices=[3, 65],
    lmax=3000,
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
    data_corr = np.load(corrected_npz, allow_pickle=False)
    data_cosmo = np.load(cosmogrid_npz, allow_pickle=False)

    shells_test = np.asarray(data_test["shells"], dtype=np.float64)
    shells_corr = np.asarray(data_corr["shells"], dtype=np.float64)
    shells_cosmo = np.asarray(data_cosmo["shells"], dtype=np.float64)

    n_shells = min(shells_test.shape[0], shells_corr.shape[0], shells_cosmo.shape[0])
    nside = hp.npix2nside(shells_cosmo.shape[1])

    # Shell info (redshift bounds)
    has_info = "info" in data_cosmo
    info = data_cosmo["info"] if has_info else None

    # Grid size for vertical lines
    grid_size = lbox / res_pm  # cMpc/h per grid cell

    ells = np.arange(lmax + 1)

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
        cl_test = compute_cl(shells_test[idx], lmax)
        cl_corr = compute_cl(shells_corr[idx], lmax)
        cl_cosmo = compute_cl(shells_cosmo[idx], lmax)

        # Ratios
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_test = np.where(cl_cosmo != 0, cl_test / cl_cosmo, np.nan)
            ratio_corr = np.where(cl_cosmo != 0, cl_corr / cl_cosmo, np.nan)

        # --- Plot ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top panel: Cl spectra
        axes[0].plot(ells, cl_test, label=label_test, lw=1.2, color="seagreen")
        axes[0].plot(ells, cl_corr, label=label_corrected, lw=1.2, color="steelblue")
        axes[0].plot(ells, cl_cosmo, label=label_cosmogrid, lw=1.2, color="tomato", linestyle="--")

        axes[0].set_ylabel(r"$C_\ell$", fontsize=12)
        axes[0].set_yscale("log")
        _grid_str = f"  |  grid={grid_size:.3f} cMpc/h"
        axes[0].set_title(
            f"Shell {idx}  |  z = [{z_lo:.4f}, {z_hi:.4f}]  |  nside={nside}{_grid_str}",
            fontsize=12,
        )
        axes[0].legend(fontsize=9)
        axes[0].grid(True, which="both", alpha=0.3)

        # Bottom panel: ratio
        axes[1].plot(ells, ratio_test, lw=1.2, color="seagreen", linestyle=":",
                     label=f"{label_test} / {label_cosmogrid}")
        axes[1].plot(ells, ratio_corr, lw=1.2, color="darkorchid",
                     label=f"{label_corrected} / {label_cosmogrid}")
        axes[1].axhline(1.0, color="k", lw=0.8, linestyle="--")
        axes[1].set_xlabel(r"Multipole $\ell$", fontsize=12)
        axes[1].set_ylabel(r"$C_\ell\,/\,C_\ell^{\rm CosmoGrid}$", fontsize=12)
        axes[1].set_ylim(0.7, 1.3)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, which="both", alpha=0.3)

        # Vertical scale lines
        for ax in axes:
            add_scale_vlines(ax, chi, nside, lmax, grid_size=grid_size)

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlim(2, lmax)

        fig.tight_layout()
        out_path = out_dir / f"cl_ratio_shell{idx:03d}_z{z_lo:.3f}-{z_hi:.3f}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    plt.close("all")
    print("Cl ratio plots done.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/capstor/scratch/cscs/damrein/cosmogridv1_test2")
    p.add_argument("--test-cosmo", type=str, required=True)
    p.add_argument("--train-script", type=str, default="train_flow_matching.py")
    p.add_argument("--apply-script", type=str, default="apply_flow_correction.py")
    p.add_argument("--python", type=str, default=sys.executable)
    
    p.add_argument("--lmax", type=int, default=1024, help="Harmonic bandlimit degree.")
    p.add_argument("--max-shells", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--ode-steps", type=int, default=10, help="Euler integration steps.")
    p.add_argument("--log-interval", type=int, default=5)

    p.add_argument("--shell-index", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--srun-torchrun", action="store_true")
    p.add_argument("--shared-tmp", type=str, default=None)
    p.add_argument("--out-root", type=str, default="./outputs")
    p.add_argument("--plot-nside", type=int, default=2048)
    p.add_argument("--plot-log", action="store_true", default=True)
    return p.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "="*80)
    print("Executing:", " ".join(str(x) for x in cmd))
    print("="*80 + "\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve()
    test_dir = data_root / args.test_cosmo

    all_cosmos = sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("cosmo_"))
    train_cosmos = [d for d in all_cosmos]
    
    test_run_dirs = [r for r in sorted(test_dir.iterdir()) if r.is_dir() and r.name.startswith("run_")]
    test_run_dir = test_run_dirs[0] if test_run_dirs else test_dir

    out_root = Path(args.out_root).expanduser()
    run_out = out_root / args.test_cosmo
    model_out, npz_out = run_out / "model", run_out / "npz"
    for d in [model_out, npz_out]: d.mkdir(parents=True, exist_ok=True)

    train_script = (root_dir / args.train_script).resolve()
    apply_script = (root_dir / args.apply_script).resolve()

    test_input = test_run_dir / "shells_nside=2048.npz"
    test_high = test_run_dir / "compressed_shells.npz"
    test_params = test_run_dir / "params.yml"
    corrected_out = npz_out / f"{args.test_cosmo}_{Path('shells_nside=2048').stem}_corrected.npz"

    shared_tmp_dir = Path(args.shared_tmp) if args.shared_tmp else None
    if shared_tmp_dir: shared_tmp_dir.mkdir(parents=True, exist_ok=True)


    print("\n" + "=" * 80)
    print("Start training flow-matching")
    print("=" * 80 + "\n")

    with tempfile.TemporaryDirectory(prefix="flow_loo_", dir=shared_tmp_dir) as tmp:
        tmp_root = Path(tmp)
        for folder in train_cosmos:
            (tmp_root / folder.name).symlink_to(folder, target_is_directory=True)

        train_args = [
            str(train_script),
            "--data-dir", str(tmp_root),
            "--lmax", str(args.lmax),
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--sigma", str(args.sigma),
            "--hidden", str(args.hidden),
            "--out-dir", str(model_out),
            "--log-interval", str(args.log_interval),
        ]

        if args.srun_torchrun:
            nnodes, gpus = os.environ.get("SLURM_NNODES", "1"), os.environ.get("GPUS_PER_NODE", "4")
            addr, port = os.environ.get("MASTER_ADDR", "127.0.0.1"), os.environ.get("MASTER_PORT", "29500")
            job_id = os.environ.get("SLURM_JOB_ID", "0")
            
            cmd_str = f"torchrun --nnodes={nnodes} --nproc_per_node={gpus} --rdzv_id={job_id} --rdzv_backend=c10d --rdzv_endpoint={addr}:{port} " + " ".join(train_args)
            train_cmd = ["srun", "bash", "-c", cmd_str]
        else:
            train_cmd = [args.python] + train_args

        run_cmd(train_cmd, cwd=root_dir)

    print("\n" + "=" * 80)
    print("Applying trained model to test cosmology")
    print("=" * 80 + "\n")

    model_path = model_out / "flow_mlp.pth"
    apply_device = "cuda:0" if args.srun_torchrun and torch.cuda.is_available() else args.device

    apply_cmd = [
        args.python, str(apply_script),
        "--model", str(model_path),
        "--input", str(test_input),
        "--params", str(test_params),
        "--steps", str(args.ode_steps),
        "--device", apply_device,
        "--out", str(corrected_out),
        "--diagnostic",
    ]
    run_cmd(apply_cmd, cwd=root_dir)

    orig_data, corr_data = np.load(test_input, allow_pickle=False), np.load(corrected_out, allow_pickle=False)
    shells_orig, shells_corr = np.asarray(orig_data["shells"], dtype=np.float32), np.asarray(corr_data["shells"], dtype=np.float32)

    cl_out_dir = run_out / "cl_ratio"
    plot_cl_ratio(
            test_npz=str(test_input),
            corrected_npz=str(corrected_out),
            cosmogrid_npz=str(test_high),
            out_dir=str(cl_out_dir),
            shell_indices=[3, 65],
            lmax=3000,
            lbox=900,
            res_pm=1664,
            label_test="DRF - Low-res input",
            label_corrected="DRF - Flow Corrected",
            label_cosmogrid="CosmoGridV1",
        )
    print("Leave-one-out Harmonic pipeline completed.")


if __name__ == "__main__":
    main()
