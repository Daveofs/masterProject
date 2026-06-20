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

vis_root = Path("/users/damrein/masterProject/vis")
if str(vis_root) not in sys.path:
    sys.path.insert(0, str(vis_root))
from visualize import plot_shells


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/capstor/scratch/cscs/damrein/cosmogridv1_test2")
    p.add_argument("--test-cosmo", type=str, required=True)
    p.add_argument("--train-script", type=str, default="train_flow_matching.py")
    p.add_argument("--apply-script", type=str, default="apply_flow_correction.py")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--low-npz", type=str, default="shells_nside=2048.npz")
    p.add_argument("--high-npz", type=str, default="compressed_shells.npz")
    
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
    train_cosmos = [d for d in all_cosmos if d.name != args.test_cosmo]
    
    test_run_dirs = [r for r in sorted(test_dir.iterdir()) if r.is_dir() and r.name.startswith("run_")]
    test_run_dir = test_run_dirs[0] if test_run_dirs else test_dir

    out_root = Path(args.out_root).expanduser()
    run_out = out_root / args.test_cosmo
    model_out, plot_out, npz_out = run_out / "model", run_out / "plots", run_out / "npz"
    for d in [model_out, plot_out, npz_out]: d.mkdir(parents=True, exist_ok=True)

    train_script = (root_dir / args.train_script).resolve()
    apply_script = (root_dir / args.apply_script).resolve()

    test_input = test_run_dir / args.low_npz
    test_high = test_run_dir / args.high_npz
    test_params = test_run_dir / "params.yml"
    corrected_out = npz_out / f"{args.test_cosmo}_{Path(args.low_npz).stem}_corrected.npz"

    shared_tmp_dir = Path(args.shared_tmp) if args.shared_tmp else None
    if shared_tmp_dir: shared_tmp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="flow_loo_", dir=shared_tmp_dir) as tmp:
        tmp_root = Path(tmp)
        for folder in train_cosmos:
            (tmp_root / folder.name).symlink_to(folder, target_is_directory=True)

        train_args = [
            str(train_script),
            "--data-dir", str(tmp_root),
            "--low-npz", args.low_npz,
            "--high-npz", args.high_npz,
            "--lmax", str(args.lmax),
            "--max-shells", str(args.max_shells),
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
    ]
    run_cmd(apply_cmd, cwd=root_dir)

    orig_data, corr_data = np.load(test_input, allow_pickle=False), np.load(corrected_out, allow_pickle=False)
    shells_orig, shells_corr = np.asarray(orig_data["shells"], dtype=np.float32), np.asarray(corr_data["shells"], dtype=np.float32)
    shells_diff = shells_corr - shells_orig

    nside_orig = hp.npix2nside(int(shells_orig.shape[1]))
    orig_named, corr_named, diff_named = npz_out / f"shells_nside={nside_orig}.npz", npz_out / f"shells_corrected_nside={nside_orig}.npz", npz_out / f"shells_diff_nside={nside_orig}.npz"

    np.savez_compressed(orig_named, shells=shells_orig)
    np.savez_compressed(corr_named, shells=shells_corr)
    np.savez_compressed(diff_named, shells=shells_diff)

    idx = int(args.shell_index)
    plot_shells(npz_path=orig_named, z_bin=idx, nside=args.plot_nside, output_dir=plot_out, plot_logarithmic=args.plot_log, name=f"{args.test_cosmo}_orig")
    plot_shells(npz_path=corr_named, z_bin=idx, nside=args.plot_nside, output_dir=plot_out, plot_logarithmic=args.plot_log, name=f"{args.test_cosmo}_corr")
    plot_shells(npz_path=diff_named, z_bin=idx, nside=args.plot_nside, output_dir=plot_out, plot_logarithmic=args.plot_log, name=f"{args.test_cosmo}_diff")
    plot_shells(npz_path=test_high,  z_bin=idx, nside=args.plot_nside, output_dir=plot_out, plot_logarithmic=args.plot_log, name=f"{args.test_cosmo}_target")

    print("Leave-one-out Harmonic pipeline completed.")


if __name__ == "__main__":
    main()
