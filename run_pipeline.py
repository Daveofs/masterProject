#!/usr/bin/env python3
"""Leave-one-out pipeline for flow matching.

Pipeline steps:
1) Build a temporary training root containing symlinks to all `cosmo_*` folders except one held-out test folder.
2) Call `ml/train_flow_matching.py` to train a model.
3) Call `apply_flow_correction.py` on the held-out folder.
4) Save helper NPZ files (`shells_nside=...`, `shells_corrected_nside=...`, and difference).
5) Plot original, corrected, and difference shells using `plot_shells()`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import healpy as hp
import numpy as np

from visualize import plot_shells


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/Users/david/testData", help="Root containing cosmo_* folders")
    p.add_argument("--test-cosmo", type=str, required=True, help="Held-out folder name, e.g. cosmo_000001")

    p.add_argument("--train-script", type=str, default="ml/train_flow_matching.py")
    p.add_argument("--apply-script", type=str, default="ml/apply_flow_correction.py")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable used to call sub-scripts")

    p.add_argument("--low-npz", type=str, default="shells_nside=512_noisy_shuffle.npz")
    p.add_argument("--high-npz", type=str, default="compressed_shells.npz")

    p.add_argument("--nside-small", type=int, default=128)
    p.add_argument("--n-patches", type=int, default=1)
    p.add_argument("--max-shells", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--shell-index", type=int, default=5, help="Shell index used for plotting")
    p.add_argument("--apply-t", type=float, default=0.0, help="t passed to apply_flow_correction.py")
    p.add_argument("--apply-power", type=float, default=0.0, help="power passed to apply_flow_correction.py")
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument(
        "--out-root",
        type=str,
        default="/Users/david/Library/CloudStorage/OneDrive-ETHZurich/ETH-Material/Master Project/github/outputs/models",
        help="Root output directory for model, corrected NPZ and plots",
    )
    p.add_argument("--plot-nside", type=int, default=128, help="nside used by plot_shells()")
    p.add_argument("--plot-log", action="store_true", default=True, help="Use logarithmic plotting in plot_shells()")
    return p.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve()
    test_dir = data_root / args.test_cosmo

    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")
    if not test_dir.exists() or not test_dir.is_dir():
        raise FileNotFoundError(f"Held-out test folder not found: {test_dir}")

    all_cosmos = sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("cosmo_"))
    train_cosmos = [d for d in all_cosmos if d.name != args.test_cosmo]
    if len(train_cosmos) == 0:
        raise RuntimeError("No training folders left after excluding test-cosmo.")

    out_root = Path(args.out_root).expanduser()
    run_out = out_root / args.test_cosmo
    model_out = run_out / "model"
    plot_out = run_out / "plots"
    npz_out = run_out / "npz"
    model_out.mkdir(parents=True, exist_ok=True)
    plot_out.mkdir(parents=True, exist_ok=True)
    npz_out.mkdir(parents=True, exist_ok=True)

    train_script = (root_dir / args.train_script).resolve()
    apply_script = (root_dir / args.apply_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"train script not found: {train_script}")
    if not apply_script.exists():
        raise FileNotFoundError(f"apply script not found: {apply_script}")

    test_input = test_dir / args.low_npz
    test_high = test_dir / args.high_npz
    test_params = test_dir / "params.yml"
    if not test_input.exists():
        raise FileNotFoundError(f"Held-out input file not found: {test_input}")
    if not test_high.exists():
        raise FileNotFoundError(f"Held-out high file not found: {test_high}")
    if not test_params.exists():
        raise FileNotFoundError(f"Held-out params.yml not found: {test_params}")

    corrected_out = npz_out / f"{args.test_cosmo}_{Path(args.low_npz).stem}_corrected.npz"

    with tempfile.TemporaryDirectory(prefix="flow_train_loo_") as tmp:
        tmp_root = Path(tmp)
        for folder in train_cosmos:
            link = tmp_root / folder.name
            link.symlink_to(folder, target_is_directory=True)

        train_cmd = [
            args.python,
            str(train_script),
            "--data-dir",
            str(tmp_root),
            "--low-npz",
            args.low_npz,
            "--high-npz",
            args.high_npz,
            "--nside-small",
            str(args.nside_small),
            "--max-shells",
            str(args.max_shells),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--sigma",
            str(args.sigma),
            "--hidden",
            str(args.hidden),
            "--n-patches",
            str(args.n_patches),
            "--out-dir",
            str(model_out),
            "--log-interval",
            str(args.log_interval),
        ]
        run_cmd(train_cmd, cwd=root_dir)

    model_path = model_out / "flow_mlp.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model not found: {model_path}")

    apply_cmd = [
        args.python,
        str(apply_script),
        "--model",
        str(model_path),
        "--input",
        str(test_input),
        "--params",
        str(test_params),
        "--nside-small",
        str(args.nside_small),
        "--n-patches",
        str(args.n_patches),
        "--power",
        str(args.apply_power),
        "--shell-index",
        "-1",
        "--t",
        str(args.apply_t),
        "--device",
        str(args.device),
        "--out",
        str(corrected_out),
    ]
    run_cmd(apply_cmd, cwd=root_dir)

    if not corrected_out.exists():
        raise FileNotFoundError(f"Corrected output not found: {corrected_out}")

    orig_data = np.load(test_input, allow_pickle=False)
    corr_data = np.load(corrected_out, allow_pickle=False)
    if "shells" not in orig_data or "shells" not in corr_data:
        raise KeyError("Both original and corrected NPZ files must contain 'shells'.")

    shells_orig = np.asarray(orig_data["shells"], dtype=np.float32)
    shells_corr = np.asarray(corr_data["shells"], dtype=np.float32)
    if shells_orig.shape != shells_corr.shape:
        raise ValueError(f"Shape mismatch: original {shells_orig.shape} vs corrected {shells_corr.shape}")

    npix = int(shells_orig.shape[1])
    if not hp.isnpixok(npix):
        raise ValueError(f"Second axis is not valid HEALPix npix: {npix}")
    nside_orig = hp.npix2nside(npix)

    shells_diff = shells_corr - shells_orig

    orig_named = npz_out / f"shells_nside={nside_orig}.npz"
    corr_named = npz_out / f"shells_corrected_nside={nside_orig}.npz"
    diff_named = npz_out / f"shells_diff_nside={nside_orig}.npz"

    np.savez_compressed(orig_named, shells=shells_orig)
    np.savez_compressed(corr_named, shells=shells_corr)
    np.savez_compressed(diff_named, shells=shells_diff)

    idx = int(args.shell_index)
    if not (0 <= idx < shells_orig.shape[0]):
        raise IndexError(f"shell-index {idx} out of range [0, {shells_orig.shape[0]-1}]")

    plot_shells(
        npz_path=orig_named,
        z_bin=idx,
        nside=args.plot_nside,
        output_dir=plot_out,
        plot_logarithmic=args.plot_log,
        name=f"{args.test_cosmo}_shells_nside{args.plot_nside}_idx{idx}",
    )
    plot_shells(
        npz_path=corr_named,
        z_bin=idx,
        nside=args.plot_nside,
        output_dir=plot_out,
        plot_logarithmic=args.plot_log,
        name=f"{args.test_cosmo}_shells_corrected_nside{args.plot_nside}_idx{idx}",
    )
    plot_shells(
        npz_path=diff_named,
        z_bin=idx,
        nside=args.plot_nside,
        output_dir=plot_out,
        plot_logarithmic=args.plot_log,
        name=f"{args.test_cosmo}_shells_diff_nside{args.plot_nside}_idx{idx}",
    )
    plot_shells(
        npz_path=test_high,
        z_bin=idx,
        nside=args.plot_nside,
        output_dir=plot_out,
        plot_logarithmic=args.plot_log,
        name=f"{args.test_cosmo}_high_npz_nside{args.plot_nside}_idx{idx}",
    )

    print("Leave-one-out pipeline completed.")
    print(f"Held-out test folder: {test_dir}")
    print(f"Model path: {model_path}")
    print(f"Original shells file: {orig_named}")
    print(f"Corrected shells file: {corr_named}")
    print(f"Difference shells file: {diff_named}")
    print(f"High shells input: {test_high}")
    print(f"Plots directory: {plot_out}")


if __name__ == "__main__":
    main()
