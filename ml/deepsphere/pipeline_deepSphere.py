#!/usr/bin/env python3
"""End-to-end DeepSphere shell-correction pipeline.

Mirrors run_pipeline.py (the flow-matching leave-one-out pipeline) but uses the
*exact* graph-CNN from deepsphere-cosmo-tf1 (``deepsphere.models.deepsphere`` /
``cgcnn``), configured for map -> map correction (see MLP.py).

Stages
------
1. Data collection : gather (low, high) shell pairs across the cosmology grid,
                      leaving the test cosmology out of training (LOO).
2. Training        : deepsphere(cgcnn).fit() in map->map mode (l2 loss).
3. Correction      : apply the trained model to the held-out test cosmology and
                      write a corrected .npz (same layout as the input shells).
4. Diagnostics     : Cl-ratio plots vs the CosmoGrid high-res reference.

DDP / multi-GPU
---------------
The deepsphere-cosmo-tf1 model is TensorFlow 1.x (a single ``tf.Session`` in
graph mode). It is NOT a PyTorch module and cannot use torch DDP / torchrun, so
training runs on a SINGLE GPU. (True TF multi-GPU would require rewriting
base_model around ``tf.distribute.MirroredStrategy`` or Horovod, which the
original code does not support.) For scaling, the natural axis here is
embarrassingly-parallel over cosmologies: launch independent single-GPU jobs.

Memory note
-----------
deepsphere loads every training map into RAM (LabeledDataset). At nside=512 a map
is ~12.6 MB (Npix=3.1M); nside=2048 (50M px, ~200 MB/map) is infeasible in-memory
for many shells. Use --nside 512 and bound the sample count with --max-cosmos /
--max-pairs. Maps at a different stored nside are ud_grade'd to --nside.

Usage
-----
  python pipeline_deepSphere.py \
      --data-root /capstor/scratch/cscs/damrein/cosmogridv1_test2 \
      --test-cosmo cosmo_000001 \
      --nside 512 --epochs 40 \
      --out-root /capstor/scratch/cscs/damrein/outputs/deepsphere/$SLURM_JOB_ID
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Route tf.keras -> tf_keras (Keras 2) only when that shim is installed (the conda
# env with TF>=2.16). The NGC TF 2.15 container has native Keras 2 and no tf_keras,
# where forcing this flag breaks Keras import. Must precede any TensorFlow import.
import importlib.util as _ilu
if _ilu.find_spec("tf_keras") is not None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# MLP.py provides the deepsphere wiring: load_shell_pairs / build_model / train /
# predict_maps / invert_prediction.
import MLP


# ===========================================================================
# Cl-ratio diagnostics (numpy/healpy/matplotlib only — adapted from run_pipeline)
# ===========================================================================

def compute_cl(shell, lmax):
    """Angular power spectrum of a HEALPix shell, computed on the overdensity."""
    mean = shell.mean()
    delta = shell if mean == 0 else shell / mean - 1.0
    return hp.anafast(delta, lmax=lmax)


def add_scale_vlines(ax, chi, nside, lmax, grid_size=None):
    pix_rad = hp.nside2resol(nside)
    ell_pix = np.pi / pix_rad
    if ell_pix <= lmax:
        ax.axvline(ell_pix, color="gray", ls="--", lw=0.7, alpha=0.6)
        ax.text(ell_pix, ax.get_ylim()[1] * 0.95, r"$\ell_{\rm pix}$",
                fontsize=8, color="gray", ha="left", va="top")
    if grid_size is not None and chi > 0:
        ell_grid = chi / grid_size * 2 * np.pi
        if 2 < ell_grid <= lmax:
            ax.axvline(ell_grid, color="green", ls="-.", lw=0.7, alpha=0.6)
            ax.text(ell_grid, ax.get_ylim()[1] * 0.90, r"$\ell_{\rm grid}$",
                    fontsize=8, color="green", ha="left", va="top")


def plot_cl_ratio(test_npz, corrected_npz, cosmogrid_npz, out_dir,
                  shell_indices=(3, 65), lmax=3000, lbox=900.0, res_pm=1664,
                  label_test="DeepSphere - Low-res input",
                  label_corrected="DeepSphere - Corrected",
                  label_cosmogrid="CosmoGridV1"):
    """Cl spectra + ratio plots for low / corrected / high-res reference shells."""
    print("\n" + "=" * 80 + "\nComputing Cl ratios\n" + "=" * 80)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_test = np.load(test_npz, allow_pickle=False)
    data_corr = np.load(corrected_npz, allow_pickle=False)
    data_cosmo = np.load(cosmogrid_npz, allow_pickle=False)

    shells_test = np.asarray(data_test["shells"], dtype=np.float64)
    shells_corr = np.asarray(data_corr["shells"], dtype=np.float64)
    shells_cosmo = np.asarray(data_cosmo["shells"], dtype=np.float64)

    n_shells = min(shells_test.shape[0], shells_corr.shape[0], shells_cosmo.shape[0])
    nside = hp.npix2nside(shells_cosmo.shape[1])
    info = data_cosmo["info"] if "info" in data_cosmo else None
    grid_size = lbox / res_pm
    ells = np.arange(lmax + 1)

    for idx in shell_indices:
        if idx < 0 or idx >= n_shells:
            print(f"  [WARN] shell {idx} out of range [0,{n_shells}); skipping.")
            continue

        if info is not None and info.dtype.names is not None:
            z_lo = float(info[idx]["lower_z"]) if "lower_z" in info.dtype.names else 0.0
            z_hi = float(info[idx]["upper_z"]) if "upper_z" in info.dtype.names else 0.0
            chi = float(info[idx]["shell_com"]) if "shell_com" in info.dtype.names else 0.0
        else:
            z_lo = z_hi = chi = 0.0

        cl_test = compute_cl(shells_test[idx], lmax)
        cl_corr = compute_cl(shells_corr[idx], lmax)
        cl_cosmo = compute_cl(shells_cosmo[idx], lmax)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_test = np.where(cl_cosmo != 0, cl_test / cl_cosmo, np.nan)
            ratio_corr = np.where(cl_cosmo != 0, cl_corr / cl_cosmo, np.nan)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(ells, cl_test, label=label_test, lw=1.2, color="seagreen")
        axes[0].plot(ells, cl_corr, label=label_corrected, lw=1.2, color="steelblue")
        axes[0].plot(ells, cl_cosmo, label=label_cosmogrid, lw=1.2, color="tomato", ls="--")
        axes[0].set_ylabel(r"$C_\ell$"); axes[0].set_yscale("log")
        axes[0].set_title(f"Shell {idx} | z=[{z_lo:.4f},{z_hi:.4f}] | nside={nside} | "
                          f"grid={grid_size:.3f} cMpc/h", fontsize=12)
        axes[0].legend(fontsize=9); axes[0].grid(True, which="both", alpha=0.3)

        axes[1].plot(ells, ratio_test, lw=1.2, color="seagreen", ls=":",
                     label=f"{label_test} / {label_cosmogrid}")
        axes[1].plot(ells, ratio_corr, lw=1.2, color="darkorchid",
                     label=f"{label_corrected} / {label_cosmogrid}")
        axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
        axes[1].set_xlabel(r"Multipole $\ell$")
        axes[1].set_ylabel(r"$C_\ell/C_\ell^{\rm CosmoGrid}$")
        axes[1].set_ylim(0.7, 1.3)
        axes[1].legend(fontsize=9); axes[1].grid(True, which="both", alpha=0.3)

        for ax in axes:
            add_scale_vlines(ax, chi, nside, lmax, grid_size=grid_size)
            ax.set_xscale("log"); ax.set_xlim(2, lmax)

        fig.tight_layout()
        out_path = out_dir / f"cl_ratio_shell{idx:03d}_z{z_lo:.3f}-{z_hi:.3f}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f"  Saved {out_path}")
    plt.close("all")
    print("Cl ratio plots done.")


# ===========================================================================
# Pipeline stages
# ===========================================================================

def collect_training_data(data_root: Path, test_cosmo: str, args, shared_tmp: Path | None):
    """Build leave-one-out training arrays from the cosmology grid.

    Symlinks every cosmology except the test one into a temp dir, then uses
    MLP.load_shell_pairs to assemble (X_low, Y_target, norm).
    """
    all_cosmos = sorted(d for d in data_root.iterdir()
                        if d.is_dir() and d.name.startswith("cosmo_"))
    # --include-test keeps the test cosmology in training (overfit sanity check).
    train_cosmos = [d for d in all_cosmos
                    if args.include_test or d.name != test_cosmo]
    if args.max_cosmos > 0:
        train_cosmos = train_cosmos[: args.max_cosmos]
    if not train_cosmos:
        raise RuntimeError(f"No training cosmologies found under {data_root}")
    mode = "INCLUDE-TEST" if args.include_test else f"LOO, excluding {test_cosmo}"
    print(f"[data] {len(train_cosmos)} training cosmologies ({mode})")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="ds_loo_", dir=shared_tmp)
    tmp_root = Path(tmp_ctx.name)
    for c in train_cosmos:
        (tmp_root / c.name).symlink_to(c, target_is_directory=True)

    X, Y, norm = MLP.load_shell_pairs(
        tmp_root, nside=args.nside,
        low_name=args.low_name, high_name=args.high_name,
        nest=args.nest, order=args.order,
        max_pairs=(args.max_pairs if args.max_pairs > 0 else None),
        standardize=True, residual=args.residual,
    )
    # Keep tmp dir alive until arrays are materialized (load_shell_pairs read
    # everything into memory already), then clean up.
    tmp_ctx.cleanup()
    return X, Y, norm


def train_streaming_stage(data_root: Path, args, test_run_dir: Path):
    """Build streaming train/val datasets and train without preloading.

    Returns (model, norm). Peak RAM is bounded by one .npz file pair regardless of
    the total number of shells, so this scales to 20k-170k shells.

    Leave-one-out by default (test cosmology excluded). With --include-test the
    test cosmology is included AND the exact applied test run is forced into the
    TRAIN split (never validation) so the overfit sanity check is meaningful.
    """
    # Candidate pairs: exclude test (LOO) unless --include-test.
    exclude = None if args.include_test else args.test_cosmo
    pairs = MLP.build_file_pairs(
        data_root, test_cosmo=exclude,
        low_name=args.low_name, high_name=args.high_name)
    if not pairs:
        raise RuntimeError("No training file pairs found (streaming).")
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    # Identify the exact file pair we will apply to (the test run).
    applied_low = (test_run_dir / args.low_name).resolve()
    applied = [p for p in pairs if p[0].resolve() == applied_low]
    others = [p for p in pairs if p[0].resolve() != applied_low]

    # Validation comes from non-applied files only; the applied test run (if
    # present, i.e. --include-test) stays in training.
    n_val = min(max(args.val_files, 1), max(len(others) - 1, 1))
    val_pairs = others[:n_val]
    train_pairs = applied + others[n_val:]
    if not train_pairs:
        raise RuntimeError("No training files left after the validation split.")

    hvd, rank, size = args._hvd, args._rank, args._size
    mode = "INCLUDE-TEST (overfit sanity check)" if args.include_test \
        else f"LOO (excludes {args.test_cosmo})"
    if rank == 0:
        print(f"[data] streaming [{mode}]: {len(train_pairs)} train files, "
              f"{len(val_pairs)} val files | applied test run in train: {bool(applied)}"
              + (f" | {size} Horovod workers" if hvd else ""))

    # Batch cap for TF's GPU sparse-matmul 2^31 index limit.
    safe = MLP._gpu_batch_for(args.nside, args.order, args.F_hidden, args.n_layers)
    batch_size = min(args.batch_size, safe)

    # Shared normalization: estimate delta_scale from the SAME first files on all
    # ranks so every worker uses identical stats.
    shared_norm = MLP.StreamingShellDataset(
        train_pairs[: max(args.stat_sample_files, 1)], nside=args.nside,
        order=args.order, nest=args.nest, residual=args.residual,
        stat_sample_files=args.stat_sample_files, verbose=(rank == 0)).norm

    if hvd is not None:
        # Data-parallel: each rank streams its own shard of the training files.
        my_pairs = train_pairs[rank::size]
        if not my_pairs:
            my_pairs = [train_pairs[rank % len(train_pairs)]]
        train_ds = MLP.StreamingShellDataset(
            my_pairs, nside=args.nside, order=args.order, nest=args.nest,
            residual=args.residual, norm=shared_norm, verbose=(rank == 0))
        total_steps = max(args.epochs * train_ds.N // batch_size, 1)
        model = MLP.build_model(
            nside=args.nside, order=args.order, n_layers=args.n_layers,
            F_hidden=args.F_hidden, K=args.K, batch_norm=not args.no_batch_norm,
            num_epochs=args.epochs, batch_size=batch_size, learning_rate=args.lr,
            total_steps=total_steps, distributed=True, verbose=(rank == 0),
            dir_name=f"ds_correction_{args.test_cosmo}_{os.environ.get('SLURM_JOB_ID', 'local')}",
        )
        losses = MLP.train_horovod(model, train_ds, args.epochs, batch_size, hvd)
        return model, shared_norm, (None, losses, losses, None)

    # Single-GPU path (deepsphere's own fit loop, with validation).
    train_ds = MLP.StreamingShellDataset(
        train_pairs, nside=args.nside, order=args.order, nest=args.nest,
        residual=args.residual, norm=shared_norm)
    val_ds = MLP.StreamingShellDataset(
        val_pairs, nside=args.nside, order=args.order, nest=args.nest,
        residual=args.residual, norm=shared_norm, shuffle=False,
        max_eval_patches=args.max_eval_patches)
    total_steps = max(args.epochs * train_ds.N // batch_size, 1)
    eval_freq = args.eval_frequency if args.eval_frequency > 0 \
        else max(total_steps // 20, 1)
    print(f"[train] total_steps={total_steps:,} | eval every {eval_freq:,} steps | "
          f"batch={batch_size} | val<= {args.max_eval_patches} patches")
    model = MLP.build_model(
        nside=args.nside, order=args.order, n_layers=args.n_layers,
        F_hidden=args.F_hidden, K=args.K, batch_norm=not args.no_batch_norm,
        num_epochs=args.epochs, batch_size=batch_size, learning_rate=args.lr,
        total_steps=total_steps, eval_frequency=eval_freq,
        dir_name=f"ds_correction_{args.test_cosmo}_{os.environ.get('SLURM_JOB_ID', 'local')}",
    )
    acc_val, loss_val, loss_train, t_step = MLP.train_streaming(model, train_ds, val_ds)
    return model, shared_norm, (acc_val, loss_val, loss_train, t_step)


def apply_to_test(model, norm: dict, test_run_dir: Path, args, out_npz: Path):
    """Correct every shell of the held-out test cosmology and save a .npz."""
    test_input = test_run_dir / args.low_name
    print(f"\n[apply] correcting {test_input}")
    data = dict(np.load(test_input, allow_pickle=False))
    shells = np.asarray(data["shells"], dtype=np.float32)  # (Nshells, Npix_in)

    npix = hp.nside2npix(args.nside)
    order = "NESTED" if args.nest else "RING"

    # Resample to the model's nside if needed.
    if shells.shape[1] != npix:
        shells = np.stack([
            hp.ud_grade(s, args.nside, order_in=order, order_out=order).astype(np.float32)
            for s in shells
        ])

    # Normalize inputs exactly as in training (per-shell overdensity), predict,
    # then invert. The high argument to overdensity_forward is unused for X.
    x_low_phys = shells.copy()
    X, _ = MLP.overdensity_forward(shells, shells, norm["delta_scale"], residual=False)

    # Partial-sphere: split each map into patches, predict per patch, stitch back.
    order = int(norm.get("order", 1))
    n_maps = X.shape[0]
    if order > 1:
        X = MLP.map_to_patches(X, order)

    t0 = time.time()
    pred = np.asarray(model.predict(X))                      # (n_samples, patch/Npix)
    if order > 1:
        pred = MLP.patches_to_maps(pred, order, n_maps)      # (Nshells, Npix)
    corrected = MLP.invert_prediction(pred, x_low_phys, norm)  # physical maps
    print(f"[apply] corrected {corrected.shape[0]} shells in {time.time()-t0:.1f}s")

    out_dict = {k: data[k] for k in data}
    out_dict["shells"] = corrected.astype(np.float32)
    out_dict["corrected_index"] = np.arange(corrected.shape[0], dtype=np.int64)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out_dict)
    print(f"[apply] saved {out_npz}")
    return out_npz


def plot_training_loss(loss_val, loss_train, out_dir: Path):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        if loss_train is not None and len(loss_train):
            ax.plot(loss_train, lw=0.8, alpha=0.7, label="train")
        if loss_val is not None and len(loss_val):
            xs = np.linspace(0, max(len(loss_train) - 1, 1), len(loss_val)) \
                if loss_train is not None and len(loss_train) else range(len(loss_val))
            ax.plot(xs, loss_val, "o-", ms=3, label="validation")
        ax.set_yscale("log"); ax.set_xlabel("eval step"); ax.set_ylabel("l2 loss")
        ax.set_title("DeepSphere correction training"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "training_loss.png", dpi=150); plt.close(fig)
        np.savez(out_dir / "training_loss.npz",
                 loss_val=np.asarray(loss_val if loss_val is not None else []),
                 loss_train=np.asarray(loss_train if loss_train is not None else []))
    except Exception as e:
        print(f"[warn] loss plot failed: {e}")


# ===========================================================================
# Orchestration
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--data-root", type=str,
                   default="/capstor/scratch/cscs/damrein/cosmogridv1_test2")
    p.add_argument("--test-cosmo", type=str, required=True)
    p.add_argument("--low-name", type=str, default="shells_nside=512.npz",
                   help="Low-res (DISCO-DJ) shell stack filename within a run dir.")
    p.add_argument("--high-name", type=str, default="compressed_shells.npz",
                   help="High-res (CosmoGrid) reference shell stack filename.")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--order", type=int, default=1,
                   help="Partial-sphere split for high nside. order=1 -> full sphere; "
                        "order>1 -> 12*order^2 patches of (nside/order)^2 px each. "
                        "Use order>=8 for nside=2048 (small-scale correction).")
    p.add_argument("--nest", action="store_true", default=True,
                   help="Stored maps are in NESTED ordering (deepsphere graph default).")
    p.add_argument("--residual", action="store_true",
                   help="Learn the high-low residual instead of the high map directly.")
    p.add_argument("--max-cosmos", type=int, default=0,
                   help="Cap #training cosmologies (0 = all). Bounds RAM.")
    p.add_argument("--max-pairs", type=int, default=0,
                   help="Cap total (low,high) shell pairs (0 = all). Bounds RAM.")
    p.add_argument("--streaming", action="store_true",
                   help="Stream shells from disk one .npz file at a time instead of "
                        "preloading all into RAM. REQUIRED for 20k-170k shells.")
    p.add_argument("--include-test", action="store_true",
                   help="SANITY CHECK: include the test cosmology in training (disables "
                        "leave-one-out). The applied test shells are kept in the TRAIN "
                        "split, so a good model should reproduce them well. If the Cl "
                        "ratio is still off here, the model/correction itself is at fault.")
    p.add_argument("--horovod", action="store_true",
                   help="Data-parallel multi-GPU training with Horovod (one MPI rank per "
                        "GPU). Files are sharded across ranks; only rank 0 applies+plots. "
                        "Launch with srun --ntasks-per-node=<#GPUs> (see run_deepsphere_horovod.sh).")
    p.add_argument("--val-files", type=int, default=1,
                   help="[streaming] #training file-pairs held out (in-memory) for "
                        "validation. Keep small.")
    p.add_argument("--stat-sample-files", type=int, default=2,
                   help="[streaming] #files sampled to estimate normalization stats.")
    # Model / training
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--K", type=int, default=5, help="Chebyshev polynomial order.")
    p.add_argument("--F-hidden", type=int, nargs="*", default=None,
                   help="Hidden feature widths (len = n_layers-1). Default auto.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-frequency", type=int, default=0,
                   help="Steps between validation passes. 0 = auto (~20 evals over "
                        "the whole run). Frequent eval is very costly — keep it sparse.")
    p.add_argument("--max-eval-patches", type=int, default=4096,
                   help="[streaming] cap on validation patches per eval (0 = all). "
                        "Full-split validation at high nside is extremely slow.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--no-batch-norm", action="store_true")
    # Output
    p.add_argument("--out-root", type=str, default="./outputs_deepsphere")
    p.add_argument("--shared-tmp", type=str, default=None)
    # Diagnostics
    p.add_argument("--shell-indices", type=int, nargs="*", default=[3, 65])
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--lbox", type=float, default=900.0)
    p.add_argument("--res-pm", type=int, default=1664)
    return p.parse_args()


def init_distributed(args):
    """Init Horovod if requested; return (hvd_or_None, rank, size) and stash on args."""
    if not args.horovod:
        args._hvd, args._rank, args._size = None, 0, 1
        return None, 0, 1
    if not args.streaming:
        raise SystemExit("--horovod requires --streaming (data is sharded per file).")
    import horovod.tensorflow as hvd
    hvd.init()
    args._hvd, args._rank, args._size = hvd, hvd.rank(), hvd.size()
    if hvd.rank() == 0:
        print(f"[horovod] initialized {hvd.size()} workers "
              f"(local_size={hvd.local_size()})", flush=True)
    return hvd, hvd.rank(), hvd.size()


def main() -> None:
    args = parse_args()
    hvd, rank, size = init_distributed(args)
    is_main = (rank == 0)

    data_root = Path(args.data_root).expanduser().resolve()
    test_dir = data_root / args.test_cosmo
    if not test_dir.exists():
        raise FileNotFoundError(f"Test cosmology dir not found: {test_dir}")

    test_run_dirs = [r for r in sorted(test_dir.iterdir())
                     if r.is_dir() and r.name.startswith("run_")]
    test_run_dir = test_run_dirs[0] if test_run_dirs else test_dir

    out_root = Path(args.out_root).expanduser()
    run_out = out_root / args.test_cosmo
    model_out, npz_out = run_out / "model", run_out / "npz"
    for d in (model_out, npz_out):
        d.mkdir(parents=True, exist_ok=True)

    shared_tmp = Path(args.shared_tmp) if args.shared_tmp else None
    if shared_tmp:
        shared_tmp.mkdir(parents=True, exist_ok=True)

    # -- 1+2. Data collection + training -----------------------------------
    gpus = f"{size} GPUs (Horovod)" if hvd else "single GPU"
    mode = "streaming (disk)" if args.streaming else "in-memory"
    if is_main:
        print("\n" + "=" * 80 + f"\nStage 1+2/4: data + training [{mode}, {gpus}]\n" + "=" * 80)
    if args.streaming:
        model, norm, (acc_val, loss_val, loss_train, t_step) = \
            train_streaming_stage(data_root, args, test_run_dir)
    else:
        X, Y, norm = collect_training_data(data_root, args.test_cosmo, args, shared_tmp)
        safe = MLP._gpu_batch_for(args.nside, args.order, args.F_hidden, args.n_layers)
        batch_size = min(args.batch_size, max(X.shape[0] - 1, 1), safe)
        n_train = int(round((1 - args.val_frac) * X.shape[0]))
        total_steps = max(args.epochs * max(n_train, 1) // batch_size, 1)
        model = MLP.build_model(
            nside=args.nside, order=args.order, n_layers=args.n_layers,
            F_hidden=args.F_hidden, K=args.K, batch_norm=not args.no_batch_norm,
            num_epochs=args.epochs, batch_size=batch_size, learning_rate=args.lr,
            total_steps=total_steps, eval_frequency=args.eval_frequency or 50,
            dir_name=f"ds_correction_{args.test_cosmo}_{os.environ.get('SLURM_JOB_ID', 'local')}",
        )
        acc_val, loss_val, loss_train, t_step = MLP.train(
            model, X, Y, val_frac=args.val_frac)
        del X, Y  # free before apply

    # Only rank 0 has the trained checkpoint -> it does apply + diagnostics.
    # Other ranks are done after training.
    if not is_main:
        return

    plot_training_loss(loss_val, loss_train, model_out)
    np.savez(model_out / "norm.npz", **{k: np.asarray(v) for k, v in norm.items()})
    print(f"[train] saved normalization stats to {model_out / 'norm.npz'}")

    # -- 3. Apply to held-out test cosmology -------------------------------
    print("\n" + "=" * 80 + "\nStage 3/4: correcting test cosmology\n" + "=" * 80)
    corrected_out = npz_out / f"{args.test_cosmo}_{Path(args.low_name).stem}_corrected.npz"
    apply_to_test(model, norm, test_run_dir, args, corrected_out)

    # -- 4. Diagnostics ----------------------------------------------------
    print("\n" + "=" * 80 + "\nStage 4/4: Cl-ratio diagnostics\n" + "=" * 80)
    test_input = test_run_dir / args.low_name
    test_high = test_run_dir / args.high_name
    plot_cl_ratio(
        test_npz=str(test_input),
        corrected_npz=str(corrected_out),
        cosmogrid_npz=str(test_high),
        out_dir=str(run_out / "cl_ratio"),
        shell_indices=tuple(args.shell_indices),
        lmax=args.lmax, lbox=args.lbox, res_pm=args.res_pm,
    )
    print("\nDeepSphere pipeline completed.")


if __name__ == "__main__":
    main()
