#!/usr/bin/env python3
"""One-time offline pre-computation script for HEALPix map -> Alm transformations.

Saves flattened real/imaginary Alm float32 vectors directly to disk as raw, 
uncompressed .npy binaries, enabling zero-RAM memory mapping (mmap_mode='r').
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import healpy as hp
from tqdm import tqdm


def _resolve_low(ld, low_name, low_glob):
    """Resolve the low (DISCO) shells path: a glob (real cosmogridv1 layout with
    disco_sim/<gpu_grid_*>/disco_shells_nside=2048.npz) or a plain filename.
    When multiple gpu_grid_* runs exist (e.g. restarts/reruns), pick the most
    recently modified one rather than the alphabetically first."""
    if low_glob:
        import glob as _g
        hits = _g.glob(str(ld / low_glob))
        if not hits:
            return None
        return Path(max(hits, key=lambda h: Path(h).stat().st_mtime))
    p = ld / low_name
    return p if p.exists() else None


def process_single_run(task_args):
    """Worker function executed by CPU subprocesses."""
    ld, low_name, high_name, lmax, low_glob, log_density = task_args

    # 1. Changed target extensions to raw .npy
    prefix = "_log" if log_density else ""
    out_low_path = ld / f"low{prefix}_alms_lmax{lmax}.npy"
    out_high_path = ld / f"high{prefix}_alms_lmax{lmax}.npy"

    if out_low_path.exists() and out_high_path.exists():
       return f"[Skipped] {ld.name} (Already transformed)"

    try:
        low_path = _resolve_low(ld, low_name, low_glob)
        high_path = ld / high_name
        if low_path is None or not high_path.exists():
            return f"[Skipped] {ld.name} (missing low/high shells)"
        low_data = np.load(low_path, allow_pickle=False)["shells"]
        high_data = np.load(high_path, allow_pickle=False)["shells"]
        n_available = min(low_data.shape[0], high_data.shape[0])

        low_alms, high_alms = [], []

        for i in range(n_available):
            # log1p(rho) is well-defined at rho=0 (-> 0, not -inf) and, unlike raw
            # rho, reconstructs via expm1 without ever needing a hard floor at 0 --
            # see transfer_function.py apply()'s --log-density branch.
            low_map = np.log1p(low_data[i]) if log_density else low_data[i]
            high_map = np.log1p(high_data[i]) if log_density else high_data[i]
            alm_low = hp.map2alm(low_map, lmax=lmax, iter=1)
            alm_high = hp.map2alm(high_map, lmax=lmax, iter=1)
            
            vec_low = np.concatenate([alm_low.real, alm_low.imag]).astype(np.float32)
            vec_high = np.concatenate([alm_high.real, alm_high.imag]).astype(np.float32)
            
            low_alms.append(vec_low)
            high_alms.append(vec_high)
            
        # 2. Save as raw, uncompressed binary blocks
        np.save(out_low_path, np.stack(low_alms))
        np.save(out_high_path, np.stack(high_alms))
        
        return f"[Success] {ld.relative_to(ld.parent.parent)} ({n_available} shells processed)"
        
    except Exception as e:
        return f"[ERROR] Failed processing {ld.name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Precompute Alms offline.")
    parser.add_argument("--data-dir", type=str, default="/Users/david/testData")
    parser.add_argument("--low-npz", type=str, default="shells_nside=2048.npz",
                        help="Low shells filename (used if --low-glob is empty).")
    parser.add_argument("--low-glob", type=str,
                        default="disco_sim/*/disco_shells_nside=2048.npz",
                        help="Glob (relative to run dir) for the DISCO low shells in the "
                             "real cosmogridv1 layout. Set '' to use --low-npz instead.")
    parser.add_argument("--high-npz", type=str, default="compressed_shells.npz")
    parser.add_argument("--lmax", type=int, default=3000)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--log-density", action="store_true",
                        help="Write low_log_alms_lmax*.npy/high_log_alms_lmax*.npy "
                             "(map2alm of log1p(rho) instead of raw rho) alongside "
                             "-- doesn't touch/require the raw-density alm files.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Multi-node: process only leaf_dirs[shard_index::"
                             "num_shards] -- each shard writes to its OWN "
                             "cosmology's output files (process_single_run's "
                             "existing skip-if-done check means no two shards "
                             "ever touch the same file), so N shards can run "
                             "concurrently on N nodes with no coordination "
                             "needed beyond a disjoint index split.")
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name.startswith("cosmo_")]
    if len(subdirs) == 0:
        subdirs = [data_dir]

    leaf_dirs = []
    for sd in subdirs:
        run_dirs = [r for r in sorted(sd.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        if run_dirs:
            leaf_dirs.extend(run_dirs)
        else:
            leaf_dirs.append(sd)

    if args.num_shards > 1:
        leaf_dirs = leaf_dirs[args.shard_index::args.num_shards]
        print(f"[shard {args.shard_index}/{args.num_shards}] {len(leaf_dirs)} leaf dirs "
              f"assigned to this shard", flush=True)

    tasks = []
    for ld in leaf_dirs:
        low_ok = _resolve_low(ld, args.low_npz, args.low_glob) is not None
        if low_ok and (ld / args.high_npz).exists():
            tasks.append((ld, args.low_npz, args.high_npz, args.lmax, args.low_glob,
                          args.log_density))

    print(f"Found {len(tasks)} valid execution targets inside: {data_dir}")
    print(f"Spawning {args.num_workers} multi-core processes...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_run, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Precomputing Alms"):
            res_message = fut.result()
            print(res_message)


if __name__ == "__main__":
    main()