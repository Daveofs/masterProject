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


def process_single_run(task_args):
    """Worker function executed by CPU subprocesses."""
    ld, low_name, high_name, lmax = task_args
    
    # 1. Changed target extensions to raw .npy
    out_low_path = ld / f"low_alms_lmax{lmax}.npy"
    out_high_path = ld / f"high_alms_lmax{lmax}.npy"
    
    if out_low_path.exists() and out_high_path.exists():
        return f"[Skipped] {ld.relative_to(ld.parent.parent)} (Already transformed)"
        
    try:
        low_data = np.load(ld / low_name, allow_pickle=False)["shells"]
        high_data = np.load(ld / high_name, allow_pickle=False)["shells"]
        n_available = min(low_data.shape[0], high_data.shape[0])
        
        low_alms, high_alms = [], []
        
        for i in range(n_available):
            alm_low = hp.map2alm(low_data[i], lmax=lmax, iter=1)
            alm_high = hp.map2alm(high_data[i], lmax=lmax, iter=1)
            
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
    parser.add_argument("--low-npz", type=str, default="shells_nside=2048.npz")
    parser.add_argument("--high-npz", type=str, default="compressed_shells.npz")
    parser.add_argument("--lmax", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
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

    tasks = []
    for ld in leaf_dirs:
        if (ld / args.low_npz).exists() and (ld / args.high_npz).exists():
            tasks.append((ld, args.low_npz, args.high_npz, args.lmax))

    print(f"Found {len(tasks)} valid execution targets inside: {data_dir}")
    print(f"Spawning {args.num_workers} multi-core processes...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_run, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Precomputing Alms"):
            res_message = fut.result()
            print(res_message)


if __name__ == "__main__":
    main()