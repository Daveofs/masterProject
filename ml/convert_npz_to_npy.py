#!/usr/bin/env python3
"""
One-time conversion of shells_nside=2048.npz and compressed_shells.npz
to plain .npy files so they can be memory-mapped in Python 3.13.

Python 3.13's zipfile module added a strict 'overlapping entries' security
check (zip-bomb mitigation) that breaks numpy's mmap_mode='r' for .npz files.
A plain .npy file (single array, no zip) can be mmapped directly via
np.load('file.npy', mmap_mode='r') with no zipfile involved.

Usage:
  python convert_npz_to_npy.py --data-root /capstor/scratch/cscs/damrein/cosmogridv1_test2
  python convert_npz_to_npy.py --data-root /capstor/scratch/cscs/damrein/cosmogridv1_test2 --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def convert(npz_path: Path, array_key: str = "shells", dry_run: bool = False) -> None:
    npy_path = npz_path.with_suffix(".npy")
    if npy_path.exists():
        print(f"  skip (exists): {npy_path.name}")
        return
    if dry_run:
        print(f"  would convert: {npz_path} → {npy_path.name}")
        return

    print(f"  loading {npz_path.name} ...", flush=True)
    data = np.load(npz_path, allow_pickle=False)
    if array_key not in data:
        print(f"  WARNING: key '{array_key}' not in {npz_path.name}, keys={list(data.files)}")
        return
    arr = data[array_key]
    print(f"    shape={arr.shape} dtype={arr.dtype}  ({arr.nbytes / 1e9:.2f} GB uncompressed)")
    print(f"  saving {npy_path.name} ...", flush=True)
    np.save(npy_path, arr)
    print(f"  done → {npy_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True,
                   help="Root directory containing cosmo_* subdirectories")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without writing files")
    args = p.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        sys.exit(f"ERROR: data-root does not exist: {data_root}")

    targets = [
        ("shells_nside=2048.npz", "shells"),
        ("compressed_shells.npz", "shells"),
    ]

    for npz_name, key in targets:
        found = sorted(data_root.rglob(npz_name))
        print(f"\n[{npz_name}] found {len(found)} files")
        for npz_path in found:
            convert(npz_path, array_key=key, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
