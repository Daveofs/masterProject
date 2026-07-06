#!/usr/bin/env python3
"""Standalone map preprocessing for the single-model sphere-flow (map-only).

Decompresses each run's DISCO low shells and CosmoGrid high shells from their
.npz archives into raw, uncompressed .npy stacks:

    low_shells_nside={nside}.npy   <- disco_sim/*/disco_shells_nside=2048.npz
    high_shells_nside={nside}.npy  <- compressed_shells.npz

Raw .npy is memory-mappable, so training random-accesses one shell at a time
(~200 MB read) instead of decompressing a 14 GB .npz per access — the main I/O
win. Zero dependency on the transfer function / spherical harmonics: this works
purely on maps. CPU-only, parallel over runs.

  python prepare_maps.py --data-dir /capstor/scratch/.../cosmogridv1 --nside 2048
"""

from __future__ import annotations
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import healpy as hp


def _to_nside(stack, nside):
    if hp.npix2nside(stack.shape[1]) == nside:
        return stack.astype(np.float32)
    return np.stack([hp.ud_grade(m.astype(np.float32), nside,
                                 order_in="NESTED", order_out="NESTED")
                     for m in stack]).astype(np.float32)


def _is_valid_npy(path: Path) -> bool:
    """True if path is a complete, readable .npy (catches truncated leftovers
    from before _atomic_save existed, or any other partial-write case)."""
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        _ = arr.shape  # touch metadata; mmap raises ValueError if file is short
        return True
    except Exception:
        return False


def _atomic_save(out_path: Path, arr: np.ndarray):
    """np.save to a temp file then os.rename into place.

    np.save is NOT atomic: if the job is killed/times out mid-write, a partial
    file is left at the final name. Since process_run's "already done" check is
    just out_path.exists(), a truncated file then poisons every future run
    (silently skipped as done -> ValueError: mmap length is greater than file
    size at training time). os.rename on the same filesystem is atomic, so
    readers only ever see a complete file at the final path.
    """
    # NOTE: np.save silently APPENDS .npy to any filename not already ending in
    # .npy — so the temp name must itself end in .npy, or np.save writes to a
    # different path than the one we then try to os.replace() from.
    tmp_path = out_path.with_name(out_path.stem + f".tmp{os.getpid()}.npy")
    np.save(tmp_path, arr)
    os.replace(tmp_path, out_path)


def process_run(task):
    ld, nside, low_glob, high_npz = task
    ld = Path(ld)
    out_low = ld / f"low_shells_nside={nside}.npy"
    out_high = ld / f"high_shells_nside={nside}.npy"
    msgs = []
    try:
        if not _is_valid_npy(out_low):
            if out_low.exists():
                msgs.append("low: found TRUNCATED, reprocessing")
            hits = glob.glob(str(ld / low_glob))
            if not hits:
                return f"[skip] {ld.name}: no DISCO shells ({low_glob})"
            # Potentially several gpu_grid_* runs exist (e.g. restarts/reruns)
            # -> Pick most recent
            latest = max(hits, key=lambda h: Path(h).stat().st_mtime)
            lo = np.load(latest, allow_pickle=False)["shells"]
            _atomic_save(out_low, _to_nside(lo, nside))
            msgs.append(f"low({lo.shape[0]})")
        if not _is_valid_npy(out_high):
            if out_high.exists():
                msgs.append("high: found TRUNCATED, reprocessing")
            hi = np.load(ld / high_npz, allow_pickle=False)["shells"]
            _atomic_save(out_high, _to_nside(hi, nside))
            msgs.append(f"high({hi.shape[0]})")
        return f"[ok] {ld}: {', '.join(msgs) if msgs else 'already prepared'}"
    except Exception as e:
        return f"[ERROR] {ld}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--low-glob", default="disco_sim/*/disco_shells_nside=2048.npz")
    p.add_argument("--high-npz", default="compressed_shells.npz")
    p.add_argument("--num-workers", type=int, default=5)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    cosmos = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_")) or [data_dir]
    tasks = []
    for c in cosmos:
        runs = [r for r in sorted(c.iterdir())
                if r.is_dir() and r.name.startswith("run_")] or [c]
        for ld in runs:
            if not (ld / args.high_npz).exists():
                continue
            if not glob.glob(str(ld / args.low_glob)):
                continue
            tasks.append((str(ld), args.nside, args.low_glob, args.high_npz))
    print(f"[prepare_maps] {len(tasks)} runs to process", flush=True)

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for f in as_completed([ex.submit(process_run, t) for t in tasks]):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
