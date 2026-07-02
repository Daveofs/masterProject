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


def process_run(task):
    ld, nside, low_glob, high_npz = task
    ld = Path(ld)
    out_low = ld / f"low_shells_nside={nside}.npy"
    out_high = ld / f"high_shells_nside={nside}.npy"
    msgs = []
    try:
        if not out_low.exists():
            hits = sorted(glob.glob(str(ld / low_glob)))
            if not hits:
                return f"[skip] {ld.name}: no DISCO shells ({low_glob})"
            lo = np.load(hits[0], allow_pickle=False)["shells"]
            np.save(out_low, _to_nside(lo, nside))
            msgs.append(f"low({lo.shape[0]})")
        if not out_high.exists():
            hi = np.load(ld / high_npz, allow_pickle=False)["shells"]
            np.save(out_high, _to_nside(hi, nside))
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
