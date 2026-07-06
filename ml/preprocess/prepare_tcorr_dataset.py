#!/usr/bin/env python3
"""Prepare the residual-training dataset: T-corrected DISCO maps + high-res maps.

For each run that has preprocessed low alms (low_alms_lmax{lmax}.npy) this script
1. applies the per-ell transfer function:  corr_alm = low_alm * T(ell, shell)
2. synthesizes the T-corrected map stack -> tcorr_shells_nside={nside}.npy (raw)
3. decompresses the CosmoGrid target once -> high_shells_nside={nside}.npy (raw)

Raw .npy files are mmap-able: training can random-access single shells (~200 MB
reads) instead of decompressing 14 GB .npz archives — the main I/O win.

The sphere-flow model then trains on the RESIDUAL high - tcorr: the transfer
function guarantees the Cl (first order); the flow learns only the non-Gaussian /
stochastic remainder. CPU-only; parallel over runs.

  python prepare_tcorr_dataset.py --data-dir /capstor/scratch/.../cosmogridv1 \
      --transfer transfer.npz --lmax 3000 --nside 2048
"""

from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import healpy as hp

# transfer_function.py lives in the sibling ../transfer/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transfer"))
from transfer_function import ell_of_flat_index


def _is_valid_npy(path: Path) -> bool:
    """True if path is a complete, readable .npy (catches truncated files left
    by a killed/timed-out job — a plain .exists() check would treat those as
    'already done' forever and poison every future run)."""
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        _ = arr.shape
        return True
    except Exception:
        return False


def process_run(task):
    ld, transfer_path, lmax, nside, high_npz = task
    ld = Path(ld)
    out_tcorr = ld / f"tcorr_shells_nside={nside}.npy"
    out_high = ld / f"high_shells_nside={nside}.npy"
    N_alm = (lmax + 1) * (lmax + 2) // 2

    msgs = []
    try:
        # ---- T-corrected DISCO maps from the preprocessed low alms ----
        if not _is_valid_npy(out_tcorr):
            if out_tcorr.exists():
                msgs.append("tcorr: found TRUNCATED, reprocessing")
            tf = np.load(transfer_path)
            T = tf["T"]                                   # (n_shells, lmax+1)
            ell = ell_of_flat_index(lmax)
            low = np.load(ld / f"low_alms_lmax{lmax}.npy", mmap_mode="r")
            n_shells = low.shape[0]
            npix = hp.nside2npix(nside)
            # Write to a temp path, then atomically rename into place: open_memmap
            # writes incrementally, so a killed job leaves a partial file at the
            # final name if written there directly.
            tmp_path = out_tcorr.with_suffix(out_tcorr.suffix + f".tmp{os.getpid()}")
            out = np.lib.format.open_memmap(tmp_path, mode="w+",
                                            dtype=np.float32, shape=(n_shells, npix))
            for i in range(n_shells):
                tvec = T[min(i, T.shape[0] - 1)][ell]     # per-mode scale
                v = np.asarray(low[i], dtype=np.float64)
                alm = v[:N_alm] * tvec + 1j * v[N_alm:] * tvec
                out[i] = hp.alm2map(alm, nside=nside, lmax=lmax).astype(np.float32)
            out.flush()
            del out
            os.replace(tmp_path, out_tcorr)
            msgs.append(f"tcorr({n_shells} shells)")
        # ---- raw high-res target (decompress the .npz once) ----
        if not _is_valid_npy(out_high):
            if out_high.exists():
                msgs.append("high: found TRUNCATED, reprocessing")
            hi = np.load(ld / high_npz, allow_pickle=False)["shells"]
            if hp.npix2nside(hi.shape[1]) != nside:
                hi = np.stack([hp.ud_grade(m.astype(np.float32), nside,
                                           order_in="NESTED", order_out="NESTED")
                               for m in hi])
            # np.save auto-appends .npy to any name not already ending in .npy —
            # the temp name must end in .npy itself (open_memmap above does NOT
            # have this quirk, only np.save).
            tmp_path = out_high.with_name(out_high.stem + f".tmp{os.getpid()}.npy")
            np.save(tmp_path, hi.astype(np.float32))
            os.replace(tmp_path, out_high)
            msgs.append(f"high({hi.shape[0]} shells)")
        return f"[OK] {ld}: {', '.join(msgs) if msgs else 'already prepared'}"
    except Exception as e:
        return f"[ERROR] {ld}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--transfer", required=True, help="transfer.npz from transfer_function.py fit")
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--nside", type=int, default=2048)
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
            if (ld / f"low_alms_lmax{args.lmax}.npy").exists() and \
               (ld / args.high_npz).exists():
                tasks.append((str(ld), args.transfer, args.lmax, args.nside, args.high_npz))
    print(f"[prepare] {len(tasks)} runs to process", flush=True)

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [ex.submit(process_run, t) for t in tasks]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
