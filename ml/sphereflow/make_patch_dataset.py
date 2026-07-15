#!/usr/bin/env python3
"""Build a (low, high) HEALPix-superpixel patch dataset for the sphere-flow.

Direct analogue of unet/make_patch_dataset.py, but the "patch" here is a
NESTED HEALPix superpixel (the unit sphere_flow's Chebyshev graph convolution
operates on), NOT a gnomonic flat image: the sphere is split into
12*order^2 contiguous NESTED superpixels of npix/(12*order^2) pixels each
(order=16, nside=2048 -> 3072 patches of 16,384 pixels), exactly the split
sphere_flow.map_to_patches does, so the graph Laplacian built on one patch
applies to every patch.

ORDERING (critical, fixed 2026-07-15): the source shell .npy stacks are stored
RING-ordered (verified by a controlled sphere-neighbour-correlation test). A
contiguous slice of a RING array is a LATITUDE ANNULUS (up to ~89 deg wide), NOT
a compact superpixel -- so we MUST reorder each shell RING->NESTED
(hp.reorder(r2n=True)) BEFORE slicing, or the "patches" don't match the nest=True
graph Laplacian and the conv mixes wrong neighbours. The stored patches are
therefore NESTED superpixels; apply_sphere_flow.correct_shell does the same
reorder on its input and the inverse (NESTED->RING) on its output.

WHY this exists (2026-07-14): training used to stream shells straight off the
14 GB per-run .npy stacks through a hand-rolled per-rank producer thread. That
had two structural problems the unet pipeline simply doesn't have:
  * every rank ran its OWN independent infinite data stream, so ranks could
    (and did) drift out of lockstep -- one rank hitting a bad shell/step while
    the others advanced is what turned into a hung NCCL collective.
  * random per-shell mmap access into huge files hammered Lustre.
Materializing a compact patch dataset up front turns training reads into cheap
contiguous memmap slices and lets a standard DistributedSampler guarantee every
rank sees exactly the same number of batches per epoch.

Storage (memmappable):
    low.npy         (N, npix_patch) float32   raw counts
    high.npy        (N, npix_patch) float32   raw counts
    cosmo.npy       (N, P)          float32   the run's params.yml numeric vector
    metadata.npy    (N,)            structured (see META_DTYPE)

Raw COUNTS are stored, not the arcsinh signal the model trains on -- the
transform is applied batched on GPU in the training loop (see
dataset.raw_to_signal_pair), same reason unet/dataset.py gives: doing it
per-sample in NumPy inside dataloader workers is data-loading-bound.

The per-shell overdensity normalization delta = rho/mean - 1 uses the mean of
the WHOLE shell (all 12*nside^2 pixels), not the patch's own mean -- that is
what sphere_flow.to_overdensity does at training time and what
apply_sphere_flow.correct_shell does at inference. So each patch's row records
its shell's global low/high means (low_shell_mean/high_shell_mean); computing a
mean from the stored patch alone would be a DIFFERENT (and wrong) normalization.

  python make_patch_dataset.py --data-dir <grid> --out-dir <patches> \\
      --nside 2048 --order 16 --n-patches 200000 --seed 0 --num-workers 32
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import healpy as hp

META_DTYPE = np.dtype([
    ("idx", "i8"),
    ("cosmo", "U16"),
    ("run", "U16"),
    ("shell_idx", "i4"),
    ("patch_idx", "i4"),          # which of the 12*order^2 NESTED superpixels
    ("n_shells", "i4"),           # that run's shell count -> normalized shell index
    ("low_shell_mean", "f8"),     # mean over the WHOLE shell (see module docstring)
    ("high_shell_mean", "f8"),
    ("lower_z", "f4"),
    ("upper_z", "f4"),
    ("nside", "i4"),
    ("order", "i4"),
])


def _is_num(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False


def cosmo_vector(params_yml: Path) -> np.ndarray:
    """The run's cosmology vector = every NUMERIC key of params.yml, sorted by
    key name. Same convention train_sphere_flow.py has always used and that
    apply_sphere_flow.cosmo_vector reproduces at inference -- keep the two in
    step or a trained model will be fed a permuted cosmology at apply time."""
    import yaml
    p = yaml.safe_load(Path(params_yml).read_text())
    keys = sorted(k for k, v in p.items() if _is_num(v))
    return np.array([float(p[k]) for k in keys], dtype=np.float32)


def find_ready_runs(data_dir: Path, nside: int):
    """Runs with both low/high .npy stacks prepared at this nside (prepare_maps.py)."""
    runs = []
    for c in sorted(data_dir.iterdir()):
        if not c.is_dir() or not c.name.startswith("cosmo_"):
            continue
        for r in sorted(c.iterdir()):
            if not r.is_dir() or not r.name.startswith("run_"):
                continue
            if ((r / f"low_shells_nside={nside}.npy").exists()
                    and (r / f"high_shells_nside={nside}.npy").exists()):
                runs.append((c.name, r.name, r))
    return runs


def load_shell_info(run_dir: Path):
    """shell_info (z bounds) from the ORIGINAL npz -- cheap (small separate array)."""
    try:
        with np.load(run_dir / "compressed_shells.npz", allow_pickle=False) as f:
            return f["shell_info"]
    except Exception:
        return None


def _process_run_shard(run_dir_str, cosmo, run_name, indices, shell_idxs, patch_idxs,
                       nside, order, n_patches_per_shell, npix_patch,
                       low_path_str, high_path_str):
    """Worker: fill every patch assigned to ONE run, writing directly into the
    shared output memmaps at each patch's global index (disjoint writes across
    workers -- safe without locking). Returns only the small metadata rows.

    Shells are grouped so each shell's 200 MB row is read from the big stack
    ONCE, no matter how many patches were drawn from it.
    """
    run_dir = Path(run_dir_str)
    lo_mm = np.load(run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
    hi_mm = np.load(run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
    shell_info = load_shell_info(run_dir)
    n_shells = int(min(lo_mm.shape[0], hi_mm.shape[0]))

    low_out = np.lib.format.open_memmap(low_path_str, mode="r+")
    high_out = np.lib.format.open_memmap(high_path_str, mode="r+")

    rows = []
    for s in np.unique(shell_idxs):
        s = int(s)
        sel = np.where(shell_idxs == s)[0]
        # ONE read per shell, reordered RING->NESTED so the contiguous slices below
        # are compact superpixels matching sphere_flow's nest=True Laplacian (the
        # .npy are RING; a raw RING slice would be a latitude annulus -- see the
        # module docstring). reorder is order-invariant for the shell means below.
        lo_shell = hp.reorder(np.asarray(lo_mm[s], dtype=np.float32), r2n=True)
        hi_shell = hp.reorder(np.asarray(hi_mm[s], dtype=np.float32), r2n=True)
        # Shell-GLOBAL means -- the normalization the model is trained/applied
        # with (see module docstring). Guard a degenerate all-zero shell.
        lo_mean = float(lo_shell.mean()) or 1.0
        hi_mean = float(hi_shell.mean()) or 1.0
        lz, uz = (float(shell_info[s]["lower_z"]), float(shell_info[s]["upper_z"])) \
            if shell_info is not None else (0.0, 0.0)

        for j in sel:
            gi = int(indices[j])
            p = int(patch_idxs[j])
            sl = slice(p * npix_patch, (p + 1) * npix_patch)
            low_out[gi] = lo_shell[sl]
            high_out[gi] = hi_shell[sl]
            rows.append((gi, cosmo, run_name, s, p, n_shells,
                         lo_mean, hi_mean, lz, uz, nside, order))

    low_out.flush()
    high_out.flush()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="grid root holding cosmo_*/run_*/{low,high}_shells_nside=*.npy "
                        "(prepare_maps.py output) + params.yml + compressed_shells.npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--order", type=int, default=16,
                   help="12*order^2 NESTED superpixels per sphere -- MUST match the "
                        "--order the model is trained/applied with (the graph Laplacian "
                        "is built for exactly this patch size).")
    p.add_argument("--n-patches", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_ready_runs(data_dir, args.nside)
    if not runs:
        raise SystemExit(f"no runs with nside={args.nside} low/high .npy pairs under "
                         f"{data_dir} -- run preprocess/prepare_maps.py first")

    n_patches_per_shell = 12 * args.order * args.order
    npix = hp.nside2npix(args.nside)
    if npix % n_patches_per_shell:
        raise SystemExit(f"nside={args.nside} npix={npix} not divisible by "
                         f"12*order^2={n_patches_per_shell}")
    npix_patch = npix // n_patches_per_shell

    n_shells = int(np.load(runs[0][2] / f"low_shells_nside={args.nside}.npy",
                           mmap_mode="r").shape[0])
    print(f"[make_patch_dataset] {len(runs)} runs | nside={args.nside} order={args.order} "
          f"-> {n_patches_per_shell} patches/shell x {npix_patch} px | ~{n_shells} shells/run",
          flush=True)

    # Draw every patch's (run, shell, patch) up front -- deterministic given --seed,
    # independent of how the work is later split across workers.
    rng = np.random.default_rng(args.seed)
    n = args.n_patches
    run_idx = rng.integers(0, len(runs), size=n)
    shell_idx = rng.integers(0, n_shells, size=n)
    patch_idx = rng.integers(0, n_patches_per_shell, size=n)

    low_path = out_dir / "low.npy"
    high_path = out_dir / "high.npy"
    np.lib.format.open_memmap(low_path, mode="w+", dtype=np.float32, shape=(n, npix_patch))
    np.lib.format.open_memmap(high_path, mode="w+", dtype=np.float32, shape=(n, npix_patch))
    gb = 2 * n * npix_patch * 4 / 1e9
    print(f"[make_patch_dataset] allocating {n:,} patches -> {gb:.1f} GB "
          f"(low.npy + high.npy)", flush=True)

    # Cosmology vector per RUN (not per patch) -- fanned out to per-patch rows below.
    cvecs = {}
    for cosmo, run_name, run_dir in runs:
        pf = run_dir / "params.yml"
        cvecs[(cosmo, run_name)] = cosmo_vector(pf) if pf.exists() else np.zeros(1, np.float32)
    P = max(len(v) for v in cvecs.values())
    cosmo_arr = np.zeros((n, P), dtype=np.float32)

    meta = np.empty(n, dtype=META_DTYPE)
    done = 0
    chunks_per_run = max(1, -(-args.num_workers // len(runs)))
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        for r in range(len(runs)):
            idxs = np.where(run_idx == r)[0]
            if len(idxs) == 0:
                continue
            cosmo, run_name, run_dir = runs[r]
            v = cvecs[(cosmo, run_name)]
            cosmo_arr[idxs, :len(v)] = v          # same vector for every patch of this run
            for sub in np.array_split(idxs, min(chunks_per_run, len(idxs))):
                if len(sub) == 0:
                    continue
                futures.append(ex.submit(
                    _process_run_shard, str(run_dir), cosmo, run_name, sub,
                    shell_idx[sub], patch_idx[sub], args.nside, args.order,
                    n_patches_per_shell, npix_patch, str(low_path), str(high_path)))

        for f in as_completed(futures):
            rows = f.result()
            for row in rows:
                meta[row[0]] = row
            done += len(rows)
            print(f"[make_patch_dataset] {done:,}/{n:,} patches "
                  f"({len(futures)} run-shards)", flush=True)

    np.save(out_dir / "metadata.npy", meta)
    np.save(out_dir / "cosmo.npy", cosmo_arr)
    print(f"[make_patch_dataset] saved {n:,} patches -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
