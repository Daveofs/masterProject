#!/usr/bin/env python3
"""Build a (low, high) HEALPix-superpixel patch dataset for the sphere-flow.

Direct analogue of unet/make_patch_dataset.py, but the "patch" here is a
ROTATED HEALPix superpixel (the shape sphere_flow's Chebyshev graph
convolution operates on), NOT a gnomonic flat image.

OVERLAP-CAPABLE PATCHES (2026-07-20, replaces the old disjoint quad-tree
scheme): each patch is drawn at a RANDOM (lon, lat, psi) -- a uniformly random
sky center + random in-plane rotation, exactly the SAME draw convention
unet/make_patch_dataset.py already uses for its gnomonic patches -- via
sphere_flow.rotated_patch_ids, which rotates canonical patch 0's own npix_patch
NESTED pixels onto that (lon, lat, psi). This is what lets
apply_sphere_flow.py reconstruct a shell from OVERLAPPING, taper-blended
patches at inference (analogous to analysis/patch_tiling.py's gnomonic overlap
for unet/diffusion) instead of the old hard, disjoint 12*order^2-block
partition: a model trained ONLY on that one fixed alignment has no reason to
behave sanely on an arbitrarily-rotated patch boundary, so training must see
the same distribution of rotations reconstruction will use. See sphere_flow.py's
"OVERLAPPING patch geometry" section for the validated rotation math.

The graph Laplacian is UNCHANGED (still built once, for canonical patch 0's
own npix_patch NESTED pixels) and applies to every rotated patch exactly as
before -- rotated_patch_ids always returns pixels in the SAME canonical
relative order, so the model architecture needed no changes, only what patches
it is trained/applied on.

ORDERING: rotated_patch_ids returns NESTED ids on the true sky; converted to
RING ids (hp.nest2ring) and gathered DIRECTLY from the shell in its native RING
order (the .npy stacks are stored RING) -- no whole-shell hp.reorder needed any
more (that was the old scheme's per-shell-group R2N reorder before slicing a
disjoint block; a rotated gather doesn't need it).

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
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sphere_flow as sf  # noqa: E402

META_DTYPE = np.dtype([
    ("idx", "i8"),
    ("cosmo", "U16"),
    ("run", "U16"),
    ("shell_idx", "i4"),
    ("center_ipix", "i8"),        # healpix index (RING, --nside) the patch was drawn at
    ("center_lon_deg", "f8"),
    ("center_lat_deg", "f8"),
    ("psi_deg", "f8"),
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


def _process_run_shard(run_dir_str, cosmo, run_name, indices, shell_idxs,
                       center_ipixs, lons, lats, psis,
                       nside, order, npix_patch,
                       low_path_str, high_path_str):
    """Worker: fill every patch assigned to ONE run, writing directly into the
    shared output memmaps at each patch's global index (disjoint writes across
    workers -- safe without locking). Returns only the small metadata rows.

    Shells are grouped so each shell's 200 MB row is read from the big stack
    ONCE, no matter how many patches were drawn from it.

    OVERLAP-CAPABLE PATCHES (2026-07-20): each patch is a rotated copy of
    canonical patch 0 (sphere_flow.rotated_patch_ids), centered at a RANDOM
    (lon, lat, psi) -- NOT one of the n_patches(order) fixed disjoint quad-tree
    blocks the old version sliced out. This is what lets apply_sphere_flow.py
    reconstruct with overlapping, taper-blended patches at inference (a model
    that only ever saw the disjoint alignment during training has no reason to
    behave sanely on an arbitrarily-rotated patch's boundary at apply time) --
    see sphere_flow.py's "OVERLAPPING patch geometry" section for the full
    rationale and the validated rotation math.

    Also FASTER than the old version: rotated_patch_ids returns NESTED ids on
    the true sky, converted to RING ids and gathered DIRECTLY from the shell in
    its native RING order -- no whole-shell hp.reorder(r2n=True) needed per
    shell group any more (that was an O(npix) call per shell; this is an
    O(npix_patch) gather per patch).
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
        # ONE read per shell, RING-ordered as stored (no reorder needed -- see
        # docstring above).
        lo_shell = np.asarray(lo_mm[s], dtype=np.float32)
        hi_shell = np.asarray(hi_mm[s], dtype=np.float32)
        # Shell-GLOBAL means -- the normalization the model is trained/applied
        # with (see module docstring). Guard a degenerate all-zero shell.
        lo_mean = float(lo_shell.mean()) or 1.0
        hi_mean = float(hi_shell.mean()) or 1.0
        lz, uz = (float(shell_info[s]["lower_z"]), float(shell_info[s]["upper_z"])) \
            if shell_info is not None else (0.0, 0.0)

        for j in sel:
            gi = int(indices[j])
            nested_ids = sf.rotated_patch_ids(nside, order, float(lons[j]), float(lats[j]),
                                              float(psis[j]))
            ring_ids = hp.nest2ring(nside, nested_ids)
            low_out[gi] = lo_shell[ring_ids]
            high_out[gi] = hi_shell[ring_ids]
            rows.append((gi, cosmo, run_name, s, int(center_ipixs[j]),
                        float(lons[j]), float(lats[j]), float(psis[j]), n_shells,
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

    n_patches_per_shell = 12 * args.order * args.order   # informational only now
    npix = hp.nside2npix(args.nside)
    if npix % n_patches_per_shell:
        raise SystemExit(f"nside={args.nside} npix={npix} not divisible by "
                         f"12*order^2={n_patches_per_shell}")
    npix_patch = npix // n_patches_per_shell

    n_shells = int(np.load(runs[0][2] / f"low_shells_nside={args.nside}.npy",
                           mmap_mode="r").shape[0])
    print(f"[make_patch_dataset] {len(runs)} runs | nside={args.nside} order={args.order} "
          f"-> {npix_patch} px/patch, RANDOM (lon,lat,psi) centers | ~{n_shells} shells/run",
          flush=True)

    # Draw every patch's (run, shell, center, psi) up front -- deterministic given
    # --seed, independent of how the work is later split across workers. center_ipix
    # (a uniformly random HEALPix pixel index -> uniform on the sphere via pix2ang)
    # + psi is the SAME draw convention unet/make_patch_dataset.py already uses for
    # its gnomonic patches -- kept identical here so the two pipelines' "random
    # patch" conventions read the same way, even though the patch SHAPE differs
    # (rotated HEALPix superpixel vs gnomonic projection).
    rng = np.random.default_rng(args.seed)
    n = args.n_patches
    run_idx = rng.integers(0, len(runs), size=n)
    shell_idx = rng.integers(0, n_shells, size=n)
    center_ipix = rng.integers(0, npix, size=n)
    center_lon, center_lat = hp.pix2ang(args.nside, center_ipix, nest=False, lonlat=True)
    psi = rng.uniform(0.0, 360.0, size=n)

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
                    shell_idx[sub], center_ipix[sub], center_lon[sub], center_lat[sub],
                    psi[sub], args.nside, args.order, npix_patch,
                    str(low_path), str(high_path)))

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
