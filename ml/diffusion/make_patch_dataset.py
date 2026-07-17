#!/usr/bin/env python3
"""Build a flat-patch (low, high) dataset from paired HEALPix shells.

DELIBERATE local duplicate of unet/make_patch_dataset.py (see
feedback-decoupled-pipeline-modules memory) -- the on-disk format (low.npy/high.npy/
metadata.npy) is byte-identical between the two, so run_diffusion.sh points
--out-dir at the SAME directory unet's run already built when nside/patch-size/
n-patches/seed match, instead of paying to rebuild it. This copy exists so the
diffusion pipeline can still build its own patch dataset independently if that
directory doesn't exist (e.g. a different patch size), without reaching into unet/.

For each patch: pick a random (cosmo, run), a random shell, a random center
pixel (uniform on the sphere via a random HEALPix index), and a random sky
orientation. Project both the low-fidelity (disco) and high-fidelity
(CosmoGrid) maps at that same pointing/orientation with a gnomonic (flat-sky)
projection, at the map's native pixel resolution — so a patch_size x patch_size
image is a like-for-like crop with no re-sampling artifacts introduced beyond
the projection itself.

Storage (memmappable, no pandas/pyarrow dependency):
    low.npy       (N, patch_size, patch_size) float32
    high.npy      (N, patch_size, patch_size) float32
    metadata.npy  (N,) structured array - enough info to exactly recreate
                  each patch (cosmo, run, shell, center pixel, orientation),
                  plus the varied cosmological parameters for that cosmo
                  (omega_m, omega_b, ns, sigma8, w0, h) from CosmoGridV1_metainfo.h5.

Recreate patch i:
    row = metadata[i]
    proj = hp.projector.GnomonicProj(rot=(row["center_lon_deg"], row["center_lat_deg"], row["psi_deg"]),
                                      xsize=row["xsize"], ysize=row["xsize"], reso=row["reso_arcmin"])
    patch = proj.projmap(shell_map, lambda x, y, z: hp.vec2pix(row["nside_source"], x, y, z, nest=False))

Note: source shells are RING-ordered (verified empirically), so nest=False
throughout - pix2ang/vec2pix must use the map's actual storage order or the
gnomonic projection samples the wrong pixel for every point except the exact
center (pix2ang/vec2pix are exact inverses of each other regardless of which
convention is passed, so a nest/ring mismatch only self-cancels at dead center).

Usage:
    python make_patch_dataset.py --data-dir /capstor/scratch/.../cosmogridv1 \
        --out-dir $sdir/sphereflow/patches/nside512_256 \
        --nside 512 --patch-size 256 --n-patches 10000 --seed 0
"""
from __future__ import annotations
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import healpy as hp
import h5py

META_DTYPE = np.dtype([
    ("idx", "i8"),
    ("cosmo", "U16"),
    ("run", "U16"),
    ("shell_idx", "i4"),
    ("lower_z", "f4"),
    ("upper_z", "f4"),
    ("shell_com", "f4"),
    ("center_ipix", "i8"),        # healpix index (RING, nside_source) of patch center
    ("center_lon_deg", "f8"),
    ("center_lat_deg", "f8"),
    ("psi_deg", "f8"),
    ("reso_arcmin", "f8"),
    ("xsize", "i4"),
    ("nside_source", "i4"),
    # the varied cosmological parameters (CosmoGridV1 fixes everything else,
    # e.g. As, m_nu, wa - not worth storing since they're constant across the grid)
    ("omega_m", "f4"),
    ("omega_b", "f4"),
    ("ns", "f4"),
    ("sigma8", "f4"),
    ("w0", "f4"),
    ("h", "f4"),
])

_COSMO_RE = re.compile(r"(cosmo_\d+)/?$")


def load_cosmo_params(metainfo_dir: Path):
    """cosmo name -> (Om, Ob, ns, s8, w0, h) from CosmoGridV1_metainfo.h5.
    Covers grid + fiducial + benchmark cosmologies (the 'all' dataset) --
    metainfo_dir is the ONE fixed location of this catalog (it is not replicated
    under every --data-dir that merely holds a subset of cosmology directories,
    e.g. /capstor/scratch/cscs/damrein/grid has none -- see --metainfo-dir)."""
    with h5py.File(metainfo_dir / "CosmoGridV1_metainfo.h5", "r") as f:
        rows = f["parameters/all"][:]
    out = {}
    for row in rows:
        path = row["path_par"].decode() if isinstance(row["path_par"], bytes) else row["path_par"]
        m = _COSMO_RE.search(path)
        if not m:
            continue
        out[m.group(1)] = (row["Om"], row["Ob"], row["ns"], row["s8"],
                            row["w0"], row["H0"] / 100.0)
    return out


def find_ready_runs(prepared_dir: Path, nside: int):
    """Runs with both low/high .npy already prepared at this nside (see prepare_maps.py
    --out-dir). Returns (cosmo, run, prepared_run_dir) - shell_info metadata is
    looked up separately from the original --data-dir (source npz, read-only)."""
    runs = []
    for c in sorted(prepared_dir.iterdir()):
        if not c.is_dir() or not c.name.startswith("cosmo_"):
            continue
        for r in sorted(c.iterdir()):
            if not r.is_dir() or not r.name.startswith("run_"):
                continue
            lo_p = r / f"low_shells_nside={nside}.npy"
            hi_p = r / f"high_shells_nside={nside}.npy"
            if lo_p.exists() and hi_p.exists():
                runs.append((c.name, r.name, r))
    return runs


def load_shell_info(source_run_dir: Path):
    """shell_info (z-bounds etc, 69 records) lives in the original npz, not the
    prepped .npy stacks - cheap to pull since npz stores it as a separate small
    array, independent of the large 'shells' array."""
    with np.load(source_run_dir / "compressed_shells.npz", allow_pickle=False) as f:
        return f["shell_info"]


def _process_run_shard(run_dir_str, cosmo, run_name, indices, shell_idxs, center_ipixs,
                        psis, nside, patch_size, reso_arcmin, data_dir_str,
                        low_path_str, high_path_str, cosmo_p):
    """Worker: compute every patch assigned to one run, write directly into the
    shared output .npy memmaps at each patch's global index (disjoint writes
    across workers - safe without locking), return only the small metadata rows.
    Runs entirely in a separate process so N runs can be processed at once."""
    run_dir = Path(run_dir_str)
    lo_mm = np.load(run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
    hi_mm = np.load(run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
    shell_info = load_shell_info(Path(data_dir_str) / cosmo / run_name)

    low_out = np.lib.format.open_memmap(low_path_str, mode="r+")
    high_out = np.lib.format.open_memmap(high_path_str, mode="r+")

    vec2pix = lambda x, y, z, _ns=nside: hp.vec2pix(_ns, x, y, z, nest=False)
    om, ob, ns, s8, w0, h = cosmo_p
    rows = []

    for idx, shell_idx, center_ipix, psi in zip(indices, shell_idxs, center_ipixs, psis):
        lon, lat = hp.pix2ang(nside, int(center_ipix), nest=False, lonlat=True)
        proj = hp.projector.GnomonicProj(rot=(lon, lat, psi), xsize=patch_size,
                                          ysize=patch_size, reso=reso_arcmin)
        low_out[idx] = proj.projmap(lo_mm[shell_idx], vec2pix).astype(np.float32)
        high_out[idx] = proj.projmap(hi_mm[shell_idx], vec2pix).astype(np.float32)

        info = shell_info[shell_idx]
        rows.append((idx, cosmo, run_name, shell_idx,
                     info["lower_z"], info["upper_z"], info["shell_com"],
                     int(center_ipix), lon, lat, psi, reso_arcmin, patch_size, nside,
                     om, ob, ns, s8, w0, h))

    low_out.flush()
    high_out.flush()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="original source data (read-only) - used only to read shell_info "
                        "z-bounds from compressed_shells.npz")
    p.add_argument("--metainfo-dir", default=None,
                   help="dir holding CosmoGridV1_metainfo.h5 (the cosmological-"
                        "parameter catalog for the WHOLE CosmoGridV1 grid, incl. "
                        "cosmologies outside --data-dir's own subset). Defaults to "
                        "--data-dir (the original cosmogridv1 layout has it there); "
                        "must be set explicitly when --data-dir is a subset dir that "
                        "doesn't carry its own copy (e.g. .../grid).")
    p.add_argument("--prepared-dir", required=True,
                   help="mirrored output of prepare_maps.py --out-dir (the low/high .npy stacks)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--nside", type=int, default=512,
                   help="nside of the prepared low/high .npy stacks to sample from")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--n-patches", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1,
                   help="parallel worker processes, one run in flight per worker "
                        "(this is CPU/IO-bound HEALPix work - a GPU node's many CPU "
                        "cores and huge page cache are what help, not the GPU itself)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    prepared_dir = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_ready_runs(prepared_dir, args.nside)
    if not runs:
        raise SystemExit(f"no runs with nside={args.nside} .npy pairs found under {prepared_dir}")
    print(f"[make_patch_dataset] {len(runs)} runs available at nside={args.nside}")

    metainfo_dir = Path(args.metainfo_dir) if args.metainfo_dir else data_dir
    cosmo_params = load_cosmo_params(metainfo_dir)
    missing = sorted({cosmo for cosmo, _, _ in runs if cosmo not in cosmo_params})
    if missing:
        raise SystemExit(f"cosmo params missing from metainfo.h5 for: {missing}")

    rng = np.random.default_rng(args.seed)
    reso_arcmin = hp.nside2resol(args.nside, arcmin=True)
    npix = hp.nside2npix(args.nside)
    n_shells = np.load(runs[0][2] / f"low_shells_nside={args.nside}.npy", mmap_mode="r").shape[0]

    n = args.n_patches
    ps = args.patch_size

    # Draw every patch's (run, shell, center, psi) up front - deterministic given
    # --seed, independent of how work gets split across workers afterwards.
    run_idx = rng.integers(0, len(runs), size=n)
    shell_idx = rng.integers(0, n_shells, size=n)
    center_ipix = rng.integers(0, npix, size=n)
    psi = rng.uniform(0.0, 360.0, size=n)

    low_path = out_dir / "low.npy"
    high_path = out_dir / "high.npy"
    np.lib.format.open_memmap(low_path, mode="w+", dtype=np.float32, shape=(n, ps, ps))
    np.lib.format.open_memmap(high_path, mode="w+", dtype=np.float32, shape=(n, ps, ps))

    meta = np.empty(n, dtype=META_DTYPE)
    done = 0

    # One task per run only gives <=len(runs) parallelism. Split each run's
    # indices into smaller sub-chunks so the number of shards scales with
    # --num-workers instead of being capped at 44 - multiple sub-shards of the
    # same run just open independent read-only mmap views onto the same file,
    # which is fine.
    chunks_per_run = max(1, -(-args.num_workers // len(runs)))  # ceil division
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        for r in range(len(runs)):
            indices = np.where(run_idx == r)[0]
            if len(indices) == 0:
                continue
            cosmo, run_name, run_dir = runs[r]
            for sub in np.array_split(indices, min(chunks_per_run, len(indices))):
                if len(sub) == 0:
                    continue
                futures.append(ex.submit(
                    _process_run_shard, str(run_dir), cosmo, run_name, sub,
                    shell_idx[sub], center_ipix[sub], psi[sub],
                    args.nside, ps, reso_arcmin, str(data_dir),
                    str(low_path), str(high_path), cosmo_params[cosmo],
                ))

        for f in as_completed(futures):
            rows = f.result()
            for row in rows:
                meta[row[0]] = row
            done += len(rows)
            print(f"[make_patch_dataset] {done}/{n} patches "
                  f"({len(futures)} run-shards total)", flush=True)

    np.save(out_dir / "metadata.npy", meta)
    print(f"[make_patch_dataset] saved {n} patches to {out_dir}")


if __name__ == "__main__":
    main()
