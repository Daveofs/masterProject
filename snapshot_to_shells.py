#!/usr/bin/env python3
"""
snapshot_to_shells.py
=====================
Convert a DISCO-DJ final snapshot (.npz) into CosmoGrid-style compressed shells.

WHY NOT UFalcon.shells.construct_shells() DIRECTLY?
----------------------------------------------------
construct_shells() reads raw binary files from a directory (pkdgrav `.lcp.*` or
l-picola `_lightcone.*`).  Our snapshot is a numpy .npz archive — there is no
supported file_format for it.  Instead we replicate the same internal pipeline
that construct_shells uses (xyz_to_spherical → searchsorted → thetaphi_to_pixelcounts)
but feed it our in-memory arrays directly.  Periodic tiling is also handled here
since construct_shells has no concept of it.

NOTE
----
A single z=0 snapshot is replicated periodically to fill the full lightcone out
to z=3.5.  This is an approximation — a proper lightcone would sample each shell
at its correct look-back time.  The output format exactly matches CosmoGridV1's
compressed_shells.npz (shells: uint16, shape (69, npix); shell_info: structured
array with redshift/comoving-distance metadata).

Usage
-----
    conda activate vir_env
    python snapshot_to_shells.py

Dependencies (all installed in vir_env):  UFalcon, astropy, h5py, healpy, numpy
"""

import numpy as np
import h5py
import healpy as hp
from astropy.cosmology import w0waCDM

import UFalcon.shells

# ─── PATHS ─────────────────────────────────────────────────────────────────────
SNAPSHOT_PATH = "/cluster/scratch/damrein/outputs/snapshots/final_snapshot_58356355.npz"
METAINFO_PATH = "/cluster/scratch/damrein/cosmogridv1/cosmo_000001/CosmoGridV1_metainfo.h5"
OUTPUT_PATH   = "/cluster/scratch/damrein/outputs/snapshots/compressed_shells_58356355.npz"

# ─── SIMULATION PARAMETERS ─────────────────────────────────────────────────────
BOXSIZE_MPH = 900.0   # simulation box side length in Mpc/h
NSIDE       = 2048    # HEALPix nside for output maps

# ─── COSMOLOGY (from params.yml) ────────────────────────────────────────────────
# w0 ≠ -1 → use w0waCDM (not FlatLambdaCDM)
cosmo = w0waCDM(
    H0=73.0,
    Om0=0.3,
    Ode0=0.7,      # flat universe: Ode0 = 1 - Om0
    w0=-1.1665,
    wa=0.0,
    Ob0=0.045,
)

h_dim   = cosmo.H0.value / 100.0    # h = 0.73
BOX_MPC = BOXSIZE_MPH / h_dim       # box side length in Mpc

# ─── 1. READ SHELL EDGES FROM CosmoGridV1 METAINFO ────────────────────────────
print("Reading shell info from CosmoGridV1 metainfo ...", flush=True)
with h5py.File(METAINFO_PATH, "r") as f:
    shell_data = np.array(f["shell_info/CosmoGrid/raw/grid/cosmo_000001"])  # (69,)

n_shells = len(shell_data)  # 69
lower_z  = shell_data["lower_z"].astype(np.float64)
upper_z  = shell_data["upper_z"].astype(np.float64)

# 70 bin edges [z0_lo, z1_lo, ..., z68_lo, z68_hi]
z_shells   = np.append(lower_z, upper_z[-1])
com_shells = cosmo.comoving_distance(z_shells).value  # Mpc, shape (70,)
r_max      = com_shells[-1]

print(f"  {n_shells} shells,  z ∈ [{z_shells[0]:.4f}, {z_shells[-1]:.4f}]")
print(f"  comoving range: [{com_shells[0]:.1f}, {r_max:.1f}] Mpc")

# ─── 2. LOAD PARTICLE POSITIONS ───────────────────────────────────────────────
print("\nLoading snapshot positions …", flush=True)
with np.load(SNAPSHOT_PATH) as snap:
    pos = np.asarray(snap["pos"], dtype=np.float32)  # (N, 3) in Mpc/h

# Handle collect_all=True snapshots with shape (n_steps, N, 3)
if pos.ndim == 3:
    print(f"  Multi-step snapshot — using last step (shape {pos.shape})")
    pos = pos[-1]

N = pos.shape[0]
print(f"  Loaded {N:,} particles")

# Mpc/h → Mpc, then centre observer at origin
pos = pos / h_dim - BOX_MPC / 2.0

# ─── 3. PRECOMPUTE VALID TILE OFFSETS ─────────────────────────────────────────
n_half     = int(np.ceil(r_max / BOX_MPC))
tile_range = range(-n_half, n_half + 1)

offsets = []
for ix in tile_range:
    for iy in tile_range:
        for iz in tile_range:
            off  = np.array([ix, iy, iz], dtype=np.float32) * BOX_MPC
            near = np.maximum(0.0, np.abs(off) - BOX_MPC / 2.0)
            if float(np.sqrt((near ** 2).sum())) <= r_max:
                far = np.abs(off) + BOX_MPC / 2.0
                if float(np.sqrt((far ** 2).sum())) >= com_shells[0]:
                    offsets.append(off)

print(f"\nPeriodic tiling: {len(offsets)} useful tiles "
      f"(of {len(tile_range)**3} candidates, box={BOX_MPC:.1f} Mpc)")

# ─── 4. ALLOCATE OUTPUT SHELL MAPS ─────────────────────────────────────────────
npix   = hp.nside2npix(NSIDE)
shells = np.zeros((n_shells, npix), dtype=np.int32)
print(f"  Shell array: {n_shells} × {npix:,} px  ({shells.nbytes / 1024**3:.1f} GB)\n")

# ─── 5. MAIN LOOP OVER TILES ──────────────────────────────────────────────────
# Mirrors the internals of UFalcon.shells.construct_shells():
#   xyz_to_spherical → sort by r → searchsorted into shell edges → thetaphi_to_pixelcounts
total_particles = 0

for t, offset in enumerate(offsets):
    print(f"  tile {t + 1:4d}/{len(offsets)}  offset=({offset[0]:+8.1f}, {offset[1]:+8.1f}, {offset[2]:+8.1f}) Mpc",
          end="\r", flush=True)

    # Shift entire snapshot to this tile replica
    xyz = pos + offset  # (N, 3), float32 — no copy of pos needed

    # Convert to spherical (r, theta, phi) — same call as inside construct_shells
    sph = UFalcon.shells.xyz_to_spherical(xyz)  # (N, 3)

    # Mask particles outside the lightcone radial range
    valid = (sph[:, 0] >= com_shells[0]) & (sph[:, 0] < r_max)
    if not valid.any():
        continue
    sph = sph[valid]

    # Sort by comoving radius — same step as inside construct_shells
    sph = sph[np.argsort(sph[:, 0])]

    # Binary-search shell boundaries — same step as inside construct_shells
    ind_shells = np.searchsorted(sph[:, 0], com_shells, side="left")

    # Accumulate pixel counts per shell — same step as inside construct_shells
    for i_shell in range(n_shells):
        i_low = ind_shells[i_shell]
        i_up  = ind_shells[i_shell + 1]
        if i_low >= i_up:
            continue
        counts = UFalcon.shells.thetaphi_to_pixelcounts(
            sph[i_low:i_up, 1], sph[i_low:i_up, 2], NSIDE
        )
        shells[i_shell, :counts.size] += counts

    total_particles += int(valid.sum())

print(f"\n\n  Total particle-shell assignments: {total_particles:,}")

# ─── 6. BUILD shell_info STRUCTURED ARRAY ──────────────────────────────────────
dt_info   = np.dtype([
    ("shell_name", "<U512"),
    ("shell_id",   "<i4"),
    ("lower_z",    "<f4"),
    ("upper_z",    "<f4"),
    ("lower_com",  "<f4"),
    ("upper_com",  "<f4"),
    ("shell_com",  "<f4"),
])
shell_info = np.empty(n_shells, dtype=dt_info)
for i in range(n_shells):
    shell_info[i]["shell_name"] = f"cosmo_000001-shell_{i:03d}"
    shell_info[i]["shell_id"]   = int(shell_data["shell_id"][i])
    shell_info[i]["lower_z"]    = float(shell_data["lower_z"][i])
    shell_info[i]["upper_z"]    = float(shell_data["upper_z"][i])
    shell_info[i]["lower_com"]  = float(shell_data["lower_com"][i])
    shell_info[i]["upper_com"]  = float(shell_data["upper_com"][i])
    shell_info[i]["shell_com"]  = float(shell_data["shell_com"][i])

# ─── 7. SAVE ───────────────────────────────────────────────────────────────────
# Clamp to uint16 range before converting (max value 65535)
overflow = int((shells > 65535).sum())
if overflow:
    print(f"  WARNING: {overflow} pixels exceed uint16 max – clamping to 65535")
    np.clip(shells, 0, 65535, out=shells)

print(f"\nSaving to {OUTPUT_PATH} …", flush=True)
np.savez_compressed(
    OUTPUT_PATH,
    shells=shells.astype(np.uint16),   # matches CosmoGridV1 format
    shell_info=shell_info,
)

non_zero = int((shells > 0).sum())
print(f"Done.  Non-zero pixels: {non_zero:,} / {n_shells * npix:,}")
