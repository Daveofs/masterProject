"""
build_lightcone_shells.py
=========================
Snapshot-based lightcone shell builder for DiscoDJ simulations.

Implements the same lightcone construction as pkdgrav3's on-the-fly HEALPix
accumulation (shell_collector.py output), but driven by per-step particle
snapshots saved by DiscoDJ.

Why not use DiscoDJ's built-in lightcone mode?
----------------------------------------------
DiscoDJ's lightcone mode only works up to r_max = lightcone_size_factor /
lightcone_size_fraction * Lbox = Lbox/2 by default (~450 Mpc/h for a 900 Mpc/h
box, i.e. z ~ 0.15).  CosmoGridV1 requires z up to 3.5, which corresponds to
~4700 Mpc/h – requiring ~5x box replications per side.

Algorithm (matching pkdgrav / CosmoGridV1 shell_collector.py)
--------------------------------------------------------------
For consecutive simulation snapshots at scale factors a_i < a_{i+1}:

  r_hi = chi(a_i)     # larger comoving radius, earlier epoch
  r_lo = chi(a_{i+1}) # smaller comoving radius, later epoch

For each particle and each periodic replica offset d = (nx,ny,nz)*L:
  r_vec = X_particle + d - X_observer      # relative to observer
  r     = |r_vec|
  If r_lo <= r < r_hi:
    direction -> HEALPix pixel -> accumulate count

The crossing position is optionally linearly interpolated from the two
snapshots (matching pkdgrav's sub-step accuracy).

Output format
-------------
One HEALPix RING FITS map per step, named:
  {prefix}-shell_z-high={z_hi}_z-low={z_lo}.fits
matching the output of shell_collector.py so the two are interchangeable.

Usage
-----
Standalone post-processor from saved snapshot NPZ files:
  python build_lightcone_shells.py \\
      --snapshots-dir  /path/to/snaps/ \\
      --output-dir     /path/to/output/ \\
      --boxsize        900.0 \\
      --z-max          3.5 \\
      --nside          2048 \\
      --cosmo          Planck15 \\
      --prefix         CosmoML

Or import LightconeShellBuilder and call inside a simulation loop
(see bottom of file for example).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import time

import numpy as np
import healpy as hp


# ---------------------------------------------------------------------------
# Comoving distance helper
# ---------------------------------------------------------------------------

def make_chi_of_a(cosmo, n_table: int = 2000):
    """
    Return a fast scalar/array function chi(a) -> comoving distance [Mpc/h],
    computed from DiscoDJ's conformal time:
        chi(a) = c * |eta(a) - eta(1)|
    where c = 299792.458 * h  [km/s * h]  and  eta = a_to_conformalt(a).

    Works with any DiscoDJ cosmology object.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    h = float(cosmo.h)
    c_kmsh = 2.99792458e5 * h  # km/s * h  (gives Mpc/h when multiplied by eta)

    a_table = np.linspace(1e-4, 1.0, n_table)
    # a_to_conformalt is normalised so eta(1) = 0
    eta_table = np.array([float(cosmo.a_to_conformalt(float(a))) for a in a_table])
    chi_table = c_kmsh * np.abs(eta_table)  # Mpc/h   (eta <= 0 for a < 1)

    chi_of_a = interp1d(a_table, chi_table, kind='linear',
                        bounds_error=False, fill_value=(chi_table[0], 0.0))
    a_of_chi = interp1d(chi_table[::-1], a_table[::-1], kind='linear',
                        bounds_error=False, fill_value=(1e-4, 1.0))
    return chi_of_a, a_of_chi


# ---------------------------------------------------------------------------
# Replica offset table  (mirrors pkdgrav's initLightConeOffsets)
# ---------------------------------------------------------------------------

def build_replica_offsets(chi_max_Mpch: float, boxsize: float) -> np.ndarray:
    """
    Return array of shape (N_rep, 3) integer tile indices whose NEAREST CORNER
    to the observer (at box centre) is within chi_max.

    Positions are expressed in units of boxsize:
      offset_Mpch = offsets * boxsize   (add to particle pos relative to observer)
    """
    n_max = int(np.ceil(chi_max_Mpch / boxsize)) + 1
    offsets = []
    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            for nz in range(-n_max, n_max + 1):
                # centre of this tile relative to observer (in Mpc/h)
                cx = nx * boxsize
                cy = ny * boxsize
                cz = nz * boxsize
                # nearest point of tile to observer:
                # tile covers [cx - L/2, cx + L/2]
                nearest_sq = (max(0.0, abs(cx) - boxsize / 2) ** 2 +
                              max(0.0, abs(cy) - boxsize / 2) ** 2 +
                              max(0.0, abs(cz) - boxsize / 2) ** 2)
                if nearest_sq <= chi_max_Mpch ** 2:
                    offsets.append((nx, ny, nz))
    offsets = np.array(offsets, dtype=np.float32)  # (N,3)
    return offsets  # multiply by boxsize to get Mpc/h


# ---------------------------------------------------------------------------
# Core shell accumulator
# ---------------------------------------------------------------------------

class LightconeShellBuilder:
    """
    Accumulates HEALPix lightcone shells from consecutive particle snapshots.

    Parameters
    ----------
    boxsize   : float   box side length [Mpc/h]
    chi_of_a  : callable   a -> chi [Mpc/h]
    nside     : int         HEALPix Nside (default 2048)
    z_max     : float       maximum redshift of the lightcone (default 3.5)
    interpolate : bool      if True, linearly interpolate crossing position
                            between the two snapshots (more accurate, slower)
    """

    def __init__(self,
                 boxsize: float,
                 chi_of_a,
                 nside: int = 2048,
                 z_min: float = 0.0,
                 z_max: float = 3.5,
                 interpolate: bool = True):
        self.L = boxsize
        self.chi_of_a = chi_of_a
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.z_min = z_min
        self.z_max = z_max
        self.chi_max = chi_of_a(1.0 / (1.0 + z_max))

        # Observer at box centre
        self.obs = np.array([boxsize / 2.0] * 3, dtype=np.float64)

        # Precompute replica offsets (integer multiples of L, relative to obs)
        self._rep_ints = build_replica_offsets(self.chi_max, boxsize)  # shape (N,3)
        self._rep_offsets = self._rep_ints * boxsize  # Mpc/h
        print(f"[LightconeShellBuilder] nside={nside}  z=[{z_min:.2f},{z_max:.2f}]  "
              f"chi_max={self.chi_max:.1f} Mpc/h  "
              f"n_replicas={len(self._rep_offsets)}")

        self.interpolate = interpolate

    def accumulate_shell(self,
                         pos_prev: np.ndarray,
                         pos_curr: np.ndarray,
                         a_prev: float,
                         a_curr: float,
                         r_lo_override: float | None = None,
                         r_hi_override: float | None = None) -> np.ndarray:
        """
        Compute a single HEALPix shell map from two consecutive snapshots.

        Parameters
        ----------
        pos_prev  : (N,3) float32  comoving positions [Mpc/h] at a_prev
        pos_curr  : (N,3) float32  comoving positions [Mpc/h] at a_curr
                    (a_curr > a_prev, i.e. later time)
        a_prev, a_curr : float  scale factors
        r_lo_override : float or None  override inner shell boundary [Mpc/h]
        r_hi_override : float or None  override outer shell boundary [Mpc/h]

        Returns
        -------
        shell_map : (npix,) float32  particle count per HEALPix pixel
        """
        r_hi = r_hi_override if r_hi_override is not None else float(self.chi_of_a(a_prev))
        r_lo = r_lo_override if r_lo_override is not None else float(self.chi_of_a(a_curr))

        if r_hi > self.chi_max:
            # Entire shell is beyond the requested z_max; skip
            return np.zeros(self.npix, dtype=np.float32)

        z_outer = 1.0 / a_prev - 1.0
        if z_outer < self.z_min:
            # Entire shell is below the requested z_min; skip
            return np.zeros(self.npix, dtype=np.float32)

        shell_map = np.zeros(self.npix, dtype=np.float32)
        obs = self.obs

        # Relative positions (Mpc/h, centred on observer, periodically wrapped
        # to nearest image is NOT needed here since we explicitly loop replicas)
        X0 = pos_prev.astype(np.float64) - obs  # (N,3)
        X1 = pos_curr.astype(np.float64) - obs  # (N,3)

        for d in self._rep_offsets:
            # Positions of this replica relative to observer
            R0 = X0 + d  # (N,3)
            R1 = X1 + d  # (N,3)

            r0 = np.linalg.norm(R0, axis=1)  # (N,)
            r1 = np.linalg.norm(R1, axis=1)  # (N,)

            # Particles whose trajectory straddles or crosses the shell.
            # A particle contributes to this shell if at some t in [0,1]
            # its distance is in [r_lo, r_hi].
            # Simple criterion: max(r0,r1) >= r_lo  and  min(r0,r1) < r_hi
            r_min = np.minimum(r0, r1)
            r_max = np.maximum(r0, r1)
            mask = (r_max >= r_lo) & (r_min < r_hi)

            if not np.any(mask):
                continue

            if self.interpolate:
                # Linearly interpolate to the inner boundary crossing (r_hi side)
                # t* = (r_hi - r0) / (r0 - r1)  when r0 > r_hi > r1
                # For simplicity use midpoint when both are within shell.
                r0_m = r0[mask]
                r1_m = r1[mask]
                R0_m = R0[mask]
                R1_m = R1[mask]

                # Interpolation fraction to r_hi crossing (clamped to [0,1])
                denom = r0_m - r1_m
                safe_denom = np.where(np.abs(denom) > 1e-10, denom, 1e-10)
                t_hi = np.clip((r_hi - r0_m) / safe_denom, 0.0, 1.0)
                # Use r_hi crossing as the representative position
                direction = (1.0 - t_hi[:, None]) * R0_m + t_hi[:, None] * R1_m
            else:
                # Use midpoint snapshot position
                direction = 0.5 * (R0[mask] + R1[mask])

            # Normalise direction and compute HEALPix pixel
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            norm = np.where(norm > 0, norm, 1.0)
            d_hat = direction / norm  # (K,3)

            theta = np.arccos(np.clip(d_hat[:, 2], -1.0, 1.0))
            phi = np.arctan2(d_hat[:, 1], d_hat[:, 0]) % (2 * np.pi)
            pix = hp.ang2pix(self.nside, theta, phi, nest=False)
            np.add.at(shell_map, pix, 1.0)

        return shell_map


# ---------------------------------------------------------------------------
# Snapshot file I/O
# ---------------------------------------------------------------------------

def save_snapshot(path: Path | str, pos: np.ndarray, a: float):
    """Save positions + scale factor to NPZ."""
    np.savez_compressed(path, pos=pos.astype(np.float32), a=np.float32(a))


def load_snapshot(path: Path | str):
    """Load snapshot; returns (pos: float32 Mpc/h, a: float)."""
    d = np.load(path)
    return d['pos'], float(d['a'])


def save_shell_fits(shell_map: np.ndarray, z_lo: float, z_hi: float,
                    output_dir: Path, prefix: str = "CosmoML"):
    """Save HEALPix shell as FITS, matching shell_collector.py naming."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"{prefix}-shell_z-high={z_hi:.6g}_z-low={z_lo:.6g}.fits"
    hp.write_map(str(fname), m=shell_map.astype(np.float32),
                 nest=False, dtype=np.float32, overwrite=True)
    return fname


# ---------------------------------------------------------------------------
# CosmoGridV1 metainfo helpers
# ---------------------------------------------------------------------------

def load_shell_info_from_metainfo(metainfo_path, cosmo_key: str | None = None):
    """
    Load the shell_info structured array from a CosmoGridV1_metainfo.h5 file.

    Parameters
    ----------
    metainfo_path : str or Path
        Path to the HDF5 metainfo file.
    cosmo_key : str or None
        Key inside ``shell_info/CosmoGrid/raw/grid/``, e.g. ``"cosmo_000001"``.
        If None, the first key (alphabetically sorted) is used.

    Returns
    -------
    shell_info : structured ndarray, shape (n_shells,)
        dtype: [('shell_id','<i4'), ('lower_z','<f4'), ('upper_z','<f4'),
                ('lower_com','<f4'), ('upper_com','<f4'), ('shell_com','<f4')]
    cosmo_key : str  (the key that was actually used)
    """
    import h5py
    with h5py.File(metainfo_path, 'r') as f:
        grp = f['shell_info/CosmoGrid/raw/grid']
        keys = sorted(grp.keys())
        if cosmo_key is None:
            cosmo_key = keys[0]
        shell_info = grp[cosmo_key][:]
    return shell_info, cosmo_key


def save_shells_npz(shells_array: np.ndarray,
                    shell_info_meta,
                    output_path,
                    prefix: str = "CosmoML"):
    """
    Save a (n_shells, npix) particle-count array as a CosmoGridV1-compatible NPZ.

    Parameters
    ----------
    shells_array : (n_shells, npix) int32
    shell_info_meta : structured array from load_shell_info_from_metainfo
        Must have fields shell_id, lower_z, upper_z, lower_com, upper_com, shell_com.
    output_path : str or Path
    prefix : str   Used to construct shell_name strings (e.g. "CosmoML").
    """
    n_shells = len(shell_info_meta)
    dt = np.dtype([
        ('shell_name',  'U64'),
        ('shell_id',    '<i4'),
        ('lower_z',     '<f4'),
        ('upper_z',     '<f4'),
        ('lower_com',   '<f4'),
        ('upper_com',   '<f4'),
        ('shell_com',   '<f4'),
    ])
    shell_info_out = np.empty(n_shells, dtype=dt)
    for field in ('shell_id', 'lower_z', 'upper_z', 'lower_com', 'upper_com', 'shell_com'):
        shell_info_out[field] = shell_info_meta[field]
    for i in range(n_shells):
        lo = float(shell_info_meta['lower_z'][i])
        hi = float(shell_info_meta['upper_z'][i])
        shell_info_out['shell_name'][i] = (
            f"{prefix}-shell_z-high={hi}_z-low={lo}.fits"
        )
    np.savez(output_path,
             shells=shells_array.astype(np.int32),
             shell_info=shell_info_out)
    print(f"[save_shells_npz] Saved {n_shells} shells → {output_path}")


# ---------------------------------------------------------------------------
# Streaming step generator  (O(N) memory, same approach as pkdgrav3)
# ---------------------------------------------------------------------------

def _streaming_steps(dj, a_steps, res_pm, stepper, method, chunk_size, nbody_kwargs):
    """
    Generator that runs the simulation ONE step at a time and yields
    ``(pos_prev, pos_curr, a_prev, a_curr)`` for each interval.

    Only two position snapshots exist in memory simultaneously – O(N) instead
    of the O(n_steps * N) that ``collect_all=True`` would require.

    Works by re-seeding DiscoDJ via ``with_external_ics(pos, vel)`` at each
    step, exactly analogous to how pkdgrav3 and build_lightcone_shells.py's
    ``accumulate_shell`` process one time-slab at a time.
    """
    import jax.numpy as jnp

    _kw = dict(res_pm=res_pm, stepper=stepper, method=method,
               light_cone=False, chunk_size=chunk_size,
               return_displacement=False, **nbody_kwargs)

    n_steps = len(a_steps) - 1

    # ── First call: collect_all=True with n_steps=1 ───────────────────────
    # This gives us BOTH the IC positions X[0]  *and*  X[1] in one trace,
    # using the full LPT / BullFrog IC initialisation that only runs once.
    a0, a1 = float(a_steps[0]), float(a_steps[1])
    result = dj.run_nbody(
        a_ini=a0, a_end=a1, n_steps=1,
        time_var=np.array([a0, a1], dtype=np.float64),
        collect_all=True, **_kw,
    )
    X2, P2, a2 = result
    pos_prev = np.asarray(X2[0]).reshape(-1, 3).astype(np.float64)
    pos_curr = np.asarray(X2[1]).reshape(-1, 3).astype(np.float64)
    vel_curr = np.asarray(P2[1]).reshape(-1, 3).astype(np.float32)
    del X2, P2

    yield pos_prev, pos_curr, float(a2[0]), float(a2[1])
    pos_prev = pos_curr

    # ── Subsequent steps: reseed from (pos, canonical_mom) ────────────────
    # with_external_ics stores pos/vel in _ics so that run_nbody uses them
    # directly (skipping the LPT init).  The canonical-momentum convention
    # matches because a_ini here equals the a_end of the previous step, so
    # the Fplus factors cancel exactly.
    for i in range(1, n_steps):
        a_s = float(a_steps[i])
        a_e = float(a_steps[i + 1])
        dj_i = dj.with_external_ics(
            pos=jnp.array(pos_prev.astype(np.float32)),
            vel=jnp.array(vel_curr),
        )
        X_n, P_n, _ = dj_i.run_nbody(
            a_ini=a_s, a_end=a_e, n_steps=1,
            time_var=np.array([a_s, a_e], dtype=np.float64),
            collect_all=False, **_kw,
        )
        pos_curr = np.asarray(X_n).reshape(-1, 3).astype(np.float64)
        vel_curr = np.asarray(P_n).reshape(-1, 3).astype(np.float32)
        del X_n, P_n

        yield pos_prev, pos_curr, a_s, a_e
        pos_prev = pos_curr


# ---------------------------------------------------------------------------
# Simulation-in-a-loop runner
# ---------------------------------------------------------------------------

def run_with_shells(
    dj,
    a_steps: np.ndarray,
    res_pm: int,
    output_dir: Path,
    nside: int = 2048,
    z_min: float = 0.0,
    z_max: float = 3.5,
    prefix: str = "CosmoML",
    snap_dir: Path | None = None,
    stepper: str = "bullfrog",
    method: str = "pm",
    interpolate: bool = True,
    chunk_size: int | None = None,
    metainfo_path: str | Path | None = None,
    cosmo_key: str | None = None,
    output_npz: str | Path | None = None,
    streaming: bool = True,
    **nbody_kwargs,
):
    """
    Run DiscoDJ and accumulate HEALPix shells step by step.

    Parameters
    ----------
    dj         : DiscoDJ object (already initialised with ICs)
    a_steps    : 1-D array of scale factors at each output step,
                 e.g. np.linspace(a_ini, 1.0, n_steps+1)
    output_dir : directory for output files (FITS or NPZ)
    snap_dir   : if given, also save intermediate snapshots as NPZ
    streaming  : if True (default) run one step at a time – O(N) memory,
                 same approach as pkdgrav3's on-the-fly lightcone.
                 If False, use ``collect_all=True`` (O(n_steps * N) memory,
                 but a single JAX trace / compilation).
    metainfo_path : optional path to CosmoGridV1_metainfo.h5.
        If provided, shell boundaries are taken from the metainfo z_bins
        and output is a single NPZ file matching CosmoGridV1 format.
        If None, falls back to individual FITS files per shell.
    cosmo_key  : which cosmology entry in metainfo to use (e.g. "cosmo_000001").
        Defaults to the first key alphabetically.
    output_npz : output path for the NPZ file when metainfo_path is given.
        Defaults to ``output_dir / f"shells_nside={nside}.npz"``.
    """

    # Build comoving-distance function from DiscoDJ cosmology
    chi_of_a, _ = make_chi_of_a(dj.cosmo)

    builder = LightconeShellBuilder(
        boxsize=dj.boxsize,
        chi_of_a=chi_of_a,
        nside=nside,
        z_min=z_min,
        z_max=z_max,
        interpolate=interpolate,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if snap_dir:
        snap_dir = Path(snap_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metainfo shell bins (NPZ mode) ──────────────────────────────────
    shell_info_meta = None
    npix = builder.npix
    if metainfo_path is not None:
        shell_info_meta, cosmo_key = load_shell_info_from_metainfo(
            metainfo_path, cosmo_key)
        n_meta_shells = len(shell_info_meta)
        shells_array = np.zeros((n_meta_shells, npix), dtype=np.int32)
        meta_lower = shell_info_meta['lower_com'].astype(np.float64)
        meta_upper = shell_info_meta['upper_com'].astype(np.float64)
        meta_ids   = shell_info_meta['shell_id'].astype(int)
        print(f"[run_with_shells] NPZ mode: {n_meta_shells} shells from "
              f"metainfo key={cosmo_key}")
        if output_npz is None:
            output_npz = output_dir / f"shells_nside={nside}.npz"
        else:
            output_npz = Path(output_npz)

    n_steps = len(a_steps) - 1
    a_ini = float(a_steps[0])
    a_end = float(a_steps[-1])
    mode_str = "streaming (O(N))" if streaming else "collect_all (O(n_steps·N))"
    print(f"Running DiscoDJ with {n_steps} steps, "
          f"a={a_ini:.4f}→{a_end:.4f}  [{mode_str}]")
    t0 = time()

    # ── Choose step data source ───────────────────────────────────────────
    if streaming:
        step_gen = _streaming_steps(
            dj, a_steps, res_pm, stepper, method, chunk_size, nbody_kwargs)
        X_all = None  # not used in streaming mode
    else:
        result = dj.run_nbody(
            a_ini=a_ini,
            a_end=a_end,
            n_steps=n_steps,
            time_var=a_steps,
            res_pm=res_pm,
            stepper=stepper,
            method=method,
            collect_all=True,
            return_displacement=False,
            light_cone=False,
            chunk_size=chunk_size,
            **nbody_kwargs,
        )
        X_all, P_all, a_all = result
        del P_all

        def _batch_gen():
            n_out = X_all.shape[0]
            for _i in range(n_out - 1):
                yield (
                    np.asarray(X_all[_i    ]).reshape(-1, 3).astype(np.float64),
                    np.asarray(X_all[_i + 1]).reshape(-1, 3).astype(np.float64),
                    float(a_all[_i]),
                    float(a_all[_i + 1]),
                )
        step_gen = _batch_gen()
        t1 = time()
        print(f"Simulation done in {t1-t0:.1f}s, now accumulating shells …")

    shells_written = 0
    last_pos_curr: np.ndarray | None = None
    last_a_curr: float | None = None

    for i, (pos_prev, pos_curr, a_prev, a_curr) in enumerate(step_gen):
        last_pos_curr = pos_curr
        last_a_curr = a_curr

        z_prev = 1.0 / a_prev - 1.0
        z_curr = 1.0 / a_curr - 1.0

        r_step_hi = float(chi_of_a(a_prev))
        r_step_lo = float(chi_of_a(a_curr))

        if shell_info_meta is None and r_step_hi > builder.chi_max:
            # FITS mode: skip if entirely beyond z_max.
            # In NPZ mode the overlap check below handles boundary steps.
            if snap_dir:
                save_snapshot(snap_dir / f"snap_{i:05d}.npz",
                              pos_prev.astype(np.float32), a_prev)
            continue
        if z_prev < builder.z_min:
            # Below z_min; skip but still optionally save snapshot
            if snap_dir:
                save_snapshot(snap_dir / f"snap_{i:05d}.npz",
                              pos_prev.astype(np.float32), a_prev)
            continue

        if snap_dir:
            save_snapshot(snap_dir / f"snap_{i:05d}.npz",
                          pos_prev.astype(np.float32), a_prev)

        t_shell = time()

        if shell_info_meta is not None:
            # ── NPZ mode: assign particles to metainfo z_bins ────────────────
            overlap = np.where(
                (meta_lower < r_step_hi) & (meta_upper > r_step_lo)
            )[0]
            for idx in overlap:
                sid = int(meta_ids[idx])
                shell_map = builder.accumulate_shell(
                    pos_prev, pos_curr, a_prev, a_curr,
                    r_lo_override=float(meta_lower[idx]),
                    r_hi_override=float(meta_upper[idx]),
                )
                shells_array[sid] += shell_map.astype(np.int32)
            dt = time() - t_shell
            print(f"  step {i+1}/{n_steps}  z=[{z_curr:.3f},{z_prev:.3f}]  "
                  f"overlapping_shells={len(overlap)}  ({dt:.1f}s)")
            shells_written += len(overlap)
        else:
            # ── FITS mode ────────────────────────────────────────────────────
            shell = builder.accumulate_shell(pos_prev, pos_curr, a_prev, a_curr)
            fname = save_shell_fits(shell, z_lo=z_curr, z_hi=z_prev,
                                    output_dir=output_dir, prefix=prefix)
            dt = time() - t_shell
            print(f"  shell {i+1}/{n_steps}  z=[{z_curr:.3f},{z_prev:.3f}]  "
                  f"n_part={int(shell.sum())}  → {fname.name}  ({dt:.1f}s)")
            shells_written += 1

    # Save final snapshot
    if snap_dir and last_pos_curr is not None:
        save_snapshot(snap_dir / f"snap_{n_steps:05d}.npz",
                      last_pos_curr.astype(np.float32), last_a_curr)

    if shell_info_meta is not None:
        # Save combined NPZ
        save_shells_npz(shells_array, shell_info_meta, output_npz, prefix=prefix)
        print(f"Done. NPZ saved → {output_npz}")
    else:
        print(f"Done. {shells_written} shells written to {output_dir}")

    return shells_written


# ---------------------------------------------------------------------------
# Stand-alone post-processor from saved snapshot NPZ files
# ---------------------------------------------------------------------------

def build_shells_from_snapshots(
    snapshots_dir: Path,
    output_dir: Path,
    boxsize: float,
    cosmo_name: str = "Planck15",
    nside: int = 2048,
    z_max: float = 3.5,
    prefix: str = "CosmoML",
    interpolate: bool = True,
):
    """
    Post-process a directory of snapshot NPZ files (as saved by save_snapshot)
    into HEALPix shells.

    Each NPZ is expected to contain:
      - 'pos': (N,3) float32 comoving positions [Mpc/h]
      - 'a'  : scalar, scale factor

    Snapshots are processed in order of increasing scale factor.
    """
    from discodj import DiscoDJ

    # Load cosmology just for chi(a)
    dj_dummy = DiscoDJ(dim=3, res=2, boxsize=boxsize, device="cpu", cosmo=cosmo_name,
                       requires_grad_wrt_cosmo=False).with_timetables()
    chi_of_a, _ = make_chi_of_a(dj_dummy.cosmo)

    builder = LightconeShellBuilder(boxsize=boxsize, chi_of_a=chi_of_a,
                                    nside=nside, z_max=z_max, interpolate=interpolate)

    snaps = sorted(Path(snapshots_dir).glob("snap_*.npz"))
    print(f"Found {len(snaps)} snapshots in {snapshots_dir}")

    pos_prev, a_prev = load_snapshot(snaps[0])

    for snap_path in snaps[1:]:
        pos_curr, a_curr = load_snapshot(snap_path)
        z_prev = 1.0 / a_prev - 1.0
        z_curr = 1.0 / a_curr - 1.0

        shell = builder.accumulate_shell(
            pos_prev.astype(np.float64), pos_curr.astype(np.float64),
            a_prev, a_curr
        )
        fname = save_shell_fits(shell, z_lo=z_curr, z_hi=z_prev,
                                output_dir=output_dir, prefix=prefix)
        print(f"  z=[{z_curr:.3f},{z_prev:.3f}]  n={int(shell.sum())}  → {fname.name}")

        pos_prev, a_prev = pos_curr, a_curr

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build HEALPix lightcone shells from DiscoDJ snapshot NPZ files."
    )
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--output-dir",    type=Path, required=True)
    parser.add_argument("--boxsize",       type=float, required=True,
                        help="Box size [Mpc/h]")
    parser.add_argument("--cosmo",         type=str, default="Planck15")
    parser.add_argument("--nside",         type=int, default=2048)
    parser.add_argument("--z-max",         type=float, default=3.5)
    parser.add_argument("--prefix",        type=str, default="CosmoML")
    parser.add_argument("--no-interpolate", action="store_true",
                        help="Use midpoint instead of crossing interpolation")
    args = parser.parse_args()

    build_shells_from_snapshots(
        snapshots_dir=args.snapshots_dir,
        output_dir=args.output_dir,
        boxsize=args.boxsize,
        cosmo_name=args.cosmo,
        nside=args.nside,
        z_max=args.z_max,
        prefix=args.prefix,
        interpolate=not args.no_interpolate,
    )
