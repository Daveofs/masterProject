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
from functools import partial
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
# JAX-native GPU helpers (lazy-loaded to keep the CPU-only path importable)
# ---------------------------------------------------------------------------

def _vec2pix_ring_jax(nside: int, x, y, z):
    """
    Pure-JAX HEALPix RING-scheme pixel index for a batch of unit vectors.
    x, y, z are JAX arrays of the same shape; nside must be a power of 2.

    Implements the healpix_bare C algorithm:  loc2hpd -> hpd2ring.
    Reference: https://github.com/ntessore/healpix (BSD-3-Clause).
    """
    import jax.numpy as jnp

    # ── normalise ─────────────────────────────────────────────────────────
    norm = jnp.sqrt(x * x + y * y + z * z)
    safe_norm = jnp.where(norm > 0.0, norm, jnp.ones_like(norm))
    x = x / safe_norm;  y = y / safe_norm;  z = z / safe_norm

    za  = jnp.abs(z)
    s   = jnp.sqrt(jnp.maximum(1.0 - z * z, 0.0))   # sin(theta)
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0.0, phi + 2.0 * jnp.pi, phi)   # [0, 2π)
    tt  = phi * (2.0 / jnp.pi)   # = 4·phi/(2π), in [0, 4)

    # Face lookup tables (size 12, indexed by face number 0-11)
    _jrll = jnp.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=jnp.int32)
    _jpll = jnp.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=jnp.int32)

    # ── loc2hpd: equatorial region  |z| ≤ 2/3 ────────────────────────────
    temp1e = 0.5 + tt                    # [0.5, 4.5)
    temp2e = z * 0.75                    # [-0.5, +0.5]
    jp_fe  = temp1e - temp2e             # ascending-edge line index  [0, 5)
    jm_fe  = temp1e + temp2e             # descending-edge line index [0, 5)
    ifp    = jnp.floor(jp_fe).astype(jnp.int32)   # face index along ascending  {0..4}
    ifm    = jnp.floor(jm_fe).astype(jnp.int32)   # face index along descending {0..4}
    xe     = (jm_fe - ifm.astype(jnp.float32)) * nside            # x within face [0, n)
    ye     = (1.0 + ifp.astype(jnp.float32) - jp_fe) * nside      # y within face [0, n)
    fe     = jnp.where(ifp == ifm,  ifp | 4,
             jnp.where(ifp <  ifm,  ifp,
                                    ifm + 8))      # face number 0-11

    # ── loc2hpd: polar regions  |z| > 2/3 ────────────────────────────────
    ntt_p  = jnp.minimum(jnp.floor(tt).astype(jnp.int32),
                         jnp.full_like(jnp.floor(tt).astype(jnp.int32), 3))
    tp_p   = tt - ntt_p.astype(jnp.float32)        # fractional part ∈ [0, 1)
    tmp_p  = s / jnp.sqrt(jnp.maximum((1.0 + za) / 3.0, 1e-30))
    jp_p   = jnp.minimum(tp_p * tmp_p, 1.0)
    jm_p   = jnp.minimum((1.0 - tp_p) * tmp_p, 1.0)
    # North polar: swap  jp←(1−jm),  jm←(1−jp)
    jp_p2  = jnp.where(z >= 0.0,  1.0 - jm_p,  jp_p)
    jm_p2  = jnp.where(z >= 0.0,  1.0 - jp_p,  jm_p)
    xp     = jp_p2 * nside
    yp     = jm_p2 * nside
    fp     = jnp.where(z >= 0.0,  ntt_p,  ntt_p + 8)

    # ── select region and floor/clamp hpd coordinates ────────────────────
    is_eq  = za <= 2.0 / 3.0
    hx_f   = jnp.where(is_eq, xe, xp)
    hy_f   = jnp.where(is_eq, ye, yp)
    hf     = jnp.where(is_eq, fe, fp).astype(jnp.int32)
    hpd_x  = jnp.minimum(jnp.floor(hx_f).astype(jnp.int32), nside - 1)
    hpd_y  = jnp.minimum(jnp.floor(hy_f).astype(jnp.int32), nside - 1)

    # ── hpd2ring ──────────────────────────────────────────────────────────
    jrll_v = jnp.take(_jrll, hf)
    jpll_v = jnp.take(_jpll, hf)
    jr     = jrll_v * nside - hpd_x - hpd_y - 1   # ring number
    nl4    = 4 * nside
    npix_  = 12 * nside * nside

    def _wrap(jp):
        return jnp.where(jp > nl4, jp - nl4,
               jnp.where(jp < 1,   jp + nl4, jp))

    # North polar cap  (jr < nside)
    jpn   = _wrap((jpll_v * jr + hpd_x - hpd_y + 1) // 2)
    ipn   = 2 * jr * (jr - 1) + jpn - 1

    # Equatorial belt  (nside ≤ jr ≤ 3·nside)
    ksh   = (jr + nside) & 1
    jpe   = _wrap((jpll_v * nside + hpd_x - hpd_y + 1 + ksh) // 2)
    ipe   = 2 * nside * (nside - 1) + (jr - nside) * nl4 + jpe - 1

    # South polar cap  (jr > 3·nside)
    jr_s  = nl4 - jr
    jps   = _wrap((jpll_v * jr_s + hpd_x - hpd_y + 1) // 2)
    ips   = npix_ - 2 * (jr_s + 1) * jr_s + jps - 1

    return jnp.where(jr < nside, ipn,
           jnp.where(jr > 3 * nside, ips, ipe))


# Module-level cache so the JIT-compiled GPU kernel is built only once per process.
_JAX_SHELL_KERNEL = None


def _get_jax_shell_kernel():
    """Return (building once) the JIT-compiled per-replica GPU scatter kernel."""
    global _JAX_SHELL_KERNEL
    if _JAX_SHELL_KERNEL is not None:
        return _JAX_SHELL_KERNEL

    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnames=['nside', 'npix', 'interpolate'])
    def _kernel(X0, X1, d, r_lo, r_hi, nside, npix, interpolate):
        """
        GPU kernel: contribution of ONE periodic replica to a HEALPix shell.

        Parameters
        ----------
        X0, X1     : (N, 3) float32  positions relative to observer [Mpc/h]
        d          : (3,)   float32  replica offset [Mpc/h]
        r_lo, r_hi : float32         shell boundaries [Mpc/h]
        nside, npix, interpolate : static compile-time constants
        """
        R0 = X0 + d
        R1 = X1 + d
        r0 = jnp.sqrt(jnp.sum(R0 * R0, axis=1))
        r1 = jnp.sqrt(jnp.sum(R1 * R1, axis=1))
        mask = (jnp.maximum(r0, r1) >= r_lo) & (jnp.minimum(r0, r1) < r_hi)

        if interpolate:
            denom      = r0 - r1
            safe_denom = jnp.where(jnp.abs(denom) > 1e-10, denom, jnp.float32(1e-10))
            t_hi       = jnp.clip((r_hi - r0) / safe_denom, 0.0, 1.0)
            direction  = (1.0 - t_hi[:, None]) * R0 + t_hi[:, None] * R1
        else:
            direction = 0.5 * (R0 + R1)

        # Steer masked-out particles to a safe sky direction so they don't NaN.
        safe_dir = jnp.where(
            mask[:, None],
            direction,
            jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                             direction.shape),
        )
        pix     = _vec2pix_ring_jax(nside, safe_dir[:, 0], safe_dir[:, 1], safe_dir[:, 2])
        pix     = jnp.clip(pix, 0, npix - 1)          # safety clamp
        weights = mask.astype(jnp.float32)
        return jnp.zeros(npix, dtype=jnp.float32).at[pix].add(weights)

    _JAX_SHELL_KERNEL = _kernel
    return _JAX_SHELL_KERNEL


# Module-level cache for the fori_loop-based GPU kernel (compiled once for all
# replica counts, using a dynamic upper bound that XLA lowers to a while-loop).
_JAX_SHELL_FORI_KERNEL = None


def _get_jax_shell_fori_kernel():
    """
    Return (building once) a JIT-compiled kernel that processes ALL active
    replicas inside a single ``jax.lax.fori_loop``.

    Unlike ``_get_jax_shell_kernel`` (one XLA dispatch *per* replica in a
    Python loop), this kernel submits a single XLA program that iterates over
    all replicas internally.  This eliminates per-replica Python dispatch
    overhead and the associated serialisation of XLA kernel launches that
    dominates at high-z (many hundreds of replicas).

    The compiled program is reused for every shell because n_active is a
    *dynamic* JAX int32 scalar: XLA lowers fori_loop to a while-loop whose
    trip count is determined at runtime, so the same compiled code handles
    any number of active replicas.
    """
    global _JAX_SHELL_FORI_KERNEL
    if _JAX_SHELL_FORI_KERNEL is not None:
        return _JAX_SHELL_FORI_KERNEL

    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnames=['nside', 'npix', 'interpolate'])
    def _fori_kernel(X0, X1, rep_offsets, n_active, r_lo, r_hi,
                     nside, npix, interpolate):
        """
        Accumulate shell contributions for ``n_active`` periodic replicas.

        Parameters
        ----------
        X0, X1       : (N, 3) float32  positions relative to observer [Mpc/h]
        rep_offsets  : (R, 3) float32  padded replica offset table;
                       only the first ``n_active`` rows are processed.
        n_active     : int32 scalar (dynamic) – replicas to process.
        r_lo, r_hi   : float32  shell boundaries [Mpc/h]
        nside, npix, interpolate : compile-time constants (static)
        """
        def body(i, shell):
            d  = rep_offsets[i]
            R0 = X0 + d
            R1 = X1 + d
            r0 = jnp.sqrt(jnp.sum(R0 * R0, axis=1))
            r1 = jnp.sqrt(jnp.sum(R1 * R1, axis=1))
            mask = (jnp.maximum(r0, r1) >= r_lo) & (jnp.minimum(r0, r1) < r_hi)

            if interpolate:
                denom      = r0 - r1
                safe_denom = jnp.where(jnp.abs(denom) > 1e-10, denom,
                                       jnp.float32(1e-10))
                t_hi       = jnp.clip((r_hi - r0) / safe_denom, 0.0, 1.0)
                direction  = (1.0 - t_hi[:, None]) * R0 + t_hi[:, None] * R1
            else:
                direction = 0.5 * (R0 + R1)

            # For masked-out particles we still go through vec2pix (to avoid
            # NaN), but we DISCARD their pixel and route to a unique trash
            # address derived from the particle index.  Spreading the zero-
            # weight adds over all pixels eliminates the catastrophic atomic
            # serialisation that occurs when all 94 % of non-contributing
            # particles are steered to the SAME pixel.
            safe_dir = jnp.where(
                mask[:, None],
                direction,
                jnp.broadcast_to(jnp.array([1., 0., 0.], dtype=jnp.float32),
                                  direction.shape),
            )
            pix      = _vec2pix_ring_jax(nside, safe_dir[:, 0],
                                         safe_dir[:, 1], safe_dir[:, 2])
            pix      = jnp.clip(pix, 0, npix - 1)
            # Unique trash pixel per particle: spreads zero-weight atomics
            # uniformly over the map → no hot-address serialisation.
            # X0.shape[0] and npix are compile-time constants inside JIT, so
            # XLA materialises trash_pix once (not per iteration).
            trash_pix = jnp.arange(X0.shape[0], dtype=jnp.int32) % jnp.int32(npix)
            safe_pix  = jnp.where(mask, pix, trash_pix)
            weights   = mask.astype(jnp.float32)
            return shell.at[safe_pix].add(weights)

        return jax.lax.fori_loop(
            jnp.int32(0), n_active, body,
            jnp.zeros(npix, dtype=jnp.float32),
        )

    _JAX_SHELL_FORI_KERNEL = _fori_kernel
    return _JAX_SHELL_FORI_KERNEL


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
                 interpolate: bool = True,
                 particle_chunk_size: int = 2_000_000):
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
        # Precompute radial bounds of each replicated cube relative to observer.
        # If a shell [r_lo, r_hi] does not intersect [rmin, rmax], this replica
        # cannot contribute and is skipped entirely.
        rep_abs = np.abs(self._rep_offsets)
        rep_near = np.maximum(0.0, rep_abs - boxsize / 2.0)
        rep_far = rep_abs + boxsize / 2.0
        self._rep_rmin = np.sqrt(np.sum(rep_near * rep_near, axis=1))
        self._rep_rmax = np.sqrt(np.sum(rep_far * rep_far, axis=1))
        print(f"[LightconeShellBuilder] nside={nside}  z=[{z_min:.2f},{z_max:.2f}]  "
              f"chi_max={self.chi_max:.1f} Mpc/h  "
              f"n_replicas={len(self._rep_offsets)}")

        self.interpolate = interpolate
        self.particle_chunk_size = int(particle_chunk_size)

    def accumulate_shell(self,
                         pos_prev: np.ndarray,
                         pos_curr: np.ndarray,
                         a_prev: float,
                         a_curr: float,
                         r_lo_override: float | None = None,
                         r_hi_override: float | None = None,
                         verbose: bool = True) -> np.ndarray:
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
        verbose   : bool  if True (default), print per-shell timing diagnostics

        Returns
        -------
        shell_map : (npix,) float32  particle count per HEALPix pixel
        """
        t_acc_start = time()
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
        t_cast = time()
        X0 = pos_prev.astype(np.float32, copy=False) - obs.astype(np.float32)
        X1 = pos_curr.astype(np.float32, copy=False) - obs.astype(np.float32)
        n_part = X0.shape[0]
        csize = max(1, self.particle_chunk_size)
        n_chunks = int(np.ceil(n_part / csize))
        t_cast_done = time()

        # Count how many replicas survive the fast-reject test
        n_rep_total = len(self._rep_offsets)
        n_rep_kept = 0
        t_loop_start = time()
        t_r_sq = 0.0
        t_mask = 0.0
        t_interp = 0.0
        t_healpix = 0.0
        n_particles_placed = 0

        for rep_idx, d in enumerate(self._rep_offsets):
            # Fast reject using replica radial bounds.
            if r_hi < self._rep_rmin[rep_idx] or r_lo > self._rep_rmax[rep_idx]:
                continue
            n_rep_kept += 1

            for i0 in range(0, n_part, csize):
                i1 = min(i0 + csize, n_part)
                R0 = X0[i0:i1] + d
                R1 = X1[i0:i1] + d

                _t = time()
                r0 = np.sqrt(np.sum(R0 * R0, axis=1, dtype=np.float32))
                r1 = np.sqrt(np.sum(R1 * R1, axis=1, dtype=np.float32))
                t_r_sq += time() - _t

                # A trajectory contributes if its radius interval overlaps shell.
                _t = time()
                r_min = np.minimum(r0, r1)
                r_max = np.maximum(r0, r1)
                mask = (r_max >= r_lo) & (r_min < r_hi)
                t_mask += time() - _t
                if not np.any(mask):
                    continue

                _t = time()
                if self.interpolate:
                    r0_m = r0[mask]
                    r1_m = r1[mask]
                    R0_m = R0[mask]
                    R1_m = R1[mask]

                    denom = r0_m - r1_m
                    safe_denom = np.where(np.abs(denom) > 1e-10, denom, 1e-10)
                    t_hi = np.clip((r_hi - r0_m) / safe_denom, 0.0, 1.0).astype(np.float32)
                    direction = (1.0 - t_hi[:, None]) * R0_m + t_hi[:, None] * R1_m
                else:
                    direction = 0.5 * (R0[mask] + R1[mask])
                t_interp += time() - _t

                # healpy's vector form is generally faster than angle conversion.
                _t = time()
                pix = hp.vec2pix(
                    self.nside,
                    direction[:, 0],
                    direction[:, 1],
                    direction[:, 2],
                    nest=False,
                )
                shell_map += np.bincount(pix, minlength=self.npix).astype(np.float32, copy=False)
                t_healpix += time() - _t
                n_particles_placed += int(np.sum(mask))

        t_total = time() - t_acc_start
        if verbose:
            print(
                f"    [accumulate_shell] r=[{r_lo:.1f},{r_hi:.1f}] Mpc/h  "
                f"n_part={n_part:,}  chunks={n_chunks}  "
                f"replicas={n_rep_kept}/{n_rep_total}  "
                f"placed={n_particles_placed:,}  "
                f"| cast={t_cast_done-t_cast:.2f}s  "
                f"rsq={t_r_sq:.2f}s  mask={t_mask:.2f}s  "
                f"interp={t_interp:.2f}s  healpix={t_healpix:.2f}s  "
                f"total={t_total:.2f}s"
            )

        return shell_map

    # ------------------------------------------------------------------
    def prepare_device_maps(self, pos_prev_jax, pos_curr_jax,
                            verbose: bool = True) -> tuple:
        """
        Transfer and centre particle positions onto every local GPU, sharding
        particles across GPUs (each GPU gets N/n_devs particles).

        This means each GPU processes N/n_devs particles across ALL replicas,
        giving true N/n_devs speedup instead of duplicating all N particles on
        every GPU.

        Returns
        -------
        (X0_devs, X1_devs, n_part_total) : tuple
            X0_devs, X1_devs – lists of per-device JAX arrays (N_local, 3)
                               where N_local = ceil(N / n_devs)
            n_part_total      – total particles across all shards
        """
        import jax
        import jax.numpy as jnp

        t_gather = time()
        devs    = jax.local_devices()
        n_devs  = len(devs)
        obs_arr = np.array(self.obs, dtype=np.float32)
        X0_flat = pos_prev_jax.reshape(-1, 3).astype(jnp.float32)
        X1_flat = pos_curr_jax.reshape(-1, 3).astype(jnp.float32)
        if jax.process_count() > 1:
            X0_np = np.concatenate(
                [np.asarray(s.data) for s in X0_flat.addressable_shards], axis=0)
            X1_np = np.concatenate(
                [np.asarray(s.data) for s in X1_flat.addressable_shards], axis=0)
        else:
            X0_np = np.asarray(X0_flat)
            X1_np = np.asarray(X1_flat)
        # Centre on observer
        X0_np = X0_np - obs_arr
        X1_np = X1_np - obs_arr
        n_part_total = X0_np.shape[0]
        # Shard particles among devices: GPU k processes particles [k*chunk:(k+1)*chunk]
        chunk = int(np.ceil(n_part_total / n_devs))
        X0_devs = []
        X1_devs = []
        for k, d in enumerate(devs):
            lo = k * chunk
            hi = min(lo + chunk, n_part_total)
            X0_devs.append(jax.device_put(X0_np[lo:hi], d))
            X1_devs.append(jax.device_put(X1_np[lo:hi], d))
        if verbose:
            print(f"    [jax_shell] gather/cast {time()-t_gather:.2f}s  "
                  f"n_part={n_part_total:,}  n_devs={n_devs}  "
                  f"part/dev={chunk:,}")
        return X0_devs, X1_devs, n_part_total

    # ------------------------------------------------------------------
    def accumulate_shell_jax(self,
                             pos_prev_jax,
                             pos_curr_jax,
                             a_prev: float,
                             a_curr: float,
                             r_lo_override: float | None = None,
                             r_hi_override: float | None = None,
                             _precast: tuple | None = None) -> np.ndarray:
        """
        GPU-accelerated shell accumulation from JAX float32 position arrays.

        Equivalent to ``accumulate_shell()`` but runs entirely on GPU:
        distances, masking, quadratic interpolation, HEALPix indexing, and
        scatter-add are all performed inside a single ``jax.lax.fori_loop``
        that iterates over all active periodic replicas without returning to
        Python between iterations.

        Parameters
        ----------
        pos_prev_jax, pos_curr_jax : JAX arrays (any leading shape, float32)
            Comoving particle positions [Mpc/h] at a_prev / a_curr.
            Will be reshaped to (N, 3) and broadcast to all available GPUs.
        a_prev, a_curr : float
        r_lo_override, r_hi_override : optional boundary overrides [Mpc/h]
        _precast : tuple or None
            Pre-computed ``(X0_devs, X1_devs, n_part)`` from
            ``prepare_device_maps()``.  When provided, the gather/cast step
            is skipped (positions are already on device).  Pass this when
            processing multiple overlapping shells for the same step to
            avoid repeating the expensive GPU→CPU→GPU round-trip.

        Returns
        -------
        shell_map : (npix,) float32 numpy array
        """
        import jax
        import jax.numpy as jnp

        r_hi = r_hi_override if r_hi_override is not None else float(self.chi_of_a(a_prev))
        r_lo = r_lo_override if r_lo_override is not None else float(self.chi_of_a(a_curr))

        if r_hi > self.chi_max:
            return np.zeros(self.npix, dtype=np.float32)
        if 1.0 / a_prev - 1.0 < self.z_min:
            return np.zeros(self.npix, dtype=np.float32)

        from concurrent.futures import ThreadPoolExecutor

        fori_kernel = _get_jax_shell_fori_kernel()

        # ---- gather/cast: bring positions to each local device -----------
        if _precast is not None:
            # Reuse pre-computed device arrays (same pos_prev/pos_curr pair).
            X0_devs, X1_devs, n_part = _precast
            devs   = jax.local_devices()
            n_devs = len(devs)
        else:
            t_gather = time()
            devs   = jax.local_devices()
            n_devs = len(devs)
            obs_arr = np.array(self.obs, dtype=np.float32)
            X0_flat = pos_prev_jax.reshape(-1, 3).astype(jnp.float32)
            X1_flat = pos_curr_jax.reshape(-1, 3).astype(jnp.float32)
            if jax.process_count() > 1:
                X0_np = np.concatenate(
                    [np.asarray(s.data) for s in X0_flat.addressable_shards], axis=0)
                X1_np = np.concatenate(
                    [np.asarray(s.data) for s in X1_flat.addressable_shards], axis=0)
            else:
                X0_np = np.asarray(X0_flat)
                X1_np = np.asarray(X1_flat)
            # Centre + shard particles across GPUs
            X0_np -= np.array(self.obs, dtype=np.float32)
            X1_np -= np.array(self.obs, dtype=np.float32)
            n_part   = X0_np.shape[0]
            chunk    = int(np.ceil(n_part / n_devs))
            X0_devs  = []
            X1_devs  = []
            for k, dev in enumerate(devs):
                lo = k * chunk
                hi = min(lo + chunk, n_part)
                X0_devs.append(jax.device_put(X0_np[lo:hi], dev))
                X1_devs.append(jax.device_put(X1_np[lo:hi], dev))
            print(f"    [jax_shell] gather/cast {time()-t_gather:.2f}s  "
                  f"r=[{r_lo:.1f},{r_hi:.1f}] Mpc/h  n_part={n_part:,}  "
                  f"n_devs={n_devs}  part/dev={chunk:,}")

        # ---- pre-collect kept replica offsets ----------------------------
        # All devices process ALL replicas, but each device only handles its
        # particle shard → true N/n_devs speedup per device.
        n_rep_total = len(self._rep_offsets)
        kept_reps = []
        for rep_idx, d in enumerate(self._rep_offsets):
            if r_hi < self._rep_rmin[rep_idx] or r_lo > self._rep_rmax[rep_idx]:
                continue
            kept_reps.append(d)
        n_rep_kept = len(kept_reps)

        # ---- dispatch per-device fori_loop in parallel threads -----------
        # Each GPU processes its own particle shard across ALL n_rep_kept
        # replicas inside a single fori_loop → results are partial shell maps
        # that are summed at the end.
        r_lo_f32  = np.float32(r_lo)
        r_hi_f32  = np.float32(r_hi)
        nside_    = self.nside
        npix_     = self.npix
        interp_   = self.interpolate

        # Build padded replica array once (fixed shape for XLA reuse)
        rep_np = np.zeros((n_rep_total, 3), dtype=np.float32)
        if n_rep_kept > 0:
            rep_np[:n_rep_kept] = np.stack(kept_reps, axis=0)
        n_active_val = np.int32(n_rep_kept)

        def _run_device(dev_i: int) -> np.ndarray:
            if X0_devs[dev_i].shape[0] == 0 or n_rep_kept == 0:
                return np.zeros(npix_, dtype=np.float32)
            rep_jax  = jax.device_put(rep_np, devs[dev_i])
            n_active = jnp.int32(n_active_val)
            return np.asarray(fori_kernel(
                X0_devs[dev_i], X1_devs[dev_i],
                rep_jax, n_active,
                r_lo_f32, r_hi_f32, nside_, npix_, interp_,
            ))

        t0 = time()
        with ThreadPoolExecutor(max_workers=n_devs) as pool:
            partial_np = list(pool.map(_run_device, range(n_devs)))
        t_gpu = time() - t0

        shell_map = sum(partial_np)

        # In a multi-process JAX distributed run each process holds a shard of
        # the particles, so shell_map contains only the partial count from this
        # process's particles.  Sum across all processes to get the global count.
        if jax.process_count() > 1:
            from jax.experimental.multihost_utils import process_allgather as _pag
            gathered = np.asarray(_pag(shell_map))  # (n_procs, npix)
            shell_map = gathered.sum(axis=0).astype(np.float32)

        print(f"    [jax_shell] r=[{r_lo:.1f},{r_hi:.1f}] Mpc/h  gpu={t_gpu:.2f}s  "
              f"replicas={n_rep_kept}/{n_rep_total}  "
              f"placed={int(shell_map.sum()):,}")
        return shell_map

def save_snapshot(path: Path | str, pos: np.ndarray, a: float):
    """Save positions + scale factor to NPZ."""
    np.savez_compressed(path, pos=pos.astype(np.float32), a=np.float32(a))


def load_snapshot(path: Path | str):
    """Load snapshot; returns (pos: float32 Mpc/h, a: float)."""
    d = np.load(path)
    return d['pos'], float(d['a'])


def save_shell_fits(shell_map: np.ndarray, z_lo: float, z_hi: float,
                    output_dir: Path, prefix: str = "Cosmo"):
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

def _streaming_steps(dj, a_steps, res_pm, stepper, method, chunk_size, nbody_kwargs,
                     return_jax: bool = False):
    """
    Generator that runs the simulation ONE step at a time and yields
    ``(pos_prev, pos_curr, a_prev, a_curr)`` for each interval.

    Only two position snapshots exist in memory simultaneously – O(N) instead
    of the O(n_steps * N) that ``collect_all=True`` would require.

    Works by re-seeding DiscoDJ via ``with_external_ics(pos, vel)`` at each
    step, exactly analogous to how pkdgrav3 and build_lightcone_shells.py's
    ``accumulate_shell`` process one time-slab at a time.

    Parameters
    ----------
    return_jax : bool
        If False (default) positions are returned as numpy float64 arrays on CPU.
        If True positions are returned as JAX float32 arrays gathered to GPU 0,
        which avoids the device→host transfer and float64 cast.  Use together
        with ``LightconeShellBuilder.accumulate_shell_jax``.
    """
    import jax
    import jax.numpy as jnp

    _kw = dict(res_pm=res_pm, stepper=stepper, method=method,
               light_cone=False, chunk_size=chunk_size,
               return_displacement=False, **nbody_kwargs)

    n_steps = len(a_steps) - 1

    # ── First call: collect_all=True with n_steps=1 ───────────────────────
    # This gives us BOTH the IC positions X[0]  *and*  X[1] in one trace,
    # using the full LPT / BullFrog IC initialisation that only runs once.
    a0, a1 = float(a_steps[0]), float(a_steps[1])
    t_nbody = time()
    result = dj.run_nbody(
        a_ini=a0, a_end=a1, n_steps=1,
        time_var=np.array([a0, a1], dtype=np.float64),
        collect_all=True, **_kw,
    )
    print(f"  [streaming] step 1/{n_steps}  a=[{a0:.4f},{a1:.4f}]  "
          f"nbody (2-snapshot init) took {time()-t_nbody:.1f}s")
    X2, P2, a2 = result
    t_gather = time()
    if return_jax:
        # Keep native sharding (all 4 GPUs) so with_external_ics stays compatible
        # with DiscoDJ's internal sharded grid self.q.  accumulate_shell_jax
        # does its own device_put to GPU 0 independently.
        pos_prev = X2[0].reshape(-1, 3).astype(jnp.float32)
        pos_curr = X2[1].reshape(-1, 3).astype(jnp.float32)
        vel_curr = P2[1].reshape(-1, 3).astype(jnp.float32)
    else:
        pos_prev = np.asarray(X2[0]).reshape(-1, 3).astype(np.float64)
        pos_curr = np.asarray(X2[1]).reshape(-1, 3).astype(np.float64)
        vel_curr = np.asarray(P2[1]).reshape(-1, 3).astype(np.float32)
    del X2, P2
    print(f"  [streaming] step 1 gather/cast took {time()-t_gather:.1f}s  "
          f"n_part={pos_prev.shape[0]:,}")

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
        t_nbody = time()
        if return_jax:
            # pos_prev is a sharded JAX float32 array (all 4 GPUs), matching
            # DiscoDJ's internal grid sharding.  Pass directly.
            dj_i = dj.with_external_ics(pos=pos_prev, vel=vel_curr)
        else:
            dj_i = dj.with_external_ics(
                pos=jnp.array(pos_prev.astype(np.float32)),
                vel=jnp.array(vel_curr),
            )
        X_n, P_n, _ = dj_i.run_nbody(
            a_ini=a_s, a_end=a_e, n_steps=1,
            time_var=np.array([a_s, a_e], dtype=np.float64),
            collect_all=False, **_kw,
        )
        print(f"  [streaming] step {i+1}/{n_steps}  a=[{a_s:.4f},{a_e:.4f}]  "
              f"nbody took {time()-t_nbody:.1f}s")
        t_gather = time()
        if return_jax:
            # Keep native sharding; same reason as step 1.
            pos_curr = X_n.reshape(-1, 3).astype(jnp.float32)
            vel_curr = P_n.reshape(-1, 3).astype(jnp.float32)
        else:
            pos_curr = np.asarray(X_n).reshape(-1, 3).astype(np.float64)
            vel_curr = np.asarray(P_n).reshape(-1, 3).astype(np.float32)
        del X_n, P_n
        print(f"  [streaming] step {i+1} gather/cast took {time()-t_gather:.1f}s")

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
    shell_chunk_size: int = 2_000_000,
    use_gpu: bool = True,
    pre_steps: int | None = None,
    z_pre_ini: float = 99.0,
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
    use_gpu    : if True (default) use the JAX GPU kernel for shell
                 accumulation (``accumulate_shell_jax``).  Avoids the
                 device→host transfer and float64 cast; runs distances,
                 masking, interpolation, HEALPix indexing and scatter-add
                 entirely on the GPU.  Set False to fall back to the original
                 NumPy CPU path (useful for debugging or CPU-only machines).
    metainfo_path : optional path to CosmoGridV1_metainfo.h5.
        If provided, shell boundaries are taken from the metainfo z_bins
        and output is a single NPZ file matching CosmoGridV1 format.
        If None, falls back to individual FITS files per shell.
    cosmo_key  : which cosmology entry in metainfo to use (e.g. "cosmo_000001").
        Defaults to the first key alphabetically.
    output_npz : output path for the NPZ file when metainfo_path is given.
        Defaults to ``output_dir / f"shells_nside={nside}.npz"``.
    pre_steps  : if given, run this many N-body steps from z=z_pre_ini to
        a_steps[0] *before* the main lightcone run (no shells accumulated).
        The evolved positions/velocities are then used as ICs for both the
        streaming and collect_all paths via ``with_external_ics``.
        Useful when the LPT ICs are seeded at high redshift (z=99) but the
        lightcone only starts at z_max (e.g. z=3.5).
    z_pre_ini  : starting redshift for the pre-step phase (default 99).
    """

    import jax  # needed here for process_index() / process_count() guards

    # Build comoving-distance function from DiscoDJ cosmology
    chi_of_a, _ = make_chi_of_a(dj.cosmo)

    builder = LightconeShellBuilder(
        boxsize=dj.boxsize,
        chi_of_a=chi_of_a,
        nside=nside,
        z_min=z_min,
        z_max=z_max,
        interpolate=interpolate,
        particle_chunk_size=shell_chunk_size,
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
        # Recompute comoving boundaries using DISCO-DJ's own chi_of_a so that
        # the shell overlap test is consistent with r_step_hi/r_step_lo.
        # The stored lower_com/upper_com come from a different integration and
        # are ~7 Mpc/h off, causing each step to overlap 2 shells (2× counts).
        _a_lower = 1.0 / (1.0 + shell_info_meta['lower_z'].astype(np.float64))
        _a_upper = 1.0 / (1.0 + shell_info_meta['upper_z'].astype(np.float64))
        meta_lower = chi_of_a(_a_lower).astype(np.float64)
        meta_upper = chi_of_a(_a_upper).astype(np.float64)
        meta_ids   = shell_info_meta['shell_id'].astype(int)
        print(f"[run_with_shells] NPZ mode: {n_meta_shells} shells from "
              f"metainfo key={cosmo_key}  (chi recomputed with DISCO-DJ chi_of_a)")
        if output_npz is None:
            output_npz = output_dir / f"shells_nside={nside}.npz"
        else:
            output_npz = Path(output_npz)

    n_steps = len(a_steps) - 1
    a_ini = float(a_steps[0])
    a_end = float(a_steps[-1])
    mode_str = "streaming (O(N))" if streaming else "collect_all (O(n_steps·N))"
    accel_str = "GPU (JAX kernel)" if use_gpu else "CPU (NumPy)"
    print(f"Running DiscoDJ with {n_steps} steps, "
          f"a={a_ini:.4f}→{a_end:.4f}  [{mode_str}]  shell_accum={accel_str}")
    t0 = time()
    t_nbody_total = 0.0
    t_shell_total = 0.0

    # ── Optional pre-step burn-in: z_pre_ini → a_steps[0], no shells ─────
    # Evolves the LPT/BullFrog ICs to the start of the lightcone window so
    # the main run (streaming or collect_all) begins with realistic positions.
    if pre_steps is not None and pre_steps > 0:
        import jax.numpy as jnp
        a_pre_ini = 1.0 / (1.0 + float(z_pre_ini))
        print(f"[run_with_shells] Pre-steps: {pre_steps} steps  "
              f"a=[{a_pre_ini:.5f}→{a_ini:.5f}]  "
              f"z=[{z_pre_ini:.1f}→{1.0/a_ini-1.0:.2f}]  (no shell accumulation)")
        t_pre = time()
        X_pre, P_pre, _ = dj.run_nbody(
            a_ini=a_pre_ini,
            a_end=a_ini,
            n_steps=pre_steps,
            res_pm=res_pm,
            stepper=stepper,
            method=method,
            collect_all=False,
            return_displacement=False,
            light_cone=False,
            chunk_size=chunk_size,
            **nbody_kwargs,
        )
        n_part_pre = int(X_pre.reshape(-1, 3).shape[0])
        print(f"[run_with_shells] Pre-steps done in {time()-t_pre:.1f}s  "
              f"n_part={n_part_pre:,}")
        dj = dj.with_external_ics(
            pos=X_pre.reshape(-1, 3).astype(jnp.float32),
            vel=P_pre.reshape(-1, 3).astype(jnp.float32),
        )
        del X_pre, P_pre

    # ── Choose step data source ───────────────────────────────────────────
    if streaming:
        step_gen = _streaming_steps(
            dj, a_steps, res_pm, stepper, method, chunk_size, nbody_kwargs,
            return_jax=use_gpu)
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
                              np.asarray(pos_prev).astype(np.float32), a_prev)
            continue
        if z_prev < builder.z_min:
            # Below z_min; skip but still optionally save snapshot
            if snap_dir and jax.process_index() == 0:
                save_snapshot(snap_dir / f"snap_{i:05d}.npz",
                              np.asarray(pos_prev).astype(np.float32), a_prev)
            continue

        if snap_dir and jax.process_index() == 0:
            save_snapshot(snap_dir / f"snap_{i:05d}.npz",
                          np.asarray(pos_prev).astype(np.float32), a_prev)

        t_step_wall = time() - t0
        t_shell_start = time()

        if shell_info_meta is not None:
            # ── NPZ mode: assign particles to metainfo z_bins ────────────────
            overlap = np.where(
                (meta_lower < r_step_hi) & (meta_upper > r_step_lo)
            )[0]
            if jax.process_index() == 0:
                print(f"  step {i+1}/{n_steps}  z=[{z_curr:.3f},{z_prev:.3f}]  "
                      f"r=[{r_step_lo:.1f},{r_step_hi:.1f}] Mpc/h  "
                      f"overlapping_shells={len(overlap)}  "
                      f"wall={t_step_wall:.1f}s")
            # Prepare device arrays once for all overlapping shells in this
            # step — avoids repeating the GPU→CPU→GPU gather/cast per shell.
            _precast = None
            if use_gpu and len(overlap) > 0:
                _precast = builder.prepare_device_maps(pos_prev, pos_curr)
            for idx in overlap:
                sid = int(meta_ids[idx])
                if use_gpu:
                    shell_map = builder.accumulate_shell_jax(
                        pos_prev, pos_curr, a_prev, a_curr,
                        r_lo_override=float(meta_lower[idx]),
                        r_hi_override=float(meta_upper[idx]),
                        _precast=_precast,
                    )
                else:
                    shell_map = builder.accumulate_shell(
                        pos_prev, pos_curr, a_prev, a_curr,
                        r_lo_override=float(meta_lower[idx]),
                        r_hi_override=float(meta_upper[idx]),
                    )
                shells_array[sid] += shell_map.astype(np.int32)
            dt_shell = time() - t_shell_start
            t_shell_total += dt_shell
            if jax.process_index() == 0:
                print(f"  step {i+1}/{n_steps}  shell_accum={dt_shell:.1f}s  "
                      f"(cumulative: shell={t_shell_total:.1f}s  "
                      f"total_wall={time()-t0:.1f}s)")
            shells_written += len(overlap)
        else:
            # ── FITS mode ────────────────────────────────────────────────────
            if jax.process_index() == 0:
                print(f"  shell {i+1}/{n_steps}  z=[{z_curr:.3f},{z_prev:.3f}]  "
                      f"r=[{r_step_lo:.1f},{r_step_hi:.1f}] Mpc/h  "
                      f"wall={t_step_wall:.1f}s")
            if use_gpu:
                shell = builder.accumulate_shell_jax(pos_prev, pos_curr, a_prev, a_curr)
            else:
                shell = builder.accumulate_shell(pos_prev, pos_curr, a_prev, a_curr)
            if jax.process_index() == 0:
                fname = save_shell_fits(shell, z_lo=z_curr, z_hi=z_prev,
                                        output_dir=output_dir, prefix=prefix)
            dt_shell = time() - t_shell_start
            t_shell_total += dt_shell
            if jax.process_index() == 0:
                print(f"  shell {i+1}/{n_steps}  n_part={int(shell.sum())}  "
                      f"→ {fname.name}  shell_accum={dt_shell:.1f}s  "
                      f"(cumulative: shell={t_shell_total:.1f}s  "
                      f"total_wall={time()-t0:.1f}s)")
            shells_written += 1

    # Save final snapshot
    if snap_dir and last_pos_curr is not None:
        if jax.process_index() == 0:
            save_snapshot(snap_dir / f"snap_{n_steps:05d}.npz",
                          np.asarray(last_pos_curr).astype(np.float32), last_a_curr)

    t_total_wall = time() - t0
    if shell_info_meta is not None:
        # Save combined NPZ — only process 0 writes; shells_array already has
        # the global per-shell counts (allreduced inside accumulate_shell_jax).
        if jax.process_index() == 0:
            save_shells_npz(shells_array, shell_info_meta, output_npz, prefix=prefix)
            print(f"Done. NPZ saved → {output_npz}")
    else:
        if jax.process_index() == 0:
            print(f"Done. {shells_written} shells written to {output_dir}")
    if jax.process_index() == 0:
        print(f"[run_with_shells] total wall time: {t_total_wall:.1f}s  "
              f"shell_accum: {t_shell_total:.1f}s  "
              f"other (nbody+IO): {t_total_wall - t_shell_total:.1f}s")

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
