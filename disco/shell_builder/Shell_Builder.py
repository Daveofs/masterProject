import numpy as np
import healpy as hp
from time import time
from shell_builder.utils import compute_replica_offsets, vec2pix_ring_jax


# ---------------------------------------------------------------------------
# Module-level JIT kernel cache (compiled once, reused for all shells)
# ---------------------------------------------------------------------------
_JAX_SHELL_FORI_KERNEL = None


def _compile_accumulate_kernel():
    """
    Return (building once) a JIT-compiled kernel that processes ALL active
    replicas inside a single ``jax.lax.fori_loop``.

    Unlike a per-replica Python loop, this submits a single XLA program that
    iterates over all replicas internally, eliminating Python dispatch overhead.
    The trip count is a dynamic JAX int32 scalar so the same compiled code
    handles any number of active replicas.
    """
    global _JAX_SHELL_FORI_KERNEL
    if _JAX_SHELL_FORI_KERNEL is not None:
        return _JAX_SHELL_FORI_KERNEL

    from functools import partial
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnames=['nside', 'npix'])
    def _fori_kernel(X0, X1, rep_offsets, n_active, r_lo, r_hi,
                     r_lo_meta, r_hi_meta,
                     nside, npix):
        """
        Accumulate shell contributions for ``n_active`` periodic replicas.

        Implements pkdgrav3's exact moving-lightcone crossing condition
        (lightcone.cxx ``pkdProcessLightCone`` ~line 240):

            vx = (r_hi - r0) / ((r_hi - r_lo) + (r1 - r0))

        Accept iff vx ∈ [0, 1) — each particle is caught in exactly one step.

        Parameters
        ----------
        X0, X1        : (N, 3) float32  positions relative to observer [Mpc/h]
        rep_offsets   : (R, 3) float32  padded replica offset table
        n_active      : int32 scalar (dynamic) – replicas to process
        r_lo, r_hi    : float32  N-body step lightcone boundaries [Mpc/h]
        r_lo_meta, r_hi_meta : float32  metainfo shell filter boundaries
        nside, npix : compile-time static constants
        """
        def body(i, shell):
            d  = rep_offsets[i]
            R0 = X0 + d
            R1 = X1 + d
            r0 = jnp.sqrt(jnp.sum(R0 * R0, axis=1))
            r1 = jnp.sqrt(jnp.sum(R1 * R1, axis=1))

            denom_lc   = (r_hi - r_lo) + (r1 - r0)
            safe_denom = jnp.where(jnp.abs(denom_lc) > 1e-10, denom_lc,
                                   jnp.float32(1e-10))
            vx        = (r_hi - r0) / safe_denom
            direction = (1.0 - vx[:, None]) * R0 + vx[:, None] * R1
            r_cross   = jnp.sqrt(jnp.sum(direction * direction, axis=1))
            mask = ((vx >= 0.0) & (vx < 1.0) &
                    (r_cross >= r_lo_meta) & (r_cross < r_hi_meta))

            safe_dir = jnp.where(
                mask[:, None],
                direction,
                jnp.broadcast_to(jnp.array([1., 0., 0.], dtype=jnp.float32),
                                 direction.shape),
            )
            pix  = vec2pix_ring_jax(nside, safe_dir[:, 0],
                                    safe_dir[:, 1], safe_dir[:, 2])
            pix  = jnp.clip(pix, 0, npix - 1)
            # Spread zero-weight atomics uniformly to avoid hot-address serialisation
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


class LightconeShellBuilder:
    """
    Accumulates HEALPix lightcone shells from consecutive particle snapshots.

    Parameters
    ----------
    boxsize   : float   box side length [Mpc/h]
    chi_of_a  : callable   a -> chi [Mpc/h]
    nside     : int         HEALPix Nside (default 2048)
    z_max     : float       maximum redshift of the lightcone (default 3.5)
    Crossing positions are linearly interpolated between consecutive snapshots.
    """

    def __init__(self,
                 boxsize: float,
                 chi_of_a,
                 nside: int = 2048,
                 z_min: float = 0.0,
                 z_max: float = 3.5,
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
        self._rep_ints = compute_replica_offsets(self.chi_max, boxsize)  # shape (N,3)
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

        self.particle_chunk_size = int(particle_chunk_size)

    def accumulate_shell_cpu(self,
                         pos_prev: np.ndarray,
                         pos_curr: np.ndarray,
                         a_prev: float,
                         a_curr: float,
                         r_lo_override: float | None = None,
                         r_hi_override: float | None = None,
                         verbose: bool = True) -> np.ndarray:
        """
        CPU PATH
        Compute a single HEALPix shell map from two consecutive snapshots

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
        # Step boundaries used for the pkdgrav vx crossing formula.
        # These MUST be the DISCO N-body step boundaries (chi(a_prev/a_curr)),
        # NOT the per-shell metainfo boundaries.  The vx ∈ [0,1) condition is
        # a partition over the full step interval: each particle is caught in
        # exactly ONE step.  If we substituted per-shell meta boundaries here,
        # an outward-moving particle near a shell boundary could satisfy
        # vx ∈ [0,1) for two consecutive shells in the same step → double count.
        r_step_hi = float(self.chi_of_a(a_prev))
        r_step_lo = float(self.chi_of_a(a_curr))

        # Metainfo shell filter (NPZ mode): overrides only used for r_cross check.
        # After the vx crossing is confirmed, r_cross selects which shell to
        # assign the particle to (pkdgrav's addToLightCone shell-assignment).
        r_hi_meta = r_hi_override if r_hi_override is not None else r_step_hi
        r_lo_meta = r_lo_override if r_lo_override is not None else r_step_lo

        if r_step_hi > self.chi_max:
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
        t_healpix = 0.0
        n_particles_placed = 0

        for rep_idx, d in enumerate(self._rep_offsets):
            # Fast reject using step-boundary replica radial bounds.
            if r_step_hi < self._rep_rmin[rep_idx] or r_step_lo > self._rep_rmax[rep_idx]:
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

                # pkdgrav moving-lightcone crossing (lightcone.cxx):
                # vx = (r_step_hi - r0) / ((r_step_hi - r_step_lo) + (r1 - r0))
                # Accept iff vx ∈ [0, 1): each particle caught in exactly one step.
                _t = time()
                denom_lc   = (r_step_hi - r_step_lo) + (r1 - r0)
                safe_denom = np.where(np.abs(denom_lc) > 1e-10, denom_lc, 1e-10)
                vx         = (r_step_hi - r0) / safe_denom
                direction  = (1.0 - vx[:, None]) * R0 + vx[:, None] * R1
                r_cross    = np.sqrt(np.sum(direction * direction, axis=1,
                                dtype=np.float32))
                # Accept: lightcone-crossing within this step AND
                # crossing radius within the metainfo shell (NPZ mode filter)
                mask = ((vx >= 0.0) & (vx < 1.0) &
                    (r_cross >= r_lo_meta) & (r_cross < r_hi_meta))
                direction = direction[mask]
                t_mask += time() - _t
                if not np.any(mask):
                    continue

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
                f"    [accumulate_shell] r_step=[{r_step_lo:.1f},{r_step_hi:.1f}]  "
                f"r_meta=[{r_lo_meta:.1f},{r_hi_meta:.1f}] Mpc/h  "
                f"n_part={n_part:,}  chunks={n_chunks}  "
                f"replicas={n_rep_kept}/{n_rep_total}  "
                f"placed={n_particles_placed:,}  "
                f"| cast={t_cast_done-t_cast:.2f}s  "
                f"rsq={t_r_sq:.2f}s  mask={t_mask:.2f}s  "
                f"healpix={t_healpix:.2f}s  "
                f"total={t_total:.2f}s"
            )

        return shell_map

    # ------------------------------------------------------------------
    def accumulate_shell_gpu(self,
                             pos_prev_jax,
                             pos_curr_jax,
                             a_prev: float,
                             a_curr: float,
                             r_lo_override: float | None = None,
                             r_hi_override: float | None = None) -> np.ndarray:
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

        Returns
        -------
        shell_map : (npix,) float32 numpy array
        """
        import jax
        import jax.numpy as jnp

        # ── Step boundaries used for the pkdgrav vx crossing formula ─────────
        # These MUST be the DISCO N-body step boundaries (chi(a_prev/a_curr)),
        # NOT the per-shell metainfo boundaries.  The vx ∈ [0,1) condition is
        # a partition over the full step interval: each particle is caught in
        # exactly ONE step.  If we substituted per-shell meta boundaries here,
        # an outward-moving particle near a shell boundary could satisfy
        # vx ∈ [0,1) for two consecutive shells in the same step → double count.
        r_step_hi = float(self.chi_of_a(a_prev))
        r_step_lo = float(self.chi_of_a(a_curr))

        # ── Metainfo shell filter boundaries ─────────────────────────────────
        # For FITS mode: r_lo/r_hi_override are None → same as step boundaries.
        # For NPZ mode:  r_lo/r_hi_override are the metainfo shell comoving
        # limits; the kernel only counts particles whose crossing radius falls
        # within this range (pkdgrav's addToLightCone shell-assignment step).
        r_hi_meta = r_hi_override if r_hi_override is not None else r_step_hi
        r_lo_meta = r_lo_override if r_lo_override is not None else r_step_lo

        if r_step_hi > self.chi_max:
            return np.zeros(self.npix, dtype=np.float32)
        if 1.0 / a_prev - 1.0 < self.z_min:
            return np.zeros(self.npix, dtype=np.float32)

        from concurrent.futures import ThreadPoolExecutor

        fori_kernel = _compile_accumulate_kernel()

        # ---- gather/cast: bring positions to each local device -----------
        t_gather = time()
        devs   = jax.local_devices()
        n_devs = len(devs)
        X0_flat = pos_prev_jax.reshape(-1, 3).astype(jnp.float32)
        X1_flat = pos_curr_jax.reshape(-1, 3).astype(jnp.float32)
        if jax.process_count() > 1:
            X0_np = np.concatenate(
                [np.asarray(s.data) for s in X0_flat.addressable_shards], axis=0)
            X1_np = np.concatenate(
                [np.asarray(s.data) for s in X1_flat.addressable_shards], axis=0)
        else:
            X0_np = np.array(X0_flat)
            X1_np = np.array(X1_flat)
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
              f"r_step=[{r_step_lo:.1f},{r_step_hi:.1f}]  "
              f"r_meta=[{r_lo_meta:.1f},{r_hi_meta:.1f}] Mpc/h  "
              f"n_part={n_part:,}  "
              f"n_devs={n_devs}  part/dev={chunk:,}")

        # ---- pre-collect kept replica offsets ----------------------------
        # Use STEP boundaries for replica culling: replicas whose nearest point
        # is beyond r_step_hi (or closest point above r_step_lo) cannot have any
        # particle caught by the lightcone during this step.
        n_rep_total = len(self._rep_offsets)
        kept_reps = []
        for rep_idx, d in enumerate(self._rep_offsets):
            if r_step_hi < self._rep_rmin[rep_idx] or r_step_lo > self._rep_rmax[rep_idx]:
                continue
            kept_reps.append(d)
        n_rep_kept = len(kept_reps)

        # ---- dispatch per-device fori_loop in parallel threads -----------
        r_lo_f32      = np.float32(r_step_lo)
        r_hi_f32      = np.float32(r_step_hi)
        r_lo_meta_f32 = np.float32(r_lo_meta)
        r_hi_meta_f32 = np.float32(r_hi_meta)
        nside_    = self.nside
        npix_     = self.npix

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
                r_lo_f32, r_hi_f32,
                r_lo_meta_f32, r_hi_meta_f32,
                nside_, npix_,
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

        print(f"    [jax_shell] r_step=[{r_step_lo:.1f},{r_step_hi:.1f}]  "
              f"r_meta=[{r_lo_meta:.1f},{r_hi_meta:.1f}] Mpc/h  gpu={t_gpu:.2f}s  "
              f"replicas={n_rep_kept}/{n_rep_total}  "
              f"placed={int(shell_map.sum()):,}")
        return shell_map

