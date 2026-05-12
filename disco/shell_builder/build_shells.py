from __future__ import annotations

import os
from pathlib import Path
from time import time

from shell_builder.Shell_Builder import LightconeShellBuilder
from shell_builder.utils import make_comoving_distance_fn

import numpy as np
import healpy as hp

# ---------------------------------------------------------------------------
# Simulation-in-a-loop runner
# ---------------------------------------------------------------------------

def build_shells(
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
    metainfo_path: str | Path | None = None,
    cosmo_key: str | None = None,
    output_npz: str | Path | None = None,
    particle_chunk_size: int = 2_000_000,
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
                 is set to 69 if metainfo_path is given (see also below)
    res_pm     : PM grid resolution 
    output_dir : directory for output files (FITS or NPZ)
    snap_dir   : if given, also save intermediate snapshots as NPZ
    use_gpu    : if True (default) use the JAX GPU kernel for shell
                 accumulation (``accumulate_shell_gpu``).  Avoids the
                 device→host transfer and float64 cast; runs distances,
                 masking, interpolation, HEALPix indexing and scatter-add
                 entirely on the GPU.  Set False to fall back to the original
                 NumPy CPU path (useful for debugging or CPU-only machines).
    metainfo_path : optional path to CosmoGridV1_metainfo.h5 (required if you 
        want to build shells matching the CosmoGridV1 z_bins)
        If provided, shell boundaries are taken from the metainfo z_bins
        and output is a single NPZ file matching CosmoGridV1 format.
        If None, falls back to individual FITS files per shell.
    cosmo_key  : which cosmology entry in metainfo to use (e.g. "cosmo_000001").
        Defaults to the first key alphabetically.
    output_npz : output path for the NPZ file when metainfo_path is given.
        Defaults to ``output_dir / f"shells_nside={nside}.npz"``.
    pre_steps  : if given, run this many N-body steps from z=z_pre_ini to
        a_steps[0] *before* the main lightcone run (no shells accumulated).
        The evolved positions/velocities are then used as ICs for the main
        shell-building path via ``with_external_ics``.
        Useful when the LPT ICs are seeded at high redshift (z=99) but the
        lightcone only starts at z_max (e.g. z=3.5).
    z_pre_ini  : starting redshift for the pre-step phase (default 99).
    """

    import jax  # needed here for process_index() / process_count() guards

    # Build comoving-distance function from DiscoDJ cosmology
    chi_of_a, _ = make_comoving_distance_fn(dj.cosmo)

    builder = LightconeShellBuilder(
        boxsize=dj.boxsize,
        chi_of_a=chi_of_a,
        nside=nside,
        z_min=z_min,
        z_max=z_max,
        particle_chunk_size=particle_chunk_size,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if snap_dir:
        snap_dir = Path(snap_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metainfo shell bins (NPZ mode) from cosmogrid ──
    shell_info_meta = None
    npix = builder.npix
    if metainfo_path is not None:
        shell_info_meta, cosmo_key = load_metainfo_shell_bins(
            metainfo_path, cosmo_key)
        n_meta_shells = len(shell_info_meta)
        shells_array = np.zeros((n_meta_shells, npix), dtype=np.int32)
        # Use the stored pkdgrav-computed comoving boundaries directly so that
        # our r_cross filter matches pkdgrav's shell assignment exactly.
        meta_lower = shell_info_meta['lower_com'].astype(np.float64)
        meta_upper = shell_info_meta['upper_com'].astype(np.float64)
        meta_ids   = shell_info_meta['shell_id'].astype(int)
        print(f"[build_shells] NPZ mode: {n_meta_shells} shells from "
              f"metainfo key={cosmo_key}  (using stored pkdgrav lower_com/upper_com)")
        if output_npz is None:
            output_npz = output_dir / f"shells_nside={nside}.npz"
        else:
            output_npz = Path(output_npz)

    n_steps = len(a_steps) - 1
    a_ini = float(a_steps[0])
    a_end = float(a_steps[-1])
    accel_str = "GPU (JAX kernel)" if use_gpu else "CPU (NumPy)"
    print(f"Running DiscoDJ with {n_steps} steps, "
            f"a={a_ini:.4f}→{a_end:.4f}  [streaming (O(N))]  shell_accum={accel_str}")
    t0 = time()
    t_nbody_total = 0.0
    t_shell_total = 0.0

    # ── Optional pre-step burn-in: z_pre_ini → a_steps[0], NO SHELLS (from z=99 to 3.5) ─────
    # Evolves the LPT/BullFrog ICs to the start of the lightcone window so
    # the main run (streaming or collect_all) begins with realistic positions.
    if pre_steps is not None and pre_steps > 0:
        import jax.numpy as jnp
        a_pre_ini = 1.0 / (1.0 + float(z_pre_ini))
        print(f"[build_shells] Pre-steps: {pre_steps} steps  "
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
            **nbody_kwargs,
        )
        n_part_pre = int(X_pre.reshape(-1, 3).shape[0])
        print(f"[build_shells] Pre-steps done in {time()-t_pre:.1f}s  "
              f"n_part={n_part_pre:,}")
        dj = dj.with_external_ics(
            pos=X_pre.reshape(-1, 3).astype(jnp.float32),
            vel=P_pre.reshape(-1, 3).astype(jnp.float32),
        )
        del X_pre, P_pre

    # ── Step data source (always streaming, O(N) memory) ─────────────────
    step_gen = _iter_nbody_steps(
        dj, a_steps, res_pm, stepper, method, nbody_kwargs,
        return_jax=use_gpu)

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
                save_particle_snapshot(snap_dir / f"snap_{i:05d}.npz",
                              np.asarray(pos_prev).astype(np.float32), a_prev)
            continue
        if z_prev < builder.z_min:
            # Below z_min; skip but still optionally save snapshot
            if snap_dir and jax.process_index() == 0:
                save_particle_snapshot(snap_dir / f"snap_{i:05d}.npz",
                              np.asarray(pos_prev).astype(np.float32), a_prev)
            continue

        if snap_dir and jax.process_index() == 0:
            save_particle_snapshot(snap_dir / f"snap_{i:05d}.npz",
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
            for idx in overlap:
                sid = int(meta_ids[idx])
                if use_gpu:
                    shell_map = builder.accumulate_shell_gpu(
                        pos_prev, pos_curr, a_prev, a_curr,
                        r_lo_override=float(meta_lower[idx]),
                        r_hi_override=float(meta_upper[idx]),
                    )
                else:
                    shell_map = builder.accumulate_shell_cpu(
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
            # ── FITS mode, all shells are in seperatly stored in file.fits ──
            if jax.process_index() == 0:
                print(f"  shell {i+1}/{n_steps}  z=[{z_curr:.3f},{z_prev:.3f}]  "
                      f"r=[{r_step_lo:.1f},{r_step_hi:.1f}] Mpc/h  "
                      f"wall={t_step_wall:.1f}s")
            if use_gpu:
                shell = builder.accumulate_shell_gpu(pos_prev, pos_curr, a_prev, a_curr)
            else:
                shell = builder.accumulate_shell_cpu(pos_prev, pos_curr, a_prev, a_curr)
            if jax.process_index() == 0:
                fname = write_shell_fits(shell, z_lo=z_curr, z_hi=z_prev,
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
            save_particle_snapshot(snap_dir / f"snap_{n_steps:05d}.npz",
                          np.asarray(last_pos_curr).astype(np.float32), last_a_curr)

    t_total_wall = time() - t0
    if shell_info_meta is not None:
        # Save combined NPZ — only process 0 writes; shells_array already has
        # the global per-shell counts (allreduced inside accumulate_shell_gpu).
        if jax.process_index() == 0:
            write_shells_npz(shells_array, shell_info_meta, output_npz, prefix=prefix)
            print(f"Done. NPZ saved → {output_npz}")
    else:
        if jax.process_index() == 0:
            print(f"Done. {shells_written} shells written to {output_dir}")
    if jax.process_index() == 0:
        print(f"[build_shells] total wall time: {t_total_wall:.1f}s  "
              f"shell_accum: {t_shell_total:.1f}s  "
              f"other (nbody+IO): {t_total_wall - t_shell_total:.1f}s")

    return shells_written

# ---------------------------------------------------------------------------
# Streaming step generator  (O(N) memory, same approach as pkdgrav3)
# ---------------------------------------------------------------------------

def _iter_nbody_steps(dj, a_steps, res_pm, stepper, method, nbody_kwargs,
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
        with ``LightconeShellBuilder.accumulate_shell_gpu``.
    """
    import jax
    import jax.numpy as jnp

    _kw = dict(res_pm=res_pm, stepper=stepper, method=method,
               light_cone=False,
               return_displacement=False, **nbody_kwargs)

    n_steps = len(a_steps) - 1

    # ── Bootstrap first interval without collect_all=True ─────────────────
    # Symplectic currently fails in DiscoDJ when collect_all=True and n_steps=1
    # (broadcast mismatch in velocity_to_Pi).  Build the first slab by taking
    # IC positions at a0 from dj._ics and evolving one step to a1 with
    # collect_all=False.
    a0, a1 = float(a_steps[0]), float(a_steps[1])

    t_ic = time()
    if "pos" in dj._ics:
        pos0_src = dj._ics["pos"]
    else:
        # Fallback for IC modes that do not store explicit positions.
        if jax.process_index() == 0:
            print("  [streaming] IC positions not found in dj._ics; "
                  "materialising X(a_ini) via zero-span run_nbody call")
        X0_tmp, _, _ = dj.run_nbody(
            a_ini=a0, a_end=a0, n_steps=1,
            time_var=np.array([a0, a0], dtype=np.float64),
            collect_all=False, **_kw,
        )
        pos0_src = X0_tmp
        del X0_tmp

    if return_jax:
        pos_prev = jnp.asarray(pos0_src).reshape(-1, 3).astype(jnp.float32)
    else:
        pos_prev = np.asarray(pos0_src).reshape(-1, 3).astype(np.float64)
    print(f"  [streaming] step 1/{n_steps}  IC snapshot gather/cast took {time()-t_ic:.1f}s  "
          f"n_part={pos_prev.shape[0]:,}")

    t_nbody = time()
    X1, P1, _ = dj.run_nbody(
        a_ini=a0, a_end=a1, n_steps=1,
        time_var=np.array([a0, a1], dtype=np.float64),
        collect_all=False, **_kw,
    )
    print(f"  [streaming] step 1/{n_steps}  a=[{a0:.4f},{a1:.4f}]  "
          f"nbody took {time()-t_nbody:.1f}s")
    t_gather = time()
    if return_jax:
        pos_curr = X1.reshape(-1, 3).astype(jnp.float32)
        vel_curr = P1.reshape(-1, 3).astype(jnp.float32)
    else:
        pos_curr = np.asarray(X1).reshape(-1, 3).astype(np.float64)
        vel_curr = np.asarray(P1).reshape(-1, 3).astype(np.float32)
    del X1, P1
    print(f"  [streaming] step 1 gather/cast took {time()-t_gather:.1f}s  "
          f"n_part={pos_prev.shape[0]:,}")

    yield pos_prev, pos_curr, a0, a1
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


def save_particle_snapshot(path: Path | str, pos: np.ndarray, a: float):
    """Save positions + scale factor to NPZ."""
    np.savez_compressed(path, pos=pos.astype(np.float32), a=np.float32(a))


def load_particle_snapshot(path: Path | str):
    """Load snapshot; returns (pos: float32 Mpc/h, a: float)."""
    d = np.load(path)
    return d['pos'], float(d['a'])


def write_shell_fits(shell_map: np.ndarray, z_lo: float, z_hi: float,
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

def load_metainfo_shell_bins(metainfo_path, cosmo_key: str | None = None):
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


def write_shells_npz(shells_array: np.ndarray,
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

