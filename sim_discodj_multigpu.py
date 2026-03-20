"""
Multi-GPU DISCO-DJ N-body simulation script.

Supports:
  - External ICs from a PKDGRAV tipsy file (--use-external-ics --ic-file <path>)
  - Internal LPT ICs (default, uses Eisenstein-Hu transfer function)

Multi-GPU: run via SLURM srun (one task per GPU).  JAX distributed init is
handled automatically when SLURM env vars are present.

Simulation parameters are passed as CLI args; see --help for the full list.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import time

# ---------------------------------------------------------------------------
# Parse args BEFORE importing JAX so we can set env-vars in time.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-GPU DISCO-DJ N-body simulation (vir_env_discodj)"
    )

    # --- device ---
    p.add_argument("--mode", choices=["gpu", "cpu"], default="gpu",
                   help="Compute backend (default: gpu)")

    # --- simulation grid / cosmology ---
    p.add_argument("--res",     type=int,   required=True,
                   help="Particle grid resolution per axis (N_part = res^3)")
    p.add_argument("--res-pm",  type=int,   required=True,
                   help="PM force grid resolution per axis")
    p.add_argument("--boxsize", type=float, required=True,
                   help="Simulation box size [Mpc/h]")
    p.add_argument("--cosmo",   type=str,   default="Planck15",
                   help="DISCO-DJ cosmology preset string (default: Planck15)")

    # --- time integration ---
    p.add_argument("--a-ini",   type=float, required=True,
                   help="Initial scale factor")
    p.add_argument("--a-end",   type=float, default=1.0,
                   help="Final scale factor (default: 1.0)")
    p.add_argument("--n-steps", type=int,   default=10,
                   help="Number of N-body timesteps (default: 10)")
    p.add_argument("--stepper",
                   choices=["bullfrog", "fastpm", "symplectic"],
                   default="bullfrog",
                   help="Time integrator (default: bullfrog)")
    p.add_argument("--time-var", type=str, default="D",
                   help="Time variable: 'a', 'lna', or 'D' (default: D)")

    # --- force solver ---
    p.add_argument("--method",
                   choices=["pm", "nufftpm"], default="pm",
                   help="Force solver method (default: pm)")
    p.add_argument("--antialias",           type=int, default=0)
    p.add_argument("--grad-kernel-order",   type=int, default=4)
    p.add_argument("--laplace-kernel-order",type=int, default=0)
    p.add_argument("--n-resample",          type=int, default=1)
    p.add_argument("--deconvolve",          action="store_true")
    p.add_argument("--num-chunks",          type=int, default=8,
                   help="Number of scatter/gather chunks (chunk_size = res^3 / num_chunks)")

    # --- lightcone (DiscoDJ built-in, low-z only) ---
    p.add_argument("--lightcone", action="store_true",
                   help="Enable DiscoDJ built-in lightcone mode (limited to ~Lbox/2 radius)")

    # --- snapshot-based shell lightcone (arbitrary z_max) ---
    p.add_argument("--build-shells", action="store_true",
                   help="Accumulate HEALPix lightcone shells from per-step snapshots "
                        "(works to high z; uses build_lightcone_shells.py)")
    p.add_argument("--shells-output-dir", type=Path, default=None,
                   help="Output directory for shell FITS files (default: output-dir/shells)")
    p.add_argument("--shells-snap-dir",   type=Path, default=None,
                   help="If set, also save per-step snapshot NPZ files here")
    p.add_argument("--shells-nside",      type=int, default=2048,
                   help="HEALPix Nside for shell maps (default: 2048)")
    p.add_argument("--shells-z-max",      type=float, default=3.5,
                   help="Maximum redshift for lightcone shells (default: 3.5)")
    p.add_argument("--shells-prefix",     type=str, default="CosmoML",
                   help="Filename prefix for shell FITS files (default: CosmoML)")

    # --- ICs ---
    p.add_argument("--ic-file", type=Path, required=False,
                   help="Path to PKDGRAV tipsy IC file (external ICs)")
    p.add_argument("--use-internal-ics", action="store_true",
                   help="Generate internal ICs (ngenic-like white noise) instead of using an external IC file")
    p.add_argument("--ngenic-seed", type=int, default=180723,
                   help="Seed for internal ngenic white-noise field (default: 180723)")
    p.add_argument("--n-order", type=int, default=3,
                   help="LPT order for internal IC generation (default: 3)")

    # --- outputs ---
    p.add_argument("--save-final", type=Path, default=None,
                   help="Save final pos/vel/a_hist as NPZ to this path")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for output plots (default: <project>/outputs/plots)")
    p.add_argument("--plot", action="store_true",
                   help="Generate a 2D density slice from the final snapshot")

    return p.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# DISCO-DJ/scripts/ must be on sys.path before ANY discodj import, because
# the editable install maps discodj.core.multigpu_utils → scripts/utils.py,
# which does bare `import utils_jens` (a peer module in the same directory).
# ---------------------------------------------------------------------------
_discodj_scripts = Path("/cluster/work/refregier/damrein/DISCO-DJ/scripts")
if str(_discodj_scripts) not in sys.path:
    sys.path.insert(0, str(_discodj_scripts))

# ---------------------------------------------------------------------------
# Update DISCO-DJ global state BEFORE importing discodj (gs is read at import
# time by scatter_and_gather.py: N = gs.N = gs.res, chunk_size = gs.chunk_size)
# ---------------------------------------------------------------------------
from discodj.core.global_state import update_global_options
update_global_options(
    res=args.res,
    res_pm=args.res_pm,
    boxsize=args.boxsize,
    a_ini=args.a_ini,
    a_end=args.a_end,
    numsteps=args.n_steps,
    cosmo=args.cosmo,
    num_chunks=args.num_chunks,
    lightcone=args.lightcone,
    run_mode=args.mode,
)

# ---------------------------------------------------------------------------
# Env-vars for JAX / CUDA (must be set before JAX is imported)
# ---------------------------------------------------------------------------
if args.mode == "gpu":
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---------------------------------------------------------------------------
# JAX distributed initialisation for multi-GPU SLURM jobs
# ---------------------------------------------------------------------------
import jax

_using_slurm = "SLURM_JOB_ID" in os.environ

if args.mode == "gpu" and _using_slurm:
    # srun launches one task per GPU; JAX reads SLURM env vars automatically
    jax.distributed.initialize(
        heartbeat_timeout_seconds=30,
        initialization_timeout=60,
    )
    _process_id = int(os.environ.get("SLURM_PROCID", 0))
    print(f"[rank {_process_id}] JAX distributed initialised")

# After init we know the full device count
_backend    = "gpu" if args.mode == "gpu" else "cpu"
num_devices = len(jax.devices(_backend))
device      = args.mode  # string passed to DiscoDJ
mode        = "gpu" if num_devices > 1 else ("singlegpu" if args.mode == "gpu" else "cpu")

print(f"Devices: {jax.devices(_backend)}  (num_devices={num_devices}, mode={mode})")

# ---------------------------------------------------------------------------
# Standard imports (after JAX is configured)
# ---------------------------------------------------------------------------
import numpy as np

# The editable install maps discodj.core.multigpu_utils → DISCO-DJ/scripts/utils.py,
# which in turn imports peer modules (utils_jens, etc.) from that same directory.
# scripts/ was already added to sys.path above — no duplicate needed here.

from discodj import DiscoDJ
from discodj.core.utils import get_sharding_none, get_mesh
from jax.experimental.multihost_utils import sync_global_devices

# Script-local helpers from the masterProject directory
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))
from read_tipsy_file import read_tipsy

try:
    from visualize import plot_density_slice
    _have_visualize = True
except ImportError:
    _have_visualize = False

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
Lbox  = args.boxsize
a_ini = args.a_ini
Npart = args.res
# Use the same chunk_size formula as gs.chunk_size (res^3 / num_chunks).
# gs.chunk_size can't be read yet (JAX not init'd → gs.num_devices hangs),
# so compute it here directly.
chunk_size = Npart ** 3 // args.num_chunks


def _sort_for_lagrangian_x(arr_pos: np.ndarray, arr_vel: np.ndarray, nx: int):
    """Sort particles by x-coordinate so axis-0 slices are monotone in x."""
    idx = np.argsort(arr_pos[:, 0])
    return arr_pos[idx], arr_vel[idx]


def _load_external_ics(ic_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PKDGRAV tipsy ICs, convert velocities, sort for Lagrangian grid."""
    print(f"Reading PKDGRAV ICs from {ic_path}")
    p_init, _ = read_tipsy(ic_path, Lbox)

    external_ics = np.c_[
        p_init["x"],  p_init["y"],  p_init["z"],
        p_init["vx"], p_init["vy"], p_init["vz"],
    ].astype(np.float32)

    # Convert PKD comoving velocities → DISCO-DJ units
    v_factor = a_ini ** 2 / np.sqrt(8 * np.pi / 3) * Lbox
    external_ics[:, 3:] *= v_factor

    pos_sorted, vel_sorted = _sort_for_lagrangian_x(
        external_ics[:, :3], external_ics[:, 3:], nx=Npart
    )
    return pos_sorted, vel_sorted


def _field_slices(field):
    """Return (mean-projection, mid-plane slice, thin-slice) of a 3-D field."""
    field_mean = field.mean(1)
    mid        = field[:, args.res_pm // 2, :]
    thickness  = max(1, args.res_pm // 128)
    thin       = field[:, args.res_pm // 2 : args.res_pm // 2 + thickness].mean(1)
    return field_mean, mid, thin


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
t0 = time()

# Only rank 0 needs to print status in multi-process setups
_rank0 = (jax.process_index() == 0)

if _rank0:
    print("=" * 60)
    print("DISCO-DJ Multi-GPU N-body simulation")
    print(f"  res={Npart}  res_pm={args.res_pm}  boxsize={Lbox} Mpc/h")
    print(f"  a_ini={a_ini}  a_end={args.a_end}  n_steps={args.n_steps}")
    print(f"  cosmo={args.cosmo}  stepper={args.stepper}  lightcone={args.lightcone}")
    print(f"  num_devices={num_devices}  multi_device={num_devices > 1}")
    print("=" * 60)

# ── Build DiscoDJ object ────────────────────────────────────────────────────
dj = DiscoDJ(
    dim=3,
    res=Npart,
    boxsize=Lbox,
    device=device,
    cosmo=args.cosmo,
    requires_grad_wrt_cosmo=False,
    multi_device=(num_devices > 1),
).with_timetables()

if _rank0:
    print("DISCO-DJ initialised:", dj)

t1 = time()

# ── Initial Conditions ────────────────────────────────────────────────────
if args.use_internal_ics:
    if _rank0:
        print("Using internal IC generator (LPT)")

    dj = dj.with_linear_ps(transfer_function="Eisenstein-Hu")
    sync_global_devices("sync_linear_ps")

    from multigpu_utils import get_white_noise_field
    # Ensure the local cache directory exists (get_white_noise_field saves
    # per-rank .npy files there using a relative path "data/...")
    Path("data").mkdir(exist_ok=True)
    white_noise = get_white_noise_field(dj, mode, Npart, args.ngenic_seed)

    dj = dj.with_ics(
        white_noise_space="real",
        white_noise_field=white_noise,
        try_to_jit=True,
    )

    with get_mesh():
        dj = dj.with_lpt(n_order=args.n_order, convert_to_numpy=False, try_to_jit=True)

elif args.ic_file is not None:
    if _rank0:
        print("Using external PKDGRAV ICs")
    pos_sorted, vel_sorted = _load_external_ics(args.ic_file)
    dj = dj.with_external_ics(pos=pos_sorted, vel=vel_sorted)
    del pos_sorted, vel_sorted

else:
    if _rank0:
        print("ERROR: No initial conditions provided. Use --ic-file or --use-internal-ics.")
    import sys as _sys; _sys.exit(2)

t2 = time()
if _rank0:
    print(f"IC setup took {t2 - t1:.2f} s")

# ── N-body run ──────────────────────────────────────────────────────────────
if _rank0:
    print("Running N-body simulation …")

# ── Snapshot-based HEALPix shell lightcone (high-z capable) ─────────────────
if args.build_shells and _rank0:
    from build_lightcone_shells import run_with_shells

    _shells_out = args.shells_output_dir
    if _shells_out is None:
        _shells_out = (_script_dir / "outputs" / "shells")
    _shells_out.mkdir(parents=True, exist_ok=True)

    # Use the same explicit a-sequence the normal run would use, so shells
    # match the pkdgrav step spacing (a_ini → a_end in n_steps equal steps)
    import numpy as _np
    _a_steps = _np.linspace(a_ini, args.a_end, args.n_steps + 1, dtype=_np.float64)

    print(f"Shell lightcone: nside={args.shells_nside}  z_max={args.shells_z_max}  "
          f"prefix={args.shells_prefix}  output={_shells_out}")

    _n_shells = run_with_shells(
        dj=dj,
        a_steps=_a_steps,
        res_pm=args.res_pm,
        output_dir=_shells_out,
        nside=args.shells_nside,
        z_max=args.shells_z_max,
        prefix=args.shells_prefix,
        snap_dir=args.shells_snap_dir,
        stepper=args.stepper,
        method=args.method,
        antialias=args.antialias,
        grad_kernel_order=args.grad_kernel_order,
        laplace_kernel_order=args.laplace_kernel_order,
        n_resample=args.n_resample,
        chunk_size=chunk_size,
        deconvolve=args.deconvolve,
    )
    print(f"Shell lightcone done: {_n_shells} shells → {_shells_out}")
    sync_global_devices("shells_done")
    import sys as _sys; _sys.exit(0)

# ── Standard N-body run (no shell accumulation) ──────────────────────────────
run_nbody_result = dj.run_nbody(
    a_ini=a_ini,
    a_end=args.a_end,
    n_steps=args.n_steps,
    res_pm=args.res_pm,
    light_cone=args.lightcone,
    time_var=args.time_var,
    stepper=args.stepper,
    method=args.method,
    antialias=args.antialias,
    grad_kernel_order=args.grad_kernel_order,
    laplace_kernel_order=args.laplace_kernel_order,
    n_resample=args.n_resample,
    chunk_size=chunk_size,
    deconvolve=args.deconvolve,
    return_displacement=True,
    convert_to_numpy=False,
)

if args.lightcone:
    psi_sim, P, a, _S = run_nbody_result
else:
    psi_sim, P, a = run_nbody_result
del run_nbody_result, P

t3 = time()
if _rank0:
    print(f"N-body run took {t3 - t2:.2f} s")

# ── Density field extraction ────────────────────────────────────────────────
# Match simple_example.py pattern: only constrain the small 2D SLICES to
# sharding_none *inside* the jitted function.  Do NOT use out_shardings on the
# whole jit — that would trigger a 1.5 GB all-gather of full particle positions
# across all GPUs and causes rank 0 to hang.

def delta_subset(dj, psi_sim):
    X_sim  = dj.get_pos_from_psi(psi_sim)
    delta  = dj.get_delta_from_pos(
        X_sim, res=args.res_pm, chunk_size=chunk_size, try_to_jit=False
    )
    # Only gather small 2D projections (512×512 each) — not the full 3-D field
    slices = jax.lax.with_sharding_constraint(_field_slices(delta), get_sharding_none())
    return slices


delta_subset = jax.jit(delta_subset)
out = delta_subset(dj, psi_sim)
delta_mean, delta_slice, delta_thin_slice = out

if _rank0:
    print("delta_mean shape:", delta_mean.shape)

# ── Save final snapshot (rank 0 gathers all shards and saves one file) ───────
if args.save_final is not None:
    args.save_final.parent.mkdir(parents=True, exist_ok=True)
    # In multi-process JAX each rank can only access its own shards.
    # Collect this rank's local slab, then send to rank 0 via jax.distributed.
    local_psi = np.concatenate(
        [np.asarray(s.data) for s in psi_sim.addressable_shards], axis=0
    )
    # process_allgather returns (nprocs, *local_shape); concatenate along axis 0
    # to get (nprocs*slab_x, res_y, res_z, 3) — the full displacement field.
    gathered = jax.experimental.multihost_utils.process_allgather(local_psi)
    full_psi = np.concatenate(gathered, axis=0)
    if _rank0:
        np.savez_compressed(args.save_final, psi=full_psi, a_hist=np.asarray(a))
        print(f"Saved full snapshot → {args.save_final}  shape={full_psi.shape}")

# ── Optional density-slice plot (rank 0 only — slices are already gathered) ─
if args.plot and _rank0:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _script_dir / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, arr in [("mean", delta_mean), ("slice", delta_slice), ("thinslice", delta_thin_slice)]:
        arr_np = np.asarray(arr)
        plt.imsave(output_dir / f"{name}_a{args.a_end:.3f}.png",
                   np.log10(1.01 + arr_np), cmap="inferno")
    print(f"Density slices saved → {output_dir}")

t4 = time()
# Proper multi-GPU shutdown sync
sync_global_devices("final_sync")
if _rank0:
    print(f"Total wall time: {t4 - t0:.2f} s")
    print("Done.")
