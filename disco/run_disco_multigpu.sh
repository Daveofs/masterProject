#!/bin/bash
# ============================================================
# Multi-GPU DISCO-DJ N-body simulation (vir_env_discodj)
#
# Runs a single process that sees all 4 GPUs; JAX uses pjit/pmap
# for intra-node multi-device sharding (faster than multi-process).
# ============================================================
#SBATCH --job-name=disco_multigpu
#SBATCH --partition=gpu.24h                   
#SBATCH --nodes=1
#SBATCH --ntasks=1                            # single process, all GPUs visible
#SBATCH --gpus=quadro_rtx_6000:8
##SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=10G                     # 1 task × 16 CPU × 10 GB = 160 GB
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/disco_multigpu_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/disco_multigpu_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR=/cluster/scratch/damrein/masterProject
SCRATCH_DIR=/cluster/scratch/damrein
VENV_DIR=/cluster/work/refregier/damrein/vir_env_discodj

# Input IC file (only used when USE_EXTERNAL_ICS=true)
IC_FILE=/cluster/scratch/damrein/outputs/ICs/000001_copy6/CosmoML.00000

# Use internal ngenic-like ICs instead of an external tipsy file
# Set to "true" to generate ICs inside the Python script
USE_INTERNAL_ICS=false
NGENIC_SEED=180723

# Initial linear power spectrum file (optional, .pk format from PKDGRAV/nbodykit)
# When set, overrides the Eisenstein-Hu transfer function for internal ICs.
# Leave empty to use Eisenstein-Hu (default).
LINEAR_PS_FILE=/capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000001/run_0/CosmoML_000001_run_0.00000.pk

# Output paths
LOG_DIR=${SCRATCH_DIR}/outputs/logs
SNAP_DIR=${SCRATCH_DIR}/outputs/snapshots
PLOT_DIR=${SCRATCH_DIR}/outputs/plots/multigpu

# ── Simulation parameters ─────────────────────────────────────────────────
MODE=gpu            # gpu | cpu
# The provided tipsy IC contains 832^3 particles. Set RES to 832 so
# the loaded positions match DISCO-DJ's expected shape (RES**3).
RES=832             # particle grid resolution per axis  (N_part = RES^3)
# Keep PM grid conservative to avoid cuFFT OOM on 24GB GPUs
RES_PM=832          # PM force grid resolution per axis (use 512 on A100/RTX Pro)
BOXSIZE=900.0       # box size [Mpc/h]
COSMO=Planck15      # DISCO-DJ cosmology preset
A_INI=0.01          # initial scale factor  (z=99 → a=0.01)
A_END=1.0           # final scale factor
N_STEPS=20          # number of N-body timesteps
STEPPER=bullfrog    # bullfrog | fastpm | symplectic
TIME_VAR=D          # time variable: a | lna | D
METHOD=pm           # pm | nufftpm
GRAD_KERNEL_ORDER=4
LAPLACE_KERNEL_ORDER=0
NUM_CHUNKS=32        # chunk_size = RES^3 / NUM_CHUNKS (must be <= RES; increase chunks to lower per-chunk memory)
LIGHTCONE=false     # set to "true" to enable lightcone mode
BUILD_SHELLS=true
SHELLS_Z_MIN=0.0   # minimum redshift for lightcone shells
SHELLS_Z_MAX=3.5    # maximum redshift for lightcone shells


# ── JAX / XLA memory settings ─────────────────────────────────────────────
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_TRACEBACK_FILTERING=off

# ── JAX persistent XLA compilation cache ──────────────────────────────────
# Compiled kernels are stored on disk so repeated runs skip recompilation.
# First run is still slow; every run after that is much faster.
export JAX_COMPILATION_CACHE_DIR=/cluster/scratch/damrein/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0   # cache everything
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# Speed-up flags for multi-GPU collectives
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_nccl_comm_splitting=true \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true"


# ── Sanity checks ─────────────────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Virtual env not found: ${VENV_DIR}" >&2; exit 1
fi

if [[ "${USE_INTERNAL_ICS}" != "true" ]]; then
    if [[ ! -f "${IC_FILE}" ]]; then
        echo "IC file not found: ${IC_FILE}. Check IC path or set USE_INTERNAL_ICS=true to generate internal ICs." >&2
        exit 2
    fi
fi

if [[ -n "${LINEAR_PS_FILE}" && ! -f "${LINEAR_PS_FILE}" ]]; then
    echo "Linear PS file not found: ${LINEAR_PS_FILE}" >&2; exit 5
fi

# ── Activate virtual environment ──────────────────────────────────────────
# Prevent any active conda / venv from shadowing the target venv
unset CONDA_DEFAULT_ENV CONDA_PREFIX VIRTUAL_ENV PYTHONHOME PYTHONPATH
source "${VENV_DIR}/bin/activate"

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found in venv: ${PYTHON_BIN}" >&2; exit 3
fi

# Verify discodj is registered in the venv (avoid triggering JAX GPU init in the check)
DISCODJ_PTH=$(find "${VENV_DIR}/lib" -name "*discodj*editable*.pth" -o -name "_discodj_editable.pth" -o -name "discodj.egg-link" 2>/dev/null | head -1)
if [[ -z "${DISCODJ_PTH}" ]]; then
    echo "discodj not installed in ${VENV_DIR}. Run: pip install -e DISCO-DJ/" >&2; exit 4
fi

# ── Create output directories ─────────────────────────────────────────────
mkdir -p "${LOG_DIR}" "${SNAP_DIR}" "${PLOT_DIR}"
# Cache directory for get_white_noise_field() (relative to PROJECT_DIR)
mkdir -p "${PROJECT_DIR}/data"

SAVE_FINAL="${SNAP_DIR}/final_multigpu_${SLURM_JOB_ID}.npz"

# ── Build srun command ─────────────────────────────────────────────────────
PYTHON_ARGS=(
    "${PROJECT_DIR}/disco/sim_discodj_multigpu.py"
    --mode        "${MODE}"
    --res         "${RES}"
    --res-pm      "${RES_PM}"
    --boxsize     "${BOXSIZE}"
    --cosmo       "${COSMO}"
    --a-ini       "${A_INI}"
    --a-end       "${A_END}"
    --n-steps     "${N_STEPS}"
    --stepper     "${STEPPER}"
    --time-var    "${TIME_VAR}"
    --method      "${METHOD}"
    --grad-kernel-order    "${GRAD_KERNEL_ORDER}"
    --laplace-kernel-order "${LAPLACE_KERNEL_ORDER}"
    --num-chunks  "${NUM_CHUNKS}"
    --save-final  "${SAVE_FINAL}"
    --output-dir  "${PLOT_DIR}"
    --plot
)

if [[ "${USE_INTERNAL_ICS}" == "true" ]]; then
    PYTHON_ARGS+=(--use-internal-ics)
    PYTHON_ARGS+=(--ngenic-seed "${NGENIC_SEED}")
else
    PYTHON_ARGS+=(--ic-file "${IC_FILE}")
fi

if [[ -n "${LINEAR_PS_FILE}" ]]; then
    PYTHON_ARGS+=(--linear-ps-file "${LINEAR_PS_FILE}")
fi

if [[ "${LIGHTCONE}" == "true" ]]; then
    PYTHON_ARGS+=(--lightcone)
    PYTHON_ARGS+=(--z-lc-ini "${A_INI}")
fi

if [[ "${BUILD_SHELLS}" == "true" ]]; then
    PYTHON_ARGS+=(--build-shells)
    PYTHON_ARGS+=(--shells-z-min "${SHELLS_Z_MIN}")
    PYTHON_ARGS+=(--shells-z-max "${SHELLS_Z_MAX}")
fi

cd "${PROJECT_DIR}"
echo "[$(date --iso-8601=seconds)] Starting DISCO-DJ multi-GPU run on $(hostname)"
echo -e "[$(date --iso-8601=seconds)] GPUs in use: \n$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"
srun "${PYTHON_BIN}" -u "${PYTHON_ARGS[@]}"
echo "[$(date --iso-8601=seconds)] Done.
