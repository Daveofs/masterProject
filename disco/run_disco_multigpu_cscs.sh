#!/bin/bash
# ============================================================
# Multi-GPU DISCO-DJ N-body simulation — CSCS Daint (Alps)
#
# Runs one SLURM task per GPU via `srun`; JAX distributed init
# is triggered automatically by the presence of SLURM_JOB_ID.
# ============================================================
#SBATCH --job-name=disco_multigpu
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # one task per GPU
#SBATCH --gpus-per-node=1           # 4 GH200 GPUs; JAX distributed assigns one per task
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/disco_multigpu_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/disco_multigpu_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR=/users/damrein/masterProject/disco
SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ENV=disco-dj
CONDA_INIT=$HOME/miniforge3/etc/profile.d/conda.sh

# Input IC file (only used when USE_EXTERNAL_ICS=true)
IC_FILE=${SCRATCH_DIR}/outputs/ICs/000001_copy6/CosmoML.00000

# Use internal ngenic-like ICs instead of an external tipsy file
USE_INTERNAL_ICS=true
NGENIC_SEED=180723

# Output paths
LOG_DIR=${SCRATCH_DIR}/outputs/logs
SNAP_DIR=${SCRATCH_DIR}/outputs/snapshots
PLOT_DIR=${SCRATCH_DIR}/outputs/plots/multigpu

# ── Simulation parameters ─────────────────────────────────────────────────
MODE=gpu
RES=832
RES_PM=1664
BOXSIZE=900.0
COSMO=Planck15
A_INI=0.01
A_END=1.0
N_STEPS=100
STEPPER=bullfrog
TIME_VAR=D
METHOD=pm
GRAD_KERNEL_ORDER=4
LAPLACE_KERNEL_ORDER=0
NUM_CHUNKS=32
LIGHTCONE=false
BUILD_SHELLS=false
# ── JAX compilation cache (avoids 20-sec first-run JIT overhead on re-runs) ──
JAX_CACHE_DIR=${SCRATCH_DIR}/jax_cache
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
mkdir -p "${JAX_CACHE_DIR}"

# ── JAX / XLA memory settings ─────────────────────────────────────────────
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_TRACEBACK_FILTERING=off

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_nccl_comm_splitting=true \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true"

export NCCL_NVLS_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ── Sanity checks ─────────────────────────────────────────────────────────
if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "Miniforge not found: ${CONDA_INIT}" >&2; exit 1
fi

if [[ "${USE_INTERNAL_ICS}" != "true" ]]; then
    if [[ ! -f "${IC_FILE}" ]]; then
        echo "IC file not found: ${IC_FILE}. Set USE_INTERNAL_ICS=true or fix IC_FILE." >&2
        exit 2
    fi
fi

# ── Activate conda environment ────────────────────────────────────────────
source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

PYTHON_BIN=$(which python)
export PYTHONPATH=/users/damrein/DISCO-DJ/scripts:${PYTHONPATH:-}

# Verify discodj is importable
if ! "${PYTHON_BIN}" -c "import discodj" 2>/dev/null; then
    echo "discodj not importable in conda env '${CONDA_ENV}'." >&2; exit 3
fi

# ── Create output directories ─────────────────────────────────────────────
mkdir -p "${LOG_DIR}" "${SNAP_DIR}" "${PLOT_DIR}" "${PROJECT_DIR}/data"

SAVE_FINAL="${SNAP_DIR}/final_multigpu_${SLURM_JOB_ID}.npz"

# ── Build srun command ─────────────────────────────────────────────────────
PYTHON_ARGS=(
    "${PROJECT_DIR}/sim_discodj_multigpu.py"
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

if [[ "${LIGHTCONE}" == "true" ]]; then
    PYTHON_ARGS+=(--lightcone)
fi

if [[ "${BUILD_SHELLS}" == "true" ]]; then
    PYTHON_ARGS+=(--build-shells)
fi

cd "${PROJECT_DIR}"
echo "[$(date --iso-8601=seconds)] Starting DISCO-DJ multi-GPU run on $(hostname)"
echo "  Nodes: ${SLURM_NODELIST}"
echo "  Tasks (processes): ${SLURM_NTASKS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

srun --output "${LOG_DIR}/disco_multigpu_${SLURM_JOB_ID}_%t.out" \
     --error  "${LOG_DIR}/disco_multigpu_${SLURM_JOB_ID}_%t.err" \
    "${PYTHON_BIN}" -u "${PYTHON_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] Done. Snapshot at: ${SAVE_FINAL}"
