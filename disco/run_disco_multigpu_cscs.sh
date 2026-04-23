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
#SBATCH --ntasks-per-node=1         # one Python process per node
#SBATCH --gpus-per-node=4           # 4 GH200 GPUs per node; process owns all local GPUs
# NOTE: RES must be divisible by (nodes * gpus-per-node).
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/disco/disco_multigpu_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/disco/disco_multigpu_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR=/users/damrein/masterProject/disco
SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ENV=disco-dj
CONDA_INIT=$HOME/miniforge3/etc/profile.d/conda.sh

# Input IC file (only used when USE_INTERNAL_ICS=false)
IC_FILE=/capstor/scratch/cscs/damrein/outputs/ICs/cosmo_000001/run_0/CosmoML.00000

# Path to the params.yml for the simulation cosmology (leave empty to use COSMO preset)
PARAMS_YML=/capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000001/run_0/params.yml

# Use internal ngenic-like ICs instead of an external tipsy file
USE_INTERNAL_ICS=false
NGENIC_SEED=180723

# Output paths
LOG_DIR=${SCRATCH_DIR}/outputs/logs
SNAP_DIR=${SCRATCH_DIR}/outputs/snapshots
DISCO_LC_DIR=${SCRATCH_DIR}/outputs/disco_lc
SHELL_DIR=${SCRATCH_DIR}/outputs/shells_no_xla_flags
PLOT_DIR=${SCRATCH_DIR}/outputs/plots/multigpu

# ── Multi-node settings ──────────────────────────────────────────────────
GPUS_PER_NODE=4              # must match --gpus-per-node above

# ── Simulation parameters ─────────────────────────────────────────────────
MODE=gpu
RES=832
RES_PM=832
BOXSIZE=900.0
COSMO=Planck15  # used only when PARAMS_YML is empty
A_INI=0.01
A_END=1.0
N_STEPS=20         # used only when SHELLS_METAINFO is empty
N_PRESTEPS=30     # sub-steps from a_ini to first shell boundary (z=99→3.5); pkdgrav3 uses ~30
STEPPER=bullfrog
TIME_VAR=D
METHOD=pm
GRAD_KERNEL_ORDER=4
LAPLACE_KERNEL_ORDER=0
NUM_CHUNKS=1
LIGHTCONE=false
BUILD_SHELLS=true
SHELLS_METAINFO="/capstor/scratch/cscs/damrein/cosmogridv1/CosmoGridV1_metainfo.h5"
#SHELLS_METAINFO=""

# # ── JAX compilation cache (avoids 20-sec first-run JIT overhead on re-runs) ──
# JAX_CACHE_DIR=${SCRATCH_DIR}/jax_cache
# export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}
# export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
# mkdir -p "${JAX_CACHE_DIR}"

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
mkdir -p "${LOG_DIR}" "${SNAP_DIR}" "${PLOT_DIR}" "${SHELL_DIR}" "${PROJECT_DIR}/data"

SAVE_FINAL="${SNAP_DIR}/final_multigpu_${SLURM_JOB_ID}.npz"
SAVE_LIGHTCONE="${DISCO_LC_DIR}/lightcone_multigpu_${SLURM_JOB_ID}.npz"

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
    --shells-output-dir   "${SHELL_DIR}"
    --gpus-per-node   "${GPUS_PER_NODE}"
    --plot
)

if [[ "${USE_INTERNAL_ICS}" == "true" ]]; then
    PYTHON_ARGS+=(--use-internal-ics)
    PYTHON_ARGS+=(--ngenic-seed "${NGENIC_SEED}")
else
    PYTHON_ARGS+=(--ic-file "${IC_FILE}")
fi

if [[ -n "${PARAMS_YML:-}" && -f "${PARAMS_YML}" ]]; then
    PYTHON_ARGS+=(--params-yml "${PARAMS_YML}")
fi

if [[ "${LIGHTCONE}" == "true" ]]; then
    PYTHON_ARGS+=(--lightcone)
    PYTHON_ARGS+=(--save-lightcone "${SAVE_LIGHTCONE}")
fi

if [[ "${BUILD_SHELLS}" == "true" ]]; then
    PYTHON_ARGS+=(--build-shells)
    PYTHON_ARGS+=(--n-presteps "${N_PRESTEPS}")
    if [[ -n "${SHELLS_METAINFO}" && -f "${SHELLS_METAINFO}" ]]; then
        PYTHON_ARGS+=(--shells-metainfo "${SHELLS_METAINFO}")
    fi
fi

cd "${PROJECT_DIR}"
echo "[$(date --iso-8601=seconds)] Starting DISCO-DJ multi-GPU run on $(hostname)"
echo -e "[$(date --iso-8601=seconds)] GPUs in use: \n$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"
# --ntasks must equal SLURM_NNODES (one process per node, each owns GPUS_PER_NODE GPUs)
N_NODES=$(( SLURM_NNODES ))
srun --ntasks="${N_NODES}" --ntasks-per-node=1 \
    "${PYTHON_BIN}" -u "${PYTHON_ARGS[@]}"
echo "[$(date --iso-8601=seconds)] Done.
