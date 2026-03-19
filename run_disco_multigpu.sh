#!/bin/bash
# ============================================================
# Multi-GPU DISCO-DJ N-body simulation (vir_env_discodj)
#
# Runs one SLURM task per GPU via `srun`; JAX distributed init
# is triggered automatically by the presence of SLURM_JOB_ID.
#
# GPU options (uncomment the pair you want):
#   4× RTX 3090 (24 GB each) on gpuhe.4h  ← default below
#   4× RTX 4090 (24 GB each) on gpuhe.4h
# ============================================================
#SBATCH --job-name=disco_multigpu
#SBATCH --partition=gpuhe.4h
#SBATCH --nodes=1
#SBATCH --ntasks=4                            # one task per GPU
#SBATCH --gpus=nvidia_geforce_rtx_3090:4      # 4× RTX 3090 on one node
##SBATCH --gpus=nvidia_geforce_rtx_4090:4     # alternative: 4× RTX 4090
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G                     # 4 tasks × 4 CPUs × 16 GB = 256 GB
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/disco_multigpu_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/disco_multigpu_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR=/cluster/scratch/damrein/masterProject
SCRATCH_DIR=/cluster/scratch/damrein
VENV_DIR=/cluster/work/refregier/damrein/vir_env_discodj

# Input IC file (only used when USE_EXTERNAL_ICS=true)
IC_FILE=/cluster/scratch/damrein/outputs/ICs/000001_copy6/CosmoML.00000

# Output paths
LOG_DIR=${SCRATCH_DIR}/outputs/logs
SNAP_DIR=${SCRATCH_DIR}/outputs/snapshots
PLOT_DIR=${SCRATCH_DIR}/outputs/plots/multigpu

# ── Simulation parameters ─────────────────────────────────────────────────
MODE=gpu            # gpu | cpu
RES=512             # particle grid resolution per axis  (N_part = RES^3)
RES_PM=512          # PM force grid resolution per axis
BOXSIZE=900.0       # box size [Mpc/h]
COSMO=Planck15      # DISCO-DJ cosmology preset
A_INI=0.01          # initial scale factor  (z=99 → a=0.01)
A_END=1.0           # final scale factor
N_STEPS=20          # number of N-body timesteps
STEPPER=bullfrog    # bullfrog | fastpm | symplectic
TIME_VAR=D          # time variable: a | lna | D
N_ORDER=1           # LPT order for internal ICs (1 or 2)
SEED=180723         # Ngenic seed (for internal ICs)
METHOD=pm           # pm | nufftpm
GRAD_KERNEL_ORDER=4
LAPLACE_KERNEL_ORDER=0
NUM_CHUNKS=8        # chunk_size = RES^3 / NUM_CHUNKS
LIGHTCONE=false     # set to "true" to enable lightcone mode

# Set to "true" to use external PKDGRAV ICs from IC_FILE, "false" for internal LPT
USE_EXTERNAL_ICS=false

# ── JAX / XLA memory settings ─────────────────────────────────────────────
export JAX_PLATFORM_NAME=gpu
# Use the platform (cudaMalloc/cudaFree) allocator so cuFFT and JAX share the
# same CUDA memory pool — avoids CUFFT_ALLOC_FAILED (error 5) when JAX's
# caching allocator occupies all VRAM before cuFFT can allocate scratch space.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_TRACEBACK_FILTERING=off

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

if [[ "${USE_EXTERNAL_ICS}" == "true" && ! -f "${IC_FILE}" ]]; then
    echo "IC file not found: ${IC_FILE}. Run run_convert_ic.sh first." >&2; exit 2
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
    --n-order     "${N_ORDER}"
    --seed        "${SEED}"
    --method      "${METHOD}"
    --grad-kernel-order    "${GRAD_KERNEL_ORDER}"
    --laplace-kernel-order "${LAPLACE_KERNEL_ORDER}"
    --num-chunks  "${NUM_CHUNKS}"
    --save-final  "${SAVE_FINAL}"
    --output-dir  "${PLOT_DIR}"
    --plot
)

if [[ "${USE_EXTERNAL_ICS}" == "true" ]]; then
    PYTHON_ARGS+=(--use-external-ics --ic-file "${IC_FILE}")
fi

if [[ "${LIGHTCONE}" == "true" ]]; then
    PYTHON_ARGS+=(--lightcone)
fi

cd "${PROJECT_DIR}"
echo "[$(date --iso-8601=seconds)] Starting DISCO-DJ multi-GPU run on $(hostname)"
echo "  GPUs requested: ${SLURM_GPUS}"
echo "  Tasks (processes): ${SLURM_NTASKS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# srun spawns one MPI-like process per task; each picks up one GPU
# Capture both stdout AND stderr per-task so XLA FATAL messages are visible
srun --output "${LOG_DIR}/disco_multigpu_${SLURM_JOB_ID}_%t.out" \
     --error  "${LOG_DIR}/disco_multigpu_${SLURM_JOB_ID}_%t.err" \
    "${PYTHON_BIN}" -u "${PYTHON_ARGS[@]}"

echo "[$(date --iso-8601=seconds)] Done. Snapshot at: ${SAVE_FINAL}"
