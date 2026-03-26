#!/bin/bash
#SBATCH --job-name=disco_gpu_cosmoIC
#SBATCH --partition=gpuhe.4h
#SBATCH --gpus=nvidia_geforce_rtx_3090:1 
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/disco_gpu_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/disco_gpu_%j.err

# GPU submission enabled in SBATCH header above 

# Currently running out of memory for GPU

set -euo pipefail

PROJECT_DIR=/cluster/scratch/damrein/masterProject
SCRATCH_DIR=/cluster/scratch/damrein
CONDA_ROOT=/cluster/scratch/damrein/miniconda3
CONDA_ENV=vir_env
IC_ARCHIVE=/cluster/scratch/damrein/outputs/ICs/000001_copy7/CosmoML.00000
PLOT_DIR=${SCRATCH_DIR}/outputs/plots/snapshots

# Activate conda in non-interactive SLURM shells
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# Prevent externally activated virtualenvs from shadowing conda python
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

ENV_PREFIX=$(conda env list | awk -v env_name="${CONDA_ENV}" '$1 == env_name {print $NF; exit}')
if [[ -z "${ENV_PREFIX}" ]]; then
	echo "Conda env '${CONDA_ENV}' not found." >&2
	exit 4
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
	echo "Python executable not found in env '${CONDA_ENV}': ${PYTHON_BIN}" >&2
	exit 5
fi

# DiscoDJ run configuration
A_INI=0.01      # pkdgrav IC is at z=99 -> a=0.01
BOXSIZE=900.0
RES=832
A_END=0.6229    # pkdgrav step 80: z=0.6059 -> a=1/(1+0.6059)
N_STEPS=10     # Could be higher for being more accurate 
RES_PM=512      # reduced from 832 to fit in GPU VRAM (832^3 PM grid = ~2.8GB alone)
STEPPER=bullfrog
METHOD=pm
DTYPE=float32

# GPU memory settings
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORM_NAME=gpu
export TF_GPU_ALLOCATOR=cuda_malloc_async   # reduce fragmentation (suggested by XLA itself)
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # don't pre-allocate all VRAM upfront
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90  # cap JAX at 90% of VRAM

cd "${PROJECT_DIR}"

# Ensure outputs directories exist before SLURM redirects stdout/stderr
mkdir -p "${SCRATCH_DIR}/outputs" "${SCRATCH_DIR}/outputs/plots" "${PLOT_DIR}" "${SCRATCH_DIR}/outputs/logs" "${SCRATCH_DIR}/outputs/snapshots"

# Path for the saved final snapshot (file, not directory)
SAVE_FINAL_FILE=${SCRATCH_DIR}/outputs/snapshots/final_snapshot_gpu_${SLURM_JOB_ID:-manual}.npz

if [[ ! -f "${IC_ARCHIVE}" ]]; then
	echo "Missing IC archive: ${IC_ARCHIVE}. Run run_convert_ic.sh first." >&2
	exit 2
fi

if ! "${PYTHON_BIN}" -c "import numpy" >/dev/null 2>&1; then
	echo "numpy is missing in conda env '${CONDA_ENV}'." >&2
	echo "Install with: conda install -n ${CONDA_ENV} numpy" >&2
	exit 3
fi

echo "[$(date --iso-8601=seconds)] Starting DiscoDJ GPU run on $(hostname)"
"${PYTHON_BIN}" sim_discodj.py \
	--ic-file "${IC_ARCHIVE}" \
	--a-ini "${A_INI}" \
	--boxsize "${BOXSIZE}" \
	--res "${RES}" \
	--a-end "${A_END}" \
	--n-steps "${N_STEPS}" \
	--res-pm "${RES_PM}" \
	--stepper "${STEPPER}" \
	--method "${METHOD}" \
	--dtype "${DTYPE}" \
	--plot \
	--save-final "${SAVE_FINAL_FILE}" \
	--output-dir "${PLOT_DIR}"
echo "[$(date --iso-8601=seconds)] Done."