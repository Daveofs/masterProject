#!/bin/bash
#SBATCH --job-name=disco_cpu_cosmoIC
#SBATCH --partition=normal.24h
#SBATCH --time=01:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/disco_cpu_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/disco_cpu_%j.err

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

# Force CPU execution even on nodes with visible GPUs
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# DiscoDJ run configuration (set to your IC metadata)
A_INI=0.01      # pkdgrav IC is at z=99 -> a=0.01
BOXSIZE=900.0
RES=832
A_END=0.6229    # pkdgrav step 80: z=0.6059 -> a=1/(1+0.6059)
N_STEPS=10     # Could be higher for being more accurate 
RES_PM=832
STEPPER=bullfrog
METHOD=pm
DTYPE=float32

cd "${PROJECT_DIR}"

mkdir -p "${SCRATCH_DIR}/outputs" "${SCRATCH_DIR}/outputs/plots" "${PLOT_DIR}" "${SCRATCH_DIR}/outputs/logs" "${SCRATCH_DIR}/outputs/snapshots"

SAVE_FINAL_FILE=${SCRATCH_DIR}/outputs/snapshots/final_snapshot_cpu_${SLURM_JOB_ID:-manual}.npz

if [[ ! -f "${IC_ARCHIVE}" ]]; then
	echo "Missing IC archive: ${IC_ARCHIVE}. Run run_convert_ic.sh first." >&2
	exit 2
fi

if ! "${PYTHON_BIN}" -c "import numpy" >/dev/null 2>&1; then
	echo "numpy is missing in conda env '${CONDA_ENV}'." >&2
	echo "Install with: conda install -n ${CONDA_ENV} numpy" >&2
	exit 3
fi

echo "[$(date --iso-8601=seconds)] Starting DiscoDJ CPU run on $(hostname)"
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