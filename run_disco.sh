#!/bin/bash
#SBATCH --job-name=disco_cosmoIC
#SBATCH --partition=gpu.4h
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/disco_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/disco_%j.err

# GPU submission enabled in SBATCH header above.

set -euo pipefail

PROJECT_DIR=/cluster/scratch/damrein/project
SRATCH_DIR=/cluster/scratch/damrein
CONDA_ROOT=/cluster/scratch/damrein/miniconda3
CONDA_ENV=vir_env
IC_ARCHIVE=/cluster/scratch/damrein/pkdgrav/pkdgrav3_dev-master/build/CosmoML.00000
PLOT_DIR=${SRATCH_DIR}/outputs/plots/snapshots

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

# DiscoDJ run configuration (set to your IC metadata)
A_INI=0.02
BOXSIZE=900.0
RES=832
A_END=1.0
N_STEPS=10
RES_PM=832
STEPPER=bullfrog
METHOD=pm
DTYPE=float32

cd "${PROJECT_DIR}"

# Ensure outputs directories exist before SLURM redirects stdout/stderr
mkdir -p "${SRATCH_DIR}/outputs" "${SRATCH_DIR}/outputs/plots" "${PLOT_DIR}"

# Path for the saved final snapshot (file, not directory)
SAVE_FINAL_FILE=${SRATCH_DIR}/outputs/final_snapshot_${SLURM_JOB_ID:-manual}.npz

if [[ ! -f "${IC_ARCHIVE}" ]]; then
	echo "Missing IC archive: ${IC_ARCHIVE}. Run run_convert_ic.sh first." >&2
	exit 2
fi

if ! "${PYTHON_BIN}" -c "import numpy" >/dev/null 2>&1; then
	echo "numpy is missing in conda env '${CONDA_ENV}'." >&2
	echo "Install with: conda install -n ${CONDA_ENV} numpy" >&2
	exit 3
fi

echo "[$(date --iso-8601=seconds)] Starting DiscoDJ on $(hostname)"
"${PYTHON_BIN}" run_sim_discodj.py \
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