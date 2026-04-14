#!/bin/bash
#SBATCH --job-name=shell_collector
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/shell_collector/shell_collector_%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/shell_collector/shell_collector_%A_%a.err

# Collects PKDGRAV lightcone output into shell FITS files

set -euo pipefail

# Run index within each cosmology (run_0 … run_6); change as needed.
RUN_ID=1

SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ENV=disco-dj
CONDA_ROOT=/users/damrein/miniforge3

# Map array task ID to zero-padded 6-digit cosmology index (1 -> 000001, 2 -> 000002, ...)
COSMO_ID=$(printf '%06d' "${SLURM_ARRAY_TASK_ID}")
PARAM_FILE=${SCRATCH_DIR}/cosmogridv1/cosmo_${COSMO_ID}/run_${RUN_ID}/cosmology.par
RUN_DIR=${SCRATCH_DIR}/outputs/ICs/cosmo_${COSMO_ID}/run_${RUN_ID}
PROJECT_DIR=/users/damrein/masterProject/pkdgrav

# Activate conda env
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh under ${CONDA_ROOT}." >&2
    exit 4
fi

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

ENV_PREFIX=$(conda env list | awk -v env_name="${CONDA_ENV}" '$1 == env_name {print $NF; exit}')
if [[ -z "${ENV_PREFIX}" ]]; then
    echo "Conda env '${CONDA_ENV}' not found." >&2
    exit 5
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found in env '${CONDA_ENV}': ${PYTHON_BIN}" >&2
    exit 6
fi

if ! "${PYTHON_BIN}" -c "import numpy, healpy" >/dev/null 2>&1; then
    echo "Required packages missing in conda env '${CONDA_ENV}' (need numpy, healpy)." >&2
    echo "Install with: conda install -n ${CONDA_ENV} numpy healpy" >&2
    exit 3
fi

if [[ ! -f "${PARAM_FILE}" ]]; then
    echo "Param file not found: ${PARAM_FILE}" >&2
    exit 2
fi

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Run directory not found: ${RUN_DIR}" >&2
    exit 7
fi

cd "${RUN_DIR}"

echo "Collecting shells for cosmo=${COSMO_ID}, run=${RUN_ID}"
echo "Param file: ${PARAM_FILE}"
"${PYTHON_BIN}" "${PROJECT_DIR}/shell_collector.py" --param_file "${PARAM_FILE}"
echo "Done."
