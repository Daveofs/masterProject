#!/bin/bash
#SBATCH --job-name=shell_collector
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/shell_collector_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/shell_collector_%j.err

# Collects PKDGRAV lightcone output into shell FITS files

set -euo pipefail

PROJECT_DIR=/cluster/scratch/damrein/masterProject
SCRATCH=/cluster/scratch/damrein
CONDA_ENV=vir_env
PARAM_FILE=${SCRATCH}/cosmogridv1/cosmo_000001/param_files/cosmology.par
RUN_DIR=${SCRATCH}/outputs/ICs/000001_copy7

if [[ -f /cluster/home/damrein/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/cluster/home/damrein/miniconda3
elif [[ -f /cluster/scratch/damrein/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/cluster/scratch/damrein/miniconda3
else
  echo "Could not find conda.sh under /cluster/home or /cluster/scratch miniconda3." >&2
  exit 4
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
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

cd "${PROJECT_DIR}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Param file not found: ${PARAM_FILE}" >&2
  exit 2
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 7
fi

cd "${RUN_DIR}"

echo "Collecting shells from output namespace in: ${PARAM_FILE}"
"${PYTHON_BIN}" "${PROJECT_DIR}/shell_collector.py" --param_file "${PARAM_FILE}"
echo "Done."
