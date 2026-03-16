#!/bin/bash
#SBATCH --job-name=write_tipsy
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/write_tipsy_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/write_tipsy_%j.err

set -euo pipefail

PROJECT_DIR=/cluster/scratch/damrein/masterProject
SCRATCH_DIR=/cluster/scratch/damrein
CONDA_ROOT=/cluster/scratch/damrein/miniconda3
CONDA_ENV=vir_env

# Input snapshot
SNAPSHOT=${SCRATCH_DIR}/outputs/snapshots/final_snapshot_cpu_60125122.npz

# Activate scratch conda
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

ENV_PREFIX=$(conda env list | awk -v env_name="${CONDA_ENV}" '$1 == env_name {print $NF; exit}')
if [[ -z "${ENV_PREFIX}" ]]; then
    echo "Conda env '${CONDA_ENV}' not found." >&2
    exit 1
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ ! -f "${SNAPSHOT}" ]]; then
    echo "Snapshot not found: ${SNAPSHOT}" >&2
    exit 3
fi

mkdir -p "${SCRATCH_DIR}/outputs/logs" "${SCRATCH_DIR}/outputs/snapshots"

echo "[$(date --iso-8601=seconds)] Starting write_tipsy on $(hostname)"
echo "Snapshot: ${SNAPSHOT}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" write_tipsy.py "${SNAPSHOT}"

echo "[$(date --iso-8601=seconds)] Done."
