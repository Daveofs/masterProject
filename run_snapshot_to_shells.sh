#!/bin/bash
#SBATCH --job-name=snapshot_to_shells
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=30G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/snapshot_to_shells_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/snapshot_to_shells_%j.err

set -euo pipefail

PROJECT_DIR=/cluster/scratch/damrein/masterProject
SCRATCH_DIR=/cluster/scratch/damrein
CONDA_ROOT=/cluster/scratch/damrein/miniconda3
CONDA_ENV=vir_env

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

# Use all allocated CPUs for numpy/healpy OpenMP operations
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# No GPU needed
export CUDA_VISIBLE_DEVICES=

echo "======================================================"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "CPUs      : ${SLURM_CPUS_PER_TASK}"
echo "Start     : $(date)"
echo "======================================================"

"${PYTHON_BIN}" "${PROJECT_DIR}/snapshot_to_shells.py"

echo "======================================================"
echo "End       : $(date)"
echo "======================================================"
