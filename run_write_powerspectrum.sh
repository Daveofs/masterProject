#!/bin/bash
#SBATCH --job-name=write_pk
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/write_pk_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/write_pk_%j.err

set -euo pipefail

SCRATCH_DIR=/cluster/scratch/damrein
CONDA_ROOT=/cluster/scratch/damrein/miniconda3
CONDA_ENV=vir_env
PY_SCRIPT=${SCRATCH_DIR}/masterProject/write_powerspectrum.py

# Input tipsy snapshot to compute P(k) for
INPUT=${SCRATCH_DIR}/outputs/snapshots/final_snapshot_tipsy.00000

mkdir -p "${SCRATCH_DIR}/outputs/powerspectrum"

# activate conda env
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
ENV_PREFIX=$(conda env list | awk -v e="${CONDA_ENV}" '$1==e{print $NF;exit}')
PYTHON_BIN="${ENV_PREFIX}/bin/python"

if [[ ! -f "${INPUT}" ]]; then
    echo "Input file not found: ${INPUT}" >&2
    exit 2
fi

echo "[$(date --iso-8601=seconds)] Starting P(k) measurement for ${INPUT}"
"${PYTHON_BIN}" "${PY_SCRIPT}" "${INPUT}"
echo "[$(date --iso-8601=seconds)] Done."
