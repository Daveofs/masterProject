#!/bin/bash
#SBATCH --job-name=pk_theory_comparison
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/pk_theory_comparison_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/pk_theory_comparison_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ROOT=/users/damrein/miniforge3
CONDA_ENV=disco-dj

PARAMS_YML="${SCRATCH_DIR}/cosmogridv1/cosmo_000001/run_0/params.yml"
HDF5_FILE="${SCRATCH_DIR}/cosmogridv1/cosmo_000001/run_0/class_processed.hdf5"
OUT_DIR="${SCRATCH_DIR}/outputs/plots"

PY_SCRIPT=/users/damrein/masterProject/tools/plot_pk_theory_comparison.py

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
mkdir -p "${SCRATCH_DIR}/outputs/logs"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
for f in "${PARAMS_YML}" "${HDF5_FILE}" "${PY_SCRIPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] File not found: ${f}" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting P(k) comparison"
echo "  Params YML: ${PARAMS_YML}"
echo "  HDF5 file:  ${HDF5_FILE}"
echo "  Output dir: ${OUT_DIR}"

python "${PY_SCRIPT}" \
    --params  "${PARAMS_YML}" \
    --hdf5    "${HDF5_FILE}" \
    --out-dir "${OUT_DIR}"

echo "[$(date --iso-8601=seconds)] Done"
