#!/bin/bash
#SBATCH --job-name=pk_two_txt_cmp
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/pk_two_txt_cmp_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/pk_two_txt_cmp_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ROOT=/users/damrein/miniforge3
CONDA_ENV=disco-dj

PK_DIR=${SCRATCH_DIR}/pk
PK_A=/capstor/scratch/cscs/damrein/pk/pk_pkd_fiducial_david.txt
PK_B=/capstor/scratch/cscs/damrein/outputs/pk_snapshot_comparison/pk_snapshot_comparison_fiducial_photons_with_bullfrog_and_symplectic_b.txt
OUTPUT_PNG=${PK_DIR}/pk_pkd_fiducial_vs_photons.png

LABEL_A="PK_A: PKD fiducial standard"
LABEL_B="PK_B: PKD fiducial photons"

PY_SCRIPT=/users/damrein/masterProject/tools/plot_two_pk_ratio.py

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p "${SCRATCH_DIR}/outputs/logs"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
for f in "${PK_A}" "${PK_B}" "${PY_SCRIPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] File not found: ${f}" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting two-file P(k) comparison"
echo "  Spectrum A: ${PK_A}"
echo "  Spectrum B: ${PK_B}"
echo "  Output PNG: ${OUTPUT_PNG}"

python "${PY_SCRIPT}" \
    --pk-a "${PK_A}" \
    --pk-b "${PK_B}" \
    --label-a "${LABEL_A}" \
    --label-b "${LABEL_B}" \
    --output "${OUTPUT_PNG}"

echo "[$(date --iso-8601=seconds)] Done. Output written to ${PK_DIR}"