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
PK_A=/capstor/scratch/cscs/damrein/outputs/pk_custom/pk_disco_concept_nu_omega_r_5.58e-5.txt
PK_B="/capstor/scratch/cscs/damrein/outputs/pk_custom/pk_disco_backscaling_omega_r_5.58e-5.txt"  # Optional additional spectrum; leave empty if not used
PK_C=/capstor/scratch/cscs/damrein/outputs/pk_custom/pk_pkd_standard_fiducial.txt
OUTPUT_PNG=${PK_DIR}/pk_backscaling_vs_concept_nu_omega_r_5.58e-5.png

LABEL_A="PK_A: DISCO concept nu omega_r=5.58e-5"
LABEL_B="PK_B: DISCO backscaling omega_r=5.58e-5"
LABEL_C="PK_C: PKD fiducial standard"

PY_SCRIPT=/users/damrein/masterProject/tools/plot_pk_from_txt.py

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
for f in "${PK_A}" "${PK_C}" "${PY_SCRIPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] File not found: ${f}" >&2
        exit 2
    fi
done

if [[ -n "${PK_B}" ]]; then
    if [[ ! -f "${PK_B}" ]]; then
        echo "[ERROR] File not found: ${PK_B}" >&2
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting two-file P(k) comparison"
echo "  Spectrum A: ${PK_A}"
if [[ -n "${PK_B}" ]]; then
    echo "  Spectrum B: ${PK_B}"
fi
echo "  Spectrum C (denominator): ${PK_C}"
echo "  Output PNG: ${OUTPUT_PNG}"

if [[ -n "${PK_B}" ]]; then
    python "${PY_SCRIPT}" \
        --pk-a "${PK_A}" \
        --pk-b "${PK_B}" \
        --pk-c "${PK_C}" \
        --label-a "${LABEL_A}" \
        --label-b "${LABEL_B}" \
        --label-c "${LABEL_C}" \
        --output "${OUTPUT_PNG}"
else
    python "${PY_SCRIPT}" \
        --pk-a "${PK_A}" \
        --pk-c "${PK_C}" \
        --label-a "${LABEL_A}" \
        --label-c "${LABEL_C}" \
        --output "${OUTPUT_PNG}"
fi

echo "[$(date --iso-8601=seconds)] Done. Output written to ${PK_DIR}"