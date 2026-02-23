#!/bin/bash
#SBATCH --job-name=downsample_cosmoIC
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --output=outputs/downsample_%j.out
#SBATCH --error=outputs/downsample_%j.err

set -euo pipefail

source /cluster/scratch/damrein/miniconda3/etc/profile.d/conda.sh
conda activate vir_env

PROJECT_DIR=/cluster/scratch/damrein/project
INPUT_ARCHIVE=${INPUT_ARCHIVE:-${PROJECT_DIR}/outputs/CosmoML_IC.npz}
TARGET_RES=${TARGET_RES:-512}
OUTPUT_ARCHIVE=${OUTPUT_ARCHIVE:-${PROJECT_DIR}/outputs/CosmoML_IC_res${TARGET_RES}.npz}
SEED=${SEED:-0}

if [[ ! -f "${INPUT_ARCHIVE}" ]]; then
    echo "Missing IC archive: ${INPUT_ARCHIVE}" >&2
    exit 2
fi

cd "${PROJECT_DIR}"

echo "[$(date --iso-8601=seconds)] Downsampling ${INPUT_ARCHIVE} -> res ${TARGET_RES}"
python tools/downsample_ic.py \
    --input "${INPUT_ARCHIVE}" \
    --output "${OUTPUT_ARCHIVE}" \
    --target-res "${TARGET_RES}" \
    --seed "${SEED}" || exit 1

echo "[$(date --iso-8601=seconds)] Done. Wrote ${OUTPUT_ARCHIVE}"
