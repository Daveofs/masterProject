#!/bin/bash
#SBATCH --job-name=pix_hist
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/hist/pix_hist_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/hist/pix_hist_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ROOT=/users/damrein/miniforge3
CONDA_ENV=disco-dj

DISCO_FILE="${SCRATCH_DIR}/outputs/shells_with_4gpu_spread_trash_multi_node/shells_nside=2048.npz"
COSMOGRID_FILE="${SCRATCH_DIR}/cosmogridv1/cosmo_000001/run_0/compressed_shells.npz"
OUT_DIR="${SCRATCH_DIR}/outputs/pixel_histogram"

PY_SCRIPT=/users/damrein/masterProject/tools/plot_pixel_histogram.py

# Shell indices to plot: 5 evenly spaced across the 69 shells (0-based)
# shell 0  z~[0.000, 0.013]
# shell 17 z~[0.285, 0.310]
# shell 34 z~[0.720, 0.780]
# shell 51 z~[1.540, 1.650]
# shell 68 z~[3.351, 3.500]
SHELL_INDICES="0 17 34 51 68"

# Number of histogram bins
NBINS=3000

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${SCRATCH_DIR}/outputs/logs/pix_hist_placeholder")"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
for f in "${DISCO_FILE}" "${COSMOGRID_FILE}" "${PY_SCRIPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] File not found: ${f}" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting pixel histogram computation"
echo "  DISCO:       ${DISCO_FILE}"
echo "  CosmoGridV1: ${COSMOGRID_FILE}"
echo "  Output dir:  ${OUT_DIR}"
echo "  Shells:      ${SHELL_INDICES}"
echo "  Bins:        ${NBINS}"

# shellcheck disable=SC2086
python "${PY_SCRIPT}" \
    --disco      "${DISCO_FILE}" \
    --cosmogrid  "${COSMOGRID_FILE}" \
    --out-dir    "${OUT_DIR}" \
    --shells     ${SHELL_INDICES} \
    --nbins      "${NBINS}"

echo "[$(date --iso-8601=seconds)] Done. Plots written to ${OUT_DIR}"
