#!/bin/bash
#SBATCH --job-name=cl_ratio
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/cl/cl_ratio_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/cl/cl_ratio_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ROOT=/users/damrein/miniforge3
CONDA_ENV=disco-dj

DISCO_FILE="/capstor/scratch/cscs/damrein/outputs/shells_with_4gpu_spread_trash_multi_node/shells_nside=2048.npz"
DISCO_FILE_1664="/capstor/scratch/cscs/damrein/outputs/shells_res_pm_1664/shells_nside=2048.npz"
COSMOGRID_FILE="${SCRATCH_DIR}/cosmogridv1/cosmo_000001/run_0/compressed_shells.npz"
OUT_DIR="${SCRATCH_DIR}/outputs/cl_ratio"

PY_SCRIPT=/users/damrein/masterProject/tools/plot_cl_ratio.py

# Shell indices to plot: 5 evenly spaced across the 69 shells (0-based)
# shell 0  z~[0.000, 0.013]
# shell 17 z~[0.285, 0.310]
# shell 34 z~[0.720, 0.780]
# shell 51 z~[1.540, 1.650]
# shell 68 z~[3.351, 3.500]
SHELL_INDICES="68"

# Maximum multipole (default: 3*2048-1 = 6143, can reduce for speed)
LMAX=3000

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${SCRATCH_DIR}/outputs/logs/cl_ratio_placeholder")"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
for f in "${DISCO_FILE}" "${DISCO_FILE_1664}" "${COSMOGRID_FILE}" "${PY_SCRIPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] File not found: ${f}" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting Cl ratio computation"
echo "  DISCO:      ${DISCO_FILE}"
echo "  DISCO_1664: ${DISCO_FILE_1664}"
echo "  CosmoGridV1: ${COSMOGRID_FILE}"
echo "  Output dir: ${OUT_DIR}"
echo "  Shells:     ${SHELL_INDICES}"
echo "  lmax:       ${LMAX}"

# shellcheck disable=SC2086
python "${PY_SCRIPT}" \
    --disco      "${DISCO_FILE}" \
    --disco-1664 "${DISCO_FILE_1664}" \
    --cosmogrid  "${COSMOGRID_FILE}" \
    --out-dir    "${OUT_DIR}" \
    --shells     ${SHELL_INDICES} \
    --lmax       "${LMAX}" \
    --lbox       900 \
    --res-pm     1664

echo "[$(date --iso-8601=seconds)] Done. Plots written to ${OUT_DIR}"
