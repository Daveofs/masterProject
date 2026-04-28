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
SHELL_INDICES="10 11 12 13 14 15 16 17 18 19 20"

# Number of histogram bins
NBINS=100

# Cosmology (units: Omega_m, h). Particle masses are computed from Lbox/res.
# Edit these values to match the simulation cosmology.
OMEGA_M=0.3
H0_H=0.73
FSKY=1.0

# Optional box parameters (set these to enable M_box calculation)
# Lbox in comoving Mpc/h (h^-1 Mpc). The Python script assumes these units by default.
# res is particles per axis (res^3 = total particles)
LBOX_DISCO="900"      # e.g. 1000.0
RES_DISCO="832"       # e.g. 1024
LBOX_COSMOGRID="900"  # e.g. 1000.0
RES_COSMOGRID="832"   # e.g. 1024

# Require box parameters to be set by the user; these are needed to compute per-particle mass
if [[ -z "${LBOX_DISCO}" || -z "${RES_DISCO}" || -z "${LBOX_COSMOGRID}" || -z "${RES_COSMOGRID}" ]]; then
    echo "[ERROR] Please set LBOX_DISCO, RES_DISCO, LBOX_COSMOGRID, and RES_COSMOGRID at the top of this script before running." >&2
    exit 2
fi

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
PY_CMD=("${PY_SCRIPT}" \
    --disco      "${DISCO_FILE}" \
    --cosmogrid  "${COSMOGRID_FILE}" \
    --out-dir    "${OUT_DIR}" \
    --shells     ${SHELL_INDICES} \
    --nbins      "${NBINS}" \
    --omega-m    "${OMEGA_M}" \
    --h          "${H0_H}" \
    --fsky       "${FSKY}")

if [[ -n "${LBOX_DISCO}" ]]; then
    PY_CMD+=(--lbox-disco "${LBOX_DISCO}")
fi
if [[ -n "${RES_DISCO}" ]]; then
    PY_CMD+=(--res-disco "${RES_DISCO}")
fi
if [[ -n "${LBOX_COSMOGRID}" ]]; then
    PY_CMD+=(--lbox-cosmogrid "${LBOX_COSMOGRID}")
fi
if [[ -n "${RES_COSMOGRID}" ]]; then
    PY_CMD+=(--res-cosmogrid "${RES_COSMOGRID}")
fi

python "${PY_CMD[@]}"

echo "[$(date --iso-8601=seconds)] Done. Plots written to ${OUT_DIR}"
