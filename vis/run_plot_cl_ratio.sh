#!/bin/bash
#SBATCH --job-name=cl_ratio
#SBATCH --account=sk037
#SBATCH --partition=debug
#SBATCH --time=00:30:00
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

DISCO_FILE="/capstor/scratch/cscs/damrein/grid/cosmo_203356/run_0/disco_sim/gpu_grid_2746372/disco_shells_nside=2048.npz"
# Set to "None" or leave empty to make it optional
DISCO_FILE_1664=None
COSMOGRID_FILE="/capstor/scratch/cscs/damrein/grid/cosmo_203356/run_0/compressed_shells.npz"
PARAMS_YML="/capstor/scratch/cscs/damrein/cosmogridv1_fiducial_test2/run_0000/params.yml"
OUT_DIR="${SCRATCH_DIR}/outputs/cl_ratio"

PY_SCRIPT=/users/damrein/masterProject/vis/plot_cl_ratio.py

# Shell indices to plot: 5 evenly spaced across the 69 shells (0-based)
SHELL_INDICES="3 65"

# Maximum multipole (default: 3*2048-1 = 6143, can reduce for speed)
LMAX=3000

# Toggle optional curves in plots
SHOW_THEORY=false
SHOW_RESID=false

# Custom legend labels
LABEL_DISCO="Disco"
LABEL_DISCO_1664="Ignore"
LABEL_COSMOGRID="Cosmogridv1 - cosmo_203356"
LABEL_THEORY="CCL theory"
LABEL_RESID="DRF - CosmoGrid (resid)"
LABEL_RESID_1664="Ignore - CosmoGrid (resid)"

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
for f in "${DISCO_FILE}" "${DISCO_FILE_1664}" "${COSMOGRID_FILE}" "${PARAMS_YML}" "${PY_SCRIPT}"; do
    # Skip validation if the file is explicitly marked as "None" or empty
    if [[ "${f}" == "None" || -z "${f}" ]]; then
        continue
    fi

    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] File not found: ${f}" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting Cl ratio computation"
echo "  DISCO:       ${DISCO_FILE}"
echo "  DISCO_1664:  ${DISCO_FILE_1664}"
echo "  CosmoGridV1: ${COSMOGRID_FILE}"
echo "  Params YML:  ${PARAMS_YML}"
echo "  Output dir:  ${OUT_DIR}"
echo "  Shells:      ${SHELL_INDICES}"
echo "  lmax:        ${LMAX}"
echo "  showTheory:  ${SHOW_THEORY}"
echo "  showResid:   ${SHOW_RESID}"

# Handle conditional flags
THEORY_FLAG="--show-theory"
if [[ "${SHOW_THEORY}" != "true" ]]; then
    THEORY_FLAG="--no-show-theory"
fi

RESID_FLAG="--show-resid"
if [[ "${SHOW_RESID}" != "true" ]]; then
    RESID_FLAG="--no-show-resid"
fi

DISCO_1664_FLAG=""
if [[ "${DISCO_FILE_1664}" != "None" && -n "${DISCO_FILE_1664}" ]]; then
    DISCO_1664_FLAG="--disco-1664 ${DISCO_FILE_1664}"
fi

# shellcheck disable=SC2086
python "${PY_SCRIPT}" \
    --disco      "${DISCO_FILE}" \
    ${DISCO_1664_FLAG} \
    --cosmogrid  "${COSMOGRID_FILE}" \
    --out-dir    "${OUT_DIR}" \
    --shells     ${SHELL_INDICES} \
    --lmax       "${LMAX}" \
    --lbox       900 \
    --res-pm     1664 \
    --params-yml "${PARAMS_YML}" \
    ${THEORY_FLAG} \
    ${RESID_FLAG} \
    --label-disco "${LABEL_DISCO}" \
    --label-disco-1664 "${LABEL_DISCO_1664}" \
    --label-cosmogrid "${LABEL_COSMOGRID}" \
    --label-theory "${LABEL_THEORY}" \
    --label-resid "${LABEL_RESID}" \
    --label-resid-1664 "${LABEL_RESID_1664}"

echo "[$(date --iso-8601=seconds)] Done. Plots written to ${OUT_DIR}"