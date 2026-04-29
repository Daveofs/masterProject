#!/usr/bin/env bash
# Validate LightconeShellBuilder against pkdgrav3's built-in lightcone (local run).
#
# Prerequisites: run_pkdgrav_local.sh must have completed, producing:
#   ${SNAP_DIR}/CosmoML_val.NNNNN   (Tipsy snapshots)
#   ${SNAP_DIR}/CosmoML_val.log     (pkdgrav3 step log)
#   ${SNAP_DIR}/CosmoML_val-shell_z-high=*_z-low=*.fits  (reference FITS shells)
#
# Usage:
#   ./run_validate_shell_builder_local.sh          # default paths
#   SNAP_DIR=/custom/path ./run_validate_shell_builder_local.sh
#   ./run_validate_shell_builder_local.sh --no-gpu # force CPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/masterProject"

# ── Paths (all overridable via environment) ────────────────────────────────
SNAP_DIR="${SNAP_DIR:-${REPO_ROOT}/outputs/pkdgrav_local}"
FITS_DIR="${FITS_DIR:-${SNAP_DIR}}"
PARAMS_YML="${PARAMS_YML:-${REPO_ROOT}/cosmogridv1/param_files/params.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/plots/shell_builder_validation}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/vir_env}"

SNAP_PREFIX="${SNAP_PREFIX:-CosmoML}"
NSIDE="${NSIDE:-2048}"        # 512 for speed; 2048 for full-resolution check
Z_MAX="${Z_MAX:-3.5}"
BOXSIZE="${BOXSIZE:-900.0}"

# ── Activate virtual environment ──────────────────────────────────────────
unset CONDA_DEFAULT_ENV CONDA_PREFIX VIRTUAL_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found in venv: ${PYTHON_BIN}" >&2; exit 3
fi

# ── Sanity checks ──────────────────────────────────────────────────────────
if [[ ! -f "${PARAMS_YML}" ]]; then
    echo "ERROR: params.yml not found: ${PARAMS_YML}" >&2
    echo "       Set PARAMS_YML to the correct path." >&2
    exit 2
fi

if [[ ! -f "${SNAP_DIR}/${SNAP_PREFIX}.log" ]]; then
    echo "ERROR: pkdgrav3 log not found: ${SNAP_DIR}/${SNAP_PREFIX}.log" >&2
    echo "       Run run_pkdgrav_local.sh first." >&2
    exit 2
fi

n_fits=$(ls "${FITS_DIR}/${SNAP_PREFIX}"-shell_z-high=*.fits 2>/dev/null | wc -l)
if [[ "${n_fits}" -eq 0 ]]; then
    echo "WARNING: No FITS reference shells found in ${FITS_DIR}." >&2
    echo "         Validation will compare counts only (no xcorr)." >&2
fi
echo "Found ${n_fits} FITS reference shells."

# ── GPU / CPU selection ────────────────────────────────────────────────────
GPU_FLAG=""
if [[ "${JAX_PLATFORM_NAME:-}" == "cpu" ]]; then
    GPU_FLAG="--no-gpu"
fi
# Allow passing --no-gpu directly to the script
for arg in "$@"; do
    [[ "$arg" == "--no-gpu" ]] && GPU_FLAG="--no-gpu"
done

export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"   # default CPU locally
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

mkdir -p "${OUTPUT_DIR}"

echo "Snap dir    : ${SNAP_DIR}"
echo "Prefix      : ${SNAP_PREFIX}"
echo "FITS dir    : ${FITS_DIR}"
echo "Params yml  : ${PARAMS_YML}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "nside       : ${NSIDE}"
echo "z_max       : ${Z_MAX}"
echo "Python      : ${PYTHON_BIN}"
echo "GPU flag    : ${GPU_FLAG:-<none, using GPU>}"
echo ""

"${PYTHON_BIN}" "${PROJECT_DIR}/pkdgrav/validate_shell_builder.py" \
    --snap-dir    "${SNAP_DIR}"    \
    --snap-prefix "${SNAP_PREFIX}" \
    --fits-dir    "${FITS_DIR}"    \
    --params-yml  "${PARAMS_YML}"  \
    --boxsize     "${BOXSIZE}"     \
    --nside       "${NSIDE}"       \
    --output-dir  "${OUTPUT_DIR}"  \
    --z-max       "${Z_MAX}"       \
    --no-gpu
    ${GPU_FLAG}   "$@"

echo ""
echo "Validation complete. Results in: ${OUTPUT_DIR}"
