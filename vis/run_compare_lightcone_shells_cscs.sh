#!/bin/bash
#SBATCH --job-name=compare_lc_shells
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/vis/compare_lc_shells_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/vis/compare_lc_shells_%j.err

# Side-by-side HEALPix comparison: DISCO-DJ lightcone vs CosmoGrid compressed_shells

set -euo pipefail

PROJECT_DIR=/users/damrein/masterProject
SCRATCH=/capstor/scratch/cscs/damrein
CONDA_ENV=disco-dj
CONDA_ROOT=/users/damrein/miniforge3
OUTPUT_DIR=${SCRATCH}/outputs/plots/shells/compare

# ── Input files ─────────────────────────────────────────────────────────────
# If LIGHTCONE is empty, the script will calculate diff map between DISCO and COSMO.
# To compute the diff, set LIGHTCONE="" (empty string) below.
LIGHTCONE=""
DISCO_SHELLS="${SCRATCH}/outputs/shells_with_4gpu_spread_trash_multi_node/shells_nside=2048.npz"
COSMO_SHELLS="${SCRATCH}/cosmogridv1/cosmo_000001/run_0/compressed_shells.npz"

# ── Visualization settings ───────────────────────────────────────────────────
# z-bin index (0-based) – must be in range [0, N_shells-1].
# The lightcone covers z~0-0.197, so valid bins are 0-13.
# Set ALL_BINS=1 to render all shells that overlap the lightcone instead.
ZBIN=5
ALL_BINS=0

NSIDE=2048
# Color scale for log10(1.01+delta); use empty string to use auto scaling
VMIN=-2.0
VMAX=2.0

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Could not find conda.sh under ${CONDA_ROOT}." >&2
  exit 4
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

ENV_PREFIX=$(conda env list | awk -v env_name="${CONDA_ENV}" '$1 == env_name {print $NF; exit}')
if [[ -z "${ENV_PREFIX}" ]]; then
  echo "Conda env '${CONDA_ENV}' not found." >&2
  exit 5
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found in env '${CONDA_ENV}': ${PYTHON_BIN}" >&2
  exit 6
fi

# Check Python packages
if ! "${PYTHON_BIN}" -c "import numpy, matplotlib, healpy" >/dev/null 2>&1; then
  echo "Required packages missing in conda env '${CONDA_ENV}' (need numpy, matplotlib, healpy)." >&2
  exit 3
fi

# Verify input files. Behavior depends on whether LIGHTCONE is provided.
if [[ -n "${LIGHTCONE}" ]]; then
  if [[ ! -f "${LIGHTCONE}" ]]; then
    echo "Input file not found: ${LIGHTCONE}" >&2
    exit 2
  fi
fi

DISCO_OK=0
COSMO_OK=0
if [[ -n "${DISCO_SHELLS}" && -f "${DISCO_SHELLS}" ]]; then DISCO_OK=1; fi
if [[ -n "${COSMO_SHELLS}" && -f "${COSMO_SHELLS}" ]]; then COSMO_OK=1; fi

if [[ -z "${LIGHTCONE}" ]]; then
  # No lightcone -> need both shells to compute a diff
  if [[ "${DISCO_OK}" -ne 1 || "${COSMO_OK}" -ne 1 ]]; then
    echo "When LIGHTCONE is not provided both DISCO_SHELLS and COSMO_SHELLS must exist." >&2
    exit 2
  fi
else
  # Lightcone provided -> need at least one shell for z-bin metadata
  if [[ "${DISCO_OK}" -ne 1 && "${COSMO_OK}" -ne 1 ]]; then
    echo "At least one of DISCO_SHELLS or COSMO_SHELLS must exist." >&2
    exit 2
  fi
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH}/outputs/logs/vis"

if [[ -n "${LIGHTCONE}" ]]; then
  echo "Lightcone  : ${LIGHTCONE}"
else
  echo "Lightcone  : (none) -- computing DISCO - COSMO diff"
fi
echo "DiscoShells: ${DISCO_SHELLS}"
echo "CosmoShells: ${COSMO_SHELLS}"
echo "z-bin     : ${ZBIN}  (all_bins=${ALL_BINS})"
echo "nside     : ${NSIDE}"
echo "Output    : ${OUTPUT_DIR}"

# ── Build argument list ───────────────────────────────────────────────────────
PY_ARGS=()
if [[ -n "${LIGHTCONE}" ]]; then PY_ARGS+=( --lightcone "${LIGHTCONE}" ); fi
if [[ -n "${DISCO_SHELLS}" ]]; then PY_ARGS+=( --disco-shells "${DISCO_SHELLS}" ); fi
if [[ -n "${COSMO_SHELLS}" ]]; then PY_ARGS+=( --cosmo-shells "${COSMO_SHELLS}" ); fi

EXTRA_ARGS=()
if [[ "${ALL_BINS}" -eq 1 ]]; then EXTRA_ARGS+=( --all-bins ); fi
if [[ -n "${VMIN}" ]]; then EXTRA_ARGS+=( --vmin "${VMIN}" ); fi
if [[ -n "${VMAX}" ]]; then EXTRA_ARGS+=( --vmax "${VMAX}" ); fi

cd "${PROJECT_DIR}"

# use --separate to write each shell to a separate file
"${PYTHON_BIN}" vis/compare_lightcone_shells.py "${PY_ARGS[@]}" \
  --z-bin       "${ZBIN}" \
  --nside       "${NSIDE}" \
  --output-dir  "${OUTPUT_DIR}" \
  --plot-logarithmic \
  --plot-counts \
  "${EXTRA_ARGS[@]}"

echo "Done."
