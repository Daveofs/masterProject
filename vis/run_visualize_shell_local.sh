#!/bin/bash

# Local macOS runner for visualize.plot_shells on compressed_shells.npz
# Usage:
#  ./run_visualize_shell_local.sh
# Optional overrides:
#   ZBIN=5 NSIDE=512 NAME_SUFFIX=my_plot PLOT_LOG=true bash run_visualize_shell_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN_DEFAULT="${REPO_ROOT}/vir_env/bin/python"
INPUT_FILE_DEFAULT="/Users/david/projects/outputs/plots/shell_builder_validation/built_maps/CosmoML-built-shell_step=00104_z-high=0.7010866_z-low=0.6713487.fits"
#INPUT_FILE_DEFAULT="/Users/david/projects/outputs/pkdgrav_local/CosmoML-shell_z-high=0.1358373_z-low=0.1211429.fits"
OUTPUT_DIR_DEFAULT="${REPO_ROOT}/outputs/plots/shells"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"
INPUT_FILE="${INPUT_FILE:-${INPUT_FILE_DEFAULT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}"

ZBIN="${ZBIN:-10}"
NSIDE="${NSIDE:-128}"
NAME_SUFFIX="${NAME_SUFFIX:-own_pkdgrav_vel}"
PLOT_LOG="${PLOT_LOG:-true}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 6
fi

if ! "${PYTHON_BIN}" -c "import numpy, matplotlib, healpy" >/dev/null 2>&1; then
  echo "Required packages missing (need numpy, matplotlib, healpy) in: ${PYTHON_BIN}" >&2
  exit 3
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Input shell file not found: ${INPUT_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "Loading shell file: ${INPUT_FILE} (z-bin ${ZBIN}, nside ${NSIDE})"
echo "Writing plot to: ${OUTPUT_DIR}"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
from visualize import plot_shells

plot_shells(
  npz_path=Path(r"${INPUT_FILE}"),
    z_bin=int(${ZBIN}),
    nside=int(${NSIDE}),
    output_dir=Path(r"${OUTPUT_DIR}"),
    plot_logarithmic=str(r"${PLOT_LOG}").lower() in {"1", "true", "yes", "y", "on"},
    name=r"${NAME_SUFFIX}",
)
PY

echo "Done."
