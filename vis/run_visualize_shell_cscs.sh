#!/bin/bash
#SBATCH --job-name=visualize_shell
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/vis/visualize_shell_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/vis/visualize_shell_%j.err

# Creates a HEALPix mollview from compressed_shells.npz on CSCS

set -euo pipefail

PROJECT_DIR=/users/damrein/masterProject
SCRATCH=/capstor/scratch/cscs/damrein
CONDA_ENV=disco-dj
CONDA_ROOT=/users/damrein/miniforge3
OUTPUT_DIR=${SCRATCH}/outputs/plots/shells

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Could not find conda.sh under ${CONDA_ROOT}." >&2
  exit 4
fi

# Activate conda environment
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

if ! "${PYTHON_BIN}" -c "import numpy, matplotlib, healpy" >/dev/null 2>&1; then
  echo "Required packages missing in conda env '${CONDA_ENV}' (need numpy, matplotlib, healpy)." >&2
  echo "Install with: conda install -n ${CONDA_ENV} numpy matplotlib healpy" >&2
  exit 3
fi

#FILENAME="CosmoML-shell_z-high=1.46305_z-low=0.980198.fits"

SHELL=/capstor/scratch/cscs/damrein/outputs/shells_with_external_ics_multinode/shells_nside=2048.npz
# Configuration for the visualization (set to your desired z-bin and nside)
ZBIN=10
NSIDE=2048
NAME_SUFFIX="disco_with_external_and_multinode_nside${NSIDE}_zbin${ZBIN}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

export NAME_SUFFIX

if [[ ! -f "${SHELL}" ]]; then
  echo "Shell file not found: ${SHELL}" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<PY
import sys, os
from pathlib import Path
sys.path.insert(0, r"${PROJECT_DIR}/vis")
sys.path.insert(0, r"${PROJECT_DIR}/disco")
from visualize import plot_shells

plot_shells(
    npz_path=Path(r"${SHELL}"),
    z_bin=${ZBIN},
    nside=${NSIDE},
    output_dir=Path(r"${OUTPUT_DIR}"),
    plot_logarithmic=True,
    normalize=True,
    name=os.environ["NAME_SUFFIX"],
)
PY

echo "Done."
