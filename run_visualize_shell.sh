#!/bin/bash
#SBATCH --job-name=visualize_shell
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/visualize_shell_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/visualize_shell_%j.err

# Creates a HEALPix mollview from compressed_shells.npz

set -euo pipefail

PROJECT_DIR=/cluster/scratch/damrein/project
SCRATCH=/cluster/scratch/damrein
CONDA_ENV=vir_env
OUTPUT_DIR=${SCRATCH}/outputs/plots/shells

if [[ -f /cluster/home/damrein/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/cluster/home/damrein/miniconda3
elif [[ -f /cluster/scratch/damrein/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/cluster/scratch/damrein/miniconda3
else
  echo "Could not find conda.sh under /cluster/home or /cluster/scratch miniconda3." >&2
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

if ! "${PYTHON_BIN}" -c "import numpy, matplotlib, healpy" >/dev/null 2>&1; then
  echo "Required packages missing in conda env '${CONDA_ENV}' (need numpy, matplotlib, healpy)." >&2
  echo "Install with: conda install -n ${CONDA_ENV} numpy matplotlib healpy" >&2
  exit 3
fi

SHELL_NPZ=${SCRATCH}/cosmogridv1/cosmo_000001/compressed_shells.npz

# Configuration for the visualization (set to your desired z-bin and nside)
ZBIN=5
NSIDE=2048

cd "${PROJECT_DIR}"

if [[ ! -f "${SHELL_NPZ}" ]]; then
  echo "Input file not found: ${SHELL_NPZ}" >&2
  exit 2
fi

echo "Loading shell file: ${SHELL_NPZ} (z-bin ${ZBIN})"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
from visualize import plot_shells

plot_shells(
    npz_path=Path(r"${SHELL_NPZ}"),
    z_bin=${ZBIN},
    nside=${NSIDE},
    output_dir=Path(r"${OUTPUT_DIR}"),
    plot_logarithmic=True
)
PY

echo "Done."
