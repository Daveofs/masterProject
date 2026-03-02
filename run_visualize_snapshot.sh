#!/bin/bash
#SBATCH --job-name=visualize_npz
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/cluster/scratch/damrein/outputs/visualize_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/visualize_%j.err

# Creates a density slice PNG in outputs/plots

set -euo pipefail

PROJECT_DIR=/cluster/scratch/damrein/project
SCRATCH=/cluster/scratch/damrein
CONDA_ENV=vir_env

if [[ -f /cluster/home/damrein/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/cluster/home/damrein/miniconda3
elif [[ -f /cluster/scratch/damrein/miniconda3/etc/profile.d/conda.sh ]]; then
  CONDA_ROOT=/cluster/scratch/damrein/miniconda3
else
  echo "Could not find conda.sh under /cluster/home or /cluster/scratch miniconda3." >&2
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

if ! "${PYTHON_BIN}" -c "import numpy, matplotlib" >/dev/null 2>&1; then
  echo "Required packages missing in conda env '${CONDA_ENV}' (need numpy and matplotlib)." >&2
  echo "Install with: conda install -n ${CONDA_ENV} numpy matplotlib" >&2
  exit 3
fi

SNAPSHOT=${SCRATCH}/outputs/snapshots/final_snapshot_58356355.npz
OUTDIR=${SCRATCH}/outputs/plots/snapshots

mkdir -p "${OUTDIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "${SNAPSHOT}" ]]; then
  echo "Snapshot not found: ${SNAPSHOT}" >&2
  exit 2
fi

echo "Loading snapshot: ${SNAPSHOT} -> writing plots to ${OUTDIR}"

"${PYTHON_BIN}" - <<PY
import numpy as np
from pathlib import Path
from visualize import plot_density_slice

snap = Path(r"${SNAPSHOT}")
data = np.load(snap, allow_pickle=False)
pos = data['pos']
boxsize = float(data.get('boxsize', 900.0))

# call the plotting helper; tweak args below as desired
plot_density_slice(
    positions=pos,
    boxsize=boxsize,
    slice_axis=2,
    slice_center=None,
    slice_thickness=5.0,
    grid=832,
    output_dir=Path(r"${OUTDIR}")
)
PY

echo "Done."
