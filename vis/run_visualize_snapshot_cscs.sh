#!/bin/bash
#SBATCH --job-name=visualize_npz
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/vis/visualize_snapshot_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/vis/visualize_snapshot_%j.err

# Creates a density slice PNG in outputs/plots

set -euo pipefail

PROJECT_DIR=/users/damrein/masterProject
SCRATCH=/capstor/scratch/cscs/damrein
CONDA_ENV=disco-dj
CONDA_ROOT=/users/damrein/miniforge3

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

if ! "${PYTHON_BIN}" -c "import numpy, matplotlib" >/dev/null 2>&1; then
  echo "Required packages missing in conda env '${CONDA_ENV}' (need numpy and matplotlib)." >&2
  echo "Install with: conda install -n ${CONDA_ENV} numpy matplotlib" >&2
  exit 3
fi

SNAPSHOT=${SCRATCH}/outputs/snapshots/final_multigpu_3009179.npz
OUTDIR=${SCRATCH}/outputs/plots/snapshots

mkdir -p "${OUTDIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "${SNAPSHOT}" ]]; then
  echo "Snapshot not found: ${SNAPSHOT}" >&2
  exit 2
fi

echo "Loading snapshot: ${SNAPSHOT} -> writing plots to ${OUTDIR}"

"${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, r"${PROJECT_DIR}/vis")
sys.path.insert(0, r"${PROJECT_DIR}/disco")

import numpy as np
from pathlib import Path
from read_tipsy_file import read_tipsy
from visualize import plot_density_slice

BOXSIZE = 900.0
snap = Path(r"${SNAPSHOT}")
outdir = Path(r"${OUTDIR}")

if snap.suffix.lower() == ".npz":
    data = np.load(snap, allow_pickle=False)
    if 'pos' in data:
        # Plain positions array
        pos = np.asarray(data['pos'])
        BOXSIZE = float(data['boxsize']) if 'boxsize' in data else BOXSIZE
        grid_val = 832
        slice_thickness_val = 5.0
    elif 'psi' in data:
        # DISCO-DJ output: psi is displacement field (res_x, res_y, res_z, 3)
        # Merge rank shards if this is a per-rank file (old format)
        import glob, re
        rank_pattern = re.sub(r'\.rank\d+\.npz$', '.rank*.npz', str(snap))
        rank_files = sorted(glob.glob(rank_pattern))
        if len(rank_files) > 1:
            print(f"Merging {len(rank_files)} rank shards: {[Path(f).name for f in rank_files]}")
            psi_all = np.concatenate([np.load(f, allow_pickle=False)['psi'] for f in rank_files], axis=0)
        else:
            psi_all = np.asarray(data['psi'])
        res_x, res_y, res_z = psi_all.shape[:3]
        print(f"psi shape: {psi_all.shape}  (res_x={res_x}, res_y={res_y}, res_z={res_z})")
        # Reconstruct Lagrangian grid q — each axis scaled to BOXSIZE independently
        ix, iy, iz = np.mgrid[0:res_x, 0:res_y, 0:res_z]
        q = np.stack([
            ix.astype(np.float32) * (BOXSIZE / res_x),
            iy.astype(np.float32) * (BOXSIZE / res_y),
            iz.astype(np.float32) * (BOXSIZE / res_z),
        ], axis=-1)
        # Eulerian positions = q + psi (periodic wrap)
        pos = ((q + psi_all).reshape(-1, 3)) % BOXSIZE
        del q, psi_all
        # Full projection along slice axis: equivalent to delta_mean from sim script
        grid_val = res_x
        slice_thickness_val = BOXSIZE
    else:
        raise KeyError(f"NPZ file {snap.name} has neither 'pos' nor 'psi' key. Keys: {list(data.keys())}")
else:
    # pkdgrav Tipsy binary (no extension or .00NNN)
    p, _ = read_tipsy(snap, BOXSIZE)
    pos = np.column_stack([p['x'], p['y'], p['z']])
    grid_val = 832
    slice_thickness_val = 5.0

plot_density_slice(
    positions=pos,
    boxsize=BOXSIZE,
    slice_axis=2,
    slice_center=None,
    slice_thickness=slice_thickness_val,
    grid=grid_val,
    input_file=snap,
    output_dir=outdir,
)
PY

echo "Done."
