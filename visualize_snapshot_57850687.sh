#!/bin/bash
# Visualization helper for final_snapshot_57850687.npz
# Creates a density slice PNG in outputs/plots

# Activate conda environment
source /cluster/scratch/damrein/miniconda3/etc/profile.d/conda.sh
conda activate vir_env

PROJECT_DIR=/cluster/scratch/damrein/project
SNAPSHOT=${PROJECT_DIR}/outputs/final_snapshot_57850687.npz
OUTDIR=${PROJECT_DIR}/outputs/plots

mkdir -p "${OUTDIR}"

if [[ ! -f "${SNAPSHOT}" ]]; then
  echo "Snapshot not found: ${SNAPSHOT}" >&2
  exit 2
fi

echo "Loading snapshot: ${SNAPSHOT} -> writing plots to ${OUTDIR}"

python - <<PY
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
    grid=256,
    output_dir=Path(r"${OUTDIR}")
)
PY

echo "Done."
