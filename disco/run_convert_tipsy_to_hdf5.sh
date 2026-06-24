#!/bin/bash
# Convert PKDGRAV tipsy ICs → HDF5 for use with simulation_run.
# Run from: /users/damrein/masterProject/disco

set -euo pipefail

CONDA_INIT=/users/damrein/miniforge3/etc/profile.d/conda.sh
CONDA_ENV=disco_lorenzo

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python - <<EOF
import sys
sys.path.insert(0, "${SCRIPT_DIR}")
from read_tipsy_file import tipsy_to_hdf5

file_ics        = '/capstor/scratch/cscs/damrein/cosmogridv1_test3/cosmo_000001/run_0/CosmoML_000001_run_0.00000'
snapshot_output = '/capstor/scratch/cscs/damrein/cosmogridv1_test3/cosmo_000001/run_0/CosmoML_000001_run_0.00000.hdf5'

# Set h, omega_m, omega_b, and omega_lambda to 0.0

hdf5_file = tipsy_to_hdf5(
    tipsy_file   = file_ics,
    output_hdf5  = snapshot_output,
    Lbox         = 900.0,
    a            = 0.01,
    h            = 0.0,
    omega_m      = 0.0,
    omega_b      = 0.0,
    omega_lambda = 0.0,
)

print(f"Done: {hdf5_file}")
EOF
