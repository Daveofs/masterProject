#!/usr/bin/env bash
set -euo pipefail

# Activate the disco-dj conda environment
source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate disco-dj

# Move to the script directory so output files land here
cd "$(dirname "$0")"

echo "============================="
echo "Running compare_pk_from_hdf5"
echo "============================="

python compare_pk_from_hdf5.py

echo ""
echo "Done. Output files written to: $(pwd)"
