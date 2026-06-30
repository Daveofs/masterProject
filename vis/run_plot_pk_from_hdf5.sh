#!/usr/bin/env bash
set -euo pipefail

# Activate the disco-dj conda environment
source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate disco-dj

# Move to the script directory so output files land relative to this script
cd "$(dirname "$0")"

# --- Defaults (edit these variables if you need different inputs) ---
PKDGRAV_FILE="/capstor/store/cscs/ska/sk037/grid_000001/CosmoML.00140.hdf5"
PKDGRAV_KEY="PartType1/Coordinates"

DISCO_BASE="/capstor/scratch/cscs/damrein/outputs/disco_custom/disco_sim/gpu_grid_3783890"
DISCO_KEY="PartType1/Coordinates"
# Number of DISCO-DJ shards (depends on number of GPUS used in the run)
DISCO_NSHARD=8 

LBOX=900.0
NGRID=832

# Where to write outputs (make sure this is writable on your system)
OUTPUT_DIR="/capstor/scratch/cscs/damrein/outputs/pk_backscaling"

mkdir -p "$OUTPUT_DIR"

echo "============================="
echo "Running plot_pk_from_hdf5"
echo "============================="

echo "Using inputs:"
echo "  PKDGRAV_FILE: $PKDGRAV_FILE"
echo "  PKDGRAV_KEY:  $PKDGRAV_KEY"
echo "  DISCO_BASE:   $DISCO_BASE"
echo "  DISCO_NSHARD: $DISCO_NSHARD"
echo "  DISCO_KEY:    $DISCO_KEY"
echo "  LBOX:         $LBOX"
echo "  NGRID:        $NGRID"
echo "  OUTPUT_DIR:   $OUTPUT_DIR"

python plot_pk_from_hdf5.py \
	--pkdgrav-file "$PKDGRAV_FILE" \
	--pkdgrav-key "$PKDGRAV_KEY" \
	--disco-base "$DISCO_BASE" \
	--disco-nshard "$DISCO_NSHARD" \
	--disco-key "$DISCO_KEY" \
	--lbox "$LBOX" \
	--ngrid "$NGRID" \
	--output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Output files written to: $OUTPUT_DIR"
