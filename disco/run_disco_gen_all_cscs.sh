#!/bin/bash
# ============================================================
# Run DISCO-CUSTOM for ALL cosmo_*/run_* directories.
# Includes tipsy → HDF5 IC conversion before the simulation.
#
# Each array task picks up:
#   - IC file  : cosmo_*/run_*/CosmoML_*.00000*
#   - params   : cosmo_*/run_*/params.yml
#   - class    : cosmo_*/run_*/class_processed.hdf5
# Output goes into the same run dir.
#
# Usage — run on the LOGIN NODE:
#
#   1. Build the job list:
#        bash run_disco_gen_all_cscs.sh --build-list
#
#   2. Submit the SLURM array:
#        N=$(( $(wc -l < /users/damrein/masterProject/disco/job_list_disco_custom.txt) - 1 ))
#        sbatch --array=0-${N} run_disco_gen_all_cscs.sh
# ============================================================

#SBATCH --job-name=disco_custom_all
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/disco_custom/disco_gen_%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/disco_custom/disco_gen_%A_%a.err

set -euo pipefail

# ---- Paths -------------------------------------------------------
COSMOGRID_DIR="/capstor/scratch/cscs/damrein/cosmogridv1_test2"
PROJECT_DIR="/users/damrein/masterProject/disco"
JOB_LIST="${PROJECT_DIR}/job_list_disco_custom.txt"
CONDA_ENV="disco_custom"
CONDA_ENV_CONVERT="disco_lorenzo"
CONDA_INIT="$HOME/miniforge3/etc/profile.d/conda.sh"
METAINFO_FILE="/capstor/scratch/cscs/damrein/cosmogridv1/CosmoGridV1_metainfo.h5"
# ------------------------------------------------------------------

# ── Simulation parameters ─────────────────────────────────────────
RES=832
RES_PM=1664
BOXSIZE=900.0
NUMSTEPS=100
A_INI=0.01
A_END=1.0
# ------------------------------------------------------------------

# ============================================================
# --build-list : scan cosmo dirs and build the job list
# ============================================================
if [[ "${1:-}" == "--build-list" ]]; then
    echo "Building job list -> ${JOB_LIST}"
    > "${JOB_LIST}"
    skipped=0
    for cosmo_dir in "${COSMOGRID_DIR}"/cosmo_*/; do
        for run_dir in "${cosmo_dir}"run_*/; do

            # Find required files flexibly
            ic_file=$(ls "${run_dir}"CosmoML_*.00000* 2>/dev/null | grep -v '\.hdf5$' | head -n 1 || true)
            params_yml="${run_dir}params.yml"
            class_processed="${run_dir}class_processed.hdf5"

            # Need IC, params.yml, and class_processed.hdf5 to run
            if [ -z "$ic_file" ] || [ ! -f "$params_yml" ] || [ ! -f "$class_processed" ]; then
                continue
            fi

            # Skip if DISCO shells already exist in the run dir
            if ls "${run_dir}"*.fits &>/dev/null; then
                (( skipped++ )) || true
                continue
            fi

            echo "${run_dir}" >> "${JOB_LIST}"
        done
    done

    if [ ! -f "${JOB_LIST}" ]; then
        N=0
    else
        N=$(wc -l < "${JOB_LIST}")
    fi

    echo "Job list built: ${N} entries to run, ${skipped} skipped (already done)"
    echo ""
    if [ "${N}" -eq 0 ]; then
        echo "Nothing to submit — all DISCO runs already exist or files are missing."
    else
        echo "Submit with:"
        echo "  sbatch --array=0-$(( N - 1 )) $(realpath "${BASH_SOURCE[0]}")"
    fi
    exit 0
fi

# ============================================================
# SLURM array task: run DISCO-CUSTOM for one (cosmo, run) entry
# ============================================================

RUN_DIR=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "${JOB_LIST}")

if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "ERROR: no valid run_dir for task ${SLURM_ARRAY_TASK_ID} (got '${RUN_DIR}')"
    exit 1
fi

# Derive IDs from path
COSMO_KEY=$(basename "$(dirname "${RUN_DIR%/}")")   # e.g., cosmo_000001
run_id=$(basename "${RUN_DIR%/}")                   # e.g., run_0000

# Find the tipsy IC (exclude any existing .hdf5)
IC_FILE_TIPSY=$(ls "${RUN_DIR}"CosmoML_*.00000* 2>/dev/null | grep -v '\.hdf5$' | head -n 1)
PARAMS_YML="${RUN_DIR}params.yml"
CLASS_PROCESSED="${RUN_DIR}class_processed.hdf5"

if [ -z "${IC_FILE_TIPSY}" ]; then
    echo "ERROR: tipsy IC file not found in ${RUN_DIR}"; exit 2
fi
if [ ! -f "${PARAMS_YML}" ]; then
    echo "ERROR: params.yml not found: ${PARAMS_YML}"; exit 3
fi
if [ ! -f "${CLASS_PROCESSED}" ]; then
    echo "ERROR: class_processed.hdf5 not found: ${CLASS_PROCESSED}"; exit 4
fi

# ── Step 1: Convert tipsy IC → HDF5 ──────────────────────────────
IC_FILE_HDF5="${IC_FILE_TIPSY}.hdf5"

source "${CONDA_INIT}"

if [ ! -f "${IC_FILE_HDF5}" ]; then
    echo "[$(date --iso-8601=seconds)] Converting tipsy IC → HDF5 ..."
    echo "  Input : ${IC_FILE_TIPSY}"
    echo "  Output: ${IC_FILE_HDF5}"

    conda activate "${CONDA_ENV_CONVERT}"

    python - <<EOF
import sys
sys.path.insert(0, "${PROJECT_DIR}")
from read_tipsy_file import tipsy_to_hdf5

hdf5_file = tipsy_to_hdf5(
    tipsy_file   = "${IC_FILE_TIPSY}",
    output_hdf5  = "${IC_FILE_HDF5}",
    Lbox         = ${BOXSIZE},
    a            = ${A_INI},
    h            = 0.0,
    omega_m      = 0.0,
    omega_b      = 0.0,
    omega_lambda = 0.0,
)
print(f"Done: {hdf5_file}")
EOF

    echo "[$(date --iso-8601=seconds)] IC conversion complete."
else
    echo "[$(date --iso-8601=seconds)] HDF5 IC already exists, skipping conversion."
fi

# ── Step 2: Activate simulation env ──────────────────────────────
conda activate "${CONDA_ENV}"

SIMRUN_BIN=$(which simulation_run)

# ── Environment Variables ─────────────────────────────────────────
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_TRACEBACK_FILTERING=off

# Execution is done inside the run directory so output files land there
cd "${RUN_DIR}"

echo "[$(date --iso-8601=seconds)] Starting DISCO-CUSTOM (${COSMO_KEY}, ${run_id})"
echo "  IC file   : ${IC_FILE_HDF5}"
echo "  params.yml: ${PARAMS_YML}"
echo "  class     : ${CLASS_PROCESSED}"
echo "  working dir: ${RUN_DIR}"

srun --ntasks=$((SLURM_NNODES * 4)) --ntasks-per-node=4 \
    "${SIMRUN_BIN}" \
    --ics-file "${IC_FILE_HDF5}" \
    --res "${RES}" \
    --res-pm "${RES_PM}" \
    --boxsize "${BOXSIZE}" \
    --numsteps "${NUMSTEPS}" \
    --run-mode gpu \
    --double \
    --no-dump-xla \
    --name grid \
    --a-ini "${A_INI}" \
    --a-end "${A_END}" \
    --no-calculate-fof \
    --save-npz-snapshot \
    --grad-kernel-order 4 \
    --n-order 1 \
    --build-shells \
    --shells-metainfo "${METAINFO_FILE}" \
    --param-file "${PARAMS_YML}" \
    --class-processed "${CLASS_PROCESSED}" \
    --shells-cosmo-key "${COSMO_KEY}" \
    --shells-nside 2048 \
    --shells-z-min 0.0 \
    --shells-z-max 3.5 \
    --pre-steps 40
rc=$?

echo "[$(date --iso-8601=seconds)] Finished (exit=${rc}, ${COSMO_KEY}, ${run_id})"
exit ${rc}
