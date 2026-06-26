#!/bin/bash

#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --job-name=discodj-custom
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/disco_custom/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/disco_custom/slurm-%j.err
#SBATCH --chdir=/capstor/scratch/cscs/damrein/outputs/disco_custom

set -euo pipefail

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_TRACEBACK_FILTERING=off

# ── ICS / cosmology ───────────────────────────────────────────────────────
SNAP_DIR=/capstor/scratch/cscs/damrein/outputs/snapshots
LOG_DIR=/capstor/scratch/cscs/damrein/outputs/logs/disco_custom
OUT_DIR=/capstor/scratch/cscs/damrein/outputs
mkdir -p "${SNAP_DIR}" "${LOG_DIR}" "${OUT_DIR}"
mkdir -p "${OUT_DIR}/disco_sim"

ICS_FILE=/capstor/scratch/cscs/damrein/cosmogridv1_test3/cosmo_000001/run_0/grid00001.00000.hdf5
PARAM_YML=/capstor/scratch/cscs/damrein/cosmogridv1_test3/cosmo_000001/run_0/params.yml
CLASS_PROCESSED=/capstor/scratch/cscs/damrein/cosmogridv1_test3/cosmo_000001/run_0/class_processed.hdf5
CONDA_INIT=/users/damrein/miniforge3/etc/profile.d/conda.sh
CONDA_ENV=disco_custom

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

PYTHON_BIN=$(which python)
SIMRUN_BIN=$(which simulation_run)

echo "[$(date --iso-8601=seconds)] Starting DISCO-CUSTOM multi-GPU"

echo "Python:         ${PYTHON_BIN}"
echo "simulation_run: ${SIMRUN_BIN}"

METAINFO_FILE=/capstor/scratch/cscs/damrein/cosmogridv1/CosmoGridV1_metainfo.h5
COSMO_KEY=cosmo_000001
RES=832
RES_PM=1664
BOXSIZE=900.0
NUMSTEPS=100
A_INI=0.01
A_END=1.0

srun --ntasks=$((SLURM_NNODES * 4)) --ntasks-per-node=4 \
    "${SIMRUN_BIN}" \
    --ics-file "${ICS_FILE}" \
    --res "${RES}" \
    --res-pm "${RES_PM}" \
    --boxsize "${BOXSIZE}" \
    --numsteps "${NUMSTEPS}" \
    --run-mode gpu \
    --precision double \
    --no-dump-xla \
    --name grid \
    --a-ini "${A_INI}" \
    --a-end "${A_END}" \
    --no-calculate-fof \
    --save-npz-snapshot \
    --grad-kernel-order 0 \
    --n-order 1 \
    --build-shells \
    --shells-metainfo "${METAINFO_FILE}" \
    --param-file "${PARAM_YML}" \
    --class-processed "${CLASS_PROCESSED}" \
    --shells-cosmo-key "${COSMO_KEY}" \
    --shells-nside 2048 \
    --shells-z-min 0.0 \
    --shells-z-max 3.5 \
    --pre-steps 40 
echo "[$(date --iso-8601=seconds)] DISCO-CUSTOM run completed"