#!/bin/bash

#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --job-name=discodj-multigpu-lorenzo
#SBATCH --partition=debug
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
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
OUT_DIR=/capstor/scratch/cscs/damrein/outputs/disco_custom
mkdir -p "${SNAP_DIR}" "${LOG_DIR}" "${OUT_DIR}"
mkdir -p "${OUT_DIR}/data/output"

ICS_FILE=/capstor/scratch/cscs/damrein/cosmogridv1_fiducial/run_0000/standard/CosmoML.00000.0.7399441599845886.hdf5
CONDA_INIT=/users/damrein/miniforge3/etc/profile.d/conda.sh
CONDA_ENV=disco_custom

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

PYTHON_BIN=$(which python)
SIMRUN_BIN=$(which simulation_run)

echo "Python:         ${PYTHON_BIN}"
echo "simulation_run: ${SIMRUN_BIN}"

METAINFO_FILE=/capstor/scratch/cscs/damrein/cosmogridv1/CosmoGridV1_metainfo.h5

srun --ntasks=$((SLURM_NNODES * 4)) --ntasks-per-node=4 \
    "${SIMRUN_BIN}" \
    --padded-sim \
    --ics-file "${ICS_FILE}" \
    --res 832 \
    --res-pm 832 \
    --boxsize 900 \
    --numsteps 100 \
    --run-mode gpu \
    --no-dump-xla \
    --name fiducial \
    --a-ini 0.01 \
    --a-end 1.0 \
    --cosmo PKdgrav_fiducial \
    --no-calculate-fof \
    --save-hdf5-snapshot \
    --grad-kernel-order 4 \
    --n-order 3 \
    --build-shells \
    --shells-metainfo "${METAINFO_FILE}"
