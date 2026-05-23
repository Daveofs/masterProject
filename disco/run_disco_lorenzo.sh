#!/bin/bash

#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --job-name=discodj-multigpu-lorenzo
#SBATCH --partition=debug
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/slurm-%j.err
#SBATCH --chdir=/capstor/scratch/cscs/damrein/outputs/disco_lorenzo

set -euo pipefail

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_TRACEBACK_FILTERING=off

# ── ICS / cosmology ───────────────────────────────────────────────────────
SNAP_DIR=/capstor/scratch/cscs/damrein/outputs/snapshots
LOG_DIR=/capstor/scratch/cscs/damrein/outputs/logs/disco_lorenzo
OUT_DIR=/capstor/scratch/cscs/damrein/outputs/disco_lorenzo

mkdir -p "${SNAP_DIR}" "${LOG_DIR}" "${OUT_DIR}"

ICS_FILE=/capstor/scratch/cscs/damrein/CosmoML_fiducial_Lorenzo.hdf5
CONDA_INIT=/users/damrein/miniforge3/etc/profile.d/conda.sh
CONDA_ENV=disco_lorenzo

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

PYTHON_BIN=$(which python)
SIMRUN_BIN=$(which simulation_run)

echo "Python:         ${PYTHON_BIN}"
echo "simulation_run: ${SIMRUN_BIN}"

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
    --n-order 3
