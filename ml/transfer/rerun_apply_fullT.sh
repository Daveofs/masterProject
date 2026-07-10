#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=apply-fullT
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/apply-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/apply-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Regenerate cosmo_000122_corrected_fullT.npz + cl_ratio_fullT plots with the
# CURRENT transfer_function.py (existing file predates the density>=0 clip fix
# added 2026-07-06, commit 8eff187 -- it has negative pixel values as a result).
# Same transfer.npz (already fit/trained) and run-dir as job 3855493, plain full
# T (no --wiener), matching the "_fullT" naming.

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=32

OUT=/capstor/scratch/cscs/damrein/outputs/transfer/3855493
python transfer/transfer_function.py apply \
    --transfer "$OUT/transfer.npz" \
    --run-dir /capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000122/run_0 \
    --nside 2048 \
    --out "$OUT/cosmo_000122_corrected_fullT.npz" \
    --plot-shells 3 30 50 \
    --plot-dir "$OUT/cl_ratio_fullT"

echo "[$(date --iso-8601=seconds)] done regenerating cosmo_000122_corrected_fullT.npz"
