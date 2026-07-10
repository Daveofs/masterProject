#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=apply-debias
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/apply-debias-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/apply-debias-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Re-run `apply` only (transfer_log.npz already fit, no need to redo preprocessing)
# with the new mean-debiasing fix (see _debias_mean in transfer_function.py):
# fixes the "map looks much brighter" issue -- flooring negative pixels at 0
# after the log-density correction still biases the mean up to +25% on the
# faintest shells, because they're so sparse most pixels sit exactly at the
# log1p(rho)=0 clip boundary already.

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=32

OUT=/capstor/scratch/cscs/damrein/outputs/transfer/logdensity_3880886
python transfer/transfer_function.py apply \
    --transfer "$OUT/transfer_log.npz" \
    --run-dir /capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000122/run_0 \
    --nside 2048 \
    --out "$OUT/cosmo_000122_corrected_logspace_debiased.npz" \
    --plot-shells 0 2 3 10 30 50 \
    --plot-dir "$OUT/cl_ratio_logspace_debiased"

echo "[$(date --iso-8601=seconds)] done"
