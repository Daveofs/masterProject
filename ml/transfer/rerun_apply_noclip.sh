#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=apply-noclip
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/apply-noclip-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/apply-noclip-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Cl-OPTIMAL product (2026-07-09): DENSITY-space transfer (3855493's emulated T),
# no positivity clipping. Measured on shell 3: mean ratio 0.998 and corrected/high
# = 1.00 / 0.99 / 0.93 at ell 50-150 / 400-800 / 800-1500 -- strictly better than
# 3855493's clipped output (1.215 and 1.17/0.88/0.74) on BOTH mean and small-scale
# Cl. Output is an OVERDENSITY field (rho<0 on faint shot-noise shells), not a
# count map; the count-map version needs lognormal-lambda + Poisson resampling.
#
# NOTE: log-density is deliberately NOT used here -- the expm1 reconstruction
# destroys small-scale DENSITY power (shell 3: 0.53/0.41 at ell 400-800/800-1500).

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=32

OUT=/capstor/scratch/cscs/damrein/outputs/transfer/noclip_${SLURM_JOB_ID}
mkdir -p "$OUT"

python transfer/transfer_function.py apply \
    --transfer /capstor/scratch/cscs/damrein/outputs/transfer/3855493/transfer.npz \
    --run-dir /capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000122/run_0 \
    --nside 2048 --no-clip \
    --out "$OUT/cosmo_000122_corrected_noclip.npz" \
    --plot-shells 0 2 3 10 30 50 \
    --plot-dir "$OUT/cl_ratio_noclip"

echo "[$(date --iso-8601=seconds)] done -> $OUT"
