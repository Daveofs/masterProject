#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=transfer-logrho
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/logrho-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/logrho-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Validate the --log-density fix (2026-07-09): the linear-density transfer function
# matches Cl_high almost perfectly BEFORE the density>=0 clip, but the clip itself
# (needed on shot-noise shells, e.g. 59% of shell 3's pixels) biases Cl by ~20% at
# ALL ell (proven: pre-clip corrected/high ~1.00, post-clip ~1.2 on shell 3). Fix:
# correct log1p(rho) instead of rho -- reconstruction via expm1 is always >= -1
# (rho >= 0) so no clip is ever needed. This refits T/R from scratch in log space
# (can't reuse the linear-space T -- tested, gives the wrong answer) using the
# `fit` (train-averaged) method first as a fast/cheap validation before touching
# the MLP emulator.

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=14

DATA=/capstor/scratch/cscs/damrein/cosmogridv1
LMAX=3000
TEST_COSMO=cosmo_000122
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/logdensity_${SLURM_JOB_ID}
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/transfer

echo "==== log-density transfer-function validation | data=$DATA | lmax=$LMAX | test=$TEST_COSMO ===="

echo "[stage 1] preprocessing log-density alms (skips runs already done)"
python preprocess/preprocess_alms.py \
    --data-dir "$DATA" \
    --low-glob "disco_sim/*/disco_shells_nside=2048.npz" \
    --high-npz compressed_shells.npz \
    --lmax $LMAX --num-workers 5 --log-density

echo "[stage 2] fit (train-averaged) T/R in log-density space, leave-one-out on $TEST_COSMO"
python transfer/transfer_function.py fit \
    --data-dir "$DATA" --lmax $LMAX --test-cosmo $TEST_COSMO \
    --log-density --out "$OUT/transfer_log.npz"

echo "[stage 3] apply to $TEST_COSMO"
python transfer/transfer_function.py apply \
    --transfer "$OUT/transfer_log.npz" \
    --run-dir "$DATA/$TEST_COSMO/run_0" \
    --nside 2048 \
    --out "$OUT/cosmo_000122_corrected_logspace.npz" \
    --plot-shells 2 3 10 30 50 \
    --plot-dir "$OUT/cl_ratio_logspace"

echo "[$(date --iso-8601=seconds)] log-density pipeline finished -> $OUT"
