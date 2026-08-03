#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=prep-tcorr
#SBATCH --partition=normal
#SBATCH --account=a0158
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/prep-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/prep-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# One-time dataset preparation for the residual sphere-flow model (CPU only):
#   1. fit the transfer function T(ell, shell) leave-one-out
#   2. tcorr_shells_nside=2048.npy  = alm2map(low_alm * T)   per run
#   3. high_shells_nside=2048.npy   = decompressed CosmoGrid target per run
# Raw .npy -> mmap per-shell random access during training (no npz decompression).
# Requires low_alms_lmax3000.npy (run_transfer_pipeline.sh stage 1 / preprocess_alms).
# ============================================================================

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=14

DATA=/capstor/scratch/cscs/damrein/cosmogridv1
LMAX=3000
TEST_COSMO=cosmo_000122
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/dataset
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

# T fit leave-one-out (so the test cosmology's tcorr baseline is honest).
python transfer_function.py fit \
    --data-dir "$DATA" --lmax $LMAX \
    --test-cosmo $TEST_COSMO --out "$OUT/transfer_loo_${TEST_COSMO}.npz"

python prepare_tcorr_dataset.py \
    --data-dir "$DATA" \
    --transfer "$OUT/transfer_loo_${TEST_COSMO}.npz" \
    --lmax $LMAX --nside 2048 --num-workers 5

echo "prepare-tcorr ${SLURM_JOB_ID} finished at $(date)"
