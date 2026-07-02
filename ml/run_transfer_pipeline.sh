#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=transfer-fn
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Per-ell TRANSFER-FUNCTION small-scale correction (DISCO -> CosmoGrid).
#
#   corrected_alm(ell,m) = low_alm(ell,m) * T(ell, shell),  T = sqrt(<Cl_high>/<Cl_low>)
#
# CPU-only (healpy). Three stages:
#   1. preprocess : map2alm the DISCO (disco_sim/.../disco_shells_nside=2048.npz) and
#                   CosmoGrid (compressed_shells.npz) shells -> low/high_alms_lmax*.npy
#                   (one-time; mmap-able -> fast downstream, like the harmonic flow).
#   2. fit        : average Cl ratio over training cosmologies (test cosmo left out).
#   3. apply      : scale the test cosmo's low alms by T, alm2map -> corrected shells,
#                   + Cl-ratio plots (low/high vs corrected/high).
# ============================================================================

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=14        # healpy map2alm is OpenMP-parallel (libsharp)

DATA=/capstor/scratch/cscs/damrein/cosmogridv1
LMAX=3000
TEST_COSMO=cosmo_000122
# Set INCLUDE_TEST=1 to fit T on the test cosmology too (sanity check: expect
# corrected/high ~ 1 by construction). Empty = proper leave-one-out.
INCLUDE_TEST=${INCLUDE_TEST:-}
FIT_FLAGS=""
[ -n "$INCLUDE_TEST" ] && FIT_FLAGS="--include-test"
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/transfer

echo "==== transfer-function pipeline | data=$DATA | lmax=$LMAX | test=$TEST_COSMO ===="

# ---- 1. Preprocess alms (skips runs already done). 5 workers x 14 OMP threads. ----
echo "[stage 1] preprocessing alms"
python preprocess_alms.py \
    --data-dir "$DATA" \
    --low-glob "disco_sim/*/disco_shells_nside=2048.npz" \
    --high-npz compressed_shells.npz \
    --lmax $LMAX --num-workers 5

# ---- 2. Fit transfer function (leave the test cosmology out) ----
echo "[stage 2] fitting T(ell, shell)"
python transfer_function.py fit \
    --data-dir "$DATA" --lmax $LMAX \
    --test-cosmo $TEST_COSMO $FIT_FLAGS --out "$OUT/transfer.npz"

# ---- 3. Apply to the held-out test cosmology + Cl plots ----
echo "[stage 3] applying to $TEST_COSMO"
python transfer_function.py apply \
    --transfer "$OUT/transfer.npz" \
    --run-dir "$DATA/$TEST_COSMO/run_0" \
    --nside 2048 --ell-min 0 \
    --plot-shells 3 30 50 --plot-dir "$OUT/cl_ratio" \
    --out "$OUT/${TEST_COSMO}_corrected.npz"

echo "transfer-function pipeline ${SLURM_JOB_ID} finished at $(date) -> $OUT"
