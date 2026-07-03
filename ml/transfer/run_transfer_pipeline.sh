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
#   2. T(ell,shell): produced by ONE of two methods (set METHOD below) ->
#        fit     : average Cl ratio over training cosmologies (test cosmo left out).
#        emulate : train an MLP emulator T=f(l,z,H0,O_cdm,Ob,Om,ns,s8,Cl_low) on the
#                  training cosmologies, then predict T for the held-out test cosmo.
#   3. apply      : scale the test cosmo's low alms by T, alm2map -> corrected shells,
#                   + Cl-ratio plots (low/high vs corrected/high).
# ============================================================================

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=14        # healpy map2alm is OpenMP-parallel (libsharp)

DATA=/capstor/scratch/cscs/damrein/cosmogridv1
LMAX=3000
TEST_COSMO=cosmo_000122
# How to build T(ell, shell): "fit" (train-averaged) or "emulate" (MLP emulator).
METHOD=${METHOD:-emulate}
# Emulator hyperparameters (METHOD=emulate only).
HIDDEN=${HIDDEN:-256,256,128}
MAX_ITER=${MAX_ITER:-200}
SAMPLE_FRAC=${SAMPLE_FRAC:-1.0}   # <1.0 subsamples ell per shell (faster training)
# Set INCLUDE_TEST=1 to fit/train T on the test cosmology too (sanity check: expect
# corrected/high ~ 1 by construction). Empty = proper leave-one-out.
INCLUDE_TEST=${INCLUDE_TEST:-}
FIT_FLAGS=""
[ -n "$INCLUDE_TEST" ] && FIT_FLAGS="--include-test"
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/transfer

echo "==== transfer-function pipeline | data=$DATA | lmax=$LMAX | test=$TEST_COSMO | method=$METHOD ===="

# ---- 1. Preprocess alms (skips runs already done). 5 workers x 14 OMP threads. ----
echo "[stage 1] preprocessing alms"
python preprocess_alms.py \
    --data-dir "$DATA" \
    --low-glob "disco_sim/*/disco_shells_nside=2048.npz" \
    --high-npz compressed_shells.npz \
    --lmax $LMAX --num-workers 5

# ---- 2. Build T(ell, shell) via the chosen method (test cosmology left out) ----
if [ "$METHOD" = "emulate" ]; then
    echo "[stage 2] training MLP emulator (hidden=$HIDDEN, max_iter=$MAX_ITER, sample_frac=$SAMPLE_FRAC)"
    python transfer/transfer_function.py train \
        --data-dir "$DATA" --lmax $LMAX \
        --test-cosmo $TEST_COSMO $FIT_FLAGS \
        --hidden "$HIDDEN" --max-iter $MAX_ITER --sample-frac $SAMPLE_FRAC \
        --out "$OUT/emulator.pkl"
    echo "[stage 2b] emulating T for $TEST_COSMO"
    python transfer/transfer_function.py emulate \
        --emulator "$OUT/emulator.pkl" \
        --run-dir "$DATA/$TEST_COSMO/run_0" \
        --out "$OUT/transfer.npz"
else
    echo "[stage 2] fitting T(ell, shell) (train-averaged)"
    python transfer/transfer_function.py fit \
        --data-dir "$DATA" --lmax $LMAX \
        --test-cosmo $TEST_COSMO $FIT_FLAGS --out "$OUT/transfer.npz"
fi

# ---- 3. Apply to the held-out test cosmology + Cl plots ----
echo "[stage 3] applying to $TEST_COSMO"
python transfer/transfer_function.py apply \
    --transfer "$OUT/transfer.npz" \
    --run-dir "$DATA/$TEST_COSMO/run_0" \
    --nside 2048 --ell-min 0 \
    --plot-shells 3 30 50 --plot-dir "$OUT/cl_ratio" \
    --out "$OUT/${TEST_COSMO}_corrected.npz"

echo "transfer-function pipeline ${SLURM_JOB_ID} finished at $(date) -> $OUT"
