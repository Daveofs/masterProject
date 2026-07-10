#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=transfer-fn
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=10:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Per-ell TRANSFER-FUNCTION small-scale correction (DISCO -> CosmoGrid), DENSITY
# space, PLUS lognormal+Poisson re-discretization into a valid count map.
#
#   corrected_alm(ell,m) = low_alm(ell,m) * T(ell, shell),  T = sqrt(<Cl_high>/<Cl_low>)
#
# CPU-only (healpy). Four stages:
#   1. preprocess : map2alm the DISCO (disco_sim/.../disco_shells_nside=2048.npz) and
#                   CosmoGrid (compressed_shells.npz) shells -> low/high_alms_lmax*.npy
#                   (one-time; mmap-able -> fast downstream, like the harmonic flow).
#                   DENSITY space only -- do NOT add --log-density: measured 2026-07-09
#                   to destroy small-scale Cl (0.41 vs 0.93 at ell 800-1500 on shell 3,
#                   because expm1 is too nonlinear for a sparse field). Dead end, kept
#                   only as a documented warning, not used here.
#   2. T(ell,shell): produced by ONE of two methods (set METHOD below) ->
#        fit     : average Cl ratio over training cosmologies (test cosmo left out).
#        emulate : train an MLP emulator T=f(l,z,H0,O_cdm,Ob,Om,ns,s8,Cl_low) on the
#                  training cosmologies, then predict T for the held-out test cosmo.
#   3. apply --no-clip : scale the test cosmo's low alms by T, alm2map -> a CONTINUOUS
#                  overdensity field (rho can go negative on faint/shot-noise shells)
#                  + Cl-ratio plots. --no-clip is the Cl-OPTIMAL choice: measured
#                  2026-07-09 that ANY positivity enforcement (clip@0, additive mean-
#                  debias) SUPPRESSES small-scale Cl (shell 3 at ell 800-1500: no-clip
#                  0.93, clip 0.74, debias 0.61) because a Gaussian-ish field with
#                  nbar~0.1 and CosmoGrid's small-scale power MUST go negative -- the
#                  true CosmoGrid shell is a sparse COUNT field (91.8% exact zeros),
#                  not Gaussian, so clipping always distorts the shape, not just the
#                  sign. This stage's output is therefore NOT a valid count map (no
#                  exact zeros, has negatives) -- correct for Cl/lensing work, wrong
#                  for anything that reads it as a density/count map (histograms,
#                  log(delta) plots, mollview will look "too bright").
#   4. poisson_resample.py : turns stage 3's output into a valid non-negative INTEGER
#                  count map via lognormal-intensity + Poisson resampling (shot-noise
#                  deconvolution -> lognormal transform -> Gaussian-random-field g,
#                  DISCO's phases below --ell-c tapered into an independent Gaussian
#                  realization above (r(ell) shows phases are noise there anyway) ->
#                  lambda=lbar*exp(g-sigma^2/2) -> Poisson draw). Measured 2026-07-09
#                  on shells 3/10/30: mean exact, sparsity within ~1pp of truth,
#                  Cl_counts/Cl_high within ~1-7% at every ell band tested -- i.e. this
#                  is the stage that recovers what --no-clip sacrifices, WITHOUT giving
#                  back the Cl accuracy (unlike clip/debias). ~150s/shell (n_avg=4 x
#                  n_iter=5 Poisson draws for the per-ell Cl calibration) -> budget
#                  ~3h for all 69 shells; that is why this script's time limit is 10h.
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
# Stage 4 (Poisson resample) cost/quality knobs -- see poisson_resample.py docstring.
# Set SKIP_POISSON=1 to stop after stage 3 (Cl-optimal continuous field only).
SKIP_POISSON=${SKIP_POISSON:-}
ELL_C=${ELL_C:-300}
TAPER=${TAPER:-100}
N_AVG=${N_AVG:-4}
N_ITER=${N_ITER:-5}
DAMP=${DAMP:-0.4}
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/transfer

echo "==== transfer-function pipeline | data=$DATA | lmax=$LMAX | test=$TEST_COSMO | method=$METHOD ===="

# ---- 1. Preprocess alms (skips runs already done). 5 workers x 14 OMP threads. ----
echo "[stage 1] preprocessing alms"
python preprocess/preprocess_alms.py \
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

# ---- 3. Apply to the held-out test cosmology, NO positivity clip (Cl-optimal) ----
echo "[stage 3] applying to $TEST_COSMO (--no-clip: Cl-optimal continuous field)"
python transfer/transfer_function.py apply \
    --transfer "$OUT/transfer.npz" \
    --run-dir "$DATA/$TEST_COSMO/run_0" \
    --nside 2048 --ell-min 0 --no-clip \
    --plot-shells 0 3 10 30 50 --plot-dir "$OUT/cl_ratio" \
    --out "$OUT/${TEST_COSMO}_corrected_noclip.npz"

# ---- 4. Lognormal + Poisson resample -> valid non-negative integer count map ----
if [ -z "$SKIP_POISSON" ]; then
    echo "[stage 4] lognormal+Poisson resample (ell_c=$ELL_C taper=$TAPER n_avg=$N_AVG n_iter=$N_ITER damp=$DAMP)"
    python transfer/poisson_resample.py \
        --corrected "$OUT/${TEST_COSMO}_corrected_noclip.npz" \
        --run-dir "$DATA/$TEST_COSMO/run_0" \
        --lmax $LMAX --nside 2048 \
        --ell-c $ELL_C --taper $TAPER --n-avg $N_AVG --n-iter $N_ITER --damp $DAMP \
        --out "$OUT/${TEST_COSMO}_counts.npz"
else
    echo "[stage 4] SKIPPED (SKIP_POISSON=1) -- output is the continuous --no-clip field only"
fi

echo "transfer-function pipeline ${SLURM_JOB_ID} finished at $(date) -> $OUT"
