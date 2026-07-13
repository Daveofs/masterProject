#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=transfer-fn
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Per-ell TRANSFER-FUNCTION small-scale correction (DISCO -> CosmoGrid), DENSITY
# space, PLUS lognormal+Poisson re-discretization into a valid count map.
#
#   corrected_alm(ell,m) = low_alm(ell,m) * T(ell, shell),  T = sqrt(<Cl_high>/<Cl_low>)
#
# CPU-only (healpy), 128 cpus (measured: a single shell's SHT calls saturate OMP
# threading around 128 threads -- 128->256 gave ZERO further speedup, measured
# directly). Four stages:
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
#                  Writes emulator.loss.png (train vs held-out LOSS, both decreasing,
#                  same shared figure -- analysis.plot_train_val_loss -- as jbucko's
#                  loss_curve.png) and a validation example-patch grid + pctile-band
#                  power ratio (plot_example_patches.py, shells 5/10/15/30/50,
#                  labeled with which cosmology was validated).
#   3. apply --poisson --ell-min-mpc 3 : the transfer-function correction AND the
#                  lognormal+Poisson re-discretization into valid non-negative
#                  INTEGER counts happen in ONE step, per shell, right after each
#                  shell's correction is computed (poisson_resample.resample_shell
#                  called directly from apply()'s loop -- no separate stage, no
#                  13.9GB continuous intermediate ever written to disk). --ell-min-mpc
#                  3 leaves comoving scales LARGER than 3 Mpc/h untouched (T=1
#                  there) -- converted to a PER-SHELL ell via each shell's own
#                  redshift + the test cosmology's params.yml (ell_min_from_mpc_h),
#                  since a fixed ell corresponds to a different physical scale at
#                  every shell (comoving distance grows with z). The Poisson step's
#                  phase-mixing weight is the ACTUAL fitted R(ell) per shell, not a
#                  fixed ell_c cutoff -- a fixed cutoff was measured to discard
#                  perfectly-good, ~100%-correlated DISCO structure on dense shells
#                  and visibly degrade those images even though band-averaged Cl
#                  still looked fine (Cl is phase-blind). No --plot-shells here
#                  (individual cl_shell*.png removed by request -- the summary
#                  plots in stage 4 are what we keep).
#                  SPEED: shells with ell_min_i>=lmax (T==1 everywhere, e.g. distant
#                  shells under --ell-min-mpc 3) are skipped for free -- exactly
#                  DISCO's own counts, unmodified, since the correction is
#                  identically zero there (not an approximation). On cosmo_000122
#                  this is 34/69 shells -> the ~2h-for-69-shells sequential cost
#                  (measured; 128 vs 288 cpus made no difference -- see
#                  poisson_resample.py's parallelism note for why) drops to ~35
#                  shells x ~90s = ~52 min. A tried 23-worker process pool was
#                  measured to be SLOWER than sequential (memory-bandwidth
#                  contention) -- --poisson-workers defaults to 1 (sequential) for
#                  that reason; do not raise it without re-validating on this
#                  hardware.
#   4. plot_example_patches.py + infer_full_sky_transfer.py : the SAME shared
#                  ../analysis/ tools (transforms/plotting/radial_power/full_sky) that
#                  unet_flow_jbucko's pipeline uses, run on stage 3's FINAL count map.
#                  patch grid = flat-patch triptych + 2D-FFT power ratio + pctile band
#                  (matches jbucko's example_patches.png); full-sky grid = gnomonic
#                  zoom + the REAL angular Cl ratio, example_full_sky.png ONLY (no
#                  individual cl_shell*.png -- --shell-indices left empty by request).
#                  End state: just the summary plots (loss+validation, patch grid,
#                  full-sky grid), no per-shell Cl clutter.
# ============================================================================

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=8         # light default; heavy stages override inline below

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
# Stage 3 (transfer correction + Poisson, merged) knobs -- see poisson_resample.py
# and transfer_function.py apply()'s --ell-min-mpc/--poisson-* docstrings.
ELL_MIN_MPC=${ELL_MIN_MPC:-3.0}
N_AVG=${N_AVG:-4}
N_ITER=${N_ITER:-5}
DAMP=${DAMP:-0.4}
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/transfer

echo "==== transfer-function pipeline | data=$DATA | lmax=$LMAX | test=$TEST_COSMO | method=$METHOD | ell_min_mpc=$ELL_MIN_MPC ===="

# ---- 1. Preprocess alms (skips runs already done). More workers -- full node. ----
echo "[stage 1] preprocessing alms"
OMP_NUM_THREADS=14 python preprocess/preprocess_alms.py \
    --data-dir "$DATA" \
    --low-glob "disco_sim/*/disco_shells_nside=2048.npz" \
    --high-npz compressed_shells.npz \
    --lmax $LMAX --num-workers 20

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

# ---- 3. Apply the correction AND Poisson-resample, merged, per shell ----
echo "[stage 3] apply + Poisson (ell_min_mpc=$ELL_MIN_MPC n_avg=$N_AVG n_iter=$N_ITER damp=$DAMP)"
COUNTS="$OUT/${TEST_COSMO}_counts.npz"
OMP_NUM_THREADS=128 python transfer/transfer_function.py apply \
    --transfer "$OUT/transfer.npz" \
    --run-dir "$DATA/$TEST_COSMO/run_0" \
    --nside 2048 --ell-min-mpc $ELL_MIN_MPC \
    --poisson --poisson-n-avg $N_AVG --poisson-n-iter $N_ITER --poisson-damp $DAMP \
    --out "$COUNTS"

# ---- 4. Shared analysis/ diagnostics on the FINAL count map (same tools as jbucko) ----
# Summaries only: example-patch grid (+ pctile band) and example_full_sky.png. No
# individual cl_shell*.png (--shell-indices left empty).
echo "[stage 4] example-patch grid (analysis.plot_example_patch_grid + pctile band)"
python transfer/plot_example_patches.py \
    --run-dir "$DATA/$TEST_COSMO/run_0" --counts "$COUNTS" \
    --shells 5 10 15 30 50 --n-per-shell 1 --patch-size 256 --nside 2048 --seed 0 \
    --out "$OUT/example_patches.png"

echo "[stage 4b] full-sky grid (analysis.plot_example_full_sky_grid), summary only"
python transfer/infer_full_sky_transfer.py \
    --run-dir "$DATA/$TEST_COSMO/run_0" --counts "$COUNTS" \
    --nside 2048 --lmax $LMAX \
    --shell-indices --example-shells 5 10 15 30 50 \
    --out-dir "$OUT/full_sky"

echo "transfer-function pipeline ${SLURM_JOB_ID} finished at $(date) -> $OUT"
