#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=transfer-fn
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
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
#   0. held-out cosmologies: a random FRACTION of cosmologies (VAL_FRAC, default
#      0.15) is held out for validation -- same whole-cosmology-split convention
#      as unet/dataset.py's split_by_cosmo (default val_frac=0.15) --
#      instead of a single fixed test cosmology, so validation isn't just one
#      cosmology getting lucky/unlucky. Override with TEST_COSMOS="cosmo_A cosmo_B"
#      for an explicit list. transfer_function.split_val_cosmos() (same function
#      `fit`/`train` use internally) is called here in bash, and the RESOLVED list
#      is then passed to `fit`/`train` via --test-cosmos explicitly (not
#      --val-frac) -- so stage 3's --run-dirs is guaranteed identical to what was
#      excluded from training, with no reliance on two separate random draws
#      agreeing by seed. Stage 3 (apply()+Poisson) is the expensive part (~90s/
#      shell x ~35 effective shells = ~52 min PER cosmology, sequential) --
#      running it on ALL held-out cosmologies would blow the time budget (e.g.
#      val_frac=0.15 on ~48 cosmologies = 7 held out = ~6h). MAX_COSMOLOGIES
#      (default 3, same eval-cost cap as jbucko's --max-cosmologies) limits how
#      many of the held-out set actually go through stages 2b/3 -- TRAINING still
#      excludes the FULL held-out set, only the expensive evaluation is capped.
#   1. preprocess : map2alm the DISCO (disco_sim/.../disco_shells_nside=2048.npz) and
#                   CosmoGrid (compressed_shells.npz) shells -> low/high_alms_lmax*.npy
#                   (one-time; mmap-able -> fast downstream, like the harmonic flow).
#                   DENSITY space only -- do NOT add --log-density: measured 2026-07-09
#                   to destroy small-scale Cl (0.41 vs 0.93 at ell 800-1500 on shell 3,
#                   because expm1 is too nonlinear for a sparse field). Dead end, kept
#                   only as a documented warning, not used here.
#   2. T(ell,shell): produced by ONE of two methods (set METHOD below) ->
#        fit     : average Cl ratio over training cosmologies (held-out set left
#                  out). ONE cosmology-independent transfer.npz, reused for every
#                  held-out cosmology in stage 3.
#        emulate : train an MLP emulator T=f(l,z,H0,O_cdm,Ob,Om,ns,s8,Cl_low) on the
#                  training cosmologies, then predict T SEPARATELY for each
#                  held-out cosmology (one transfer_<cosmo>.npz per cosmology).
#                  Writes emulator.loss.png (train vs held-out LOSS, both decreasing,
#                  same shared figure -- analysis.plot_train_val_loss -- as jbucko's
#                  loss_curve.png). The validation example-patch grid + pctile-band
#                  power ratio for shells 5/10/15/30/50 (labeled with the full
#                  held-out cosmology set) is produced in stage 3 below, POOLED
#                  across all held-out cosmologies' corrected output.
#   3. apply_transfer.py --poisson --ell-min-mpc 3 : ONE script (mirroring
#                  unet/apply_flow.py) that embeds apply() (the
#                  transfer-function correction + lognormal+Poisson
#                  re-discretization into valid non-negative INTEGER counts, one
#                  step per shell, no 13.9GB continuous intermediate ever written
#                  to disk) AND every diagnostic plot, "connected cleanly" by
#                  passing the in-memory `corrected` arrays straight to the
#                  plotting functions instead of round-tripping through disk.
#                  Runs once per held-out cosmology (--run-dirs takes the whole
#                  list); pctile-band power ratio and full-sky moments/histograms
#                  POOL patches/pixels across ALL held-out cosmologies (see
#                  apply_transfer.py's plot_patches/plot_full_sky docstrings), the
#                  visual example grids use the first held-out cosmology, labeled
#                  with the full set.
#                  --ell-min-mpc 3 leaves comoving scales LARGER than 3 Mpc/h
#                  untouched (T=1 there) -- converted to a PER-SHELL ell via each
#                  shell's own redshift + that cosmology's params.yml
#                  (ell_min_from_mpc_h), since a fixed ell corresponds to a
#                  different physical scale at every shell (comoving distance
#                  grows with z) -- confirmed against real Disco/CosmoGrid Cl-
#                  ratio diagnostics (2026-07-13). The Poisson step's phase-mixing
#                  weight is the ACTUAL fitted R(ell) per shell, not a fixed ell_c
#                  cutoff -- a fixed cutoff was measured to discard perfectly-
#                  good, ~100%-correlated DISCO structure on dense shells and
#                  visibly degrade those images even though band-averaged Cl
#                  still looked fine (Cl is phase-blind). No individual
#                  cl_shell*.png, and no example_full_sky.png (both removed by
#                  request). cl_ratio_by_zbin_grid.png is THE Cl diagnostic: the
#                  GENUINE multi-cosmology check (one row per held-out cosmology,
#                  one column per redshift bin, pctile band) -- the same statistic
#                  + shared plotting code as unet/apply_flow.py's
#                  example_full_sky.png uses (analysis.plot_cl_ratio_pctile_grid +
#                  zbin_shell_samples). Our old example_full_sky.png only ever
#                  showed ONE cosmology at one fixed sky position, so its Cl panel
#                  was strictly subsumed by the zbin grid.
#                  SPEED: shells with ell_min_i>=lmax (T==1 everywhere, e.g. distant
#                  shells under --ell-min-mpc 3) are skipped for free -- exactly
#                  DISCO's own counts, unmodified, since the correction is
#                  identically zero there (not an approximation). On cosmo_000122
#                  this is 34/69 shells -> the ~2h-for-69-shells sequential cost
#                  (measured; 128 vs 288 cpus made no difference -- see
#                  poisson_resample.py's parallelism note for why) drops to ~35
#                  shells x ~90s = ~52 min PER held-out cosmology.
#                  A tried 23-worker process pool was measured to be SLOWER than
#                  sequential (memory-bandwidth contention) -- --poisson-workers
#                  defaults to 1 (sequential) for that reason; do not raise it
#                  without re-validating on this hardware.
#                  Diagnostics use the SAME shared ../analysis/ tools
#                  (transforms/plotting/radial_power/full_sky) that
#                  unet's pipeline uses. End state, per held-out set:
#                    example_patches.png            flat-patch triptych + 2D-FFT
#                                                   power ratio (matches jbucko's)
#                    patch_power_ratio_pctile_band.png  pooled over all cosmologies
#                    cl_ratio_by_zbin_grid.png      THE Cl check (cosmology x zbin)
#                    moments_vs_shell.png /         one-point PDF (pooled), the
#                    example_histograms.png         check Cl structurally can't do
#                    kappa_cl_{per_cosmology,pctile_band}.png +
#                    kappa_moments_scatter.png      weak lensing (--kappa)
#                  NOTE --kappa-nside/--kappa-lmax default to 1024/2048, NOT the
#                  old 128/350: nside=128 caps kappa at ell~383, but the transfer
#                  function is ~1 below ell 350 and does ALL its work above it
#                  (measured max|T-1| on shells 10-30: 0.002-0.025 for ell<=350 vs
#                  0.15-1.10 for ell 351-3000), so the old kappa plots were blind
#                  to the entire correction by construction.
# ============================================================================

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=8         # light default; heavy stages override inline below

# apply_transfer.py (stage 3) now ALSO builds weak-lensing kappa maps via UFalcon
# (analysis.weak_lensing, --kappa) -- UFalcon is installed in sphereflow (the SAME
# venv unet's pipeline uses), not deepSphere (which lacks it, and whose
# sklearn-based MLP emulator stage 2 needs is in turn not in sphereflow) -- so ONLY
# stage 3's invocation below switches env, via the same uenv+venv activation
# unet/run_flow.sh uses, rather than installing UFalcon separately into
# two environments that could drift out of sync.
export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
SPHEREFLOW_VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow

DATA=/capstor/scratch/cscs/damrein/cosmogridv1
LMAX=3000
# How to build T(ell, shell): "fit" (train-averaged) or "emulate" (MLP emulator).
METHOD=${METHOD:-emulate}
# Emulator hyperparameters (METHOD=emulate only).
HIDDEN=${HIDDEN:-256,256,128}
MAX_ITER=${MAX_ITER:-200}
SAMPLE_FRAC=${SAMPLE_FRAC:-1.0}   # <1.0 subsamples ell per shell (faster training)
# Set INCLUDE_TEST=1 to fit/train T on the held-out cosmologies too (sanity check:
# expect corrected/high ~ 1 by construction). Empty = proper leave-one-out.
INCLUDE_TEST=${INCLUDE_TEST:-}
FIT_FLAGS=""
[ -n "$INCLUDE_TEST" ] && FIT_FLAGS="--include-test"
# Stage 3 (transfer correction + Poisson, merged) knobs -- see poisson_resample.py
# and transfer_function.py apply()'s --ell-min-mpc/--poisson-* docstrings.
ELL_MIN_MPC=${ELL_MIN_MPC:-5.0}
N_AVG=${N_AVG:-4}
N_ITER=${N_ITER:-3}
DAMP=${DAMP:-0.4}
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/transfer

# ---- 0. Resolve the held-out validation cosmologies (multiple, not just one) ----
VAL_FRAC=${VAL_FRAC:-0.15}
VAL_SEED=${VAL_SEED:-0}
if [ -n "$TEST_COSMOS" ]; then
    read -ra COSMOS_ARR <<< "$TEST_COSMOS"
else
    COSMOS_ARR=($(python -c "
import sys; sys.path.insert(0, 'transfer')
from transfer_function import split_val_cosmos
print(' '.join(split_val_cosmos('$DATA', $VAL_FRAC, $VAL_SEED)))
"))
fi
# Pass the RESOLVED list explicitly to fit/train (not --val-frac) -- see the
# header comment above for why.
TEST_COSMOS_FLAGS="--test-cosmos ${COSMOS_ARR[@]}"

# Cap how many held-out cosmologies actually go through the EXPENSIVE stage
# 2b/3 (apply()+Poisson) -- training above already excludes the FULL set.
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-3}
EVAL_COSMOS_ARR=("${COSMOS_ARR[@]:0:$MAX_COSMOLOGIES}")

echo "==== transfer-function pipeline | data=$DATA | lmax=$LMAX | method=$METHOD | ell_min_mpc=$ELL_MIN_MPC ===="
echo "held out from training (${#COSMOS_ARR[@]}): ${COSMOS_ARR[@]}"
echo "evaluated in stages 2b/3 (${#EVAL_COSMOS_ARR[@]} of them, MAX_COSMOLOGIES=$MAX_COSMOLOGIES): ${EVAL_COSMOS_ARR[@]}"

# ---- 1. Preprocess alms (skips runs already done). More workers -- full node. ----
echo "[stage 1] preprocessing alms"
OMP_NUM_THREADS=14 python preprocess/preprocess_alms.py \
    --data-dir "$DATA" \
    --low-glob "disco_sim/*/disco_shells_nside=2048.npz" \
    --high-npz compressed_shells.npz \
    --lmax $LMAX --num-workers 20

# ---- 2. Build T(ell, shell) via the chosen method (held-out cosmologies left out) ----
# OMP_NUM_THREADS=128 here (not the file-wide default of 8): the gather loop in
# `train` is dominated by hp.alm2cl (OpenMP-parallel healpy SHT calls) over 38+
# training runs x 69 shells -- running that at 8 threads while 128 cpus are
# allocated left >90% of the node idle (measured: job 4200458 took ~1.3 min/run
# just to gather Cls at OMP_NUM_THREADS=8, on track for ~50 min for this step alone).
if [ "$METHOD" = "emulate" ]; then
    echo "[stage 2] training MLP emulator (hidden=$HIDDEN, max_iter=$MAX_ITER, sample_frac=$SAMPLE_FRAC)"
    OMP_NUM_THREADS=128 python transfer/transfer_function.py train \
        --data-dir "$DATA" --lmax $LMAX \
        $TEST_COSMOS_FLAGS $FIT_FLAGS \
        --hidden "$HIDDEN" --max-iter $MAX_ITER --sample-frac $SAMPLE_FRAC \
        --gather-workers 32 \
        --out "$OUT/emulator.pkl"
    TRANSFER_FILES=""
    OK_COSMOS_ARR=()
    for c in "${EVAL_COSMOS_ARR[@]}"; do
        echo "[stage 2b] emulating T for $c"
        # Check exit status -- do NOT blindly append a transfer file that may not
        # exist: split_val_cosmos already filters out cosmologies with no
        # disco_sim/ data (see its docstring), but this is defense in depth
        # against any OTHER emulate failure (corrupt alms, etc.) so one bad
        # cosmology can't poison --transfer/--run-dirs for stage 3.
        if OMP_NUM_THREADS=128 python transfer/transfer_function.py emulate \
            --emulator "$OUT/emulator.pkl" \
            --run-dir "$DATA/$c/run_0" \
            --out "$OUT/transfer_${c}.npz"; then
            TRANSFER_FILES="$TRANSFER_FILES $OUT/transfer_${c}.npz"
            OK_COSMOS_ARR+=("$c")
        else
            echo "[stage 2b] WARNING: emulate failed for $c -- excluding it from stage 3"
        fi
    done
    EVAL_COSMOS_ARR=("${OK_COSMOS_ARR[@]}")
else
    echo "[stage 2] fitting T(ell, shell) (train-averaged)"
    OMP_NUM_THREADS=128 python transfer/transfer_function.py fit \
        --data-dir "$DATA" --lmax $LMAX \
        $TEST_COSMOS_FLAGS $FIT_FLAGS --gather-workers 32 --out "$OUT/transfer.npz"
    TRANSFER_FILES="$OUT/transfer.npz"   # ONE file, broadcast to every held-out cosmology
fi
if [ ${#EVAL_COSMOS_ARR[@]} -eq 0 ]; then
    echo "ERROR: no held-out cosmology has a usable transfer function -- aborting before stage 3"
    exit 1
fi

# ---- 3. apply_transfer.py: correction + Poisson + ALL diagnostics, one script ----
# Embeds apply() + the former plot_example_patches.py/infer_full_sky_transfer.py,
# connected in-memory (see apply_transfer.py's docstring) -- mirrors
# unet/apply_flow.py's one-script pattern. Runs once per held-out
# cosmology (--run-dirs), pooling diagnostics across all of them. --kappa's held-out
# coverage is therefore bounded by MAX_COSMOLOGIES too (it reuses the in-memory
# `corrected` this stage already computed -- giving it every held-out cosmology
# would mean re-running the expensive apply()+Poisson step, ~52min each, on more of
# them, contradicting MAX_COSMOLOGIES' whole reason for existing).
# Runs under sphereflow (uenv+venv, same env unet/run_flow.sh uses),
# NOT deepSphere (stages 0-2 above) -- UFalcon (--kappa) is only installed there;
# apply_transfer.py's own deps (numpy/healpy/scipy/yaml/matplotlib) all work under
# sphereflow too (checked), so this is a clean env switch, not a partial one.
RUN_DIRS=""
for c in "${EVAL_COSMOS_ARR[@]}"; do RUN_DIRS="$RUN_DIRS $DATA/$c/run_0"; done
echo "[stage 3] apply_transfer.py: correction + Poisson (ell_min_mpc=$ELL_MIN_MPC n_avg=$N_AVG n_iter=$N_ITER damp=$DAMP) + diagnostics (incl. kappa), ${#EVAL_COSMOS_ARR[@]} held-out cosmologies"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${SPHEREFLOW_VENV}/bin/activate
    OMP_NUM_THREADS=128 python transfer/apply_transfer.py \
        --transfer $TRANSFER_FILES \
        --run-dirs $RUN_DIRS \
        --nside 2048 --lmax $LMAX --ell-min-mpc $ELL_MIN_MPC \
        --poisson --poisson-n-avg $N_AVG --poisson-n-iter $N_ITER --poisson-damp $DAMP \
        --out-counts-dir '$OUT/counts' \
        --patch-shells 5 10 15 30 50 --n-per-shell 1 --patch-size 256 --seed 0 \
        --fullsky-shells 5 10 15 30 50 \
        --kappa \
        --out-dir '$OUT/eval'
"

echo "transfer-function pipeline ${SLURM_JOB_ID} finished at $(date) -> $OUT"
