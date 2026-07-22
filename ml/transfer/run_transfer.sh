#!/bin/bash
#SBATCH --nodes=10
#SBATCH --job-name=transfer-fn
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Per-ell TRANSFER-FUNCTION small-scale correction (DISCO -> CosmoGrid), DENSITY
# space.
#
#   corrected_alm(ell,m) = low_alm(ell,m) * T(ell, shell),  T = sqrt(<Cl_high>/<Cl_low>)
#
# CPU-only (healpy), 128 cpus/node (measured: a single shell's SHT calls saturate
# OMP threading around 128 threads -- 128->256 gave ZERO further speedup, measured
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
#      agreeing by seed. Stage 3 (apply()) is the expensive part (SHT-bound, one
#      full lightcone per held-out cosmology) -- running it on ALL held-out
#      cosmologies would blow the time budget. MAX_COSMOLOGIES (default 3, same
#      eval-cost cap as jbucko's --max-cosmologies) limits how many of the
#      held-out set actually go through stages 2b/3 -- TRAINING still excludes the
#      FULL held-out set, only the expensive evaluation is capped.
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
#   3. apply_transfer.py --no-clip --ell-min-mpc 5 : ONE script (mirroring
#                  unet/apply_flow.py) that embeds apply() (the transfer-function
#                  correction) AND every diagnostic plot, "connected cleanly" by
#                  passing the in-memory `corrected` arrays straight to the
#                  plotting functions instead of round-tripping through disk.
#                  Runs once per held-out cosmology (--run-dirs takes the whole
#                  list); pctile-band power ratio and full-sky moments/histograms
#                  POOL patches/pixels across ALL held-out cosmologies (see
#                  apply_transfer.py's plot_patches/plot_full_sky docstrings), the
#                  visual example grids use the first held-out cosmology, labeled
#                  with the full set.
#                  --ell-min-mpc 5 leaves comoving scales LARGER than 5 Mpc/h
#                  untouched (T=1 there) -- converted to a PER-SHELL ell via each
#                  shell's own redshift + that cosmology's params.yml
#                  (ell_min_from_mpc_h), since a fixed ell corresponds to a
#                  different physical scale at every shell (comoving distance
#                  grows with z) -- confirmed against real Disco/CosmoGrid Cl-
#                  ratio diagnostics (2026-07-13).
#                  --no-clip: emits the continuous, Cl-optimal overdensity field
#                  instead of a positivity-clipped/Poisson-resampled count map.
#                  poisson_resample.py (lognormal+Poisson re-discretization into
#                  valid counts) was REMOVED from this pipeline (2026-07-16):
#                  empirically, for kappa specifically, --no-clip is both cheaper
#                  AND more accurate (kappa Cl ratio to truth: no-clip ~0.94-1.01
#                  vs poisson ~0.81-1.08 across 5 log-ell bands on 2 held-out
#                  cosmologies, one of which also exposed a real Poisson tail-
#                  calibration bug -- shell 30 max count 27847 vs truth 3529).
#                  clip-at-0 was also tested as a positivity-only middle ground and
#                  is WORSE (injects +14-23% spurious large-scale power from filling
#                  in spatially-correlated voids). See apply_transfer.py's --no-clip
#                  help text for the full writeup. This flag ONLY makes sense while
#                  --patch-shells/--fullsky-shells (which DO need real per-pixel
#                  count realism) are off/not being trusted for that purpose.
#                  SPEED: shells with ell_min_i>=lmax (T==1 everywhere, e.g. distant
#                  shells under --ell-min-mpc 5) are skipped for free -- exactly
#                  DISCO's own field, unmodified, since the correction is
#                  identically zero there (not an approximation).
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
#
# MULTI-NODE (2026-07-21): this script scales stages 2 and 3 across however many
# nodes SLURM actually allocates ($SLURM_JOB_NUM_NODES) -- NOT a variable you set
# inside the script, since #SBATCH pragmas can't reference shell variables.
# Request more than the default 1 node AT SUBMISSION TIME:
#   sbatch --nodes=4 transfer/run_transfer.sh
# With N>1 nodes:
#   stage 2 (gather): each node processes a disjoint SHARD of the training runs
#     (transfer_function.py gather-shard --shard-index i --num-shards N, launched
#     in the background via `srun --nodes=1 --ntasks=1 --exclusive`), then ONE
#     `gather-merge` call combines every shard into the same transfer.npz/
#     emulator.pkl a single-node run would have produced (byte-identical for
#     `fit`; same total training data, different array-order-dependent random
#     train/val split for `emulate` -- see gather-merge's docstring).
#   stage 3 (apply): each node runs apply_transfer.py in COMPUTE-ONLY mode
#     (correction only, no plots) on its own slice of the held-out cosmologies,
#     writing corrected shells to a SHARED --out-counts-dir. Once every node's
#     background srun finishes, ONE final apply_transfer.py call (single process,
#     no srun) reloads every cosmology's counts via --reuse-counts (near-instant --
#     apply() is deterministic given (transfer, --seed, --ell-min-mpc, --no-clip),
#     so this is exact, not an approximation) and produces the actual diagnostic
#     plots, pooling across the FULL held-out set exactly as the single-node path
#     does.
# N=1 (the default) takes neither branch -- this script's single-node behavior is
# completely unchanged from before this section existed.
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

DATA=/capstor/scratch/cscs/damrein/grid
# nside=2048 supports SHT up to ell~3*nside-1=6143 -- lmax=3000 (the long-running
# default) leaves everything above it in every corrected map untouched (apply()
# adds the correction as a residual on the native nside=2048 map, only touching
# ell<=lmax -- see apply()'s own comment on this). Raising LMAX lets the
# correction reach further into small scales -- e.g. LMAX=4096 or the full
# 6143 -- at the cost of re-running EVERYTHING lmax-dependent: preprocessing
# (new low/high_alms_lmax<N>.npy, doesn't overwrite the lmax=3000 ones, but
# costs a full new SHT pass over every shell) and fit/train/apply (T/R/emulator
# are all shaped by lmax too). SHT cost scales ~linearly with lmax at fixed
# nside, so LMAX=6143 is roughly 2x the wall-clock of the lmax=3000 default per
# shell. Override at submission time, e.g. `sbatch --export=LMAX=4096 ...`; NOT
# yet confirmed to fix the "washed out" patch look (2026-07-21) -- this is the
# knob to test that hypothesis with, not a validated fix.
LMAX=${LMAX:-6143}
# Stage 1's per-node SHT worker count. 20 was sized for lmax=3000; each worker's
# map2alm/alm2map memory footprint grows with lmax (roughly linearly, at fixed
# nside=2048), so running 20 of them CONCURRENTLY at a much higher lmax can
# exceed a node's memory -- confirmed 2026-07-21 (job 4255910, lmax=6143):
# stage 1 OOM-killed after 14m37s with 0/198 shells done (sacct: 710GB peak RSS
# on the unwrapped main-script step), which then cascaded through every later
# stage as "file not found" (nothing to gather/merge/emulate). Scale down
# automatically above lmax=3000 (proportionally, floor of 4); override
# explicitly if this still isn't right for your node's actual memory.
NUM_WORKERS_DEFAULT=$(( LMAX > 3000 ? 20 * 3000 / LMAX : 20 ))
[ "$NUM_WORKERS_DEFAULT" -lt 4 ] && NUM_WORKERS_DEFAULT=4
NUM_WORKERS=${NUM_WORKERS:-$NUM_WORKERS_DEFAULT}
# How to build T(ell, shell): "fit" (train-averaged) or "emulate" (MLP emulator).
METHOD=${METHOD:-emulate}
# Emulator hyperparameters (METHOD=emulate only).
HIDDEN=${HIDDEN:-256,256,128}
MAX_ITER=${MAX_ITER:-200}
SAMPLE_FRAC=${SAMPLE_FRAC:-1.0}   # <1.0 subsamples ell per shell (faster training)
SMOOTH_WINDOW=${SMOOTH_WINDOW:-21}
# Set INCLUDE_TEST=1 to fit/train T on the held-out cosmologies too (sanity check:
# expect corrected/high ~ 1 by construction). Empty = proper leave-one-out.
INCLUDE_TEST=${INCLUDE_TEST:-}
FIT_FLAGS=""
[ -n "$INCLUDE_TEST" ] && FIT_FLAGS="--include-test"
# Stage 3 (transfer correction) knobs -- see transfer_function.py apply()'s
# --ell-min-mpc docstring.
# --hp-transition: raised-cosine hand-over band above ell_min (fraction of lmax),
# same convention/default as diffusion/run_diffusion.sh's HP_TRANSITION -- see
# highpass_ell_ramp's docstring. 0 recovers the old hard ell_min step exactly.
ELL_MIN_MPC=${ELL_MIN_MPC:-5.0}
HP_TRANSITION=${HP_TRANSITION:-0.10}
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT" /capstor/scratch/cscs/damrein/outputs/logs/transfer

# Actual allocated node count -- see the MULTI-NODE header comment above. 1 unless
# `sbatch --nodes=N` overrode the #SBATCH pragma at submission time.
N_NODES=${SLURM_JOB_NUM_NODES:-1}

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
# 2b/3 (apply()) -- training above already excludes the FULL set.
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-30}
EVAL_COSMOS_ARR=("${COSMOS_ARR[@]:0:$MAX_COSMOLOGIES}")

echo "==== transfer-function pipeline | data=$DATA | lmax=$LMAX | method=$METHOD | ell_min_mpc=$ELL_MIN_MPC | nodes=$N_NODES ===="
echo "held out from training (${#COSMOS_ARR[@]}): ${COSMOS_ARR[@]}"
echo "evaluated in stages 2b/3 (${#EVAL_COSMOS_ARR[@]} of them, MAX_COSMOLOGIES=$MAX_COSMOLOGIES): ${EVAL_COSMOS_ARR[@]}"

# ---- 1. Preprocess alms (skips runs already done). Embarrassingly parallel
# across cosmologies (process_single_run's own skip-if-done check means
# concurrent shards never touch the same output file) -- sharded across every
# allocated node when N_NODES>1, same srun-background+wait pattern as stages
# 2/3, instead of the whole dataset landing on one node. OMP_NUM_THREADS is
# sized so NUM_WORKERS*OMP_NUM_THREADS doesn't oversubscribe a node's 128 cpus.
OMP_NUM_THREADS_PP=$(( 128 / NUM_WORKERS ))
[ "$OMP_NUM_THREADS_PP" -lt 1 ] && OMP_NUM_THREADS_PP=1
if [ "$N_NODES" -gt 1 ]; then
    echo "[stage 1] preprocessing alms across $N_NODES nodes (sharded, $NUM_WORKERS workers/node)"
    for ((i = 0; i < N_NODES; i++)); do
        srun --nodes=1 --ntasks=1 --exclusive bash -c "
            OMP_NUM_THREADS=$OMP_NUM_THREADS_PP python preprocess/preprocess_alms.py \
                --data-dir '$DATA' \
                --low-glob 'disco_sim/*/disco_shells_nside=2048.npz' \
                --high-npz compressed_shells.npz \
                --lmax $LMAX --num-workers $NUM_WORKERS \
                --shard-index $i --num-shards $N_NODES
        " &
    done
    wait
else
    echo "[stage 1] preprocessing alms ($NUM_WORKERS workers)"
    OMP_NUM_THREADS=$OMP_NUM_THREADS_PP python preprocess/preprocess_alms.py \
        --data-dir "$DATA" \
        --low-glob "disco_sim/*/disco_shells_nside=2048.npz" \
        --high-npz compressed_shells.npz \
        --lmax $LMAX --num-workers $NUM_WORKERS
fi
N_ALM_FILES=$(find "$DATA" -maxdepth 3 -name "low_alms_lmax${LMAX}.npy" 2>/dev/null | wc -l)
if [ "$N_ALM_FILES" -eq 0 ]; then
    echo "ERROR: stage 1 produced no low_alms_lmax${LMAX}.npy anywhere under $DATA -- "
    echo "aborting before stage 2 (see slurm-${SLURM_JOB_ID}.err for the real traceback, "
    echo "e.g. a BrokenProcessPool from an OOM'd worker -- try a smaller NUM_WORKERS)."
    exit 1
fi
echo "[stage 1] done -- $N_ALM_FILES low_alms_lmax${LMAX}.npy files present"

# ---- 2. Build T(ell, shell) via the chosen method (held-out cosmologies left out) ----
# OMP_NUM_THREADS=128 here (not the file-wide default of 8): the gather loop in
# `train`/`fit` is dominated by hp.alm2cl (OpenMP-parallel healpy SHT calls) over
# many training runs -- running that at 8 threads while 128 cpus are allocated
# left >90% of the node idle (measured: job 4200458 took ~1.3 min/run just to
# gather Cls at OMP_NUM_THREADS=8).
GATHER_METHOD="fit"; [ "$METHOD" = "emulate" ] && GATHER_METHOD="train"
if [ "$N_NODES" -gt 1 ]; then
    echo "[stage 2] gathering '$GATHER_METHOD' across $N_NODES nodes (sharded)"
    TRAIN_SHARD_FLAGS=""
    [ "$GATHER_METHOD" = "train" ] && \
        TRAIN_SHARD_FLAGS="--smooth-window $SMOOTH_WINDOW --sample-frac $SAMPLE_FRAC"
    for ((i = 0; i < N_NODES; i++)); do
        srun --nodes=1 --ntasks=1 --exclusive bash -c "
            OMP_NUM_THREADS=128 python transfer/transfer_function.py gather-shard \
                --data-dir '$DATA' --lmax $LMAX --method $GATHER_METHOD \
                $TEST_COSMOS_FLAGS $FIT_FLAGS $TRAIN_SHARD_FLAGS \
                --gather-workers 32 --shard-index $i --num-shards $N_NODES \
                --out '$OUT/shard_${i}.pkl'
        " &
    done
    wait
    SHARD_FILES=""
    for ((i = 0; i < N_NODES; i++)); do
        [ -f "$OUT/shard_${i}.pkl" ] || {
            echo "ERROR: $OUT/shard_${i}.pkl missing -- that node's gather-shard failed (see slurm-${SLURM_JOB_ID}.err), aborting before gather-merge."
            exit 1
        }
        SHARD_FILES="$SHARD_FILES $OUT/shard_${i}.pkl"
    done
    if [ "$METHOD" = "emulate" ]; then
        echo "[stage 2 merge] training MLP emulator from $N_NODES shards"
        OMP_NUM_THREADS=128 python transfer/transfer_function.py gather-merge \
            --shards $SHARD_FILES --hidden "$HIDDEN" --max-iter $MAX_ITER \
            --out "$OUT/emulator.pkl" || { echo "ERROR: gather-merge failed, aborting before stage 2b/3."; exit 1; }
    else
        echo "[stage 2 merge] fitting T(ell,shell) from $N_NODES shards"
        python transfer/transfer_function.py gather-merge \
            --shards $SHARD_FILES --out "$OUT/transfer.npz" \
            || { echo "ERROR: gather-merge failed, aborting before stage 2b/3."; exit 1; }
    fi
elif [ "$METHOD" = "emulate" ]; then
    echo "[stage 2] training MLP emulator (hidden=$HIDDEN, max_iter=$MAX_ITER, sample_frac=$SAMPLE_FRAC)"
    OMP_NUM_THREADS=128 python transfer/transfer_function.py train \
        --data-dir "$DATA" --lmax $LMAX \
        $TEST_COSMOS_FLAGS $FIT_FLAGS \
        --hidden "$HIDDEN" --max-iter $MAX_ITER --sample-frac $SAMPLE_FRAC \
        --gather-workers 32 \
        --out "$OUT/emulator.pkl"
else
    echo "[stage 2] fitting T(ell, shell) (train-averaged)"
    OMP_NUM_THREADS=128 python transfer/transfer_function.py fit \
        --data-dir "$DATA" --lmax $LMAX \
        $TEST_COSMOS_FLAGS $FIT_FLAGS --gather-workers 32 --out "$OUT/transfer.npz"
fi

# ---- 2b. Per-cosmology transfer file (emulate: one predict call per held-out
# cosmology; fit: the single transfer.npz broadcasts to all) ----
TRANSFER_ARR=()
OK_COSMOS_ARR=()
if [ "$METHOD" = "emulate" ]; then
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
            TRANSFER_ARR+=("$OUT/transfer_${c}.npz")
            OK_COSMOS_ARR+=("$c")
        else
            echo "[stage 2b] WARNING: emulate failed for $c -- excluding it from stage 3"
        fi
    done
else
    for c in "${EVAL_COSMOS_ARR[@]}"; do
        TRANSFER_ARR+=("$OUT/transfer.npz")
        OK_COSMOS_ARR+=("$c")
    done
fi
EVAL_COSMOS_ARR=("${OK_COSMOS_ARR[@]}")
if [ ${#EVAL_COSMOS_ARR[@]} -eq 0 ]; then
    echo "ERROR: no held-out cosmology has a usable transfer function -- aborting before stage 3"
    exit 1
fi

# ---- 3. apply_transfer.py: correction + ALL diagnostics, one script ----
# Embeds apply() + the former plot_example_patches.py/infer_full_sky_transfer.py,
# connected in-memory (see apply_transfer.py's docstring) -- mirrors
# unet/apply_flow.py's one-script pattern. Pools diagnostics across every held-out
# cosmology in EVAL_COSMOS_ARR (bounded by MAX_COSMOLOGIES).
# Runs under sphereflow (uenv+venv, same env unet/run_flow.sh uses),
# NOT deepSphere (stages 0-2 above) -- UFalcon (--kappa) is only installed there;
# apply_transfer.py's own deps (numpy/healpy/scipy/yaml/matplotlib) all work under
# sphereflow too (checked), so this is a clean env switch, not a partial one.
if [ "$N_NODES" -gt 1 ]; then
    echo "[stage 3a] correcting ${#EVAL_COSMOS_ARR[@]} held-out cosmologies across $N_NODES nodes (compute-only)"
    declare -a BATCH_RUN_DIRS BATCH_TRANSFERS
    for ((i = 0; i < N_NODES; i++)); do BATCH_RUN_DIRS[$i]=""; BATCH_TRANSFERS[$i]=""; done
    for ((j = 0; j < ${#EVAL_COSMOS_ARR[@]}; j++)); do
        node=$((j % N_NODES))
        BATCH_RUN_DIRS[$node]="${BATCH_RUN_DIRS[$node]} $DATA/${EVAL_COSMOS_ARR[$j]}/run_0"
        BATCH_TRANSFERS[$node]="${BATCH_TRANSFERS[$node]} ${TRANSFER_ARR[$j]}"
    done
    for ((i = 0; i < N_NODES; i++)); do
        [ -z "${BATCH_RUN_DIRS[$i]}" ] && continue    # fewer cosmologies than nodes
        srun --nodes=1 --ntasks=1 --exclusive \
            uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
                source ${SPHEREFLOW_VENV}/bin/activate
                OMP_NUM_THREADS=128 python transfer/apply_transfer.py \
                    --transfer ${BATCH_TRANSFERS[$i]} \
                    --run-dirs ${BATCH_RUN_DIRS[$i]} \
                    --nside 2048 --lmax $LMAX --ell-min-mpc $ELL_MIN_MPC \
                    --hp-transition $HP_TRANSITION \
                    --no-clip --seed 0 \
                    --out-counts-dir '$OUT/counts' \
                    --patch-shells --fullsky-shells --n-zbins 0 \
                    --out-dir '$OUT/eval_shard_${i}'
            " &
    done
    wait
    REUSE_FLAG="--reuse-counts $OUT/counts"
else
    REUSE_FLAG=""
fi

RUN_DIRS=""
TRANSFER_FILES=""
for ((j = 0; j < ${#EVAL_COSMOS_ARR[@]}; j++)); do
    RUN_DIRS="$RUN_DIRS $DATA/${EVAL_COSMOS_ARR[$j]}/run_0"
    TRANSFER_FILES="$TRANSFER_FILES ${TRANSFER_ARR[$j]}"
done
echo "[stage 3] apply_transfer.py: correction (--no-clip, ell_min_mpc=$ELL_MIN_MPC, hp_transition=$HP_TRANSITION) + diagnostics (incl. kappa), ${#EVAL_COSMOS_ARR[@]} held-out cosmologies"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${SPHEREFLOW_VENV}/bin/activate
    OMP_NUM_THREADS=128 python transfer/apply_transfer.py \
        --transfer $TRANSFER_FILES \
        --run-dirs $RUN_DIRS \
        --nside 2048 --lmax $LMAX --ell-min-mpc $ELL_MIN_MPC \
        --hp-transition $HP_TRANSITION \
        --no-clip --seed 0 \
        $REUSE_FLAG \
        --out-counts-dir '$OUT/counts' \
        --patch-shells 5 10 15 30 50 --n-per-shell 1 --patch-size 256 \
        --fullsky-shells 5 10 15 30 50 \
        --kappa \
        --out-dir '$OUT/eval'
"

echo "transfer-function pipeline ${SLURM_JOB_ID} finished at $(date) -> $OUT"
