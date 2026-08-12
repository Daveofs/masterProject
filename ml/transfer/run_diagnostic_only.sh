#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=transfer-diag
#SBATCH --partition=normal
#SBATCH --account=a0158
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/diag-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/diag-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Diagnostic-only run against an ALREADY-BUILT transfer.npz (from a prior fit/train
# stage) -- runs apply_transfer.py's full correction + ALL diagnostics WITHOUT
# redoing alm preprocessing or emulator training. Mirrors
# unet/run_diagnostics_only.sh's pattern (reuse an existing checkpoint,
# just re-run apply+eval -- e.g. after changing --ell-min-mpc).
#
# Validates on the SAME MULTIPLE held-out cosmologies run_transfer.sh used
# (auto-discovered from the prior job's output -- see stage 0 below), not just one.
#
# Point TRANSFER_JOB at the SLURM job id of a prior run_transfer.sh run whose
# transfer.npz (fit) / transfer_<cosmo>.npz (emulate) you want to reuse:
#   sbatch --export=TRANSFER_JOB=4199680 transfer/run_diagnostic_only.sh
# DATA/ELL_MIN_MPC currently default to the grid run_transfer.sh job 4215699's
# config (grid dataset, ell_min_mpc=5.0) -- see the block below for why. For a
# DIFFERENT prior job (e.g. a cosmogridv1 run_transfer.sh run), override all of
# these together at submission time, since REUSE_COUNTS is only an EXACT match if
# they agree with whatever originally produced the counts:
#   sbatch --export=TRANSFER_JOB=4201972,DATA=/capstor/scratch/cscs/damrein/cosmogridv1,ELL_MIN_MPC=3.0 \
#       transfer/run_diagnostic_only.sh

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=8

# apply_transfer.py itself now runs under sphereflow (see run_transfer.sh's stage 3
# comment) -- UFalcon (--kappa) is only installed there. deepSphere stays active
# here only for this script's own cosmology-resolution bash/numpy logic below.
export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
SPHEREFLOW_VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
# Single source of truth for node cpu count -- see run_transfer.sh's identical
# comment (128 on Alps, 288 on Clariden; SLURM sets this from --cpus-per-task).
CPUS_PER_NODE=${SLURM_CPUS_PER_TASK:-288}

DATA=${DATA:-/capstor/scratch/cscs/damrein/grid}
# Must match TRANSFER_JOB's own LMAX -- unlike run_transfer.sh (where raising
# LMAX triggers a full fresh preprocess+fit/train+apply), this script REUSES an
# EXISTING transfer.npz/emulator.pkl and low/high_alms_lmax<N>.npy built at
# whatever lmax that prior job used; a mismatched LMAX here will simply fail to
# find those files (they're named low_alms_lmax<N>.npy) or silently reload the
# wrong alm set if a coincidentally-matching one exists elsewhere.
LMAX=${LMAX:-1500}
N_NODES=${SLURM_JOB_NUM_NODES:-1}

# Defaults currently point at the grid run_transfer.sh job 2884508 (10 held-out
# grid cosmologies, MAX_COSMOLOGIES=10, ell_min_mpc=5.0, lmax=3000) -- so plain
# `sbatch transfer/run_diagnostic_only.sh` re-runs/recovers ITS diagnostics with no
# --export needed. To target a DIFFERENT prior job (e.g. a cosmogridv1 run),
# override at submission time -- see the header comment above for the full
# --export= form (TRANSFER_JOB/DATA/ELL_MIN_MPC must all match together).
TRANSFER_JOB=${TRANSFER_JOB:-2884508}
# No suffix (2026-07-24, was "${TRANSFER_JOB}_cos200"): current run_transfer.sh
# writes OUT=.../transfer/${SLURM_JOB_ID} with no suffix -- the old "_cos200"
# pattern was stale and silently broke TRANSFER_DIR resolution for every job
# produced by the script as it exists today.
TRANSFER_DIR="/capstor/scratch/cscs/damrein/outputs/transfer/${TRANSFER_JOB}"

# ---- 0. Resolve held-out cosmologies + matching transfer file(s) from the prior job ----
if ls "$TRANSFER_DIR"/transfer_cosmo_*.npz >/dev/null 2>&1; then
    # emulate method: one transfer_<cosmo>.npz per held-out cosmology.
    if [ -z "$TEST_COSMOS" ]; then
        COSMOS_ARR=()
        for f in "$TRANSFER_DIR"/transfer_cosmo_*.npz; do
            name=$(basename "$f" .npz); COSMOS_ARR+=("${name#transfer_}")
        done
    else
        read -ra COSMOS_ARR <<< "$TEST_COSMOS"
    fi
    TRANSFER_ARR=()
    for c in "${COSMOS_ARR[@]}"; do
        f="$TRANSFER_DIR/transfer_${c}.npz"
        [ -f "$f" ] || { echo "ERROR: $f not found (no emulated T for $c in job $TRANSFER_JOB)"; exit 1; }
        TRANSFER_ARR+=("$f")
    done
elif [ -f "$TRANSFER_DIR/transfer.npz" ]; then
    # fit method: ONE cosmology-independent transfer.npz, broadcast to every cosmology.
    if [ -z "$TEST_COSMOS" ]; then
        COSMOS_ARR=($(python -c "
import numpy as np
d = np.load('$TRANSFER_DIR/transfer.npz')
print(' '.join(d['test_cosmos'].tolist()) if 'test_cosmos' in d.files else '')
"))
    else
        read -ra COSMOS_ARR <<< "$TEST_COSMOS"
    fi
    TRANSFER_ARR=()
    for c in "${COSMOS_ARR[@]}"; do TRANSFER_ARR+=("$TRANSFER_DIR/transfer.npz"); done
else
    echo "ERROR: no transfer.npz or transfer_<cosmo>.npz found under $TRANSFER_DIR"
    exit 1
fi
if [ ${#COSMOS_ARR[@]} -eq 0 ]; then
    echo "ERROR: could not determine held-out cosmologies for job $TRANSFER_JOB -- "
    echo "the transfer.npz predates test_cosmos being saved. Set TEST_COSMOS explicitly, e.g.:"
    echo "  sbatch --export=TRANSFER_JOB=$TRANSFER_JOB,TEST_COSMOS='cosmo_000001 cosmo_000003' \\"
    echo "      transfer/run_diagnostic_only.sh"
    exit 1
fi

# Same apply-stage knobs as run_transfer.sh -- override via --export= to re-run
# diagnostics with different settings against the SAME trained transfer(s).
# REUSE_COUNTS below is only an EXACT match if this agrees with whatever originally
# produced the counts -- job 4215699 and the other pre-2026-08-11 runs used
# ELL_MIN_MPC=5.0 AND a raised-cosine hand-over, so their counts are NOT reusable
# under the current default; let them recompute.
# ELL_MIN_MPC: leave comoving scales LARGER than this untouched. Converted to a
# PER-SHELL ell_min = 2*pi*chi/L, so the correction starts where the particle-mesh
# deficit actually starts on that shell. 17.0 Mpc/h is the measured onset
# (17.0 +- 1.1 Mpc/h, constant to 4% over 25 shells spanning a factor 22 in chi);
# the previous 5.0 started 3.4x too late and left the near shells uncorrected.
# The HP_TRANSITION raised-cosine hand-over was REMOVED 2026-08-11 (as were the
# generative pipelines' fixed-angular hpc/hpt): it held the correction below full
# strength for ~0.10*lmax past each ell_min, exactly the band the kappa spectra
# were worst in. The hand-over is now a hard step at ell_min.
ELL_MIN_MPC=${ELL_MIN_MPC:-17.0}
# Shell resolution the correction is applied at. Was hard-coded to 2048; the thesis
# scores all three pipelines at 512 (common-footing comparison), and a mismatch here
# silently compares two different resolutions.
NSIDE=${NSIDE:-512}
if [ "$LMAX" -gt $((3 * NSIDE - 1)) ]; then
    echo "[abort] LMAX=$LMAX exceeds band limit 3*NSIDE-1=$((3 * NSIDE - 1)) for NSIDE=$NSIDE" >&2
    exit 1
fi
# ---- common comparison footing (thesis Sec. "protocol") --------------------------
# ALL pipelines are scored at N_side=512 -> lmax=1500, and the kappa maps are built at
# the SAME resolution as the shells they integrate. Building kappa at a higher nside
# than the shells is pure upsampling: it invents no information and produces spectra
# past the shells' own band limit (3*512-1 = 1535) that are interpolation and
# pixel-window artefacts. Derived, not hard-coded, so the two cannot drift apart.
KAPPA_NSIDE=${KAPPA_NSIDE:-$NSIDE}
KAPPA_LMAX=${KAPPA_LMAX:-$LMAX}
if [ "$KAPPA_LMAX" -gt $((3 * KAPPA_NSIDE - 1)) ]; then
    echo "[abort] KAPPA_LMAX=$KAPPA_LMAX exceeds band limit 3*KAPPA_NSIDE-1=$((3 * KAPPA_NSIDE - 1))" >&2
    exit 1
fi
# apply_transfer.py's --max-cosmologies caps cl_ratio_by_zbin_grid.png's rows
# (default 3, sized for a single-cosmology-at-a-time budget). This script exists
# specifically to re-run diagnostics against ALREADY-COMPUTED (or freshly
# multi-node-computed) counts, so default to showing EVERY held-out cosmology
# resolved above -- there's no compute reason to cap it here. Override explicitly
# to go back to a smaller grid.
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-${#COSMOS_ARR[@]}}
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT"

# REUSE_COUNTS: point at a previous job's counts/ dir to skip apply() and go
# straight to the plots -- the corrected counts are deterministic given
# (transfer, seed, ELL_MIN_MPC, --no-clip), so this is EXACT as long as none of
# those changed. This is the fast path when only the DIAGNOSTICS changed
# (new/updated plots), which is most iterations. Only consulted when --nodes=1
# (the default) -- see the MULTI-NODE header comment above for why N>1 always
# does a full parallel recompute instead.
#   sbatch --export=TRANSFER_JOB=4201972,REUSE_COUNTS=/capstor/scratch/cscs/damrein/outputs/transfer/4201972/counts \
#       transfer/run_diagnostic_only.sh
# Defaults to TRANSFER_JOB's OWN counts/ dir if that exists (same job that produced
# the transfer files also produced counts under the identical knobs).
if [ "$N_NODES" -eq 1 ] && [ -z "$REUSE_COUNTS" ] && [ -d "$TRANSFER_DIR/counts" ]; then
    REUSE_COUNTS="$TRANSFER_DIR/counts"
    echo "[info] auto-reusing counts from $REUSE_COUNTS (set REUSE_COUNTS='' to force recompute)"
fi
REUSE_FLAG=""
[ "$N_NODES" -eq 1 ] && [ -n "$REUSE_COUNTS" ] && REUSE_FLAG="--reuse-counts '$REUSE_COUNTS'"

echo "==== transfer diagnostics-only | reusing transfer(s) from job $TRANSFER_JOB | ell_min_mpc=$ELL_MIN_MPC | nodes=$N_NODES ===="
echo "held-out cosmologies (${#COSMOS_ARR[@]}): ${COSMOS_ARR[@]}"
echo "kappa: nside=$KAPPA_NSIDE lmax=$KAPPA_LMAX"

# ---- compute-only correction, parallelized across nodes when N_NODES>1 ----
if [ "$N_NODES" -gt 1 ]; then
    echo "[stage a] correcting ${#COSMOS_ARR[@]} held-out cosmologies across $N_NODES nodes (compute-only)"
    declare -a BATCH_RUN_DIRS BATCH_TRANSFERS
    for ((i = 0; i < N_NODES; i++)); do BATCH_RUN_DIRS[$i]=""; BATCH_TRANSFERS[$i]=""; done
    for ((j = 0; j < ${#COSMOS_ARR[@]}; j++)); do
        node=$((j % N_NODES))
        BATCH_RUN_DIRS[$node]="${BATCH_RUN_DIRS[$node]} $DATA/${COSMOS_ARR[$j]}/run_0"
        BATCH_TRANSFERS[$node]="${BATCH_TRANSFERS[$node]} ${TRANSFER_ARR[$j]}"
    done
    for ((i = 0; i < N_NODES; i++)); do
        [ -z "${BATCH_RUN_DIRS[$i]}" ] && continue
        srun --nodes=1 --ntasks=1 --exclusive \
            uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
                source ${SPHEREFLOW_VENV}/bin/activate
                OMP_NUM_THREADS=$CPUS_PER_NODE python transfer/apply_transfer.py \
                    --transfer ${BATCH_TRANSFERS[$i]} \
                    --run-dirs ${BATCH_RUN_DIRS[$i]} \
                    --nside $NSIDE --lmax $LMAX --ell-min-mpc $ELL_MIN_MPC \
                    --no-clip --seed 0 \
                    --out-counts-dir '$OUT/counts' \
                    --patch-shells --fullsky-shells --n-zbins 0 \
                    --out-dir '$OUT/eval_shard_${i}'
            " &
    done
    wait
    REUSE_FLAG="--reuse-counts '$OUT/counts'"
fi

RUN_DIRS=""
TRANSFER_FILES=""
for ((j = 0; j < ${#COSMOS_ARR[@]}; j++)); do
    RUN_DIRS="$RUN_DIRS $DATA/${COSMOS_ARR[$j]}/run_0"
    TRANSFER_FILES="$TRANSFER_FILES ${TRANSFER_ARR[$j]}"
done

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${SPHEREFLOW_VENV}/bin/activate
    OMP_NUM_THREADS=$CPUS_PER_NODE python transfer/apply_transfer.py \
        --transfer $TRANSFER_FILES \
        --run-dirs $RUN_DIRS \
        --nside $NSIDE --lmax $LMAX --ell-min-mpc $ELL_MIN_MPC \
        --no-clip --seed 0 \
        $REUSE_FLAG \
        --out-counts-dir '$OUT/counts' \
        --patch-shells 5 10 15 30 50 --n-per-shell 1 --patch-size 256 \
        --fullsky-shells 5 10 15 30 50 --max-cosmologies $MAX_COSMOLOGIES \
        --kappa --kappa-nside $KAPPA_NSIDE --kappa-lmax $KAPPA_LMAX \
        --out-dir '$OUT/eval'
"

echo "transfer diagnostics-only ${SLURM_JOB_ID} finished at $(date) -> $OUT/eval"
