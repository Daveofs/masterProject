#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=transfer-diag
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/diag-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/diag-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Diagnostic-only run against an ALREADY-BUILT transfer.npz (from a prior fit/train
# stage) -- runs apply_transfer.py's full correction + Poisson + ALL diagnostics
# WITHOUT redoing alm preprocessing or emulator training. Mirrors
# unet/run_diagnostics_only.sh's pattern (reuse an existing checkpoint,
# just re-run apply+eval -- e.g. after changing --ell-min-mpc or a Poisson knob).
#
# Validates on the SAME MULTIPLE held-out cosmologies run_transfer_pipeline.sh used
# (auto-discovered from the prior job's output -- see stage 0 below), not just one.
#
# Point TRANSFER_JOB at the SLURM job id of a prior run_transfer_pipeline.sh /
# run_transfer.sh run whose transfer.npz (fit) / transfer_<cosmo>.npz (emulate) you
# want to reuse:
#   sbatch --export=TRANSFER_JOB=4199680 transfer/run_diagnostic_only.sh
# DATA/ELL_MIN_MPC/N_ITER currently default to the grid run_transfer.sh job
# 4215699's config (grid dataset, ell_min_mpc=5.0, n_iter=3) -- see the block below
# for why. For a DIFFERENT prior job (e.g. a cosmogridv1 run_transfer_pipeline.sh
# run), override all of these together at submission time, since REUSE_COUNTS is
# only an EXACT match if they agree with whatever originally produced the counts:
#   sbatch --export=TRANSFER_JOB=4201972,DATA=/capstor/scratch/cscs/damrein/cosmogridv1,ELL_MIN_MPC=3.0,N_ITER=5 \
#       transfer/run_diagnostic_only.sh
# Override the held-out cosmology set explicitly (space-separated) if you want a
# different/smaller set than what that job evaluated:
#   sbatch --export=TRANSFER_JOB=4199680,TEST_COSMOS='cosmo_000001 cosmo_000003' \
#       transfer/run_diagnostic_only.sh

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=8

# apply_transfer.py itself now runs under sphereflow (see run_transfer_pipeline.sh's
# stage 3 comment) -- UFalcon (--kappa) is only installed there. deepSphere stays
# active here only for this script's own cosmology-resolution bash/numpy logic below.
export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
SPHEREFLOW_VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow

DATA=${DATA:-/capstor/scratch/cscs/damrein/grid}
LMAX=3000

# Defaults currently point at the grid run_transfer.sh job 4215699 (10 held-out
# grid cosmologies, MAX_COSMOLOGIES=10, ell_min_mpc=5.0 n_iter=3) -- so plain
# `sbatch transfer/run_diagnostic_only.sh` re-runs/recovers ITS diagnostics with no
# --export needed. To target a DIFFERENT prior job (e.g. a cosmogridv1 run),
# override at submission time -- see the header comment above for the full
# --export= form (TRANSFER_JOB/DATA/ELL_MIN_MPC/N_ITER must all match together).
TRANSFER_JOB=${TRANSFER_JOB:-4215699}
TRANSFER_DIR="/capstor/scratch/cscs/damrein/outputs/transfer/${TRANSFER_JOB}_cos200"

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
    TRANSFER_FILES=""
    for c in "${COSMOS_ARR[@]}"; do
        f="$TRANSFER_DIR/transfer_${c}.npz"
        [ -f "$f" ] || { echo "ERROR: $f not found (no emulated T for $c in job $TRANSFER_JOB)"; exit 1; }
        TRANSFER_FILES="$TRANSFER_FILES $f"
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
    TRANSFER_FILES="$TRANSFER_DIR/transfer.npz"
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

# Same apply-stage knobs as run_transfer_pipeline.sh/run_transfer.sh -- override via
# --export= to re-run diagnostics with different settings against the SAME trained
# transfer(s). Defaults here match job 4215699's actual grid-run config
# (ELL_MIN_MPC=5.0, N_ITER=3) -- REUSE_COUNTS below is only an EXACT match if these
# agree with what originally produced the counts being reused.
ELL_MIN_MPC=${ELL_MIN_MPC:-5.0}
N_AVG=${N_AVG:-4}
N_ITER=${N_ITER:-3}
DAMP=${DAMP:-0.4}
KAPPA_NSIDE=${KAPPA_NSIDE:-1024}
KAPPA_LMAX=${KAPPA_LMAX:-2048}
# apply_transfer.py's --max-cosmologies caps cl_ratio_by_zbin_grid.png's rows
# (default 3, sized for the EXPENSIVE apply()+Poisson case). This script exists
# specifically to re-run diagnostics against ALREADY-COMPUTED counts, so default
# to showing EVERY held-out cosmology resolved above -- there's no compute reason
# to cap it here. Override explicitly to go back to a smaller grid.
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-${#COSMOS_ARR[@]}}
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/${SLURM_JOB_ID}
mkdir -p "$OUT"

# REUSE_COUNTS: point at a previous job's counts/ dir to skip apply()+Poisson
# (~50 min PER cosmology) and go straight to the plots -- the corrected counts are
# deterministic given (transfer, seed, ELL_MIN_MPC, N_AVG/N_ITER/DAMP), so this is
# EXACT as long as none of those changed. This is the fast path when only the
# DIAGNOSTICS changed (new/updated plots), which is most iterations.
#   sbatch --export=TRANSFER_JOB=4201972,REUSE_COUNTS=/capstor/scratch/cscs/damrein/outputs/transfer/4201972/counts \
#       transfer/run_diagnostic_only.sh
# Defaults to TRANSFER_JOB's OWN counts/ dir if that exists (same job that produced
# the transfer files also produced counts under the identical knobs).
if [ -z "$REUSE_COUNTS" ] && [ -d "$TRANSFER_DIR/counts" ]; then
    REUSE_COUNTS="$TRANSFER_DIR/counts"
    echo "[info] auto-reusing counts from $REUSE_COUNTS (set REUSE_COUNTS='' to force recompute)"
fi
REUSE_FLAG=""
[ -n "$REUSE_COUNTS" ] && REUSE_FLAG="--reuse-counts '$REUSE_COUNTS'"

echo "==== transfer diagnostics-only | reusing transfer(s) from job $TRANSFER_JOB | ell_min_mpc=$ELL_MIN_MPC ===="
echo "held-out cosmologies (${#COSMOS_ARR[@]}): ${COSMOS_ARR[@]}"
echo "kappa: nside=$KAPPA_NSIDE lmax=$KAPPA_LMAX"

RUN_DIRS=""
for c in "${COSMOS_ARR[@]}"; do RUN_DIRS="$RUN_DIRS $DATA/$c/run_0"; done

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${SPHEREFLOW_VENV}/bin/activate
    OMP_NUM_THREADS=128 python transfer/apply_transfer.py \
        --transfer $TRANSFER_FILES \
        --run-dirs $RUN_DIRS \
        --nside 2048 --lmax $LMAX --ell-min-mpc $ELL_MIN_MPC \
        --no-poisson --no-clip \
        $REUSE_FLAG \
        --out-counts-dir '$OUT/counts' \
        --patch-shells 5 10 15 30 50 --n-per-shell 1 --patch-size 256 --seed 0 \
        --fullsky-shells 5 10 15 30 50 --max-cosmologies $MAX_COSMOLOGIES \
        --kappa --kappa-nside $KAPPA_NSIDE --kappa-lmax $KAPPA_LMAX \
        --out-dir '$OUT/eval'
"

echo "transfer diagnostics-only ${SLURM_JOB_ID} finished at $(date) -> $OUT/eval"
