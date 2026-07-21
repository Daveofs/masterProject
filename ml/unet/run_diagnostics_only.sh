#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unet-diag
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/unet/diag-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/unet/diag-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Diagnostic-only run against the ALREADY-TRAINED checkpoint in OUT_DIR below (no
# retraining): apply_flow.py's full eval suite, including the full-sky section
# (--data-root given: full-sky reconstruction + real angular Cl) and --kappa.
# Emits the SAME figure set as transfer/run_diagnostic_only.sh and
# sphereflow/run_diagnostics_only.sh -- same shells (5 10 15 30 50), same kappa
# resolution (nside=1024, lmax=2048, apply_flow.py's defaults) -- so the three
# pipelines' eval dirs can be diffed plot-by-plot.

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
UNET=/users/damrein/masterProject/ml/unet
# Overridable so this can target ANY finished run without editing the file, e.g.
#   sbatch --export=ALL,PATCH_DIR=...,OUT_DIR=...,DATA_ROOT=... unet/run_diagnostics_only.sh
# (PATCH_DIR/OUT_DIR must be the pair that were actually trained together --
# apply_flow.py reads the source nside from PATCH_DIR's metadata, so a mismatched
# pair is a silent resolution error, not a crash; DATA_ROOT must be wherever
# PATCH_DIR's own held-out cosmologies actually live, e.g. grid for a grid-trained
# checkpoint, cosmogridv1 for a cosmogridv1-trained one -- see run_flow.sh's
# DATA_TAG note for why a mismatch here silently breaks the full-sky/kappa sections).
# Defaults below point at job 4219071's completed grid run (confirmed working
# end-to-end 2026-07-15) -- the OLD default (flow_nside2048_patch256_..._cond) was
# never actually trained (only apply_flow.py's own eval/ dir existed there), so
# every run_diagnostics_only.sh submission failed loading best.pt from it.
DATA_ROOT=${DATA_ROOT:-/capstor/scratch/cscs/damrein/cosmogridv1}
PATCH_DIR=${PATCH_DIR:-/capstor/scratch/cscs/damrein/outputs/flowpatches/cosmogridv1_nside512_256_100000}
OUT_DIR=${OUT_DIR:-/capstor/scratch/cscs/damrein/outputs/flowruns/flow_cosmogridv1_nside512_patch256_n100000_ch32_b248_e40}

export PYTHONUNBUFFERED=1
# EVAL_GPUS defined here (used below by both OMP_NUM_THREADS and the torchrun launch)
# -- apply_flow.py splits its zbin-grid/kappa sections (dominant cost) across this
# many GPUs via torch.distributed, single-node intra-NVLink. Drop to 1 to fall back
# to the original single-process path.
EVAL_GPUS=${EVAL_GPUS:-4}
# Pin OpenMP-using CPU work (healpy's Cl/anafast, UFalcon's kappa-map construction)
# to the cores SLURM actually allocated (--cpus-per-task=128 above), DIVIDED across
# the EVAL_GPUS concurrent rank processes -- the parallel zbin-grid/kappa sections
# run all ranks' CPU-bound work at once, so giving every rank the full core count
# would oversubscribe (EVAL_GPUS x 128 threads contending for 128 cores).
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-128} / EVAL_GPUS ))
# weak-lensing kappa map diagnostic -- off by default, see run_flow.sh's KAPPA note.
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa"
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-10}
# --amp defaults to True in apply_flow.py (added 2026-07-20, UNTESTED at the time) --
# AMP=0 here lets a diagnostics-only rerun test whether bf16 autocast in
# flow_model.sample_ode is responsible for a suspected regression (job 4248318's
# cl_ratio_by_zbin_grid.png: catastrophic percentile-band blowup on faint shells +
# a systematic below-baseline bias) before assuming it's a tiling/model issue.
AMP=${AMP:-1}
AMP_FLAG=""; [ "${AMP}" = "0" ] && AMP_FLAG="--no-amp"
# --no-amp (tested 2026-07-20, job 4248443) reproduced the SAME cl_ratio_by_zbin_grid.png
# failure as --amp (job 4248318) -- catastrophic low-ell percentile-band collapse on
# faint shells + a systematic below-baseline bias -- so AMP is ruled out. Next
# hypothesis: --taper-power (already a supported flag, default 1.0 -- see
# apply_flow.py's help, reasoned from unet's ODE being deterministic so overlapping
# tiles "should" agree). That reasoning may break down on faint/low-count shells,
# where each tile normalizes by its OWN local mean before integrating -- a boundary
# landing differently on a sparse shell can make overlapping tiles genuinely
# disagree (not from stochastic noise, but window-dependent normalization), and a
# plain average (taper_power=1) blends that disagreement destructively. Test
# TAPER_POWER=32 (diffusion's tuned value) here before assuming it transfers as-is.
TAPER_POWER=${TAPER_POWER:-}
TAPER_POWER_FLAG=""; [ -n "${TAPER_POWER}" ] && TAPER_POWER_FLAG="--taper-power ${TAPER_POWER}"

# SPEED: job 4201387 TIMED OUT at 10h having done only 61 shell reconstructions --
# full-sky tiling was re-projecting all 12288 gnomonic tiles from scratch for every
# shell (~8 min/shell of single-threaded CPU, dwarfing the GPU work it was feeding).
# The tile geometry is value-independent, so it is now built once and reused across
# every shell AND cosmology (analysis.patch_tiling.gnomonic_index_maps): ~108x faster
# per shell (~500s -> ~5s), leaving the GPU flow-ODE integration (~1.5 min/shell) as
# the limiter. --kappa-max-cosmologies is therefore the real cost knob now, not the
# tiling. NOTE: the index cache costs ~3.2GB RAM at nside=2048/patch=256.

srun --nodes=1 --ntasks=1 --gres=gpu:${EVAL_GPUS} uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${UNET}/plot_flow_loss.py --run-dir '${OUT_DIR}'
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=${EVAL_GPUS} \
    ${UNET}/apply_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${OUT_DIR}/eval' \
    --steps 8 \
    --data-root '${DATA_ROOT}' \
    --shell-indices --example-shells 5 10 15 30 50 \
    --fullsky-patch-size 256 --max-cosmologies ${MAX_COSMOLOGIES} \
    --kappa-max-cosmologies ${MAX_COSMOLOGIES} ${KAPPA_FLAG} ${AMP_FLAG} ${TAPER_POWER_FLAG}
"
echo "unet diagnostics-only job \${SLURM_JOB_ID} finished at \$(date) -> ${OUT_DIR}/eval"
