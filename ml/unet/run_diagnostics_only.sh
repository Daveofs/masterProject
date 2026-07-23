#!/bin/bash
#SBATCH --nodes=4
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

# MULTI-NODE EVAL (2026-07-21, matching sphereflow/run_diagnostics_only.sh's
# identical pattern): the --nodes=1 header above is just the sbatch default --
# override at submission (`sbatch --nodes=4 unet/run_diagnostics_only.sh`) to
# scale past one node's 4 GPUs. WORLD_SIZE below is derived from however many
# nodes SLURM actually gave this job (SLURM_NNODES x GPUS_PER_NODE), via a genuine
# multi-node torchrun c10d rendezvous (MASTER_ADDR from the node list, srun spans
# every node). This is a DIFFERENT risk profile from unet TRAINING's multi-node
# conversion (run_flow.sh): this eval job's own distributed calls are a couple of
# dist.all_gather_object per diagnostic stage (zbin-grid, kappa) -- a handful of
# collectives total, not thousands of sustained gradient-allreduce steps -- so
# it's a materially lighter cross-node workload. Not proven crash-proof, just a
# different, lower-frequency usage pattern.

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
DATA_ROOT=${DATA_ROOT:-/capstor/scratch/cscs/damrein/grid}
PATCH_DIR=${PATCH_DIR:-/capstor/scratch/cscs/damrein/outputs/flowpatches/grid_nside512_256_100000}
OUT_DIR=${OUT_DIR:-/capstor/scratch/cscs/damrein/outputs/flowruns/flow_delta_grid_nside512_patch256_n100000_ch32_b32_e200_lr3e-5_hp0.10_0.20_lossw}
# Where the figures go (default: the run's own eval/). Override for an A/B rerun
# against a checkpoint whose eval/ already holds the baseline you want to KEEP --
# e.g. EVAL_DIR=${OUT_DIR}/eval_taper32 for the taper_power A/B, so the existing
# taper=1 figures survive for side-by-side comparison instead of being overwritten.
EVAL_DIR=${EVAL_DIR:-${OUT_DIR}/eval}

export PYTHONUNBUFFERED=1
# GPUS_PER_NODE x SLURM_NNODES = total ranks -- apply_flow.py splits its two
# dominant-cost sections (zbin-grid, kappa) across ALL of them via
# torch.distributed. SLURM_NNODES reflects whatever --nodes this job was actually
# submitted with (see the multi-node note above); GPUS_PER_NODE=4 is this
# cluster's fixed per-node GPU count, not a tunable.
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-1}
EVAL_GPUS=$(( GPUS_PER_NODE * NNODES ))
# Pin OpenMP-using CPU work (healpy's Cl/anafast, UFalcon's kappa-map construction)
# to the cores SLURM actually allocated PER NODE (--cpus-per-task=128 above),
# DIVIDED across the GPUS_PER_NODE concurrent rank processes THAT NODE RUNS -- the
# parallel zbin-grid/kappa sections run all ranks' CPU-bound work at once, so
# giving every rank the full per-node core count would oversubscribe.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-128} / GPUS_PER_NODE ))
# weak-lensing kappa map diagnostic -- off by default, see run_flow.sh's KAPPA note.
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa"
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-30}
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

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

echo "==== unet diagnostics | job ${SLURM_JOB_ID} | ${NNODES} node(s) x ${GPUS_PER_NODE} GPU = ${EVAL_GPUS} ranks ===="

# srun spans EVERY allocated node (one task/node, torchrun fans out to
# GPUS_PER_NODE local ranks) -- NOT --nodes=1 like the old single-node version.
srun --ntasks=${NNODES} --ntasks-per-node=1 --gres=gpu:${GPUS_PER_NODE} uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${UNET}/plot_flow_loss.py --run-dir '${OUT_DIR}'
  python -m torch.distributed.run \
    --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${UNET}/apply_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${EVAL_DIR}' \
    --steps 8 \
    --data-root '${DATA_ROOT}' \
    --shell-indices --example-shells 5 10 15 30 50 \
    --fullsky-patch-size 256 --max-cosmologies ${MAX_COSMOLOGIES} \
    --kappa-max-cosmologies ${MAX_COSMOLOGIES} ${KAPPA_FLAG} ${AMP_FLAG} ${TAPER_POWER_FLAG}
"
echo "unet diagnostics-only job \${SLURM_JOB_ID} finished at \$(date) -> ${EVAL_DIR}"
