#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=unet-flow
#SBATCH --partition=normal
#SBATCH --account=a0158
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/unet/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/unet/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# The patch-based conditional flow-matching (UNet) pipeline. Everything lives in
# unet/ (renamed from the old unet_flow_jbucko/ -- job 4208375 died because these
# scripts still pointed at the old path):
#   flow_model.py, dataset.py, make_patch_dataset.py, train_flow.py (model/data)
#   plot_flow_loss.py, apply_flow.py                                (plot + eval)
#
#   stage 0  prepare low/high shell stacks (our preprocess/prepare_maps.py)
#   stage 1  build flat gnomonic (low,high) patch dataset  (make_patch_dataset.py)
#   stage 2  DDP train the low->high flow                  (train_flow.py, ${SLURM_NNODES} nodes x 4 GPU)
#   stage 3  plot train/val loss + apply on HELD-OUT patches
#
# NODES: ${SLURM_NNODES} (header #SBATCH --nodes=4), GENUINE MULTI-NODE (2026-07-21,
# by explicit request) -- stage 2 now uses a from-scratch torchrun/c10d rendezvous
# across every allocated node, same pattern as diffusion/run_diffusion.sh's stage 2
# and sphereflow/run_sphere_flow.sh's (converted alongside this one). THIS IS A
# KNOWN RISK, not an oversight: sphereflow's original multi-node trainer crashed
# DETERMINISTICALLY at ~step 13.2k, TWICE (jobs 3852435, 4210107), always a
# cross-node Slingshot/libfabric (CXI) error ("NET/OFI ... PTLTE_NOT_FOUND", NCCL
# SIGABRT) that no amount of FI_CXI_*/NCCL_* tuning fixed -- the fix that actually
# worked was going single-node (this script ran single-node, --standalone, from
# 2026-07-14 until this change). No CXI/NCCL env tuning is added below (that tuning
# specifically did NOT fix the earlier crashes) -- if this dies with a similar
# cross-node NCCL/libfabric error late into training, that is the same known
# failure mode recurring, not a new bug. Stages 0/1/3 stay single-node
# (--nodes=1): they are single-task CPU/prep/eval-launcher work that does not
# parallelize across nodes, so giving them more would just leave nodes idle.
#
# Why the flow (not our deterministic diff model): a deterministic MSE regressor's
# optimum is the conditional mean, which shrinks the correction to corr*target
# (measured corr~0.25-0.30) -> small-scale power deficit. A flow SAMPLE carries the
# full high-field variance, so small-scale power is right by construction.
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="${DATA_ROOT:-/capstor/scratch/cscs/damrein/grid}"
# CosmoGridV1_metainfo.h5 (the cosmological-parameter catalog make_patch_dataset.py
# needs) lives ONLY under cosmogridv1/, not replicated under grid/ or any other
# subset dir -- pass it separately regardless of which DATA_ROOT is active (this is
# what job 4219025 was missing: metainfo-dir defaulted to DATA_ROOT=grid, which has
# no such file).
METAINFO_DIR="/capstor/scratch/cscs/damrein/cosmogridv1"
UNET=/users/damrein/masterProject/ml/unet

NSIDE=${NSIDE:-1024}
PATCH_SIZE=${PATCH_SIZE:-256}
NPATCH=${NPATCH:-100000}
EPOCHS=${EPOCHS:-200}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
# NO_LR_SCALING=1 drops train_flow.py's world_size multiplier (16 GPUs -> the
# configured LEARNING_RATE is used AS-IS, not x16 = 4.8e-4). Recurring pattern
# across pipelines (sphereflow job 4251268, diffusion job 4256969): scaled LR with
# no warmup caused a large early-training instability spike. Default ON (0) here
# means scaling is STILL applied by default -- set NO_LR_SCALING=1 to test the
# unscaled LR directly.
NO_LR_SCALING=${NO_LR_SCALING:-0}
NO_LR_SCALING_FLAG=""; [ "${NO_LR_SCALING}" = "1" ] && NO_LR_SCALING_FLAG="--no-lr-scaling"
BATCH=${BATCH:-32}
BASE_CH=${BASE_CH:-32}
STEPS=${STEPS:-8}
# High-pass residual formulation (2026-07-21, ported from diffusion/run_diffusion.sh
# after comparing all three pipelines' cl_ratio_by_zbin_grid/kappa_cl_pctile_band --
# see flow_model.py's module docstring): the flow target now only adds small-scale
# content, large scales pinned to the low map. Passed to BOTH train (builds the
# target) and, via the checkpoint, apply (composes the corrected map).
HP_CUTOFF=${HP_CUTOFF:-0.05}
HP_TRANSITION=${HP_TRANSITION:-0.12}
# cosmology+redshift conditioning at the bottleneck (flow_model.FlowUNet) -- on by
# default; set USE_COSMO_COND=0 for an A/B run against the unconditioned model.
USE_COSMO_COND=${USE_COSMO_COND:-1}
COSMO_FLAG="--use-cosmo-cond"; COSMO_SUFFIX=""
if [ "${USE_COSMO_COND}" = "0" ]; then
  COSMO_FLAG="--no-use-cosmo-cond"; COSMO_SUFFIX="_nocosmo"
fi
# DATA_ROOT is folded into PATCH_DIR/RUN_NAME below -- PATCH_DIR used to be named
# ONLY from NSIDE/PATCH_SIZE/NPATCH, so switching DATA_ROOT (e.g. cosmogridv1 ->
# grid) while keeping those the same silently reused the OTHER data root's stale
# patch dataset (stage 1's "already exists, skip" check only looks at PATCH_DIR).
# That is exactly what broke job 4218868: training silently ran on cosmogridv1
# patches while DATA_ROOT=grid, so stage 3's full-sky eval then looked for those
# cosmogridv1 cosmologies' low_shells_nside=*.npy under grid/, where they don't
# exist -> FileNotFoundError. Tagging both paths with DATA_ROOT's basename makes a
# data-root switch always produce a fresh, correctly-matched, distinctly-named
# patch dir + run (and never overwrites the other data root's existing outputs).
DATA_TAG=$(basename "${DATA_ROOT}")
# 'delta' (linear overdensity) is the space analysis.full_sky.od_cl actually
# measures -- same fix diffusion made 2026-07-18 (see dataset.raw_to_delta_pair).
# Assigned BEFORE RUN_NAME below (it is folded into it -- when it was assigned
# after, every auto run-name silently expanded ${SPACE} to "" -> flow__grid_...).
SPACE=${SPACE:-delta}
# SPACE folded in (matching diffusion/run_diffusion.sh's RUN_NAME convention) so a
# highpass-residual delta-space retrain never collides with an old pre-2026-07-21
# full-field checkpoint at the same name -- apply_flow.py's hp_cutoff guard would
# reject the old one anyway, but this keeps the two from silently overwriting.
RUN_NAME=${RUN_NAME:-flow_${SPACE}_${DATA_TAG}_nside${NSIDE}_patch${PATCH_SIZE}_n${NPATCH}_ch${BASE_CH}_b${BATCH}_e${EPOCHS}_lr${LEARNING_RATE}_hpc${HP_CUTOFF}_hpt${HP_TRANSITION}${COSMO_SUFFIX}}
# weak-lensing kappa map diagnostic (analysis.weak_lensing, apply_flow.py --kappa):
# ON by default. It reconstructs every usable shell (z<~1.05, ~47/69) via full-sky
# tiling for --kappa-max-cosmologies held-out cosmologies, which used to be far too
# expensive to run by default -- but the tile geometry is now built once and reused
# across every shell/cosmology (analysis.patch_tiling.gnomonic_index_maps, ~108x
# faster per shell), so it fits comfortably. KAPPA=0 skips it.
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa"
# held-out cosmologies used by BOTH the zbin-grid diagnostic (--max-cosmologies)
# and the kappa diagnostic (--kappa-max-cosmologies) -- apply_flow.py defaults
# each to 3 independently; one knob here keeps them in sync unless overridden.
MAX_COSMO=${MAX_COSMO:-30}

PATCH_DIR="/capstor/scratch/cscs/damrein/outputs/flowpatches/${DATA_TAG}_nside${NSIDE}_${PATCH_SIZE}_${NPATCH}"
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/flowruns/${RUN_NAME}"
mkdir -p "$PATCH_DIR" "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/unet

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
ulimit -c 0

echo "==== unet-flow | job ${SLURM_JOB_ID} | nside=${NSIDE} patch=${PATCH_SIZE} n=${NPATCH} use_cosmo_cond=${USE_COSMO_COND} ===="

# ---- stage 0: nside-512 low/high shell stacks (skip runs already prepared) ----
# Single-node: single-task CPU prep, does not parallelize across nodes -- extra
# allocated nodes would just sit idle (see header note on why only stage 2 scales).
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside ${NSIDE} --num-workers 5
"

# ---- stage 1: build the (low, high) patch dataset (his make_patch_dataset.py) ----
# metadata.npy is the LAST file make_patch_dataset.py writes (low.npy/high.npy are
# allocated as empty memmaps up front, before any patch is filled in) -- checking
# low.npy here would treat a killed/interrupted stage-1 run as complete and skip
# straight to training against a metadata.npy that was never written.
if [ ! -f "${PATCH_DIR}/metadata.npy" ]; then
  srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${VENV}/bin/activate
    python ${UNET}/make_patch_dataset.py \
      --data-dir     '${DATA_ROOT}' \
      --metainfo-dir '${METAINFO_DIR}' \
      --prepared-dir '${DATA_ROOT}' \
      --out-dir      '${PATCH_DIR}' \
      --nside ${NSIDE} --patch-size ${PATCH_SIZE} \
      --n-patches ${NPATCH} --seed 0 \
      --num-workers ${SLURM_CPUS_PER_TASK}
  "
else
  echo '[stage1] patch dataset already exists, skipping'
fi

# ---- stage 2: DDP train the low->high flow, GENUINE multi-node (${SLURM_NNODES}
# nodes x 4 GPU) -- see the header warning above. c10d rendezvous: one torchrun
# launcher task per node (--ntasks-per-node=1, NOT one task per GPU -- torchrun
# itself spawns the 4 per-GPU worker processes on each node), all pointed at
# node 0's address (same pattern as diffusion/run_diffusion.sh's stage 2). ----
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=29500
echo "[stage2] multi-node rendezvous: ${SLURM_NNODES} nodes, master=${MASTER_ADDR}:${MASTER_PORT}"

srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 --gres=gpu:4 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  cd ${UNET}
  python -m torch.distributed.run \
    --nnodes=${SLURM_NNODES} --nproc_per_node=4 \
    --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --out-dir   '${OUT_DIR}' \
    --epochs ${EPOCHS} --batch-size ${BATCH} --base-channels ${BASE_CH} --lr ${LEARNING_RATE} \
    --hp-cutoff ${HP_CUTOFF} --hp-transition ${HP_TRANSITION} --space ${SPACE} \
    --num-workers $((SLURM_CPUS_PER_TASK / 4)) ${COSMO_FLAG} ${NO_LR_SCALING_FLAG}
"

# ---- stage 3: loss/val plot + apply on held-out test patches + full-sky Cl (glue) ----
# plot_flow_loss.py: train vs validation flow-matching MSE (formula in the title),
# comparable to transfer_function.py train()'s emulator.loss.png.
# apply_flow.py: flat patch-grid diagnostic (analysis.plot_example_patch_grid, 2D-FFT
# power ratio bounded by that patch's own Nyquist ell) AND, since --data-root is
# given, the full-sky reconstruction + REAL angular Cl (analysis.patch_tiling +
# analysis.full_sky.od_cl, analysis.plot_example_full_sky_grid) -- one script, one
# --example-shells list shared by both diagnostics so they can never silently
# diverge. Full-sky reconstruction tiles the WHOLE sphere via one flow ODE
# integration per patch -- far more expensive per shell than the CPU-only transfer
# pipeline's real Cl. --shell-indices is left empty (the lone-Cl plots are redundant
# with cl_ratio_by_zbin_grid.png); --example-shells 5 10 15 30 50 is the SAME shell
# set transfer/apply_transfer.py and sphereflow/apply_sphere_flow.py use, so the
# three pipelines' figures compare directly.
# MULTI-NODE EVAL (2026-07-21, matching diffusion/run_diffusion.sh's stage 3
# EXACTLY, for consistency): apply_flow.py splits its two dominant-cost sections
# (zbin-grid, kappa) across ALL of GPUS_PER_NODE x SLURM_NNODES ranks via genuine
# multi-node torchrun c10d rendezvous, reusing the SAME MASTER_ADDR/MASTER_PORT
# stage 2 already established above. This eval's own distributed calls are a
# couple of dist.all_gather_object per diagnostic stage -- a handful of
# collectives, not thousands of sustained gradient-allreduce steps like stage 2 --
# a materially lighter cross-node workload. GPUS_PER_NODE=4 is this cluster's
# fixed per-node GPU count.
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-1}
EVAL_GPUS=$(( GPUS_PER_NODE * NNODES ))
srun --ntasks=${NNODES} --ntasks-per-node=1 --gres=gpu:${GPUS_PER_NODE} uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${UNET}/plot_flow_loss.py --run-dir '${OUT_DIR}'
  python -m torch.distributed.run \
    --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${UNET}/apply_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${OUT_DIR}/eval' \
    --steps ${STEPS} \
    --data-root '${DATA_ROOT}' \
    --shell-indices --example-shells 5 10 15 30 50 \
    --fullsky-patch-size ${PATCH_SIZE} ${KAPPA_FLAG} \
    --max-cosmologies ${MAX_COSMO} --kappa-max-cosmologies ${MAX_COSMO}
"
echo "unet-flow job ${SLURM_JOB_ID} finished at $(date) -> ${OUT_DIR}/eval"
