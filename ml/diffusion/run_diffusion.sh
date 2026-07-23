#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=diffusion
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/diffusion/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/diffusion/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# The patch-based EDM conditional-diffusion pipeline (see diffusion/model.py for why
# this is a genuinely different generative process from unet/'s and sphereflow/'s
# rectified flow matching -- real multi-step Heun ODE from pure noise, not a 2-point
# straight-line flow). Everything lives in diffusion/:
#   model.py, dataset.py, make_patch_dataset.py, train_diffusion.py (model/data)
#   plot_diffusion_loss.py, apply_diffusion.py                       (plot + eval)
#
#   stage 0  prepare low/high shell stacks (our preprocess/prepare_maps.py)
#   stage 1  build flat gnomonic (low,high) patch dataset (make_patch_dataset.py)
#   stage 2  DDP train the denoiser                (train_diffusion.py, multi-node x 4 GPU)
#   stage 3  plot train/val loss + apply on HELD-OUT patches
#
# STAGE 2 IS GENUINE MULTI-NODE (SLURM_NNODES x 4 GPU), BY EXPLICIT REQUEST --
# this is a KNOWN RISK on this cluster, not an oversight: sphereflow's original
# multi-node trainer crashed DETERMINISTICALLY at ~step 13.2k, TWICE (jobs 3852435,
# 4210107), always a cross-node Slingshot/libfabric (CXI) error ("NET/OFI ...
# PTLTE_NOT_FOUND", NCCL SIGABRT) that no amount of FI_CXI_*/NCCL_* tuning fixed --
# see sphereflow-model-survey / diffusion-pipeline-build memories. The fix that
# actually worked there was going single-node (NCCL stays on intra-node NVLink,
# never touches the fabric) -- unet/run_flow.sh and sphereflow/run_sphere_flow.sh
# are BOTH single-node today for that reason, and there is no currently-working
# multi-node recipe anywhere in this repo to copy. This script's stage 2 rendezvous
# below is a from-scratch standard torchrun/c10d multi-node launch (no CXI/NCCL env
# tuning added, since that tuning specifically did NOT fix the earlier crashes) --
# if it dies with a similar cross-node NCCL/libfabric error late into training,
# that is the same known failure mode, not a new bug in this pipeline.
#
# PATCH_DIR below uses the EXACT SAME naming convention as unet/run_flow.sh's, so
# when unet has already built a patch dataset at this (DATA_ROOT, NSIDE, PATCH_SIZE,
# NPATCH) combination, stage 1's "already exists, skip" check finds it and this
# pipeline trains on it directly -- no need to pay for a second, byte-identical
# patch-build job. If it doesn't exist yet (e.g. a diffusion-only patch size),
# diffusion/make_patch_dataset.py builds its own independently (it's a deliberate
# local duplicate of unet's, not an import -- see feedback-decoupled-pipeline-modules
# memory, so this pipeline never breaks if unet/ changes).
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="${DATA_ROOT:-/capstor/scratch/cscs/damrein/grid}"
METAINFO_DIR="/capstor/scratch/cscs/damrein/cosmogridv1"
DIFFUSION=/users/damrein/masterProject/ml/diffusion

# nside=2048 by DEFAULT: this is where the actual small-scale correction lives (faint
# shells, ell~800-1500) and where transfer/ and sphereflow/ were evaluated, so the
# figures are comparable and there's real headroom to win. The first run was nside=512,
# where DISCO is already ~perfect (low/high Cl ratio ~1 almost everywhere) so there was
# almost nothing to correct -- see diffusion-pipeline-build memory.
NSIDE=${NSIDE:-512}
PATCH_SIZE=${PATCH_SIZE:-256}
NPATCH=${NPATCH:-100000}
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-32}
BASE_CH=${BASE_CH:-32}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
# NO_LR_SCALING=1 drops train_diffusion.py's world_size multiplier (16 GPUs -> the
# configured LEARNING_RATE is used AS-IS, not x16 = 4.8e-4). Recurring pattern
# across pipelines (sphereflow job 4251268, this pipeline's own job 4256969):
# scaled LR with no warmup caused a large early-training instability spike (and,
# for 4256969, a loss that oscillated without converging for all 200 epochs).
# Default ON (0) here means scaling is STILL applied by default -- set
# NO_LR_SCALING=1 to test the unscaled LR directly.
NO_LR_SCALING=${NO_LR_SCALING:-1}
NO_LR_SCALING_FLAG=""; [ "${NO_LR_SCALING}" = "1" ] && NO_LR_SCALING_FLAG="--no-lr-scaling"
# EDM sampler: far more steps than the flow pipelines' ~8 -- see model.sample_heun.
STEPS=${STEPS:-32}
SIGMA_MIN=${SIGMA_MIN:-0.002}
SIGMA_MAX=${SIGMA_MAX:-80.0}
RHO=${RHO:-7.0}
# EDM training noise schedule: ln(sigma) ~ N(P_MEAN, P_STD^2), Karras et al. defaults.
P_MEAN=${P_MEAN:--1.2}
P_STD=${P_STD:-1.2}
# High-pass residual formulation (see model.py): the model diffuses only
# highpass(high_log-low_log); scales below HP_CUTOFF*Nyquist are pinned to the low
# (DISCO) map, fixing the large-scale kappa Cl the first full-field run destroyed.
# Fractions of patch Nyquist. Passed to BOTH train (builds the target) and, via the
# checkpoint, apply (composes the corrected map) -- they can't drift.
HP_CUTOFF=${HP_CUTOFF:-0.05}
HP_TRANSITION=${HP_TRANSITION:-0.12}
# Field the residual is modelled in. 'delta' (linear overdensity) is the space
# analysis.full_sky.od_cl actually measures. Training on 'log1p' was a FORMULATION BUG:
# log1p compresses the density peaks so DISCO already looks ~correct there (low/high
# power ratio 0.93-1.05) while the real deficit in linear delta reaches 0.62 -- the
# model then scored ~1.02 on the (log-space) patch diagnostic while the full-sky Cl it
# is judged on stayed at 0.57. See dataset.raw_to_delta_pair.
SPACE=${SPACE:-delta}
# left empty by default -> train_diffusion.py MEASURES it from real training data
# instead of guessing (see estimate_sigma_data); set explicitly to reuse a known
# value across resumed/related runs without re-measuring.
SIGMA_DATA=${SIGMA_DATA:-}
SIGMA_DATA_FLAG=""; [ -n "${SIGMA_DATA}" ] && SIGMA_DATA_FLAG="--sigma-data ${SIGMA_DATA}"

USE_COSMO_COND=${USE_COSMO_COND:-1}
COSMO_FLAG="--use-cosmo-cond"; COSMO_SUFFIX=""
if [ "${USE_COSMO_COND}" = "0" ]; then
  COSMO_FLAG="--no-use-cosmo-cond"; COSMO_SUFFIX="_nocosmo"
fi

DATA_TAG=$(basename "${DATA_ROOT}")
RUN_NAME=${RUN_NAME:-diffusion_${SPACE}_${DATA_TAG}_nside${NSIDE}_patch${PATCH_SIZE}_n${NPATCH}_ch${BASE_CH}_b${BATCH}_e${EPOCHS}_lr${LEARNING_RATE}_hpc${HP_CUTOFF}_hpt${HP_TRANSITION}${COSMO_SUFFIX}}
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa"
# Stage-3 cost knobs. Each full-sky shell reconstruction is one Heun ODE per tile, so
# these are what set the eval walltime: the Cl grid costs
# MAX_COSMOLOGIES x N_ZBINS x N_SHELLS_PER_ZBIN reconstructions, and kappa costs
# KAPPA_MAX_COSMOLOGIES x ~47 (every usable shell) -- kappa is by far the most
# expensive section and runs LAST, so if the job runs out of time the Cl plots are
# already written.
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-30}
KAPPA_MAX_COSMOLOGIES=${KAPPA_MAX_COSMOLOGIES:-30}
N_ZBINS=${N_ZBINS:-3}
N_SHELLS_PER_ZBIN=${N_SHELLS_PER_ZBIN:-5}

# SAME path unet/run_flow.sh uses -- see the header note above.
PATCH_DIR="/capstor/scratch/cscs/damrein/outputs/flowpatches/${DATA_TAG}_nside${NSIDE}_${PATCH_SIZE}_${NPATCH}"
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/diffusionruns/${RUN_NAME}"
mkdir -p "$PATCH_DIR" "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/diffusion

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
ulimit -c 0

echo "==== diffusion | job ${SLURM_JOB_ID} | nside=${NSIDE} patch=${PATCH_SIZE} n=${NPATCH} steps=${STEPS} use_cosmo_cond=${USE_COSMO_COND} ===="

# ---- stage 0: nside-512 low/high shell stacks (skip runs already prepared) ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside ${NSIDE} --num-workers 5
"

# ---- stage 1: build the (low, high) patch dataset (reused from unet's if present) ----
if [ ! -f "${PATCH_DIR}/metadata.npy" ]; then
  srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${VENV}/bin/activate
    python ${DIFFUSION}/make_patch_dataset.py \
      --data-dir     '${DATA_ROOT}' \
      --metainfo-dir '${METAINFO_DIR}' \
      --prepared-dir '${DATA_ROOT}' \
      --out-dir      '${PATCH_DIR}' \
      --nside ${NSIDE} --patch-size ${PATCH_SIZE} \
      --n-patches ${NPATCH} --seed 0 \
      --num-workers ${SLURM_CPUS_PER_TASK}
  "
else
  echo '[stage1] patch dataset already exists (possibly built by unet/run_flow.sh), skipping'
fi

# ---- stage 2: DDP train the denoiser, GENUINE multi-node (${SLURM_NNODES} nodes x
# 4 GPU) -- see the header warning above. c10d rendezvous: one torchrun launcher
# task per node (--ntasks-per-node=1, NOT one task per GPU -- torchrun itself spawns
# the 4 per-GPU worker processes on each node), all pointed at node 0's address. ----
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=29500
echo "[stage2] multi-node rendezvous: ${SLURM_NNODES} nodes, master=${MASTER_ADDR}:${MASTER_PORT}"

srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 --gres=gpu:4 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  cd ${DIFFUSION}
  python -m torch.distributed.run \
    --nnodes=${SLURM_NNODES} --nproc_per_node=4 \
    --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train_diffusion.py \
    --patch-dir '${PATCH_DIR}' \
    --out-dir   '${OUT_DIR}' \
    --epochs ${EPOCHS} --batch-size ${BATCH} --base-channels ${BASE_CH} --lr ${LEARNING_RATE} \
    --p-mean ${P_MEAN} --p-std ${P_STD} ${SIGMA_DATA_FLAG} \
    --hp-cutoff ${HP_CUTOFF} --hp-transition ${HP_TRANSITION} --space ${SPACE} \
    --num-workers $((SLURM_CPUS_PER_TASK / 4)) ${COSMO_FLAG} ${NO_LR_SCALING_FLAG}
"

# ---- stage 3: loss/val plot + apply on held-out test patches + full-sky Cl (glue) ----
# MULTI-NODE EVAL (2026-07-21, matching diffusion/run_diagnostics_only.sh's
# identical pattern): apply_diffusion.py splits its two dominant-cost sections
# (zbin-grid, kappa) across ALL of GPUS_PER_NODE x SLURM_NNODES ranks via genuine
# multi-node torchrun c10d rendezvous -- SLURM_NNODES reflects whatever --nodes
# this job's header (or an override at submission) actually requested. This eval
# job's own distributed calls are a couple of dist.all_gather_object per
# diagnostic stage -- a handful of collectives, not thousands of sustained
# gradient-allreduce steps like stage 2 -- so it's a materially lighter cross-node
# workload. GPUS_PER_NODE=4 is this cluster's fixed per-node GPU count.
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-1}
EVAL_GPUS=$(( GPUS_PER_NODE * NNODES ))
srun --ntasks=${NNODES} --ntasks-per-node=1 --gres=gpu:${GPUS_PER_NODE} uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${DIFFUSION}/plot_diffusion_loss.py --run-dir '${OUT_DIR}'
  python -m torch.distributed.run \
    --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${DIFFUSION}/apply_diffusion.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${OUT_DIR}/eval' \
    --steps ${STEPS} --sigma-min ${SIGMA_MIN} --sigma-max ${SIGMA_MAX} --rho ${RHO} \
    --data-root '${DATA_ROOT}' \
    --shell-indices --example-shells 5 10 15 30 50 \
    --n-zbins ${N_ZBINS} --n-shells-per-zbin ${N_SHELLS_PER_ZBIN} \
    --max-cosmologies ${MAX_COSMOLOGIES} --kappa-max-cosmologies ${KAPPA_MAX_COSMOLOGIES} \
    --fullsky-patch-size ${PATCH_SIZE} ${KAPPA_FLAG}
"
echo "diffusion job ${SLURM_JOB_ID} finished at $(date) -> ${OUT_DIR}/eval"
