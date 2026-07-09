#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unet-diff
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/unetdiff/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/unetdiff/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Deterministic residual-correction UNet pipeline (unet/unet_diff.py):
#   stage 0  prepare raw npy maps (shared dataset)
#   stage 1  DDP train: predict diff = signal(high) - signal(DISCO); corrected =
#            DISCO + diff; pixel-MSE loss; validation curve on held-out runs
#   stage 2  correct the held-out TEST cosmology, plot Cl + maps + loss curve,
#            and run the SANITY CHECK (fails the job if the correction doesn't help)
#
# 1 node (4 GPUs, NVLink-only NCCL): avoids the confirmed multi-node Slingshot hang.
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"

RUN_NAME=${RUN_NAME:-diff_small}
DOWNSCALE=${DOWNSCALE:-16}       # e.g. DOWNSCALE=8 for a fast dev run (16x16 patches)
ORDER=${ORDER:-16}               # n_patches(order) = 12*order^2; fewer/bigger patches
                                 # per shell at lower order -> ~4x fewer steps/epoch at order=8
LAMBDA_SPEC=${LAMBDA_SPEC:-0.5} # weight of the radial-power-spectrum loss term
HUBER_DELTA=${HUBER_DELTA:-0.1} # robust-loss transition point for the pixel term
EPOCHS=${EPOCHS:-8}
# Model width/depth: keep matched to the patch size actually being trained on. The
# defaults (base=64, ch_mult 1,2,4,8 -> 512-wide bottleneck) suit the full-res 128x128
# patch. At large DOWNSCALE the patch shrinks a lot (e.g. 16x16 at DOWNSCALE=8) and the
# same big model collapses to a ~1x1 bottleneck and overfits almost instantly (observed:
# train loss -> 0, validation loss flat and far worse than doing nothing) -- override
# BASE/CH_MULT down for fast-dev runs, e.g. BASE=16 CH_MULT=1,2,4 at DOWNSCALE=8.
BASE=${BASE:-64}
CH_MULT=${CH_MULT:-1,2,4,8}
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/unetdiff/${RUN_NAME}"
TEST_COSMO=cosmo_000122
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/unetdiff

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29502
export GPUS_PER_NODE=4
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_RX_MATCH_MODE=software
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

echo "==== unet-diff | job ${SLURM_JOB_ID} | nodes=${SLURM_NNODES} ===="

# ---- stage 0: ensure raw npy dataset exists ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside 2048 --num-workers 5
"

# ---- stage 1: DDP training (with validation curve) ----
srun uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python -m torch.distributed.run \
           --nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} \
           --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    unet/train_unet_diff.py \
      --data-root  '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --nside 2048 --order ${ORDER} --downscale ${DOWNSCALE} \
      --base ${BASE} --ch-mult ${CH_MULT} --bottleneck 64 \
      --lambda-spec ${LAMBDA_SPEC} --huber-delta ${HUBER_DELTA} \
      --n-val 3 --epochs ${EPOCHS} --batch-size 128 --patch-frac 0.5 --lr 1e-4 \
      --log-every 50 --val-every 500 --ckpt-every 500 \
      --out-dir '${OUT_DIR}'
"

# ---- stage 2: correction + evaluation plots + sanity check (1 GPU) ----
srun --nodes=1 --ntasks=1 --gres=gpu:1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python unet/apply_unet_diff.py \
      --model-dir  '${OUT_DIR}' \
      --data-root  '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --shell-indices 3 30 50 \
      --out-dir '${OUT_DIR}/eval'
"
echo "sanity/eval exit code: $?"
echo "unet-diff job ${SLURM_JOB_ID} finished at $(date)"
