#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=unet-flow
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/unetflow/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/unetflow/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Simple 2D-UNet flow-matching generator (unet_flow.py) on HEALPix patches.
# Same data path as sphere-flow; plain 2D convs instead of graph convs.
#
# DEFAULT: 1 node (4 GPUs, NVLink-only NCCL). The DeepSphere/sphere-flow runs
# proved a deterministic multi-node Slingshot/NCCL collective hang at 4 nodes
# (the last node stops posting a collective ~mid-run, healthy values) -- a fabric
# issue, not a code bug. 1 node avoids it entirely and is plenty for 44 runs.
# The trainer IS full DDP: bump --nodes once the fabric issue is resolved/ticketed.
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"

RUN_NAME=${RUN_NAME:-unet_v1}
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/unetflow/${RUN_NAME}"
TEST_COSMO=cosmo_000122
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/unetflow

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29501
export GPUS_PER_NODE=4
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_RX_MATCH_MODE=software
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0                     # never dump multi-GB core files into (home) cwd

echo "==== unet-flow | job ${SLURM_JOB_ID} | nodes=${SLURM_NNODES} ===="

# ---- stage 0: ensure raw npy dataset exists (shared with sphere-flow) ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside 2048 --num-workers 5
"

# ---- stage 1: DDP training ----
srun uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python -m torch.distributed.run \
           --nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} \
           --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    sphereflow/train_unet_flow.py \
      --data-root  '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --include-test \
      --nside      2048 \
      --order      16 \
      --base       64 \
      --ch-mult    1,2,2 \
      --epochs     8 \
      --batch-size 128 \
      --patch-frac 0.5 \
      --lr         2e-4 \
      --log-every  50 \
      --ckpt-every 500 \
      --out-dir    '${OUT_DIR}'
"

# ---- stage 2: correction + evaluation plots (rank 0 / 1 GPU) ----
srun --nodes=1 --ntasks=1 --gres=gpu:1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python sphereflow/apply_unet_flow.py \
      --model-dir  '${OUT_DIR}' \
      --data-root  '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --shell-indices 3 30 50 \
      --steps      50 \
      --out-dir    '${OUT_DIR}/eval'
"

echo "unet-flow job ${SLURM_JOB_ID} finished at $(date)"
