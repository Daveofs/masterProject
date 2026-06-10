#!/bin/bash
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --job-name=flow-matching
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/flow_matching/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/flow_matching/slurm-%j.err
#SBATCH --chdir=/capstor/scratch/cscs/damrein/outputs/logs/flow_matching

# ============================================================
# Environment setup
# ============================================================
source /users/damrein/miniforge3/bin/activate

# ============================================================
# Multi-node / multi-GPU config
# ============================================================
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

GPUS_PER_NODE=4
NNODES=${SLURM_NNODES}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "========================================"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES:      ${NNODES}"
echo "WORLD_SIZE:  ${WORLD_SIZE}"
echo "========================================"

# Optional: improve NCCL performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="/users/damrein/masterProject/ml"
DATA_DIR="/capstor/scratch/cscs/damrein/cosmogrid"
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/flow_matching/${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"
mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/flow_matching

# ============================================================
# Launch: one torchrun per node via srun
# ============================================================
srun bash -c "\
    torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --rdzv_id=${SLURM_JOB_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${SCRIPT_DIR}/train_flow_matching.py \
        --data-dir ${DATA_DIR} \
        --low-npz disco_shells.npz \
        --high-npz compressed_shells.npz \
        --max-shells 10 \
        --batch-size 8 \
        --epochs 5 \
        --lr 1e-3 \
        --sigma 0.01 \
        --hidden 512 \
        --num-workers 4 \
        --out-dir ${OUT_DIR} \
        --log-interval 10
"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
