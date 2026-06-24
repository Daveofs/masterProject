#!/bin/bash
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --job-name=flow-loo-harmonic
#SBATCH --partition=debug
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/flow_matching/slurm-harmonic-pipeline-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/flow_matching/slurm-harmonic-pipeline-%j.err
#SBATCH --chdir=/capstor/scratch/cscs/damrein/outputs/flow_matching

# ============================================================
# Environment setup
# ============================================================
source /users/damrein/miniforge3/bin/activate

# ============================================================
# Multi-node / multi-GPU config
# ============================================================
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export GPUS_PER_NODE=4
export NNODES=${SLURM_NNODES}
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "========================================"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES:      ${NNODES}"
echo "WORLD_SIZE:  ${WORLD_SIZE}"
echo "========================================"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="/users/damrein/masterProject"
DATA_DIR="/capstor/scratch/cscs/damrein/cosmogridv1_test2"
OUT_ROOT="/capstor/scratch/cscs/damrein/outputs/flow_matching/${SLURM_JOB_ID}"
SHARED_TMP="/capstor/scratch/cscs/damrein/outputs/tmp/${SLURM_JOB_ID}"

mkdir -p "$OUT_ROOT"
mkdir -p "$SHARED_TMP"
mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/flow_matching

# ============================================================
# Launch Harmonic Pipeline Wrapper
# ============================================================
# Executes run_pipeline.py on the head node; triggers srun + torchrun 
# for the DDP Alm training phase.

python ${SCRIPT_DIR}/ml/run_pipeline.py \
    --data-root ${DATA_DIR} \
    --test-cosmo cosmo_000001 \
    --out-root ${OUT_ROOT} \
    --train-script train_flow_matching.py \
    --apply-script apply_flow_correction.py \
    --shared-tmp ${SHARED_TMP} \
    --srun-torchrun \
    --max-shells 1000 \
    --batch-size 1 \
    --epochs 2 \
    --lr 2e-4 \
    --sigma 0.01 \
    --hidden 64 \
    --lmax 3000 \
    --ode-steps 25 \
    --plot-nside 2048 \
    --device cuda:0 


echo "Harmonic Pipeline ${SLURM_JOB_ID} finished at $(date)"

# Cleanup the shared temp directory
rm -rf "$SHARED_TMP"