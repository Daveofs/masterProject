#!/bin/bash
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --job-name=flow-loo-patch
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/flow_matching/slurm-patch-pipeline-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/flow_matching/slurm-patch-pipeline-%j.err
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Slingshot / OFI transport — NCCL was falling back to TCP Socket (10-20x slower).
# First check which module provides the OFI plugin:  module avail 2>&1 | grep -i nccl
# Then load it above the activate line, e.g.:  module load aws-ofi-nccl
#                                          or:  module load cray-mpich cray-nccl
export NCCL_IB_DISABLE=1              # skip InfiniBand probing (not present on Alps)
export NCCL_SOCKET_IFNAME=hsn         # TCP fallback uses Slingshot HSN, not mgmt NIC
export FI_CXI_ATS=0                   # required for HPE Cray Slingshot CXI fabric
export FI_PROVIDER=cxi                # force CXI (Slingshot) libfabric provider
export FI_MR_CACHE_MONITOR=memhooks  # avoids CXI MR cache stalls under fork

export LD_LIBRARY_PATH=/opt/cray/pe/lib64:/opt/cray/libfabric/lib64:$LD_LIBRARY_PATH

export NCCL_DEBUG=WARN                # was INFO — suppresses the per-rank startup noise
export NCCL_CROSS_NIC=1

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
    --epochs 1 \
    --lr 5e-4 \
    --chunk-size 2000000 \
    --sigma 0.01 \
    --hidden 512  \
    --ode-steps 25 \
    --plot-nside 2048 \
    --device cuda:0 \
    --log-interval 10 


echo "Harmonic Pipeline ${SLURM_JOB_ID} finished at $(date)"

# Cleanup the shared temp directory
rm -rf "$SHARED_TMP"