#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=sphere-flow
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# SINGLE-MODEL sphere-flow (v3, formulation=direct): DeepSphere-conv flow
# matching conditioned on the raw DISCO map, generating the high-res signal.
#
# v3 fixes vs the failed v2 run (3825944):
#   * direct formulation (no global resid_scale — the per-shell heteroscedastic
#     residual scale was miscalibrating faint vs dense shells)
#   * shell-index conditioning restored (model knows faint vs dense regime)
#   * batch 512 -> 128: best measured throughput AND ~5x more optimizer steps
#     (v2 made only 1,035 updates — undertrained)
#   * background prefetch thread (CPU shell prep overlaps GPU compute)
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow

DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/sphereflow/${SLURM_JOB_ID}"
TEST_COSMO=cosmo_000122
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_ATS=0
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=memhooks
export LD_LIBRARY_PATH=/opt/cray/pe/lib64:/opt/cray/libfabric/lib64:$LD_LIBRARY_PATH
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== sphere-flow v3 (single model, direct) | job ${SLURM_JOB_ID} ===="

# ---- stage 0: ensure raw npy dataset exists (decompress DISCO+high npz; CPU) ----
#      map-only, no transfer function / SHT. Skips runs already prepared.
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python prepare_maps.py --data-dir '${DATA_ROOT}' --nside 2048 --num-workers 5
"

# ---- stage 1: DDP training (4 nodes x 4 GPUs) ----
srun uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python -m torch.distributed.run \
           --nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} \
           --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train_sphere_flow.py \
      --data-root   '${DATA_ROOT}' \
      --test-cosmo  ${TEST_COSMO} \
      --include-test \
      --formulation direct \
      --nside       2048 \
      --order       16 \
      --hidden      64 \
      --n-layers    6 \
      --K           5 \
      --epochs      8 \
      --batch-size  128 \
      --patch-frac  0.5 \
      --lr          2e-4 \
      --log-every   50 \
      --out-dir     '${OUT_DIR}'
"

# ---- stage 2: evaluate flow vs DISCO baseline vs CosmoGrid ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python apply_sphere_flow.py \
      --model-dir '${OUT_DIR}' \
      --data-root '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --shell-indices 3 30 50 --steps 50 --lmax 3000
"

echo "sphere-flow v3 job ${SLURM_JOB_ID} finished at $(date)"
