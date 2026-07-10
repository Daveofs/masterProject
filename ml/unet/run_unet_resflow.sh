#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=unet-resflow
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/unetresflow/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/unetresflow/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Conditional flow matching on the DISCO->CosmoGrid RESIDUAL.
#
# The deterministic diff model is provably capped: its MSE optimum shrinks the
# prediction to corr*target (measured corr ~0.25-0.30, std ratio ~0.37-0.41), which IS
# the small-scale Cl deficit; converged relative loss 0.7348 vs 0.750 for "predict
# zero" => only ~2% of the residual is deterministically predictable. A flow SAMPLE
# instead carries the full residual variance, so small-scale power is right by
# construction (at the cost of not being pixel-exact -- which is impossible anyway).
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"

RUN_NAME=${RUN_NAME:-resflow_v2}
DOWNSCALE=${DOWNSCALE:-1}
ORDER=${ORDER:-16}
EPOCHS=${EPOCHS:-8}
BASE=${BASE:-64}
CH_MULT=${CH_MULT:-1,2,2}
STEPS=${STEPS:-50}          # Euler steps for the ODE sampler at eval time
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/unetresflow/${RUN_NAME}"
TEST_COSMO=cosmo_000122
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/unetresflow

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29503
export GPUS_PER_NODE=4
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_RX_MATCH_MODE=software
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0

echo "==== unet-resflow | job ${SLURM_JOB_ID} | nodes=${SLURM_NNODES} ===="

srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside 2048 --num-workers 5
"

srun uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python -m torch.distributed.run \
           --nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} \
           --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    unet/train_unet_resflow.py \
      --data-root  '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --nside 2048 --order ${ORDER} --downscale ${DOWNSCALE} \
      --base ${BASE} --ch-mult ${CH_MULT} \
      --n-val 3 --epochs ${EPOCHS} --batch-size 128 --patch-frac 0.5 --lr 1e-4 \
      --log-every 50 --val-every 500 --ckpt-every 500 \
      --out-dir '${OUT_DIR}'
"

srun --nodes=1 --ntasks=1 --gres=gpu:1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python unet/apply_unet_resflow.py \
      --model-dir  '${OUT_DIR}' \
      --data-root  '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --shell-indices 3 30 50 \
      --steps ${STEPS} \
      --out-dir '${OUT_DIR}/eval'
"
echo "unet-resflow job ${SLURM_JOB_ID} finished at $(date)"
