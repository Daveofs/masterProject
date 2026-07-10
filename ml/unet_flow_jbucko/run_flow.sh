#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=flow-jbucko
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/flowjbucko/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/flowjbucko/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Pipeline that EMBEDS jbucko's conditional flow-matching files (leaving them
# untouched in unet_flow_jbucko/):  flow_model.py, dataset.py, make_patch_dataset.py,
# train_flow.py. Glue only lives in flow_pipeline/ (plot + apply).
#
#   stage 0  prepare nside-512 low/high shell stacks (our preprocess/prepare_maps.py)
#   stage 1  build flat gnomonic (low,high) patch dataset  (his make_patch_dataset.py)
#   stage 2  DDP train the low->high flow                  (his train_flow.py, 1 node/4 GPU)
#   stage 3  plot train/val loss + apply on HELD-OUT patches (flow_pipeline glue)
#
# Why the flow (not our deterministic diff model): a deterministic MSE regressor's
# optimum is the conditional mean, which shrinks the correction to corr*target
# (measured corr~0.25-0.30) -> small-scale power deficit. A flow SAMPLE carries the
# full high-field variance, so small-scale power is right by construction.
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"
JBUCKO=/users/damrein/masterProject/ml/unet_flow_jbucko
PIPE=/users/damrein/masterProject/ml/unet_flow_jbucko

NSIDE=${NSIDE:-2048}
PATCH_SIZE=${PATCH_SIZE:-256}
NPATCH=${NPATCH:-20000}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-32}
BASE_CH=${BASE_CH:-32}
STEPS=${STEPS:-8}
RUN_NAME=${RUN_NAME:-flow_v1}

PATCH_DIR="/capstor/scratch/cscs/damrein/outputs/flowpatches/nside${NSIDE}_${PATCH_SIZE}_${NPATCH}"
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/flowruns/${RUN_NAME}"
mkdir -p "$PATCH_DIR" "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/flowjbucko

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
ulimit -c 0

echo "==== flow-jbucko | job ${SLURM_JOB_ID} | nside=${NSIDE} patch=${PATCH_SIZE} n=${NPATCH} ===="

# ---- stage 0: nside-512 low/high shell stacks (skip runs already prepared) ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside ${NSIDE} --num-workers 5
"

# ---- stage 1: build the (low, high) patch dataset (his make_patch_dataset.py) ----
if [ ! -f "${PATCH_DIR}/low.npy" ]; then
  srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${VENV}/bin/activate
    python ${JBUCKO}/make_patch_dataset.py \
      --data-dir     '${DATA_ROOT}' \
      --prepared-dir '${DATA_ROOT}' \
      --out-dir      '${PATCH_DIR}' \
      --nside ${NSIDE} --patch-size ${PATCH_SIZE} \
      --n-patches ${NPATCH} --seed 0 \
      --num-workers ${SLURM_CPUS_PER_TASK}
  "
else
  echo '[stage1] patch dataset already exists, skipping'
fi

# ---- stage 2: DDP train the low->high flow (his train_flow.py, 1 node / 4 GPU) ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  cd ${JBUCKO}
  python -m torch.distributed.run --standalone --nproc_per_node=4 train_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --out-dir   '${OUT_DIR}' \
    --epochs ${EPOCHS} --batch-size ${BATCH} --base-channels ${BASE_CH} \
    --num-workers $((SLURM_CPUS_PER_TASK / 4))
"

# ---- stage 3: loss/val plot + apply on held-out test patches (glue) ----
srun --nodes=1 --ntasks=1 --gres=gpu:1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${PIPE}/plot_flow_loss.py --run-dir '${OUT_DIR}'
  python ${PIPE}/apply_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${OUT_DIR}/eval' \
    --steps ${STEPS}
"
echo "flow-jbucko job ${SLURM_JOB_ID} finished at $(date)"
