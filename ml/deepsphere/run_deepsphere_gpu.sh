#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=deepsphere-gpu
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/deepsphere/slurm-gpu-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/deepsphere/slurm-gpu-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# DeepSphere correction pipeline on the GH200 GPU, via the NGC TF container.
#
# Runs inside nvidia/tensorflow:24.03-tf2-py3 (TF 2.15, Keras 2, CUDA-enabled for
# aarch64). The conda `deepSphere` env is NOT used here — the container provides
# TensorFlow. healpy + pyyaml are pip-installed into the container at job start
# (NGC TF already ships numpy/scipy/matplotlib).
#
# Prerequisite (one-time): import the image with enroot, e.g.
#   ENROOT_DATA_PATH=/capstor/scratch/cscs/damrein/.enroot-data \
#   enroot import -o /capstor/scratch/cscs/damrein/tf_ngc_2403.sqsh \
#                 docker://nvcr.io#nvidia/tensorflow:24.03-tf2-py3
# The EDF tf_ngc.toml points at that .sqsh.
# ============================================================================

EDF=/users/damrein/masterProject/ml/tf_ngc.toml

DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1_test2"
OUT_ROOT="/capstor/scratch/cscs/damrein/outputs/deepsphere/${SLURM_JOB_ID}"
TEST_COSMO="cosmo_000001"

mkdir -p "$OUT_ROOT" /capstor/scratch/cscs/damrein/outputs/logs/deepsphere

echo "========================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Container:  NGC TF 24.03 (GPU)"
echo "Data root:  ${DATA_ROOT}"
echo "Out root:   ${OUT_ROOT}"
echo "========================================"

# ---- 0. Sanity: confirm TF sees the GPU inside the container ----
srun --environment="${EDF}" bash -c '
  pip install --no-cache-dir healpy pyyaml matplotlib scipy >/dev/null 2>&1
  python -c "import tensorflow as tf; print(\"TF\", tf.__version__, \"| CUDA\", tf.test.is_built_with_cuda(), \"| GPUs\", tf.config.list_physical_devices(\"GPU\"))"
'

# ---- GPU utilization sampler (background, shares the allocation) ----
# Confirms the Chebyshev convs actually run on the H100 (not silently on CPU).
( for i in $(seq 1 20); do
    srun --overlap --jobid="${SLURM_JOB_ID}" nvidia-smi \
       --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
       | sed "s/^/[gpu ${i}] /"
    sleep 30
  done ) &
GPU_SAMPLER=$!

# ---- 1. Run the pipeline on GPU (STREAMING) ----
#   --streaming reads one .npz file pair at a time, so RAM is bounded by a single
#   run-file regardless of total shell count -> scales to 20k-170k shells.
#   nside=2048 + order=8 keeps small scales (768 patches of 256x256 px per shell).
#   --val-files holds out a couple of run-files (in memory) for validation.
srun --environment="${EDF}" bash -c '
  pip install --no-cache-dir healpy pyyaml matplotlib scipy >/dev/null 2>&1
  python -u pipeline_deepSphere.py \
      --data-root   "'"${DATA_ROOT}"'" \
      --test-cosmo  "'"${TEST_COSMO}"'" \
      --out-root    "'"${OUT_ROOT}"'" \
      --low-name    "shells_nside=2048.npz" \
      --high-name   "compressed_shells.npz" \
      --nside       2048 \
      --order       8 \
      --streaming \
      --include-test \
      --residual \
      --max-pairs   4 \
      --val-files   1 \
      --stat-sample-files 1 \
      --max-eval-patches 2048 \
      --n-layers    5 \
      --K           5 \
      --epochs      3 \
      --batch-size  64 \
      --lr          2e-4 \
      --shell-indices 3 65 \
      --lmax        4000
'

kill "${GPU_SAMPLER}" 2>/dev/null
echo "DeepSphere GPU pipeline ${SLURM_JOB_ID} finished at $(date)"
