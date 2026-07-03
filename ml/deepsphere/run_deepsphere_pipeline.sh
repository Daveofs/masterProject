#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=deepsphere-correct
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/deepsphere/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/deepsphere/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# DeepSphere shell-correction pipeline (data -> train -> apply -> Cl ratio).
#
# Uses the exact deepsphere-cosmo-tf1 graph-CNN (TensorFlow 1.x API via
# tensorflow.compat.v1) configured for map->map small-scale correction.
#
# !!! COMPUTE NOTE — READ ME !!!
# The TensorFlow installed in the `deepSphere` conda env is CPU-ONLY
# (tf.test.is_built_with_cuda() == False). The conda-forge aarch64 builds are
# all `cpu_*`, and there is no pip CUDA wheel for TF on aarch64/Hopper. So this
# job runs the model on the Grace CPU cores (fast: 72-core ARM, high BW memory),
# NOT on the H100. The --gres=gpu:1 above just reserves the node's GPU; it is
# unused by TF unless you switch to a CUDA-enabled TF (see bottom of file).
#
# DDP: deepsphere(cgcnn) is a single-tf.Session TF1 model — no torch DDP / no
# torchrun. Training is single-process. Scale by launching independent jobs over
# test cosmologies (an array job), not by data-parallelism within one fit().
# ============================================================================

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere

# Keras 2 shim for the TF1 compat layers deepsphere uses (tf.layers.*).
export TF_USE_LEGACY_KERAS=1
# Where the deepsphere-cosmo-tf1 checkout lives (MLP.py imports from here).
export DEEPSPHERE_PATH=/users/damrein/deepsphere-cosmo-tf1

# ---- CPU threading: use the Grace cores well ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-64}
export TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=1   # quieten TF info logs
export PYTHONUNBUFFERED=1        # stream stdout to the .out file live (no block buffering)

# ============================================================================
# Paths / config
# ============================================================================
DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1_test2"
OUT_ROOT="/capstor/scratch/cscs/damrein/outputs/deepsphere/${SLURM_JOB_ID}"
TEST_COSMO="cosmo_000001"

mkdir -p "$OUT_ROOT"
mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/deepsphere

echo "========================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Data root:  ${DATA_ROOT}"
echo "Out root:   ${OUT_ROOT}"
echo "Test cosmo: ${TEST_COSMO}"
echo "CPUs/task:  ${SLURM_CPUS_PER_TASK}"
echo "========================================"

# ============================================================================
# Run the pipeline
#   nside=2048 + order=8  -> 768 patches of 256x256 px each (keeps small scales).
#   --max-pairs bounds RAM (every full map is held in memory; ~200 MB each at
#   nside=2048). Raise it as memory allows; 60 maps ~= 12 GB per array.
# ============================================================================
python -u pipeline_deepSphere.py \
    --data-root   "${DATA_ROOT}" \
    --test-cosmo  "${TEST_COSMO}" \
    --out-root    "${OUT_ROOT}" \
    --low-name    "shells_nside=2048.npz" \
    --high-name   "compressed_shells.npz" \
    --nside       2048 \
    --order       8 \
    --max-pairs   60 \
    --n-layers    5 \
    --K           5 \
    --epochs      40 \
    --batch-size  16 \
    --lr          2e-4 \
    --val-frac    0.1 \
    --shell-indices 3 65 \
    --lmax        4000

echo "DeepSphere pipeline ${SLURM_JOB_ID} finished at $(date)"

# ============================================================================
# OPTIONAL — true GPU via NVIDIA NGC TF container (ARM-SBSA / GH200)
# ----------------------------------------------------------------------------
# The conda TF is CPU-only. To use the H100, run inside NVIDIA's ARM TF image
# with enroot (available on Alps). Sketch:
#
#   enroot import docker://nvcr.io#nvidia/tensorflow:24.10-tf2-py3      # ARM SBSA
#   enroot create --name tf_ngc nvidia+tensorflow+24.10-tf2-py3.sqsh
#   srun --gres=gpu:1 enroot start --mount /users/damrein:/users/damrein \
#        --mount /capstor:/capstor tf_ngc \
#        bash -c 'pip install healpy pyyaml && \
#                 TF_USE_LEGACY_KERAS=1 DEEPSPHERE_PATH=/users/damrein/deepsphere-cosmo-tf1 \
#                 python /users/damrein/masterProject/ml/pipeline_deepSphere.py ...'
#
# Note: NGC TF images ship TF2; deepsphere uses tf.compat.v1 which those images
# still provide. Verify tf.test.is_built_with_cuda() is True inside the container.
# ============================================================================
