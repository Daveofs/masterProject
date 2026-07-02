#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sphere-flow
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# DeepSphere-conv FLOW MATCHING (generative small-scale correction), multi-GPU.
#
# PyTorch DDP via torchrun inside the CSCS pytorch uenv (native GPU + NCCL over
# Slingshot). A venv on top of the uenv provides healpy/scipy/pyyaml.
# One process per GPU; shells are streamed + sharded per rank.
#
# One-time venv setup (already done once; recreate if missing):
#   export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
#   uenv run pytorch/v2.9.1:v2 --view=default -- bash -c '
#     python -m venv --system-site-packages /capstor/scratch/cscs/damrein/venvs/sphereflow
#     source /capstor/scratch/cscs/damrein/venvs/sphereflow/bin/activate
#     pip install healpy scipy pyyaml'
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow

DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1_test2"
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/sphereflow/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

# ---- rendezvous + Slingshot NCCL (same as the existing flow-matching runs) ----
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
export NCCL_CROSS_NIC=1
export PYTHONUNBUFFERED=1

echo "==== job ${SLURM_JOB_ID} on $(hostname) | ${SLURM_NNODES} node(s) x ${GPUS_PER_NODE} GPU ===="

# Use 'python -m torch.distributed.run' (NOT the uenv 'torchrun' binary) so DDP
# workers spawn with the VENV python (sys.executable) and can import healpy. The
# uenv torchrun would launch the uenv python, which lacks healpy.
srun uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python -m torch.distributed.run \
           --nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} \
           --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    train_sphere_flow.py \
      --data-root   '${DATA_ROOT}' \
      --test-cosmo  cosmo_000001 \
      --low-name    'shells_nside=2048.npz' \
      --high-name   'compressed_shells.npz' \
      --nside       2048 \
      --order       8 \
      --include-test \
      --hidden      64 \
      --n-layers    6 \
      --K           5 \
      --epochs      5 \
      --batch-size  32 \
      --lr          2e-4 \
      --log-every   20 \
      --out-dir     '${OUT_DIR}'
"

echo "sphere-flow job ${SLURM_JOB_ID} finished at $(date)"
