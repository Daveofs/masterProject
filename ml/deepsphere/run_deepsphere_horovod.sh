#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=deepsphere-hvd
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=4          # one MPI rank per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/deepsphere/slurm-hvd-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/deepsphere/slurm-hvd-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Multi-GPU DeepSphere training with Horovod (data-parallel over shells).
#
# One MPI rank per GPU. Training files are sharded across ranks; gradients are
# all-reduced every step (deepsphere's optimizer is wrapped in
# hvd.DistributedOptimizer). Only rank 0 applies the model + makes Cl plots.
#
# Scale out: raise --nodes (4 GPUs each). Effective batch = batch_per_gpu * ranks,
# and the LR is linearly scaled by the number of ranks.
#
# Runs in the NGC TF 24.03 container (TF2.15 + Horovod + OpenMPI, CUDA aarch64).
# ============================================================================

EDF=/users/damrein/masterProject/ml/tf_ngc.toml

DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1_test2"
OUT_ROOT="/capstor/scratch/cscs/damrein/outputs/deepsphere/${SLURM_JOB_ID}"
TEST_COSMO="cosmo_000001"

mkdir -p "$OUT_ROOT" /capstor/scratch/cscs/damrein/outputs/logs/deepsphere

echo "======================================== job ${SLURM_JOB_ID} on $(hostname)"
echo "nodes=${SLURM_NNODES}  tasks=${SLURM_NTASKS}  data=${DATA_ROOT}"

# ---- Multi-GPU training (rank per GPU). --mpi=pmix so Horovod's MPI works.
#      GPUs are NOT split per task (each task sees all 4); each rank pins to its
#      hvd.local_rank() inside the code.
#      Deps are pip-installed per rank with PLAIN pip (no --target): this respects
#      the container's NumPy 1.x that TF/Horovod were compiled against. Using
#      --target would install NumPy 2.x in an isolated dir and break TF's C-exts.
srun --mpi=pmix --ntasks-per-node=4 --gres=gpu:4 --environment="${EDF}" bash -c "
  sleep \$(( \${SLURM_LOCALID:-0} * 8 ))   # stagger parallel installs (per-rank)
  pip install --no-cache-dir healpy pyyaml matplotlib scipy >/dev/null 2>&1
  python -u pipeline_deepSphere.py \
      --data-root   '${DATA_ROOT}' \
      --test-cosmo  '${TEST_COSMO}' \
      --out-root    '${OUT_ROOT}' \
      --low-name    'shells_nside=2048.npz' \
      --high-name   'compressed_shells.npz' \
      --nside       2048 \
      --order       8 \
      --streaming \
      --horovod \
      --residual \
      --n-layers    5 \
      --K           4 \
      --F-hidden    16 32 32 16 \
      --epochs      3 \
      --batch-size  64 \
      --lr          2e-4 \
      --shell-indices 3 65 \
      --lmax        4000
"

echo "DeepSphere Horovod job ${SLURM_JOB_ID} finished at $(date)"
