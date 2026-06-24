#!/bin/bash
#SBATCH --job-name=get_tf
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/get_tf/get_tf_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/get_tf/get_tf_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/damrein/outputs/class

SCRIPT=/users/damrein/masterProject/pkdgrav/get_transfer_function.py

# Activate miniforge3 base environment
CONDA_ROOT=/users/damrein/miniforge3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MPLBACKEND=Agg

mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/get_tf
PARAM_DIR=/capstor/scratch/cscs/damrein/cosmogridv1_test3/cosmo_000001/run_0

start_time=$(date +%s)
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting get_transfer_function.py (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
python "${SCRIPT}" --param_dir "${PARAM_DIR}"
rc=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished; exit=${rc}; elapsed=${elapsed}s"
exit ${rc}
