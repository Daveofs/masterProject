#!/bin/bash
#SBATCH --job-name=pkdgrav_gpu
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # one MPI rank per GPU
#SBATCH --gpus-per-node=4           # 4 GH200 GPUs per node
#SBATCH --cpus-per-task=16          # CPU threads available to each rank
#SBATCH --array=1
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/pkdgrav/pkdgrav_gpu_%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/pkdgrav/pkdgrav_gpu_%A_%a.err

# Run index within each cosmology (run_0 … run_6); change as needed.
RUN_ID=0

SCRATCH_DIR=/capstor/scratch/cscs/damrein
PKDGRAV_BIN=/users/damrein/pkdgrav/pkdgrav_gpu/build/pkdgrav3
CONDA_ROOT=/users/damrein/miniforge3
CONDA_ENV=pkdgrav

# Activate the pkdgrav conda env to pull in runtime libs (OpenMPI, FFTW, HDF5, CUDA)
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

# Expose CUDA runtime and toolkit libraries installed via conda
CUDA_TARGETS=${CONDA_ROOT}/envs/${CONDA_ENV}/targets/sbsa-linux
export LD_LIBRARY_PATH="${CUDA_TARGETS}/lib:${CONDA_ROOT}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Map array task ID to zero-padded 6-digit cosmology index (1 -> 000001, 2 -> 000002, ...)
COSMO_ID=$(printf '%06d' "${SLURM_ARRAY_TASK_ID}")
OUTPUT_DIR=${SCRATCH_DIR}/outputs/ICs/cosmo_gpu_${COSMO_ID}/run_${RUN_ID}
PARAM_FILE=${SCRATCH_DIR}/cosmogridv1/cosmo_${COSMO_ID}/run_${RUN_ID}/cosmology.par

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"  # pkdgrav3 writes output relative to cwd

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

MPI_RANKS="${SLURM_NTASKS:-4}"

start_time=$(date +%s)
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting srun GPU (cosmo=${COSMO_ID}, run=${RUN_ID}, mpiranks=${MPI_RANKS}, gpus/node=${SLURM_GPUS_PER_NODE:-4})"

# Each rank gets one GPU; CUDA_VISIBLE_DEVICES is set per-rank by SLURM via --gpus-per-task
srun --mpi=pmix -n "${MPI_RANKS}" --gpus-per-task=1 --cpu_bind=cores "${PKDGRAV_BIN}" "${PARAM_FILE}"

rc=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished srun; exit=${rc}; elapsed=${elapsed}s"
exit ${rc}
