#!/bin/bash
#SBATCH --job-name=pkdgrav
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=250
#SBATCH --array=1
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/pkdgrav/pkdgrav_%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/pkdgrav/pkdgrav_%A_%a.err

# Run index within each cosmology (run_0 … run_6); change as needed.
RUN_ID=0

SCRATCH_DIR=/capstor/scratch/cscs/damrein
PKDGRAV_BIN=/users/damrein/pkdgrav/pkdgrav3_dev-master/build/pkdgrav3

# Load the same module stack used to build pkdgrav3 so the binary is run
# against the correct OpenMPI 4.1.6 runtime (not conda's OpenMPI 5.x).
# Deactivate conda first so its LD_LIBRARY_PATH/PATH don't override the modules.
CONDA_ROOT=/users/damrein/miniforge3
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
fi

#unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH
#module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
#module load fftw/3.3.10
#module load hdf5/1.14.3

# Map array task ID to zero-padded 6-digit cosmology index (1 -> 000001, 2 -> 000002, ...)
COSMO_ID=$(printf '%06d' "${SLURM_ARRAY_TASK_ID}")
OUTPUT_DIR=${SCRATCH_DIR}/outputs/ICs/cosmo_${COSMO_ID}/run_${RUN_ID}
PARAM_FILE=${SCRATCH_DIR}/cosmogridv1/cosmo_${COSMO_ID}/run_${RUN_ID}/cosmology.par

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}" # PKDGRAV writes output to the current directory, so cd there first

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# With --ntasks-per-node=4 and --nodes=4, SLURM_NTASKS == 16 (4 MPI ranks per node).
MPI_RANKS="${SLURM_NTASKS:-1}"

start_time=$(date +%s)
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting srun (cosmo=${COSMO_ID}, run=${RUN_ID}, mpiranks=${MPI_RANKS}, cpus/task=${SLURM_CPUS_PER_TASK:-1})"
# Use --mpi=pmi2 because pkdgrav3 is built against PMI1/2, not PMIx.
#srun --mpi=pmi2 -n "${MPI_RANKS}" --cpu_bind=cores "${PKDGRAV_BIN}" "${PARAM_FILE}"
srun "${PKDGRAV_BIN}" "${PARAM_FILE}"
rc=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished srun; exit=${rc}; elapsed=${elapsed}s"
exit ${rc}
