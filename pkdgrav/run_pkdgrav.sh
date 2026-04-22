#!/bin/bash
#SBATCH --job-name=pkdgrav
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/pkdgrav_%A_%a.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/pkdgrav_%A_%a.err

SCRATCH_DIR=/cluster/scratch/damrein
PKDGRAV_BIN=/cluster/home/damrein/pkdgrav/pkdgrav3_dev-master/build/pkdgrav3

# Load the same module stack used to build pkdgrav3 so the binary is run
# against the correct OpenMPI 4.1.6 runtime (not conda's OpenMPI 5.x).
# Deactivate conda first so its LD_LIBRARY_PATH/PATH don't override the modules.
CONDA_ROOT=${SCRATCH_DIR}/miniconda3
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
fi
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load fftw/3.3.10
module load hdf5/1.14.3

# Map array task ID to zero-padded 6-digit cosmology index (1 -> 000001, 2 -> 000002, ...)
COSMO_ID=$(printf '%06d' "${SLURM_ARRAY_TASK_ID}")
OUTPUT_DIR=${SCRATCH_DIR}/outputs/ICs/cosmo_${COSMO_ID}
PARAM_FILE=/cluster/work/refregier/damrein/cosmogridv1/cosmo_000001/param_files/cosmology.par

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}" # PKDGRAV writes output to the current directory, so cd there first

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# With --ntasks-per-node=1, SLURM_NTASKS == SLURM_JOB_NUM_NODES, so one MPI rank per node.
MPI_RANKS="${SLURM_NTASKS:-1}"

start_time=$(date +%s)
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting mpirun (mpirank=${MPI_RANKS}, cpus/task=${SLURM_CPUS_PER_TASK})"
# Map policy: one MPI rank per node; PE= tells OpenMPI how many CPU threads each rank will use
# so hwloc can set up correct NUMA-aware binding without crashing on certain node topologies.
mpirun -np "${MPI_RANKS}" --map-by "ppr:1:node:PE=${SLURM_CPUS_PER_TASK:-1}" --bind-to core "${PKDGRAV_BIN}" "${PARAM_FILE}"
rc=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished mpirun; exit=${rc}; elapsed=${elapsed}s"
exit ${rc}
