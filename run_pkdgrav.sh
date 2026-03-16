#!/bin/bash
#SBATCH --job-name=pkdgrav
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/pkdgrav_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/pkdgrav_%j.err

SCRATCH_DIR=/cluster/scratch/damrein
OUTPUT_DIR=${SCRATCH_DIR}/outputs/ICs/000001_copy8
PKDGRAV_BIN=${SCRATCH_DIR}/pkdgrav/pkdgrav3_dev-master/build/pkdgrav3
PARAM_FILE=${SCRATCH_DIR}/cosmogridv1/cosmo_000001/param_files/cosmology.par

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}" # PKDGRAV writes output to the current directory, so cd there first

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

NTASKS="${SLURM_NTASKS:-1}"
# If you want one MPI rank per node (ppr:1:node), set MPI_RANKS to the number of allocated nodes.
# Fall back to NTASKS for interactive/testing when SLURM_JOB_NUM_NODES is not set.
MPI_RANKS="${SLURM_JOB_NUM_NODES:-${NTASKS}}"

start_time=$(date +%s)
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting mpirun (mpirank=${MPI_RANKS}, ntasks=${NTASKS}, cpus/task=${SLURM_CPUS_PER_TASK})"
# Map policy: one MPI rank per node, advertise per-rank CPU allocation (PE) so hybrid MPI+OpenMP works.
# This requests ppr:1:node and assigns PE equal to SLURM_CPUS_PER_TASK (fallback 1).
mpirun -np "${MPI_RANKS}" --map-by "ppr:1:node:PE=${SLURM_CPUS_PER_TASK:-1}" --bind-to core --report-bindings "${PKDGRAV_BIN}" "${PARAM_FILE}"
rc=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished mpirun; exit=${rc}; elapsed=${elapsed}s"
exit ${rc}
