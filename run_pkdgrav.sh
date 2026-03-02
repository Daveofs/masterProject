#!/bin/bash
#SBATCH --job-name=pkdgrav
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/pkdgrav_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/pkdgrav_%j.err

SCRATCH_DIR=/cluster/scratch/damrein
OUTPUT_DIR=${SCRATCH_DIR}/outputs/ICs/000001_copy11
PKDGRAV_BIN=${SCRATCH_DIR}/pkdgrav/pkdgrav3_dev-master/build/pkdgrav3
PARAM_FILE=${SCRATCH_DIR}/cosmogridv1/cosmo_000001/param_files/cosmology.par

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}" # PKDGRAV writes output to the current directory, so cd there first

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

NTASKS="${SLURM_NTASKS:-1}"

start_time=$(date +%s)
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting mpirun (ntasks=${NTASKS}, cpus/task=${SLURM_CPUS_PER_TASK})"
mpirun -np "${NTASKS}" --map-by node --bind-to none "${PKDGRAV_BIN}" "${PARAM_FILE}"
rc=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished mpirun; exit=${rc}; elapsed=${elapsed}s"
exit ${rc}
