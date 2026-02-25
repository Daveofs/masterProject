#!/bin/bash
#SBATCH --job-name=pkdgrav
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/pkdgrav_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/pkdgrav_%j.err

set -euo pipefail

SCRATCH_DIR=/cluster/scratch/damrein
OUTPUT_DIR=${SCRATCH_DIR}/outputs/ICs/000001
PKDGRAV_BIN=${SCRATCH_DIR}/pkdgrav/pkdgrav3_dev-master/build/pkdgrav3
PARAM_FILE=${SCRATCH_DIR}/cosmogridv1/cosmo_000001/param_files/cosmology.par

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}" # PKDGRAV writes output to the current directory, so cd there first

srun "${PKDGRAV_BIN}" "${PARAM_FILE}"