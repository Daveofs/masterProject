#!/bin/bash
#SBATCH --job-name=build_pkdgrav
#SBATCH --partition=normal.4h
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/damrein/outputs/logs/build_pkdgrav_%j.out
#SBATCH --error=/cluster/scratch/damrein/outputs/logs/build_pkdgrav_%j.err

set -euo pipefail

PKDGRAV_SRC=/cluster/scratch/damrein/pkdgrav_latest/pkdgrav3

# Deactivate any active conda environment so its include/library paths do not
# shadow the pkdgrav3 bundled libraries (e.g. fmt) during compilation.
CONDA_ROOT=/cluster/scratch/damrein/miniconda3
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
fi
# Also clear any compiler search-path variables conda may have exported
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH

# Load required modules (same stack used during cmake configuration)
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load fftw/3.3.10
module load hdf5/1.14.3
module load boost/1.83.0

echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting cmake build (jobs=${SLURM_CPUS_PER_TASK})"
cd "${PKDGRAV_SRC}"

cmake --build build -- -j"${SLURM_CPUS_PER_TASK}"
rc=$?

echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] cmake build finished; exit=${rc}"
exit ${rc}
