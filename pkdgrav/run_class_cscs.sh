#!/bin/bash
#SBATCH --job-name=class
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/class/class_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/class/class_%j.err

CLASS_DIR=/users/damrein/class/class_public
INI_FILE=/users/damrein/class_exact.ini

# Deactivate conda so its libraries don't interfere
CONDA_ROOT=/users/damrein/miniforge3
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

mkdir -p "$(dirname /capstor/scratch/cscs/damrein/outputs/logs/class/x)"

cd "${CLASS_DIR}"

start_time=$(date +%s)
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting CLASS (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
./class "${INI_FILE}"
rc=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished CLASS; exit=${rc}; elapsed=${elapsed}s"
exit ${rc}
