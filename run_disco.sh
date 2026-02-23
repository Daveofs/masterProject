#!/bin/bash
#SBATCH --job-name=disco_cosmoIC
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --output=outputs/disco_%j.out
#SBATCH --error=outputs/disco_%j.err

# Activate your Conda environment directly
source /cluster/scratch/damrein/miniconda3/etc/profile.d/conda.sh
conda activate vir_env

export JAX_PLATFORM_NAME=cpu

PROJECT_DIR=/cluster/scratch/damrein/project
IC_ARCHIVE=${PROJECT_DIR}/outputs/CosmoML_IC_res512.npz

cd "${PROJECT_DIR}"

# Ensure outputs directories exist before SLURM redirects stdout/stderr
mkdir -p "${PROJECT_DIR}/outputs" "${PROJECT_DIR}/outputs/plots"

# Path for the saved final snapshot (file, not directory)
SAVE_FINAL_FILE=${PROJECT_DIR}/outputs/final_snapshot_${SLURM_JOB_ID:-manual}.npz

if [[ ! -f "${IC_ARCHIVE}" ]]; then
	echo "Missing IC archive: ${IC_ARCHIVE}. Run run_convert_ic.sh first." >&2
	exit 2
fi

echo "[$(date --iso-8601=seconds)] Starting running simulation discodj on $(hostname)"
python run_sim_discodj.py --ic-file "${IC_ARCHIVE}" --plot --save-final "${SAVE_FINAL_FILE}"
echo "[$(date --iso-8601=seconds)] Done."