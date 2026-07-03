#!/bin/bash
#SBATCH --job-name=ic_prep
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/ic_prep/ic_prep_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/ic_prep/ic_prep_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/damrein/cosmogridv1
#
# Submit with:  sbatch run_IC_preparation.sh
#
# All the actual work (deleting extra run_* dirs, unzipping param_files.tar.gz,
# patching cosmology.par and baryonification_params.py, computing sigma8 /
# OmegaRad via CLASS) happens in IC_preparation.py, which runs as a single
# Python process and fans the per-directory work out across many workers.
# Patching stays idempotent: each change is only applied if still needed.

# Under sbatch the script is copied to /var/spool/slurmd, so BASH_SOURCE is
# useless here. Use the submit dir if set, else the known source location.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-/users/damrein/masterProject/pkdgrav}"
PY_SCRIPT="${SCRIPT_DIR}/IC_preparation.py"
[ -f "$PY_SCRIPT" ] || PY_SCRIPT="/users/damrein/masterProject/pkdgrav/IC_preparation.py"

# Activate conda environment
CONDA_ROOT=/users/damrein/miniforge3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate base


mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/ic_prep

# Each CLASS solve is kept single-threaded; we parallelize across directories
# instead. Give the pool as many workers as the cores we were allocated.
export OMP_NUM_THREADS=1
export IC_PREP_WORKERS=$(( ${SLURM_NTASKS_PER_NODE:-4} * ${SLURM_CPUS_PER_TASK:-64} ))

echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting IC_preparation.py (IC_PREP_WORKERS=${IC_PREP_WORKERS})"
start_time=$(date +%s)

python3 "$PY_SCRIPT"
rc=$?

elapsed=$(( $(date +%s) - start_time ))
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Finished (rc=${rc}, elapsed=${elapsed}s)"
exit $rc
