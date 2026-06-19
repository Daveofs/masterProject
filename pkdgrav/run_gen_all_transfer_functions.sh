#!/bin/bash
#SBATCH --job-name=get_tf
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/get_tf/get_tf_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/get_tf/get_tf_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/damrein/outputs/class

SCRIPT=/users/damrein/masterProject/pkdgrav/get_transfer_function.py
COSMOGRID_DIR=/capstor/scratch/cscs/damrein/cosmogridv1_test4
MAX_PARALLEL="${SLURM_NTASKS:-4}"

# Activate miniforge3 base environment
CONDA_ROOT=/users/damrein/miniforge3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MPLBACKEND=Agg

mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/get_tf

echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting get_transfer_function.py across all cosmo_*/run_*/"
echo "  MAX_PARALLEL=${MAX_PARALLEL}, OMP_NUM_THREADS=${OMP_NUM_THREADS}"

start_time=$(date +%s)

# Collect all run directories
dirs=()
for d in "${COSMOGRID_DIR}"/cosmo_*/run_*/; do
    [ -d "$d" ] && dirs+=("$d")
done

total=${#dirs[@]}
echo "  Found ${total} run directories"

completed=0
failed=0

# Process directories in batches of MAX_PARALLEL
for ((i = 0; i < total; i += MAX_PARALLEL)); do
    pids=()
    batch_dirs=()

    for ((j = i; j < i + MAX_PARALLEL && j < total; j++)); do
        run_dir="${dirs[$j]}"
        batch_dirs+=("$run_dir")
        python "${SCRIPT}" --param_dir "${run_dir}" &
        pids+=($!)
    done

    # Wait for batch and collect exit codes
    for k in "${!pids[@]}"; do
        wait "${pids[$k]}"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  FAILED (rc=${rc}): ${batch_dirs[$k]}"
            ((failed++))
        else
            ((completed++))
        fi
    done

    # Progress update every batch
    done_so_far=$((completed + failed))
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Progress: ${done_so_far}/${total} (${failed} failed)"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "============================================"
echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] All done."
echo "  Total: ${total} | Completed: ${completed} | Failed: ${failed}"
echo "  Elapsed: ${elapsed}s"
echo "============================================"

[ "$failed" -gt 0 ] && exit 1
exit 0
