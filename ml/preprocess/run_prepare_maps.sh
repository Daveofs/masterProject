#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=prep-maps
#SBATCH --partition=normal
#SBATCH --account=a0158
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/prep/prep-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/prep/prep-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml
# Standalone stage 0: degrade the native nside=2048 shells to NSIDE.
#
# Exists so the three pipelines do not each run prepare_maps.py themselves and
# race on the same output files (their internal stage 0 then simply skips,
# because the files already exist). Run this ONCE, then start the models.
#
# NUM_WORKERS is memory-bound, not cpu-bound: each worker holds a full
# 69 x 50M-pixel float32 stack (~13.9 GB) in memory at a time, so 8 workers is
# ~120 GB peak. Raising it much further risks OOM on the node.
set -euo pipefail
DATA=${DATA:-/capstor/scratch/cscs/damrein/grid}
NSIDE=${NSIDE:-512}
NUM_WORKERS=${NUM_WORKERS:-8}
mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/prep

source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere

echo "==== prepare_maps | job ${SLURM_JOB_ID} | nside=${NSIDE} | data=${DATA} ===="
python preprocess/prepare_maps.py --data-dir "$DATA" --nside "$NSIDE" \
       --num-workers "$NUM_WORKERS"
N=$(find "$DATA" -maxdepth 3 -name "low_shells_nside=${NSIDE}.npy" | wc -l)
echo "[done] low_shells_nside=${NSIDE}.npy present for $N runs"
