#!/bin/bash
#SBATCH --job-name=copy_to_scratch
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/cluster/home/damrein/project/outputs/copy_to_scratch_%j.out
#SBATCH --error=/cluster/home/damrein/project/outputs/copy_to_scratch_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch run_copy_to_scratch.sh SRC_PATH [SRC_PATH ...]" >&2
  exit 2
fi

PROJECT_DIR=/cluster/home/damrein/project
DEST_BASE=/cluster/scratch/damrein
mkdir -p "${PROJECT_DIR}/outputs" "${DEST_BASE}"

RSYNC_OPTS=( -aHAX --partial --inplace --progress --delete )

echo "[$(date --iso-8601=seconds)] Copy job starting on $(hostname)"
echo "Sources: $*"
echo "Destination base: ${DEST_BASE}"

for src in "$@"; do
  if [[ ! -e "${src}" ]]; then
    echo "Source not found: ${src}, skipping." >&2
    continue
  fi
  base=$(basename "${src}")
  dest=${DEST_BASE}/${base}
  mkdir -p "${dest}"

  echo "--- Syncing ${src} -> ${dest} ---"
  rsync "${RSYNC_OPTS[@]}" "${src%/}/" "${dest%/}/"

  echo "Source size:"; du -sh "${src}" || true
  echo "Dest size:"; du -sh "${dest}" || true
  echo "--- Done ${base} ---"
done

echo "[$(date --iso-8601=seconds)] Copy job finished."
