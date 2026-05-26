#!/bin/bash
#SBATCH --job-name=pk_snapshot_cmp
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/pk_snapshot_comparison_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/pk_snapshot_comparison_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH_DIR=/capstor/scratch/cscs/damrein
CONDA_ROOT=/users/damrein/miniforge3
CONDA_ENV=disco-dj

SNAPSHOT_A=/capstor/scratch/cscs/damrein/outputs/snapshots/final_multigpu_3536615.npz
SNAPSHOT_B=/capstor/scratch/cscs/damrein/cosmogridv1_fiducial/run_0000/standard/CosmoML.00140
SNAPSHOT_C=/capstor/scratch/cscs/damrein/outputs/snapshots/final_multigpu_3536615.npz
OUT_DIR=${SCRATCH_DIR}/outputs/pk_snapshot_comparison

PY_SCRIPT=/users/damrein/masterProject/tools/plot_pk_from_npz.py

LBOX=900
NGRID=512
THREADS=8

LABEL_A="DISCO-DJ snapshot - backscaling with Omega_nu and hdf5 ICs"
LABEL_B="PKDGRAV Tipsy - standard fiducial"
LABEL_C="DISCO-DJ snapshot - same"
TITLE="Snapshot Power Spectrum Comparison"
OUTPUT_NAME="pk_snapshot_backscaling_omega_nu_hdf5.png"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
mkdir -p "${SCRATCH_DIR}/outputs/logs"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
for f in "${SNAPSHOT_A}" "${SNAPSHOT_B}" "${SNAPSHOT_C}" "${PY_SCRIPT}"; do
	if [[ ! -f "${f}" ]]; then
		echo "[ERROR] File not found: ${f}" >&2
		exit 2
	fi
done

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting snapshot P(k) comparison"
echo "  Snapshot A: ${SNAPSHOT_A}"
echo "  Snapshot B: ${SNAPSHOT_B}"
echo "  Snapshot C: ${SNAPSHOT_C}"
echo "  Output dir: ${OUT_DIR}"
echo "  Lbox:       ${LBOX}"
echo "  Ngrid:      ${NGRID}"
echo "  Threads:    ${THREADS}"

python "${PY_SCRIPT}" \
	--snapshot-a "${SNAPSHOT_A}" \
	--snapshot-b "${SNAPSHOT_B}" \
	--snapshot-c "${SNAPSHOT_C}" \
	--out-dir    "${OUT_DIR}" \
	--lbox       "${LBOX}" \
	--ngrid      "${NGRID}" \
	--threads    "${THREADS}" \
	--label-a    "${LABEL_A}" \
	--label-b    "${LABEL_B}" \
	--label-c    "${LABEL_C}" \
	--title      "${TITLE}" \
	--output-name "${OUTPUT_NAME}"

echo "[$(date --iso-8601=seconds)] Done. Outputs written to ${OUT_DIR}"


