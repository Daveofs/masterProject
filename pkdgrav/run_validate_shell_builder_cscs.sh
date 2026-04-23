#!/bin/bash
#SBATCH --job-name=validate_shells
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/vis/validate_shells_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/vis/validate_shells_%j.err

# Validate LightconeShellBuilder against pkdgrav3's own built-in lightcone.
#
# Prerequisites: run_pkdgrav_validation_cscs.sh must have completed, producing:
#   ${SNAP_DIR}/CosmoML_val.NNNNN   (141 Tipsy snapshots)
#   ${SNAP_DIR}/CosmoML_val.log     (pkdgrav3 step log)
#   ${SNAP_DIR}/CosmoML_val-shell_z-high=*_z-low=*.fits  (reference FITS shells)
#
# Runtime estimate:
#   ~70 lightcone steps × (~90 s GPU/step) ≈ 1.5–2 h
#
# Memory: two 20 GB Tipsy files held simultaneously → ~45 GB RAM peak.
#         (Adjust --mem if needed.)

set -euo pipefail

PROJECT_DIR=/users/damrein/masterProject
SCRATCH=/capstor/scratch/cscs/damrein
CONDA_ENV=disco-dj
CONDA_ROOT=/users/damrein/miniforge3

SNAP_DIR=${SCRATCH}/outputs/pkdgrav_validation
PARAMS_YML=${SCRATCH}/cosmogridv1/cosmo_000001/run_0/params.yml
OUTPUT_DIR=${SCRATCH}/outputs/plots/shell_builder_validation

SNAP_PREFIX=CosmoML_val
NSIDE=512       # Use 512 for speed; change to 2048 for full-resolution check
Z_MAX=3.5
BOXSIZE=900.0

# ── Sanity checks ─────────────────────────────────────────────────────────
for F in "${SNAP_DIR}/${SNAP_PREFIX}.log" "${PARAMS_YML}"; do
    if [[ ! -f "${F}" ]]; then
        echo "Required file not found: ${F}" >&2
        echo "Run run_pkdgrav_validation_cscs.sh first." >&2
        exit 2
    fi
done

n_fits=$(ls "${SNAP_DIR}/${SNAP_PREFIX}"-shell_z-high=*.fits 2>/dev/null | wc -l)
if [[ "${n_fits}" -eq 0 ]]; then
    echo "No FITS reference shells found in ${SNAP_DIR}." >&2
    echo "Ensure shell_collector.py ran successfully as part of run_pkdgrav_validation_cscs.sh." >&2
    exit 3
fi
echo "Found ${n_fits} FITS reference shells."

# ── Activate conda env ────────────────────────────────────────────────────
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

ENV_PREFIX=$(conda env list | awk -v env_name="${CONDA_ENV}" '$1 == env_name {print $NF; exit}')
if [[ -z "${ENV_PREFIX}" ]]; then
    echo "Conda env '${CONDA_ENV}' not found." >&2; exit 5
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"

# ── GPU env vars ──────────────────────────────────────────────────────────
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH}/outputs/logs/vis"

echo "Snap dir    : ${SNAP_DIR}"
echo "Prefix      : ${SNAP_PREFIX}"
echo "FITS dir    : ${SNAP_DIR}"
echo "Params yml  : ${PARAMS_YML}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "nside       : ${NSIDE}"
echo "z_max       : ${Z_MAX}"
echo ""

"${PYTHON_BIN}" "${PROJECT_DIR}/pkdgrav/validate_shell_builder.py" \
    --snap-dir    "${SNAP_DIR}"    \
    --snap-prefix "${SNAP_PREFIX}" \
    --fits-dir    "${SNAP_DIR}"    \
    --params-yml  "${PARAMS_YML}"  \
    --boxsize     "${BOXSIZE}"     \
    --nside       "${NSIDE}"       \
    --output-dir  "${OUTPUT_DIR}"  \
    --z-max       "${Z_MAX}"

echo ""
echo "Validation complete. Results in: ${OUTPUT_DIR}"
