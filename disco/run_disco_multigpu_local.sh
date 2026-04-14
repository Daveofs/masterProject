#!/bin/bash
# ============================================================
# Multi-GPU DISCO-DJ N-body simulation — local macOS runner
#
# Runs the multi-GPU Python script on CPU (no SLURM, no GPUs).
# JAX distributed init is skipped (no SLURM env vars present).
# Run with bash run_disco_multigpu_local.sh from the disco/ directory.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR="${SCRIPT_DIR}"
VENV_DIR="${REPO_ROOT}/vir_env"

# Input IC file (only used when USE_INTERNAL_ICS=false)
IC_FILE="/Users/david/testData/cosmo_000010/CosmoML.00000"

# Use internal ngenic-like ICs instead of an external tipsy file
USE_INTERNAL_ICS=true
NGENIC_SEED=180723

# Output paths
LOG_DIR="${REPO_ROOT}/outputs/logs"
SNAP_DIR="${REPO_ROOT}/outputs/snapshots"
PLOT_DIR="${REPO_ROOT}/outputs/plots/multigpu"

# ── Simulation parameters ─────────────────────────────────────────────────
MODE=cpu            # cpu (no GPU available on local machine)
RES=64              # keep small for a local test run (N_part = RES^3)
RES_PM=64           # PM force grid resolution
BOXSIZE=450.0       # box size [Mpc/h]
COSMO=Planck15      # DISCO-DJ cosmology preset
A_INI=0.2222        # initial scale factor (z=3.5 → a=1/4.5≈0.2222)
A_END=1.0           # final scale factor (z=0)
N_STEPS=10         # if meta_info is given n_steps = 70 is used
STEPPER=bullfrog    # bullfrog | fastpm | symplectic
TIME_VAR=D          # time variable: a | lna | D
METHOD=pm           # pm | nufftpm
GRAD_KERNEL_ORDER=4
LAPLACE_KERNEL_ORDER=0
NUM_CHUNKS=8        # chunk_size = RES^3 / NUM_CHUNKS
LIGHTCONE=false        # differentiable built-in LC (needs --n-lightcone-replicas, expensive)
BUILD_SHELLS=false
SHELLS_NSIDE=512
SHELLS_Z_MAX=3.5
# Path to CosmoGridV1_metainfo.h5 for NPZ output (set to "" to use FITS mode)
SHELLS_METAINFO="${REPO_ROOT}/CosmoGridV1_metainfo.h5"

# ── JAX / XLA settings ────────────────────────────────────────────────────
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_TRACEBACK_FILTERING=off

# ── Sanity checks ─────────────────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Virtual env not found: ${VENV_DIR}" >&2; exit 1
fi

if [[ "${USE_INTERNAL_ICS}" != "true" ]]; then
    if [[ ! -f "${IC_FILE}" ]]; then
        echo "IC file not found: ${IC_FILE}. Set USE_INTERNAL_ICS=true or update IC_FILE." >&2
        exit 2
    fi
fi

# ── Activate virtual environment ──────────────────────────────────────────
unset CONDA_DEFAULT_ENV CONDA_PREFIX VIRTUAL_ENV PYTHONHOME PYTHONPATH 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found in venv: ${PYTHON_BIN}" >&2; exit 3
fi

# ── Create output directories ─────────────────────────────────────────────
mkdir -p "${SNAP_DIR}" "${PLOT_DIR}"

_TS="$(date +%Y%m%d_%H%M%S)"
SAVE_FINAL="${SNAP_DIR}/final_multigpu_local_${_TS}.npz"
SAVE_LIGHTCONE="${SNAP_DIR}/lightcone_S_${_TS}.npz"

# ── Build Python args ──────────────────────────────────────────────────────
PYTHON_ARGS=(
    "${PROJECT_DIR}/sim_discodj_multigpu.py"
    --mode        "${MODE}"
    --res         "${RES}"
    --res-pm      "${RES_PM}"
    --boxsize     "${BOXSIZE}"
    --cosmo       "${COSMO}"
    --a-ini       "${A_INI}"
    --a-end       "${A_END}"
    --n-steps     "${N_STEPS}"
    --stepper     "${STEPPER}"
    --time-var    "${TIME_VAR}"
    --method      "${METHOD}"
    --grad-kernel-order    "${GRAD_KERNEL_ORDER}"
    --laplace-kernel-order "${LAPLACE_KERNEL_ORDER}"
    --num-chunks  "${NUM_CHUNKS}"
    --save-final      "${SAVE_FINAL}"
    --save-lightcone  "${SAVE_LIGHTCONE}"
    --output-dir      "${PLOT_DIR}"
    --plot
)

if [[ "${USE_INTERNAL_ICS}" == "true" ]]; then
    PYTHON_ARGS+=(--use-internal-ics)
    PYTHON_ARGS+=(--ngenic-seed "${NGENIC_SEED}")
else
    PYTHON_ARGS+=(--ic-file "${IC_FILE}")
fi

if [[ "${LIGHTCONE}" == "true" ]]; then
    PYTHON_ARGS+=(--lightcone)
    PYTHON_ARGS+=(--z-lc-ini "${A_INI}")
fi

if [[ "${BUILD_SHELLS}" == "true" ]]; then
    PYTHON_ARGS+=(--build-shells)
    PYTHON_ARGS+=(--shells-nside   "${SHELLS_NSIDE}")
    PYTHON_ARGS+=(--shells-z-max   "${SHELLS_Z_MAX}")
    PYTHON_ARGS+=(--shells-output-dir "${REPO_ROOT}/outputs/shells")
    if [[ -n "${SHELLS_METAINFO}" && -f "${SHELLS_METAINFO}" ]]; then
        PYTHON_ARGS+=(--shells-metainfo "${SHELLS_METAINFO}")
    fi
fi

cd "${PROJECT_DIR}"
echo "[$(date -Iseconds)] Starting DISCO-DJ multi-GPU (CPU) local run on $(hostname)"

"${PYTHON_BIN}" -u "${PYTHON_ARGS[@]}"

echo "[$(date -Iseconds)] Done. Snapshot at: ${SAVE_FINAL}"
[[ "${LIGHTCONE}" == "true" ]] && echo "  Lightcone S at: ${SAVE_LIGHTCONE}"
