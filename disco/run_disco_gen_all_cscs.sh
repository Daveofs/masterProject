#!/bin/bash
# ============================================================
# Run DISCO-DJ for ALL cosmo_*/run_* directories.
#
# Each array task picks up:
#   - IC file  : cosmo_*/run_*/CosmoML_XXXXXX_run_Y.00000
#   - params   : cosmo_*/run_*/params.yml
# Output (shells, final snapshot) goes into the same run dir,
# prefixed disco_XXXXXX_run_Y.
#
# Usage — run on the LOGIN NODE:
#
#   1. Build the job list:
#        bash run_disco_gen_all_cscs.sh --build-list
#
#   2. Submit the SLURM array:
#        N=$(( $(wc -l < /users/damrein/masterProject/disco/job_list_disco_gen.txt) - 1 ))
#        sbatch --array=0-${N} run_disco_gen_all_cscs.sh
# ============================================================

#SBATCH --job-name=disco_gen_all
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/disco/disco_gen_%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/disco/disco_gen_%A_%a.err

# ---- Paths -------------------------------------------------------
COSMOGRID_DIR="/capstor/scratch/cscs/damrein/cosmogridv1"
PROJECT_DIR="/users/damrein/masterProject/disco"
JOB_LIST="${PROJECT_DIR}/job_list_disco_gen.txt"
CONDA_ENV="disco-dj"
CONDA_INIT="$HOME/miniforge3/etc/profile.d/conda.sh"
# ------------------------------------------------------------------

# ── Simulation parameters (mirror run_disco_multigpu_cscs.sh) ─────
MODE=gpu
RES=832
RES_PM=832
BOXSIZE=900.0
N_STEPS=20
N_PRESTEPS=100
STEPPER=bullfrog
TIME_VAR=D
METHOD=pm
GRAD_KERNEL_ORDER=4
LAPLACE_KERNEL_ORDER=0
NUM_CHUNKS=1
GPUS_PER_NODE=4
A_INI=0.01
A_END=1.0
BUILD_SHELLS=true
SHELLS_METAINFO="${COSMOGRID_DIR}/CosmoGridV1_metainfo.h5"
# ------------------------------------------------------------------

# ============================================================
# --build-list : scan cosmo dirs and build the job list
# ============================================================
if [[ "${1}" == "--build-list" ]]; then
    echo "Building job list -> ${JOB_LIST}"
    > "${JOB_LIST}"
    skipped=0
    for cosmo_dir in "${COSMOGRID_DIR}"/cosmo_*/; do
        cosmo_id=$(basename "$cosmo_dir")          # cosmo_000001
        cosmo_num="${cosmo_id#cosmo_}"              # 000001
        for run_dir in "${cosmo_dir}"run_*/; do
            run_id=$(basename "$run_dir")           # run_0
            ic_file="${run_dir}CosmoML_${cosmo_num}_${run_id}.00000"
            params_yml="${run_dir}params.yml"

            # Need both IC and params.yml to run
            if [ ! -f "$ic_file" ] || [ ! -f "$params_yml" ]; then
                continue
            fi

            # Skip if DISCO shells already exist in the run dir
            if ls "${run_dir}"*.fits &>/dev/null; then
                (( skipped++ )) || true
                continue
            fi

            echo "${run_dir}" >> "${JOB_LIST}"
        done
    done
    N=$(wc -l < "${JOB_LIST}")
    echo "Job list built: ${N} entries to run, ${skipped} skipped (already done)"
    echo ""
    if [ "${N}" -eq 0 ]; then
        echo "Nothing to submit — all DISCO runs already exist."
    else
        echo "Submit with:"
        echo "  sbatch --array=0-$(( N - 1 )) $(realpath "${BASH_SOURCE[0]}")"
    fi
    exit 0
fi

# ============================================================
# SLURM array task: run DISCO-DJ for one (cosmo, run) entry
# ============================================================

RUN_DIR=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "${JOB_LIST}")

if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "ERROR: no valid run_dir for task ${SLURM_ARRAY_TASK_ID} (got '${RUN_DIR}')"
    exit 1
fi

# Derive IDs from path
cosmo_id=$(basename "$(dirname "${RUN_DIR%/}")")   # cosmo_000001
cosmo_num="${cosmo_id#cosmo_}"                      # 000001
run_id=$(basename "${RUN_DIR%/}")                   # run_0

IC_FILE="${RUN_DIR}CosmoML_${cosmo_num}_${run_id}.00000"
PARAMS_YML="${RUN_DIR}params.yml"

if [ ! -f "${IC_FILE}" ]; then
    echo "ERROR: IC file not found: ${IC_FILE}"; exit 2
fi
if [ ! -f "${PARAMS_YML}" ]; then
    echo "ERROR: params.yml not found: ${PARAMS_YML}"; exit 3
fi

# Shells go directly into the run dir alongside all other files
SHELL_DIR="${RUN_DIR%/}"

# ── Activate conda ────────────────────────────────────────────────
source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"

PYTHON_BIN=$(which python)
export PYTHONPATH=/users/damrein/DISCO-DJ/scripts:${PYTHONPATH:-}

if ! "${PYTHON_BIN}" -c "import discodj" 2>/dev/null; then
    echo "ERROR: discodj not importable in conda env '${CONDA_ENV}'." >&2; exit 4
fi

# ── JAX / XLA / NCCL settings ────────────────────────────────────
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_TRACEBACK_FILTERING=off
export NCCL_NVLS_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_nccl_comm_splitting=true \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true"

# RUN_DIR already exists; nothing to create

PYTHON_ARGS=(
    "${PROJECT_DIR}/sim_discodj_multigpu.py"
    --mode                  "${MODE}"
    --res                   "${RES}"
    --res-pm                "${RES_PM}"
    --boxsize               "${BOXSIZE}"
    --a-ini                 "${A_INI}"
    --a-end                 "${A_END}"
    --n-steps               "${N_STEPS}"
    --stepper               "${STEPPER}"
    --time-var              "${TIME_VAR}"
    --method                "${METHOD}"
    --grad-kernel-order     "${GRAD_KERNEL_ORDER}"
    --laplace-kernel-order  "${LAPLACE_KERNEL_ORDER}"
    --num-chunks            "${NUM_CHUNKS}"
    --gpus-per-node         "${GPUS_PER_NODE}"
    --ic-file               "${IC_FILE}"
    --params-yml            "${PARAMS_YML}"
    --shells-output-dir     "${SHELL_DIR}"
)

if [[ "${BUILD_SHELLS}" == "true" ]]; then
    PYTHON_ARGS+=(--build-shells)
    PYTHON_ARGS+=(--n-presteps "${N_PRESTEPS}")
    if [[ -n "${SHELLS_METAINFO}" && -f "${SHELLS_METAINFO}" ]]; then
        PYTHON_ARGS+=(--shells-metainfo "${SHELLS_METAINFO}")
    fi
fi

cd "${PROJECT_DIR}"
echo "[$(date --iso-8601=seconds)] Starting DISCO-DJ (cosmo=${cosmo_num}, ${run_id})"
echo "  IC file   : ${IC_FILE}"
echo "  params.yml: ${PARAMS_YML}"
echo "  shells dir: ${RUN_DIR} (alongside existing files)"
srun --ntasks=1 --ntasks-per-node=1 \
    "${PYTHON_BIN}" -u "${PYTHON_ARGS[@]}"
rc=$?

echo "[$(date --iso-8601=seconds)] Finished (exit=${rc}, cosmo=${cosmo_num}, ${run_id})"
exit ${rc}
