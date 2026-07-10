#!/bin/bash
# ============================================================
# Generate ICs for ALL cosmo_*/run_* directories using pkdgrav.
#
# Kills pkdgrav as soon as "Output file has been successfully
# written" appears in its stdout — no full simulation needed.
# The IC file (achOutName.00000) is written directly into
# cosmogridv1/cosmo_*/run_*/ (pkdgrav cwd = that directory).
#
# Usage — run these two commands on the LOGIN NODE:
#
#   1. Build the job list (only needed once, or after new cosmo dirs):
#        bash run_pkdgrav_gen_all_cscs.sh --build-list
#
#   2. Submit the SLURM array:
#        N=$(( $(wc -l < /users/damrein/masterProject/pkdgrav/job_list_gen_all.txt) - 1 ))
#        sbatch --array=0-${N} run_pkdgrav_gen_all_cscs.sh
# ============================================================

#SBATCH --job-name=pkdgrav_gen_all
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/pkdgrav/pkdgrav_gen_%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/pkdgrav/pkdgrav_gen_%A_%a.err

# ---- Paths -------------------------------------------------------
COSMOGRID_DIR="/capstor/scratch/cscs/damrein/grid"
PKDGRAV_BIN="/users/damrein/pkdgrav/pkdgrav_latest/pkdgrav3/build/pkdgrav3"
JOB_LIST="/users/damrein/masterProject/pkdgrav/job_list_gen_all.txt"
LOG_DIR="/capstor/scratch/cscs/damrein/outputs/logs/pkdgrav"
# ------------------------------------------------------------------

# ============================================================
# Before --build-list we define a mask for parameters
# ============================================================

OMEGA_M_MIN=0.1;  OMEGA_M_MAX=0.5
SIGMA_8_MIN=0.6;  SIGMA_8_MAX=1.0
W0_MIN=-1.4;       W0_MAX=-0.6
H0_MIN=62.0;       H0_MAX=78.0
OMEGA_B_MIN=0.046; OMEGA_B_MAX=0.051
N_S_MIN=0.94;      N_S_MAX=0.98

# Returns 0 (true) if params.yml values are inside the mask ranges, 1 otherwise.
check_mask() {
    local params_file="$1"
    python3 - "$params_file" \
        "$OMEGA_M_MIN" "$OMEGA_M_MAX" \
        "$SIGMA_8_MIN" "$SIGMA_8_MAX" \
        "$W0_MIN" "$W0_MAX" \
        "$H0_MIN" "$H0_MAX" \
        "$OMEGA_B_MIN" "$OMEGA_B_MAX" \
        "$N_S_MIN" "$N_S_MAX" <<'PYEOF'
import sys
import yaml

params_file = sys.argv[1]
omega_m_min, omega_m_max = float(sys.argv[2]), float(sys.argv[3])
sigma8_min,  sigma8_max  = float(sys.argv[4]), float(sys.argv[5])
w0_min,      w0_max      = float(sys.argv[6]), float(sys.argv[7])
h0_min,      h0_max      = float(sys.argv[8]), float(sys.argv[9])
omega_b_min, omega_b_max = float(sys.argv[10]), float(sys.argv[11])
ns_min,      ns_max      = float(sys.argv[12]), float(sys.argv[13])

try:
    with open(params_file) as f:
        p = yaml.safe_load(f)

    omega_m = float(p["Om"])
    sigma8  = float(p["s8"])
    w0      = float(p["w0"])
    h0      = float(p["H0"])
    omega_b = float(p["Ob"])
    ns      = float(p["ns"])

    # Check all ranges
    if not (omega_m_min <= omega_m <= omega_m_max):
        print(f"  Mask check failed: Om={omega_m} not in [{omega_m_min}, {omega_m_max}]")
        sys.exit(1)
    if not (sigma8_min <= sigma8 <= sigma8_max):
        print(f"  Mask check failed: sigma8={sigma8} not in [{sigma8_min}, {sigma8_max}]")
        sys.exit(1)
    if not (w0_min <= w0 <= w0_max):
        print(f"  Mask check failed: w0={w0} not in [{w0_min}, {w0_max}]")
        sys.exit(1)
    if not (h0_min <= h0 <= h0_max):
        print(f"  Mask check failed: H0={h0} not in [{h0_min}, {h0_max}]")
        sys.exit(1)
    if not (omega_b_min <= omega_b <= omega_b_max):
        print(f"  Mask check failed: Ob={omega_b} not in [{omega_b_min}, {omega_b_max}]")
        sys.exit(1)
    if not (ns_min <= ns <= ns_max):
        print(f"  Mask check failed: ns={ns} not in [{ns_min}, {ns_max}]")
        sys.exit(1)

    # All checks passed
    sys.exit(0)

except Exception as e:
    print(f"  Warning: mask check failed for {params_file}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}


# ============================================================
# --build-list : generate the job list (run on login node only)
# ============================================================
if [[ "${1}" == "--build-list" ]]; then
    echo "Building job list -> ${JOB_LIST}"
    > "${JOB_LIST}"
    skipped=0
    masked=0
    excluded=0
    for cosmo_dir in "${COSMOGRID_DIR}"/cosmo_*/; do
        # Skip cosmologies flagged by IC_preparation.py: CLASS could not compute
        # sigma8/omega_rad for these (omega_b outside the default BBN YHe table), so
        # their cosmology.par never got the correct cosmological-parameter block --
        # generating ICs for them would use wrong/stale cosmology. The flag travels
        # with the cosmo_* directory regardless of which grid it lives under.
        if [ -f "${cosmo_dir}EXCLUDED_BBN_OMEGAB.flag" ]; then
            (( excluded++ )) || true
            continue
        fi
        for run_dir in "${cosmo_dir}"run_*/; do

            echo "Checking ${run_dir} ..."
            cos_file="${run_dir}cosmology.par"
            par_file="${run_dir}params.yml"
            if [ ! -f "$cos_file" ] || [ ! -f "$par_file" ]; then
                continue
            fi
  
            if ! check_mask "$par_file"; then
                (( masked++ )) || true
                continue
            fi

            # Skip if IC file already exists
            ach_out=$(grep '^achOutName' "$cos_file" | sed 's/.*= *"\(.*\)"/\1/')
            if [ -n "$ach_out" ] && [ -f "${run_dir}${ach_out}.00000" ]; then
                (( skipped++ )) || true
                continue
            fi
            echo "${run_dir}" >> "${JOB_LIST}"
        done
    done
    N=$(wc -l < "${JOB_LIST}")
    echo "Job list built: ${N} entries to run, ${skipped} skipped (IC already exists)"
    echo "  ${masked} entries skipped due to mask"
    echo "  ${excluded} cosmologies skipped (EXCLUDED_BBN_OMEGAB.flag)"
    echo ""
    if [ "${N}" -eq 0 ]; then
        echo "Nothing to submit."
    else
        echo "Submit with:"
        echo "  sbatch --array=0-$(( N - 1 )) $(realpath "${BASH_SOURCE[0]}")"
    fi
    exit 0
fi

# ============================================================
# SLURM array task: generate ICs for one (cosmo, run) entry
# ============================================================

# Read the run directory for this task (0-indexed)
RUN_DIR=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "${JOB_LIST}")
PARAM_FILE="${RUN_DIR}cosmology.par"

if [ -z "${RUN_DIR}" ] || [ ! -f "${PARAM_FILE}" ]; then
    echo "ERROR: no valid run_dir for task ${SLURM_ARRAY_TASK_ID} (got '${RUN_DIR}')"
    exit 1
fi

# Deactivate conda so its OpenMPI doesn't override the system one
CONDA_ROOT=/users/damrein/miniforge3
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda deactivate 2>/dev/null || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
MPI_RANKS="${SLURM_NTASKS:-1}"

# IC file lands in the same dir as cosmology.par (pkdgrav cwd = RUN_DIR)
cd "${RUN_DIR}"

# Check if IC already exists by reading achOutName from the param file
ACH_OUT_NAME=$(grep '^achOutName' "${PARAM_FILE}" | sed 's/.*= *"\(.*\)"/\1/')
IC_FILE="${RUN_DIR}${ACH_OUT_NAME}.00000"
if [ -f "${IC_FILE}" ]; then
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] SKIP: IC already exists at ${IC_FILE}"
    exit 0
fi

# Per-task live log
mkdir -p "${LOG_DIR}"
LIVE_LOG="${LOG_DIR}/pkdgrav_gen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] Starting IC generation"
echo "  run_dir    : ${RUN_DIR}"
echo "  param_file : ${PARAM_FILE}"
echo "  mpi_ranks  : ${MPI_RANKS}"
echo "  live_log   : ${LIVE_LOG}"

# Launch pkdgrav in the background, capturing all output to the live log
srun --mpi=pmix -n "${MPI_RANKS}" --cpu_bind=cores \
    "${PKDGRAV_BIN}" "${PARAM_FILE}" > "${LIVE_LOG}" 2>&1 &
SRUN_PID=$!

# Monitor: once the IC is written, cancel this SLURM job step via scancel.
# scancel is the most reliable way to stop all MPI ranks cleanly in SLURM.
while kill -0 "${SRUN_PID}" 2>/dev/null; do
    if grep -q "Output file has been successfully written" "${LIVE_LOG}" 2>/dev/null; then
        echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] IC written — cancelling job step via scancel"
        scancel "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 2>/dev/null || \
            scancel "${SLURM_JOB_ID}" 2>/dev/null
        # Exit immediately — scancel sends SIGTERM to this script too,
        # but racing it with an explicit exit ensures we don't block on wait.
        echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] SUCCESS: ICs generated for ${RUN_DIR}"
        exit 0
    fi
    sleep 2
done

# srun exited on its own — check if IC was written
wait "${SRUN_PID}" 2>/dev/null
if grep -q "Output file has been successfully written" "${LIVE_LOG}" 2>/dev/null; then
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] SUCCESS: ICs generated for ${RUN_DIR}"
    exit 0
else
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] WARNING: pkdgrav exited without writing ICs for ${RUN_DIR}"
    exit 1
fi
