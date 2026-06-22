#!/bin/bash
# For all cosmo_*/run_* directories:
#   1. Unzip param_files.tar.gz if cosmology.par is missing
#   2. Patch cosmology.par (achTfFile, bClass, cosmological params, bWriteIC, bParaWrite, nSteps4)
#   3. Patch baryonification_params.py (transfct path)
# Patching is idempotent: each change is only applied if still needed.

COSMOGRID_DIR="/capstor/scratch/cscs/damrein/cosmogrid_w0_test"

# Activate conda environment
source /users/damrein/miniforge3/bin/activate

# --- Safety Check: Ensure the directory actually exists ---
if [ ! -d "$COSMOGRID_DIR" ]; then
    echo "Error: Directory $COSMOGRID_DIR does not exist."
    exit 1
fi

# Move into the target directory so run_*/ can be found
cd "$COSMOGRID_DIR" || exit 1

# Now cosmo_dir will correctly be "/capstor/scratch/cscs/damrein/cosmogridv1_fiducial_test1"
cosmo_dir=$(pwd)
cosmo_id=$(basename "$cosmo_dir")   

# Extract a Python variable value from baryonification_params.py
extract_py_param() {
    local file="$1"
    local var="$2"
    grep -E "^\s*${var}\s*=" "$file" | head -1 | sed "s/.*=\s*//" | tr -d ' "'
}

# Read Omega_rad from class_processed.hdf5 using h5py
# Returns the present-day Omega_r(z=0) = rho_g(z=0) / rho_crit(z=0)
extract_omega_rad() {
    local hdf5_file="$1"

    if [ ! -f "$hdf5_file" ]; then
        echo ""
        return
    fi

    python3 - "$hdf5_file" <<'PYEOF'
import sys
import numpy as np
import h5py

class_processed = sys.argv[1]

try:
    with h5py.File(class_processed, "r") as f:
        bg = f['background'] if 'background' in f else f

        z = bg['z'][:]
        rho_g = bg['rho_g'][:]
        rho_crit = bg['rho_crit'][:]

        # Calculate Omega_r(z) across all redshifts
        omega_r_z = rho_g / rho_crit

        # Find the present-day value (where z is closest to 0)
        idx_today = np.argmin(np.abs(z))
        omega_r_0 = omega_r_z[idx_today]

        print(f"{omega_r_0:.10e}")
except Exception as e:
    print("", file=sys.stderr)
    print(f"  Warning: Failed to read OmegaRad from {class_processed}: {e}", file=sys.stderr)
PYEOF
}


# Compute sigma8 from class_processed.hdf5 and params.yml
# Uses the weighted (cdm+b + ncdm) matter power spectrum at z=0
extract_sigma8() {
    local hdf5_file="$1"
    local params_file="$2"

    if [ ! -f "$hdf5_file" ] || [ ! -f "$params_file" ]; then
        echo ""
        return
    fi

    python3 - "$hdf5_file" "$params_file" <<'PYEOF'
import sys
import numpy as np
import h5py
import yaml
from pathlib import Path

hdf5_file = sys.argv[1]
params_file = sys.argv[2]
k_pivot = 0.05

def _tophat_W(x):
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3

def compute_sigma8(k_hMpc, Pk, R=8.0):
    x = k_hMpc * R
    W = _tophat_W(x)
    integrand = Pk * k_hMpc**2 * W**2 / (2 * np.pi**2)
    return np.sqrt(np.trapezoid(integrand, k_hMpc))

try:
    # --- Read cosmological parameters from params.yml ---
    yml_path = Path(params_file)
    with yml_path.open() as f:
        p = yaml.safe_load(f)

    A_s = float(p["As"])
    n_s = float(p["ns"])
    h   = float(p["H0"]) / 100

    # --- Read HDF5 data ---
    with h5py.File(hdf5_file, "r") as f:
        k      = f["perturbations/k"][:]
        a_pert = f["perturbations/a"][:]
        d_cb   = f["perturbations/delta_cdm+b"][:]

    i_a0 = np.argmin(np.abs(a_pert - 1.0))
    delta_cb = d_cb[i_a0, :]

    P_prim     = A_s * (k / k_pivot)**(n_s - 1)
    Pk_hdf5_cb = (2 * np.pi**2 / k**3) * P_prim * delta_cb**2 * h**3

    k_hMpc = k / h
    sigma8 = compute_sigma8(k_hMpc, Pk_hdf5_cb)
    print(f"{sigma8:.10e}")
except Exception as e:
    print("", file=sys.stderr)
    print(f"  Warning: Failed to compute sigma8 from {hdf5_file}: {e}", file=sys.stderr)
PYEOF
}

extract_omega_0() {
    local params_file="$1"

    if [ ! -f "$params_file" ]; then
        echo ""
        return
    fi

    python3 - "$params_file" <<'PYEOF'
import sys
import yaml
from pathlib import Path

yml_path = Path(sys.argv[1])
try:
    with yml_path.open() as f:
        p = yaml.safe_load(f)

    omega_0 = float(p["O_cdm"]) + float(p.get("O_nu", 0.0)) + float(p["Ob"])
    print(f"{omega_0:.12f}")
except Exception as e:
    print("", file=sys.stderr)
    print(f"  Warning: Failed to read Omega0 from {yml_path}: {e}", file=sys.stderr)
PYEOF
}

extract_w_0() {
    local params_file="$1"

    if [ ! -f "$params_file" ]; then
        echo ""
        return
    fi

    python3 - "$params_file" <<'PYEOF'
import sys
import yaml
from pathlib import Path

yml_path = Path(sys.argv[1])
try:
    with yml_path.open() as f:
        p = yaml.safe_load(f)

    w0 = float(p.get("w0"))
    print(f"{w0:.12f}")
except Exception as e:
    print("", file=sys.stderr)
    print(f"  Warning: Failed to read w0 from {yml_path}: {e}", file=sys.stderr)
PYEOF
}

patch_cosmology_par() {
    local par_file="$1"
    local abs_class="$2"
    local out_name="$3"
    local bary_file="$4"
    local params_file="$5"
    local changed=0

    if grep -qE '^achOutName\s*=' "$par_file"; then
        sed -i "s|^achOutName\s*=.*|achOutName = \"${out_name}\"|" "$par_file"
        changed=1
    fi

    if grep -q 'achClassFilename' "$par_file"; then
        sed -i 's|^achClassFilename.*|achTfFile = "transfer_fiducial_cb.dat"|' "$par_file"
        changed=1
    fi

    if grep -qE '^bClass\s*=\s*1' "$par_file"; then
        sed -i 's|^bClass\s*=\s*1|bClass = 0|' "$par_file"
        changed=1
    elif ! grep -qE '^bClass' "$par_file"; then
        if grep -q 'achTfFile' "$par_file"; then
            sed -i '/^achTfFile/a bClass = 0' "$par_file"
        else
            echo "bClass = 0" >> "$par_file"
        fi
        changed=1
    fi

    for remove_param in nGridLin achLinSpecies achPkSpecies; do
        if grep -qE "^${remove_param}\s*=" "$par_file"; then
            sed -i "/^${remove_param}\s*=/d" "$par_file"
            changed=1
        fi
    done

    if [ -f "$bary_file" ] && [ -n "$abs_class" ]; then
        local h_val omega_0 omega_b_val ns_val sigma8_val w0_val

        omega_rad=$(extract_omega_rad "$abs_class")
        h_val=$(extract_py_param "$bary_file" "par.cosmo.h0")
        omega_0=$(extract_omega_0 "$params_file")
        omega_b_val=$(extract_py_param "$bary_file" "par.cosmo.Ob")
        ns_val=$(extract_py_param "$bary_file" "par.cosmo.ns")
        sigma8_val=$(extract_sigma8 "$abs_class" "$params_file")
        w0_val=$(extract_w_0 "$params_file")

        cat >> "$par_file" <<EOF
        
# Cosmological parameters (from baryonification_params.py)
dOmegaRad        = ${omega_rad}
h                = ${h_val}
dOmega0          = ${omega_0}
dLambda          = $(awk "BEGIN {printf \"%.10g\", 1 - ${omega_0}}")
dOmegab          = ${omega_b_val}
dSpectral        = ${ns_val}
dSigma8          = ${sigma8_val}
w0               = ${w0_val}
EOF
            changed=1
    fi

    if grep -qE '^nSteps4' "$par_file"; then
        sed -i 's|^\(nSteps4\)|#\1|' "$par_file"
        changed=1
    fi

    if grep -qE 'bWriteIC\s*=\s*0' "$par_file"; then
        sed -i 's|bWriteIC.*|bWriteIC         = 1|' "$par_file"
        changed=1
    fi

    if grep -qE 'bParaWrite\s*=\s*0' "$par_file"; then
        sed -i 's|bParaWrite.*|bParaWrite       = 1|' "$par_file"
        changed=1
    fi

    [ "$changed" -eq 1 ] && echo "  Patched cosmology.par"
}

patch_baryonification_params() {
    local py_file="$1"
    local abs_class="$2"

    if grep -q 'transfct        = "class_processed.hdf5"' "$py_file"; then
        sed -i "s|transfct        = \"class_processed.hdf5\"|transfct        = \"${abs_class}\"|" "$py_file"
        echo "  Patched baryonification_params.py"
    fi
}

# --- Loop over run directories inside COSMOGRID_DIR ---
for cosmo_dir in "$COSMOGRID_DIR"/cosmo_*/; do
    for run_dir in "$cosmo_dir"/run_*/; do
        # Ensure it's actually a directory
        [ -d "$run_dir" ] || continue

        echo "Processing ${run_dir}..."

        tarball="${run_dir}param_files.tar.gz"
        cos_file="${run_dir}cosmology.par"
        params_file="${run_dir}params.yml"
        bary_file="${run_dir}baryonification_params.py"
        abs_class="${run_dir}class_processed.hdf5"

        run_id=$(basename "$run_dir")      
        cosmo_id=$(basename "$cosmo_dir")
        
        if [[ "$cosmo_id" == cosmo_* ]]; then
            out_name="CosmoML_${cosmo_id#cosmo_}_${run_id}"
        else
            out_name="CosmoML_${cosmo_id}_${run_id}"
        fi

        if [ ! -f "$tarball" ]; then
            echo "  Warning: $tarball not found. Skipping."
            continue
        fi

        # Step 1: unzip if not yet extracted
        if [ ! -f "$cos_file" ]; then
            tar -xzf "$tarball" -C "$run_dir"
            echo "  Unzipped: $run_dir"
        fi

        # Step 2: patch cosmology.par
        if [ -f "$cos_file" ]; then
            patch_cosmology_par "$cos_file" "$abs_class" "$out_name" "$bary_file" "$params_file"
        fi

        # Step 3: patch baryonification_params.py (path adjustements)
        if [ -f "$bary_file" ]; then
            patch_baryonification_params "$bary_file" "$abs_class"
        fi
    done
done

echo "Done."