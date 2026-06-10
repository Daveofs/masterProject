#!/bin/bash
# For all cosmo_*/run_* directories:
#   1. Unzip param_files.tar.gz if cosmology.par is missing
#   2. Patch cosmology.par (achTfFile, bClass, cosmological params, bWriteIC, bParaWrite, nSteps4)
#   3. Patch baryonification_params.py (transfct path)
# Patching is idempotent: each change is only applied if still needed.

COSMOGRID_DIR="/capstor/scratch/cscs/damrein/cosmogridv1"

# Extract a Python variable value from baryonification_params.py
# Usage: extract_py_param "filename" "varname"
extract_py_param() {
    local file="$1"
    local var="$2"
    grep -E "^\s*${var}\s*=" "$file" | head -1 | sed "s/.*=\s*//" | tr -d ' "'
}

patch_cosmology_par() {
    local par_file="$1"
    local abs_class="$2"
    local out_name="$3"
    local bary_file="$4"
    local changed=0

    # FIX: Use regex to safely overwrite the entire achOutName line regardless of what's inside
    if grep -qE '^achOutName\s*=' "$par_file"; then
        sed -i "s|^achOutName\s*=.*|achOutName = \"${out_name}\"|" "$par_file"
        changed=1
    fi

    # Replace achClassFilename with achTfFile pointing to transferfunction.dat
    if grep -q 'achClassFilename' "$par_file"; then
        sed -i 's|^achClassFilename.*|achTfFile = "transferfunction.dat"|' "$par_file"
        changed=1
    fi

    # Set bClass = 0 (False)
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

    # --- Remove unwanted parameters ---
    for remove_param in nGridLin achLinSpecies achPkSpecies; do
        if grep -qE "^${remove_param}\s*=" "$par_file"; then
            sed -i "/^${remove_param}\s*=/d" "$par_file"
            changed=1
        fi
    done

    # --- Add cosmological parameters from baryonification_params.py ---
    if [ -f "$bary_file" ]; then
        # FIX: Adjusted variable names to match 'h0' and 's8' used in the Python file
        local h_val omega_m_val omega_b_val ns_val sigma8_val

        h_val=$(extract_py_param "$bary_file" "par.cosmo.h0")
        omega_m_val=$(extract_py_param "$bary_file" "par.cosmo.Om")
        omega_b_val=$(extract_py_param "$bary_file" "par.cosmo.Ob")
        ns_val=$(extract_py_param "$bary_file" "par.cosmo.ns")
        sigma8_val=$(extract_py_param "$bary_file" "par.cosmo.s8")

        # Only patch if we got values and they aren't already in the file
        if [ -n "$h_val" ] && ! grep -qE '^\s*h\s*=' "$par_file"; then
            cat >> "$par_file" <<EOF

# Cosmological parameters (from baryonification_params.py)
h                = ${h_val}
dOmega0          = ${omega_m_val}
dLambda          = $(awk "BEGIN {printf \"%.10g\", 1 - ${omega_m_val}}")
dOmegab          = ${omega_b_val}
dSpectral        = ${ns_val}
dSigma8          = ${sigma8_val}
EOF
            changed=1
        fi
    fi

    # Comment out nSteps4 if uncommented
    if grep -qE '^nSteps4' "$par_file"; then
        sed -i 's|^\(nSteps4\)|#\1|' "$par_file"
        changed=1
    fi

    # Enable IC writing (handles potential variation in spaces/tabs)
    if grep -qE 'bWriteIC\s*=\s*0' "$par_file"; then
        sed -i 's|bWriteIC.*|bWriteIC         = 1|' "$par_file"
        changed=1
    fi

    # Enable parallel writing (handles potential variation in spaces/tabs)
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

for cosmo_dir in "$COSMOGRID_DIR"/cosmo_*/; do
    for run_dir in "$cosmo_dir"run_*/; do
        tarball="${run_dir}param_files.tar.gz"
        par_file="${run_dir}cosmology.par"
        bary_file="${run_dir}baryonification_params.py"
        abs_class="${run_dir}class_processed.hdf5"

        # Derive cosmo and run IDs from directory names for achOutName
        cosmo_id=$(basename "$cosmo_dir")   # e.g. cosmo_000244
        run_id=$(basename "$run_dir")       # e.g. run_1
        out_name="CosmoML_${cosmo_id#cosmo_}_${run_id}"  # e.g. CosmoML_000244_run_1

        if [ ! -f "$tarball" ]; then
            continue
        fi

        # Step 1: unzip if not yet extracted
        if [ ! -f "$par_file" ]; then
            tar -xzf "$tarball" -C "$run_dir"
            echo "Unzipped: $run_dir"
        fi

        # Step 2: patch cosmology.par
        if [ -f "$par_file" ]; then
            patch_cosmology_par "$par_file" "$abs_class" "$out_name" "$bary_file"
        fi

        # Step 3: patch baryonification_params.py
        if [ -f "$bary_file" ]; then
            patch_baryonification_params "$bary_file" "$abs_class"
        fi
    done
done

echo "Done."
