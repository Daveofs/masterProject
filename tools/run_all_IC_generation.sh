#!/bin/bash
# For all cosmo_*/run_* directories:
#   1. Unzip param_files.tar.gz if cosmology.par is missing
#   2. Patch cosmology.par (achClassFilename, bWriteIC, bParaWrite, nSteps4)
#   3. Patch baryonification_params.py (transfct path)
# Patching is idempotent: each change is only applied if still needed.

COSMOGRID_DIR="/capstor/scratch/cscs/damrein/cosmogridv1"

patch_cosmology_par() {
    local par_file="$1"
    local abs_class="$2"
    local out_name="$3"
    local changed=0

    # Patch achOutName to include cosmo/run identifiers
    if grep -q 'achOutName = "CosmoML"' "$par_file"; then
        sed -i "s|achOutName = \"CosmoML\"|achOutName = \"${out_name}\"|" "$par_file"
        changed=1
    fi

    # Fix achClassFilename to absolute path (only if still relative)
    if grep -q 'achClassFilename = "class_processed.hdf5"' "$par_file"; then
        sed -i "s|achClassFilename = \"class_processed.hdf5\"|achClassFilename = \"${abs_class}\"|" "$par_file"
        changed=1
    fi

    # Comment out nSteps4 if uncommented
    if grep -qE '^nSteps4' "$par_file"; then
        sed -i 's|^\(nSteps4\)|#\1|' "$par_file"
        changed=1
    fi

    # Enable IC writing
    if grep -q 'bWriteIC         = 0' "$par_file"; then
        sed -i 's|bWriteIC         = 0|bWriteIC         = 1|' "$par_file"
        changed=1
    fi

    # Enable parallel writing
    if grep -q 'bParaWrite       = 0' "$par_file"; then
        sed -i 's|bParaWrite       = 0|bParaWrite       = 1|' "$par_file"
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
            patch_cosmology_par "$par_file" "$abs_class" "$out_name"
        fi

        # Step 3: patch baryonification_params.py
        if [ -f "$bary_file" ]; then
            patch_baryonification_params "$bary_file" "$abs_class"
        fi
    done
done

echo "Done."
