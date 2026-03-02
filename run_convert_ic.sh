#!/bin/bash
#SBATCH --job-name=convert_ic
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/cluster/home/damrein/project/outputs/convert_ic_%j.out
#SBATCH --error=/cluster/home/damrein/project/outputs/convert_ic_%j.err

set -euo pipefail

PROJECT_DIR=/cluster/home/damrein/project
IC_INPUT=/cluster/home/damrein/pkdgrav/pkdgrav3_dev-master/build/CosmoML.00000
IC_OUTPUT=/cluster/home/damrein/project/outputs/CosmoML_IC.npz
ENV_NAME=vir_env
PYTHON_BIN=/cluster/home/damrein/miniconda3/envs/${ENV_NAME}/bin/python

mkdir -p "${PROJECT_DIR}/outputs"

# shellcheck disable=SC1091
echo "[$(date --iso-8601=seconds)] Starting conversion on $(hostname)"
"${PYTHON_BIN}" -m pip show pynbody || true
"${PYTHON_BIN}" "${PROJECT_DIR}/tools/convert_pkdgrav_ic.py" "${IC_INPUT}" -o "${IC_OUTPUT}"
echo "[$(date --iso-8601=seconds)] Done. Output: ${IC_OUTPUT}"
