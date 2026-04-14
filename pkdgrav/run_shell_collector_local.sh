#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths
PARAM_FILE="${PARAM_FILE:-/Users/david/projects/cosmogridv1/param_files/cosmology.par}"
RUN_DIR="${RUN_DIR:-/Users/david/projects/outputs/pkdgrav_local}"
VENV_DIR="${VENV_DIR:-/Users/david/projects/vir_env}"

# ── Activate virtual environment ──────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "ERROR: param file not found: ${PARAM_FILE}" >&2
  exit 1
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

cd "${RUN_DIR}"
echo "Run dir    : ${RUN_DIR}"
echo "Param file : ${PARAM_FILE}"
exec "${SCRIPT_DIR}/shell_collector.py" --param_file "${PARAM_FILE}"
