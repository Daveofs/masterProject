#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths
PARAM_FILE="${PARAM_FILE:-${REPO_ROOT}/cosmology.par}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/pkdgrav_local}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "ERROR: param file not found: ${PARAM_FILE}" >&2
  exit 1
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import numpy, healpy" >/dev/null 2>&1; then
  echo "ERROR: required packages missing (numpy, healpy) in: ${PYTHON_BIN}" >&2
  echo "Install with: pip install numpy healpy" >&2
  exit 1
fi

cd "${RUN_DIR}"
echo "Run dir    : ${RUN_DIR}"
echo "Param file : ${PARAM_FILE}"
echo "Python     : ${PYTHON_BIN}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/shell_collector.py" --param_file "${PARAM_FILE}"
