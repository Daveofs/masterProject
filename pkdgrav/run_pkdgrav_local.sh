#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths
PARAM_FILE="${PARAM_FILE:-/Users/david/projects/cosmogridv1/param_files/cosmology.par}"
OUTPUT_DIR="${OUTPUT_DIR:-/Users/david/projects/outputs/pkdgrav_local}"

# Locate pkdgrav binary
PKDGRAV_BIN="${PKDGRAV_BIN:-}"
if [[ -z "${PKDGRAV_BIN}" ]]; then
  for p in \
    "${REPO_ROOT}/pkdgrav3_dev-master/build/pkdgrav3" \
    /opt/homebrew/bin/pkdgrav3 \
    /usr/local/bin/pkdgrav3; do
    if [[ -x "$p" ]]; then
      PKDGRAV_BIN="$p"
      break
    fi
  done
fi
if [[ ! -x "${PKDGRAV_BIN:-}" ]]; then
  echo "ERROR: pkdgrav binary not found. Set PKDGRAV_BIN to the full path." >&2
  exit 1
fi

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "ERROR: param file not found: ${PARAM_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# pkdgrav writes output relative to the working directory
cd "${OUTPUT_DIR}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Param file : ${PARAM_FILE}"
echo "Binary     : ${PKDGRAV_BIN}"
exec "${PKDGRAV_BIN}" "${PARAM_FILE}" "$@"
