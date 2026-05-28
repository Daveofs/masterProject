#!/usr/bin/env bash
set -euo pipefail
# Prepend the local project root so Python imports prefer the local
# `discodj_examples` package over any installed package.
export PYTHONPATH="/users/damrein/masterProject/disco:${PYTHONPATH:-}"

# Prefer running the installed `simulation_run` wrapper which performs a
# direct import and calls `cli_entry()` (this avoids runpy's `-m` warnings).
if command -v simulation_run >/dev/null 2>&1; then
	exec simulation_run "$@"
else
	# Fallback to absolute path of the env-installed wrapper
	exec /users/damrein/miniforge3/envs/disco_lorenzo/bin/simulation_run "$@"
fi
