#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=rebuild-cl
#SBATCH --partition=normal
#SBATCH --account=a0158
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml
#
# Rebuild transfer's cl_ratio_by_zbin figure + its data npz from the corrected
# shells already on disk in <RUN>/counts/, WITHOUT re-running the correction.
#
# Why this exists: apply_transfer.py (before 2026-08-03) did not persist
# cl_ratio_by_zbin_data.npz, so any plot-only change -- a legend that overflows the
# panel, a relabelled axis -- forced a full ~1.5 h pipeline rerun. The Cl ratio is a
# pure function of (low, high, corrected), and all three are already stored, so it
# can be recomputed in minutes instead. apply_transfer.py now writes the npz itself;
# this script is the retrofit for runs that predate that.
#
# MUST run as a batch job, not on the login node: the login-node watchdog killed an
# interactive attempt after 19 of 30 cosmologies (the shell reported exit 0 because
# the status came from the tail(1) at the end of the pipe, not from python -- so the
# failure looked like a success and left no npz behind).
#
#   RUN=/capstor/scratch/cscs/damrein/outputs/transfer/2986895 sbatch run_rebuild_cl_figure.sh

set -euo pipefail
RUN=${RUN:-/capstor/scratch/cscs/damrein/outputs/transfer/2986895}
NSIDE=${NSIDE:-512}
LMAX=${LMAX:-1500}
GRID=${GRID:-/capstor/scratch/cscs/damrein/grid}
PY=${PY:-/users/damrein/miniforge3/bin/python}

echo "[rebuild] RUN=$RUN NSIDE=$NSIDE LMAX=$LMAX"
RUN="$RUN" NSIDE="$NSIDE" LMAX="$LMAX" GRID="$GRID" $PY -u - <<'PYEOF'
import os, sys, gc
import numpy as np
from pathlib import Path

ML = Path("/users/damrein/masterProject/ml")
sys.path.insert(0, str(ML))
from analysis.full_sky import od_cl, zbin_shell_samples, shell_redshifts as fs_z
from analysis.plotting import plot_cl_ratio_pctile_grid

RUN = Path(os.environ["RUN"]); GRID = Path(os.environ["GRID"])
NSIDE = int(os.environ["NSIDE"]); LMAX = int(os.environ["LMAX"])
INFO = "compressed_shells.npz"
OUT = RUN / "eval"; OUT.mkdir(parents=True, exist_ok=True)

pairs = []
for c in sorted(RUN.glob("counts/cosmo_*_counts.npz")):
    r = GRID / c.name.replace("_counts.npz", "") / "run_0"
    if (r / f"low_shells_nside={NSIDE}.npy").exists():
        pairs.append((c, r))
print(f"[rebuild] {len(pairs)} cosmologies with prepared nside={NSIDE} inputs", flush=True)

n_tot = np.load(pairs[0][1] / f"low_shells_nside={NSIDE}.npy", mmap_mode="r").shape[0] - 1
zbins = zbin_shell_samples(n_tot, 5, 3, 5, shell_z=fs_z(pairs[0][1], INFO))
print(f"[rebuild] zbins {[b[0] for b in zbins]}", flush=True)

ells = np.arange(LMAX + 1)
grid, names = [], []
dump = {"lmax": np.array(LMAX),
        "bin_labels": np.array([b[0] for b in zbins]),
        "bin_shells": np.array([np.asarray(b[1]) for b in zbins], dtype=object)}

for i, (cpath, run) in enumerate(pairs):
    cname = run.parent.name
    low = np.load(run / f"low_shells_nside={NSIDE}.npy", mmap_mode="r")
    high = np.load(run / f"high_shells_nside={NSIDE}.npy", mmap_mode="r")
    # mmap_mode is SILENTLY IGNORED for .npz -- this decompresses the whole
    # (69, npix) stack, so pull out only the shells we need and drop it at once.
    want = sorted({int(s) for _, sh in zbins for s in sh})
    with np.load(cpath) as z:
        corr = {s: np.asarray(z["shells"][s], np.float32) for s in want}
    panels = []
    for lab, shells in zbins:
        lo_s, co_s = [], []
        for s in shells:
            s = int(s)
            hi = od_cl(np.asarray(high[s], np.float32), LMAX)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo_s.append(od_cl(np.asarray(low[s], np.float32), LMAX) / hi)
                co_s.append(od_cl(corr[s], LMAX) / hi)
            dump[f"low_{cname}_s{s}"] = lo_s[-1]
            dump[f"corrected_{cname}_s{s}"] = co_s[-1]
        panels.append((lab, shells, ells, np.array(lo_s), np.array(co_s)))
    grid.append((f"{cname}/{run.name}", panels)); names.append(cname)
    del corr, low, high; gc.collect()
    print(f"  [{i+1}/{len(pairs)}] {cname}", flush=True)

dump["cosmos"] = np.array([f"{n}/run_0" for n in names])
np.savez(OUT / "cl_ratio_by_zbin_data.npz", **dump)
print(f"[rebuild] wrote {OUT/'cl_ratio_by_zbin_data.npz'}", flush=True)

plot_cl_ratio_pctile_grid(grid, OUT / "cl_ratio_by_zbin_grid.png",
                          corrected_label="corrected (transfer (no-clip)) / true (after)")

print("\n[summary] median Cl ratio pooled over cosmologies+shells", flush=True)
for bi, (lab, shells) in enumerate(zbins):
    lo = np.array([dump[f"low_{n}_s{int(s)}"] for n in names for s in shells])
    co = np.array([dump[f"corrected_{n}_s{int(s)}"] for n in names for s in shells])
    print(f"  {lab}")
    for a, b in [(30, 100), (100, 300), (300, 700), (700, 1100), (1100, 1500)]:
        print(f"    ell {a:5d}-{b:5d}:  low {np.nanmedian(lo[:, a:b]):.4f}"
              f"   corrected {np.nanmedian(co[:, a:b]):.4f}", flush=True)
PYEOF
echo "[rebuild] done"
