#!/usr/bin/env python3
"""Plot WHERE the validation cosmologies sit in cosmological-parameter space.

Resolves the runnable cosmology pool under --data-root (those with both a DISCO
low input and the CosmoGrid high npz -- same availability rule every pipeline
uses), reads each one's params.yml, and draws analysis.plotting's
plot_cosmo_param_matrix: a corner scatter of the pool (gray) with the HELD-OUT
validation set highlighted + a table of their parameter values.

The held-out set is either:
  * recomputed from the SHARED split convention every pipeline follows
    (sorted cosmology list -> np.random.default_rng(--seed).shuffle -> first
    round(N * --val-frac); transfer's split_val_cosmos, unet's and sphereflow's
    split_by_cosmo all implement exactly this), or
  * pinned explicitly with --cosmos c1 c2 ... (e.g. read from a sphereflow run's
    meta.npz test_cosmos, to plot exactly what THAT model held out).

  python analysis/plot_cosmo_params.py \\
      --data-root /capstor/scratch/cscs/damrein/grid \\
      --out /capstor/scratch/cscs/damrein/outputs/grid_eval/heldout_cosmo_params.png
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.plotting import plot_cosmo_param_matrix  # noqa: E402


def runnable_cosmos(data_root: Path, low_glob: str, high_npz: str) -> list[str]:
    """Sorted cosmologies that have BOTH sim sides -- mirrors
    transfer_function.split_val_cosmos's availability filter (kept as a local
    copy: analysis/ must not import from a pipeline dir)."""
    out = []
    for c in sorted(d for d in data_root.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_")):
        runs = [r for r in sorted(c.iterdir())
                if r.is_dir() and r.name.startswith("run_")] or [c]
        if any(next(r.glob(low_glob), None) is not None and (r / high_npz).exists()
               for r in runs):
            out.append(c.name)
    return out


def read_params(data_root: Path, cosmo: str, keys) -> dict | None:
    c = data_root / cosmo
    runs = [r for r in sorted(c.iterdir())
            if r.is_dir() and r.name.startswith("run_")] or [c]
    for r in runs:
        f = r / "params.yml"
        if f.exists():
            p = yaml.safe_load(f.read_text())
            return {k: float(p[k]) for k in keys}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/capstor/scratch/cscs/damrein/grid")
    ap.add_argument("--low-glob", default="disco_sim/*/disco_shells_nside=2048.npz")
    ap.add_argument("--high-npz", default="compressed_shells.npz")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cosmos", nargs="*", default=None,
                    help="pin the held-out set explicitly instead of recomputing "
                         "the shared --val-frac/--seed split")
    ap.add_argument("--params", nargs="+", default=["Om", "s8", "w0", "H0", "ns", "Ob"],
                    help="params.yml keys, in plot order (default matches the "
                         "CosmoGridV1 corner-plot ordering).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    pool_names = runnable_cosmos(data_root, args.low_glob, args.high_npz)
    if not pool_names:
        raise SystemExit(f"no runnable cosmologies under {data_root}")

    if args.cosmos:
        held_names = sorted(args.cosmos)
        missing = [c for c in held_names if c not in pool_names]
        if missing:
            print(f"[warn] pinned cosmologies not in the runnable pool: {missing}")
    else:
        cosmos = np.array(pool_names)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(cosmos)
        n_val = max(1, int(round(len(cosmos) * args.val_frac)))
        held_names = sorted(cosmos[:n_val].tolist())

    pool, held = {}, {}
    for c in pool_names:
        p = read_params(data_root, c, args.params)
        if p is None:
            continue
        pool[c] = p
        if c in held_names:
            held[c] = p
    print(f"[plot_cosmo_params] pool={len(pool)} cosmologies, "
          f"held-out={len(held)}: {sorted(held)}", flush=True)

    plot_cosmo_param_matrix(
        pool, held, args.out, params=tuple(args.params),
        pool_label=f"pool ({len(pool)} cosmologies, {data_root.name})",
        held_label=f"held-out / validation ({len(held)})",
        suptitle=f"Validation cosmologies in parameter space -- {data_root.name}: "
                 f"{len(held)} held out of {len(pool)} "
                 f"(whole-cosmology split, val_frac={args.val_frac}, seed={args.seed})")


if __name__ == "__main__":
    main()
