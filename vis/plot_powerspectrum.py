#!/usr/bin/env python3
"""Plot power spectrum .pk files.

Reads files where column 0 is k and column 1 is P(k).
Multiple inputs are overlaid on the same plot with a legend.

Usage:
  python plot_powerspectrum.py file.pk
  python plot_powerspectrum.py file1.pk file2.pk file3.pk
  python plot_powerspectrum.py dir_with_pk_files/
  python plot_powerspectrum.py file.pk dir/ -o my_output/ --name comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np


def find_pk_files(path: Path):
    if path.is_dir():
        return sorted(path.glob("*.pk"))
    if path.is_file() and path.suffix == ".pk":
        return [path]
    return sorted(Path('.').glob(str(path)))


def load_k_p(path: Path):
    try:
        data = np.loadtxt(path)
    except Exception:
        data = np.genfromtxt(path, comments="#")
    if data.ndim == 1 and data.size >= 2:
        k, p = data[0], data[1]
    else:
        if data.shape[1] < 2:
            raise ValueError(f"File {path} has fewer than 2 columns")
        k, p = data[:, 0], data[:, 1]
    mask = np.isfinite(k) & np.isfinite(p) & (k > 0) & (p > 0)
    return k[mask], p[mask]


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Plot .pk power spectrum files (multi-input overlay)")
    parser.add_argument("inputs", nargs='*',
                        default=["/cluster/scratch/damrein/outputs/ICs/000001_copy7/CosmoML.00080.pk", "/cluster/scratch/damrein/outputs/powerspectrum/final_snapshot_tipsy.pk"],
                        help=".pk files or directories (can mix multiple)")
    parser.add_argument("-o", "--outdir", default="/cluster/scratch/damrein/outputs/plots/powerspectrum",
                        help="output directory")
    parser.add_argument("--name", default=None,
                        help="output filename stem (default: auto from input names)")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # collect all files from all inputs
    all_files = []
    for inp in args.inputs:
        found = find_pk_files(Path(inp))
        if not found:
            print(f"Warning: no .pk files found for '{inp}'", file=sys.stderr)
        all_files.extend(found)

    if not all_files:
        print("No .pk files found.", file=sys.stderr)
        return 2

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for f in all_files:
        try:
            k, pwr = load_k_p(f)
        except Exception as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)
            continue
        ax.loglog(k, pwr, linewidth=1, label=f.name)
        plotted += 1

    if plotted == 0:
        print("No files could be plotted.", file=sys.stderr)
        return 1

    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    if plotted > 1:
        ax.legend(fontsize="small", loc="best")

    # determine output filename
    if args.name:
        stem = args.name
    elif len(all_files) == 1:
        stem = all_files[0].stem
    else:
        stem = "powerspectrum_comparison"

    outpath = outdir / (stem + ".png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Wrote {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
