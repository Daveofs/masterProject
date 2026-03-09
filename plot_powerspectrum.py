#!/usr/bin/env python3
"""Plot power spectrum .pk files.

Reads files where column 0 is k and column 1 is P(k). Saves plots to
../outputs/plots/powerspectrum relative to this script's directory.

Usage:
  python plot_powerspectrum.py file.pk
  python plot_powerspectrum.py dir_with_pk_files/

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
    # try glob if user passed patterns
    return sorted(Path('.').glob(str(path)))


def load_k_p(path: Path):
    try:
        data = np.loadtxt(path)
    except Exception:
        data = np.genfromtxt(path, comments="#")
    if data.ndim == 1 and data.size >= 2:
        k = data[0]
        p = data[1]
    else:
        if data.shape[1] < 2:
            raise ValueError(f"File {path} has fewer than 2 columns")
        k = data[:, 0]
        p = data[:, 1]
    # filter NaNs and non-positive k or p for log plotting
    mask = np.isfinite(k) & np.isfinite(p) & (k > 0) & (p > 0)
    return k[mask], p[mask]


def ensure_outdir(script_dir: Path) -> Path:
    out = (script_dir.parent / "../damrein/outputs/plots/powerspectrum").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_and_save(k, p, outpath: Path, title: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(k, p, linewidth=1)
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot .pk power spectrum files")
    p.add_argument("input", nargs='?', default="/cluster/scratch/damrein/outputs/ICs/000001_copy6/CosmoML.00027.pk", help=".pk file or directory containing .pk files (optional)")
    p.add_argument("-o", "--outdir", default="/cluster/scratch/damrein/outputs/plots/powerspectrum", help="output directory (overrides default)")
    args = p.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    default_out = ensure_outdir(script_dir)
    outdir = Path(args.outdir).resolve() if args.outdir else default_out
    outdir.mkdir(parents=True, exist_ok=True)

    src = Path(args.input)
    files = find_pk_files(src)
    if not files:
        print(f"No .pk files found for: {src}")
        return 2

    for f in files:
        try:
            k, pwr = load_k_p(f)
        except Exception as e:
            print(f"Skipping {f}: failed to load ({e})", file=sys.stderr)
            continue
        outname = f.stem + ".png"
        outpath = outdir / outname
        title = f.name
        try:
            plot_and_save(k, pwr, outpath, title=title)
            print(f"Wrote {outpath}")
        except Exception as e:
            print(f"Failed to plot {f}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
