#!/usr/bin/env python3
"""Compare two .pk files: shapes, median ratio, and plots.

Saves compare_P.png and ratio_P1_over_P2.png to outputs/plots/powerspectrum
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys


def load_k_p(path: Path):
    data = np.genfromtxt(path, comments="#")
    if data.ndim == 1 and data.size >= 2:
        return np.array([data[0]]), np.array([data[1]])
    return data[:, 0], data[:, 1]


f1 = Path("/cluster/scratch/damrein/outputs/ICs/000001_copy7/CosmoML.00080.pk")
f2 = Path("/cluster/scratch/damrein/outputs/powerspectrum/final_snapshot_tipsy.pk")

for f in (f1, f2):
    if not f.exists():
        print(f"Missing: {f}", file=sys.stderr)
        sys.exit(3)

k1, p1 = load_k_p(f1)
k2, p2 = load_k_p(f2)

print("Shapes:", k1.shape, p1.shape, k2.shape, p2.shape)

# Align k: use intersection if available, otherwise union + interpolation
common_k = np.intersect1d(k1, k2)
if common_k.size == 0:
    uk = np.union1d(k1, k2)
    p1i = np.interp(uk, k1, p1)
    p2i = np.interp(uk, k2, p2)
else:
    uk = common_k
    p1i = np.interp(uk, k1, p1)
    p2i = np.interp(uk, k2, p2)

ratio = p1i / p2i
print("Median ratio (P1/P2):", float(np.median(ratio)))
print("Mean ratio (P1/P2):", float(np.mean(ratio)))

outdir = Path("/cluster/scratch/damrein/outputs/plots/powerspectrum")
outdir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8,5))
plt.loglog(uk, p1i, label=f1.name)
plt.loglog(uk, p2i, label=f2.name)
plt.legend()
plt.xlabel('k')
plt.ylabel('P(k)')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(outdir / 'compare_P.png', dpi=150)
plt.close()

plt.figure(figsize=(8,4))
plt.semilogx(uk, ratio)
plt.axhline(np.median(ratio), color='k', ls='--')
plt.xlabel('k')
plt.ylabel('P1/P2')
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / 'ratio_P1_over_P2.png', dpi=150)
plt.close()

print("Wrote:", outdir / 'compare_P.png')
print("Wrote:", outdir / 'ratio_P1_over_P2.png')
