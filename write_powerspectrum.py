"""
Compute P(k) from a TIPSY dark-matter snapshot using a pure-Python/numpy FFT
pipeline. This version follows PKDGRAV3's `measure_pk` choices for the
FFT normalization and mass-assignment correction.

All configuration is set in-script (no argv). Adjust the constants below to
match your simulation.
"""
import os
import math
from pathlib import Path
import numpy as np

# --- Configuration (set these to match your run / pkdgrav measure_pk) ---
SNAPSHOT = "/cluster/scratch/damrein/outputs/snapshots/final_snapshot_tipsy.00000"
import sys as _sys
# Allow specifying the snapshot file as the single command-line argument.
if len(_sys.argv) > 1:
    SNAPSHOT = _sys.argv[1]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs/powerspectrum")
NGRID = 512          # pkd: nGrid (grid used for P(k))
LBOX = 900.0         # box size in Mpc/h (adjust if needed)
ASSIGN_ORDER = 2     # iAssignment: 2 -> CIC (set to match pkd)
INTERLACE = False    # bInterlace
N_BINS = 300         # nBins (pkd default-like)
# ---------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
stem = Path(SNAPSHOT).stem
out_pk = Path(OUT_DIR) / f"{stem}.pk"


def read_tipsy_dark(snapshot_path):
    # Minimal loader for tipsy format used elsewhere in this repo (big-endian)
    header_dtype = np.dtype([
        ('time',    '>f8'),
        ('nBodies', '>u4'),
        ('nDim',    '>u4'),
        ('nSph',    '>u4'),
        ('nDark',   '>u4'),
        ('nStar',   '>u4'),
        ('pad',     '>u4'),
    ])
    dark_dtype = np.dtype([
        ('mass', '>f4'),
        ('pos',  '>f4', (3,)),
        ('vel',  '>f4', (3,)),
        ('eps',  '>f4'),
        ('phi',  '>f4'),
    ])
    with open(snapshot_path, 'rb') as f:
        header = np.fromfile(f, dtype=header_dtype, count=1)[0]
        nSph = int(header['nSph'])
        nDark = int(header['nDark'])
        f.seek(header_dtype.itemsize + nSph * 48)
        dark = np.fromfile(f, dtype=dark_dtype, count=nDark)
    return dark


def window_correction(nGrid, iAssignment):
    # Follow PKDGRAV's Window constructor: W[i] = (win(i))^iAssignment
    W = np.empty(nGrid, dtype=float)
    for i in range(nGrid):
        win = math.pi * i / nGrid
        if win > 0.1:
            win = win / math.sin(win)
        else:
            # series expansion to avoid numerical cancellation
            win = 1.0 / (1.0 - win * win / 6.0 * (1.0 - win * win / 20.0 * (1.0 - win * win / 76.0)))
        W[i] = win ** iAssignment
    return W


print(f"Loading {SNAPSHOT} ...")
dark = read_tipsy_dark(SNAPSHOT)
pos = dark['pos'].astype('f8')
pos += 0.5
pos *= NGRID
pos = np.clip(pos, 0.0, NGRID - 1e-6)

print("CIC mass assignment ...")
# CIC (Order=2): matches pkdgrav's iAssignment=2 / aweights.hpp Order=2.
# The weight formula is: rr = r - 0.5, i = floor(rr), h = rr - i
# cell i: (1-h), cell i+1: h  (per axis, trilinearly for 3-D)
pos_sh = pos - 0.5          # rr = r - 0.5  (pos is already in [0, NGRID))
ix0 = np.floor(pos_sh[:, 0]).astype(int)
iy0 = np.floor(pos_sh[:, 1]).astype(int)
iz0 = np.floor(pos_sh[:, 2]).astype(int)
hx = pos_sh[:, 0] - ix0.astype(float)   # weight toward ix0+1
hy = pos_sh[:, 1] - iy0.astype(float)
hz = pos_sh[:, 2] - iz0.astype(float)
ix0 %= NGRID;  ix1 = (ix0 + 1) % NGRID
iy0 %= NGRID;  iy1 = (iy0 + 1) % NGRID
iz0 %= NGRID;  iz1 = (iz0 + 1) % NGRID
delta = np.zeros((NGRID, NGRID, NGRID), dtype='f8')
np.add.at(delta, (ix0, iy0, iz0), (1-hx)*(1-hy)*(1-hz))
np.add.at(delta, (ix1, iy0, iz0),    hx *(1-hy)*(1-hz))
np.add.at(delta, (ix0, iy1, iz0), (1-hx)*   hy *(1-hz))
np.add.at(delta, (ix0, iy0, iz1), (1-hx)*(1-hy)*   hz )
np.add.at(delta, (ix1, iy1, iz0),    hx *   hy *(1-hz))
np.add.at(delta, (ix1, iy0, iz1),    hx *(1-hy)*   hz )
np.add.at(delta, (ix0, iy1, iz1), (1-hx)*   hy *   hz )
np.add.at(delta, (ix1, iy1, iz1),    hx *   hy *   hz )
mean = delta.mean()
delta = delta / mean - 1.0

print("FFT ...")
# Use real FFT for memory; result shape = (NGRID, NGRID, NGRID//2+1)
dk = np.fft.rfftn(delta)

# Apply PKDGRAV-like normalization and assignment-window correction
W = window_correction(NGRID, ASSIGN_ORDER)
iNyquist = NGRID // 2

# Build integer-index arrays matching PKDGRAV indexing conventions.
# numpy rfftn halves the LAST axis (kz), so:
#   axis 0 (kx): 0..NGRID-1  — indices > iNyquist are negative frequencies → fold
#   axis 1 (ky): 0..NGRID-1  — same → fold
#   axis 2 (kz): 0..iNyquist — already non-negative, no fold needed
i = np.arange(NGRID, dtype=int)
j = np.arange(NGRID, dtype=int)
k = np.arange(iNyquist + 1, dtype=int)
I, J, K = np.meshgrid(i, j, k, indexing='ij')

# Fold both I and J (pkdgrav folds j and k because FFTW halves axis-0 instead)
I_fold = np.where(I > iNyquist, NGRID - I, I)
J_fold = np.where(J > iNyquist, NGRID - J, J)
K_fold = K  # K already in [0, iNyquist]

Wcorr = W[I_fold] * W[J_fold] * W[K_fold]

# Following PKDGRAV: v1 = dk * (1/nGrid^3) * Wcorr, then fPower += |v1|^2
scale_fft = 1.0 / (NGRID ** 3)
v1 = dk * scale_fft * Wcorr
pk3d = np.abs(v1) ** 2

print("Binning P(k) ...")
# Bin according to PKDGRAV's logarithmic scheme (no LINEAR_PK).
# pkdgrav computes: ks_int = int(ak), then bin = floor(log(ks_int) * scale)
# — it bins using the *integer* k-magnitude, not the float one.
nBins = N_BINS
iNy = iNyquist
scale = nBins / math.log(iNy + 1.0)

fK = np.zeros(nBins, dtype=float)
fPower = np.zeros(nBins, dtype=float)
nPower = np.zeros(nBins, dtype=int)

# Vectorised loop: use folded integer magnitudes for filtering and bin assignment
ak = np.sqrt(I_fold.astype(float)**2 + J_fold.astype(float)**2 + K_fold.astype(float)**2)
int_ak = ak.astype(int)          # truncate like C's int() — matches pkdgrav's `ks = int(ak)`
valid = (int_ak >= 1) & (int_ak <= iNy)
ak_valid    = ak[valid]
pk_valid    = pk3d[valid]
int_ak_valid = int_ak[valid]

# bin index uses log of the *integer* magnitude (pkdgrav: floor(log(ks)*scale))
ks = np.floor(np.log(int_ak_valid.astype(float)) * scale).astype(int)
mask = (ks >= 0) & (ks < nBins)
for kk, pval, akv in zip(ks[mask], pk_valid[mask], ak_valid[mask]):
    fK[kk] += math.log(akv)
    fPower[kk] += pval
    nPower[kk] += 1

# finalize bins
with np.errstate(divide='ignore', invalid='ignore'):
    k_centers = np.exp(np.where(nPower > 0, fK / nPower, 0.0))
    power_avg = np.where(nPower > 0, fPower / nPower, 0.0)

# Convert integer-k centers to physical k: k_phys = k_centers * (2*pi / LBOX)
k_phys = k_centers * (2.0 * math.pi / LBOX)

# PKDGRAV multiplies |delta_k|^2 by Lbox^3 to get physical P(k)
pk_phys = power_avg * (LBOX ** 3)

# write output (k [1/Mpc] , P(k) , n_modes)
with open(out_pk, 'w') as f:
    for k_val, pk_val, n in zip(k_phys, pk_phys, nPower):
        if n > 0:
            f.write(f"{k_val:.6e} {pk_val:.6e} {int(n)}\n")

print(f"Power spectrum written to {out_pk}")
