import h5py
import numpy as np
import matplotlib.pyplot as plt

import Pk_library as PKL
import MAS_library as MASL


def get_pk(pos, Lbox, Ngrid):
    
    pos = pos.astype(np.float32)
    BoxSize = Lbox
    grid = Ngrid

    # Build density field
    delta = np.zeros((grid, grid, grid), dtype=np.float32)

    # Mass assignment with CIC
    MASL.MA(pos, delta, BoxSize, MAS="CIC")

    # Convert density to overdensity
    delta /= np.mean(delta)
    delta -= 1.0

    # Compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis=0, MAS="CIC", threads=4)

    k = Pk.k3D
    Pk_vals = Pk.Pk[:, 0]

    print("k:", k.shape)
    print("P(k):", Pk_vals.shape)

    return k, Pk_vals


# =========================
# User parameters
# =========================

pkdgrav_file = "/capstor/scratch/cscs/damrein/cosmogridv1_fiducial/run_0000/standard/CosmoML.00140.hdf5"
pkdgrav_key = "PartType1/Coordinates"

disco_base = "/users/damrein/masterProject/disco/data/output/gpu_fiducial_3523591"
disco_files = [f"{disco_base}/snapshot.{i}.hdf5" for i in range(8)]
disco_key = "PartType1/Coordinates"

Lbox = 900.0

# Density grid resolution.
Ngrid = 832

# Plot up to this multiple of the Nyquist frequency.
# Used only internally to define the x-axis range.
k_ny_plot_factor = 1.6


# =========================
# Read PKDGRAV
# =========================

print("\n==============================")
print("Reading PKDGRAV")
print("==============================")

with h5py.File(pkdgrav_file, "r") as f:
    pos_pkd = f[pkdgrav_key][:]

print("PKDGRAV raw shape:", pos_pkd.shape)
print("PKDGRAV raw dtype:", pos_pkd.dtype)
print("PKDGRAV raw min:", np.min(pos_pkd, axis=0))
print("PKDGRAV raw max:", np.max(pos_pkd, axis=0))

# PKDGRAV is in [-0.5, 0.5] -> convert to [0, Lbox)
pos_pkd = (pos_pkd + 0.5) * Lbox
pos_pkd = np.mod(pos_pkd, Lbox)

print("PKDGRAV converted min:", np.min(pos_pkd, axis=0))
print("PKDGRAV converted max:", np.max(pos_pkd, axis=0))


# =========================
# Read DISCO-DJ
# =========================

print("\n==============================")
print("Reading DISCO-DJ")
print("==============================")

all_pos_dis = []

total_raw = 0
total_used = 0
total_removed = 0

for filename in disco_files:
    print(f"\nReading DISCO-DJ shard: {filename}")

    with h5py.File(filename, "r") as f:
        pos = f[disco_key][:]

    print("raw shape:", pos.shape)
    print("raw dtype:", pos.dtype)
    print("raw min:", np.nanmin(pos, axis=0))
    print("raw max:", np.nanmax(pos, axis=0))

    total_raw += pos.shape[0]

    # Remove invalid rows: this removes both NaN and inf
    finite = np.isfinite(pos).all(axis=1)
    n_used = int(finite.sum())
    n_removed = int(pos.shape[0] - n_used)

    print("finite particles:", n_used)
    print("removed non-finite:", n_removed)

    pos = pos[finite]

    total_used += pos.shape[0]
    total_removed += n_removed

    all_pos_dis.append(pos)

print("\nConcatenating DISCO-DJ shards...")
pos_dis = np.concatenate(all_pos_dis, axis=0)

print("\nDISCO-DJ summary:")
print("total raw entries:", total_raw)
print("total used finite particles:", total_used)
print("total removed non-finite:", total_removed)
print("DISCO-DJ concatenated shape:", pos_dis.shape)
print("DISCO-DJ min:", np.min(pos_dis, axis=0))
print("DISCO-DJ max:", np.max(pos_dis, axis=0))

# DISCO-DJ is already in physical coordinates [0, Lbox)
pos_dis = np.mod(pos_dis, Lbox)

print("DISCO-DJ wrapped min:", np.min(pos_dis, axis=0))
print("DISCO-DJ wrapped max:", np.max(pos_dis, axis=0))


# =========================
# Compute power spectra
# =========================

print("\n==============================")
print("Computing PKDGRAV P(k)")
print("==============================")
k_pkd, pk_pkd = get_pk(pos_pkd, Lbox, Ngrid)

print("\n==============================")
print("Computing DISCO-DJ P(k)")
print("==============================")
k_dis, pk_dis = get_pk(pos_dis, Lbox, Ngrid)


# =========================
# Common k handling
# =========================

k_ny = np.pi * Ngrid / Lbox
k_plot_max = k_ny_plot_factor * k_ny

# Ideally the two k arrays should be identical because same box and same grid.
if np.allclose(k_pkd, k_dis, rtol=1e-5, atol=0):
    k = k_pkd
    pk_dis_on_pkd = pk_dis
else:
    print("\nWARNING: k bins differ. Interpolating DISCO-DJ P(k) onto PKDGRAV k bins.")
    k = k_pkd
    pk_dis_on_pkd = np.interp(k_pkd, k_dis, pk_dis)

mask_plot = (
    np.isfinite(k)
    & np.isfinite(pk_pkd)
    & np.isfinite(pk_dis_on_pkd)
    & (k > 0)
    & (pk_pkd > 0)
    & (pk_dis_on_pkd > 0)
    & (k <= k_plot_max)
)

k_plot = k[mask_plot]
pk_pkd_plot = pk_pkd[mask_plot]
pk_dis_plot = pk_dis_on_pkd[mask_plot]

ratio = pk_dis_plot / pk_pkd_plot

print("\nPlotting summary:")
print(f"Kept {len(k_plot)} bins out of {len(k)}")
print(f"k min plotted = {k_plot.min():.6f}")
print(f"k max plotted = {k_plot.max():.6f}")


# =========================
# Save P(k) values to txt
# =========================

out_pkd_txt = "pk_pkd_fiducial.txt"
out_disco_txt = "pk_disco_fiducial.txt"

np.savetxt(
    out_pkd_txt,
    np.column_stack([k_plot, pk_pkd_plot]),
    header="k[h/Mpc] Pk[(Mpc/h)^3]",
    fmt="%.10e",
)

np.savetxt(
    out_disco_txt,
    np.column_stack([k_plot, pk_dis_plot]),
    header="k[h/Mpc] Pk[(Mpc/h)^3]",
    fmt="%.10e",
)

print("\nSaved:")
print(out_pkd_txt)
print(out_disco_txt)


# =========================
# Plot 1: Power spectra
# =========================

out_pk_png = "pkd_disco_pks.png"

plt.figure(figsize=(8, 5.5))

plt.loglog(k_plot, pk_pkd_plot, label="PKDGRAV")
plt.loglog(k_plot, pk_dis_plot, label="DISCO-DJ")

plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
plt.title("PKDGRAV vs DISCO-DJ power spectrum")
plt.xlim(k_plot.min(), k_plot_max)
plt.legend()
plt.tight_layout()
plt.savefig(out_pk_png, dpi=200)
plt.close()

print("\nSaved:")
print(out_pk_png)


# =========================
# Plot 2: Ratio
# =========================

out_ratio_png = "pkd_disco_ratio.png"

plt.figure(figsize=(8, 5.5))

plt.semilogx(
    k_plot,
    ratio,
    label=r"$P_\mathrm{DISCO-DJ}(k) / P_\mathrm{PKDGRAV}(k)$",
)

# Central reference line
plt.axhline(
    1.0,
    linestyle="-",
    linewidth=1.2,
    color="black",
    label="ratio = 1",
)

# +/- 3% lines (more prominent)
plt.axhline(
    0.97,
    linestyle="--",
    linewidth=1.2,
    color="gray",
    label=r"$\pm 3\%$",
)
plt.axhline(
    1.03,
    linestyle="--",
    linewidth=1.2,
    color="gray",
)

# +/- 5% lines 
plt.axhline(
    0.95,
    linestyle=":",
    linewidth=1.0,
    color="gray",
    label=r"$\pm 5\%$",
)
plt.axhline(
    1.05,
    linestyle=":",
    linewidth=1.0,
    color="gray",
)

plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$P_\mathrm{DISCO-DJ}(k) / P_\mathrm{PKDGRAV}(k)$")
plt.title("Power spectrum ratio")
plt.xlim(k_plot.min(), k_plot_max)
plt.ylim(0.5, 1.5)

plt.legend()
plt.tight_layout()
plt.savefig(out_ratio_png, dpi=200)
plt.close()

print("\nSaved:")
print(out_ratio_png)
