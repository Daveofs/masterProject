import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from visualize import plot_density_slice
from pathlib import Path


print("\n==============================")
print("Reading DISCO-DJ")
print("==============================")

disco_base = '/capstor/scratch/cscs/damrein/outputs/disco_custom/data/output/gpu_grid_3697757'
disco_files = [f"{disco_base}/snapshot.{i}.hdf5" for i in range(4)]
disco_key = 'PartType1/Coordinates'
Boxsize = 900.0
slice_thickness_val = 5.0
grid_val = 832
output_dir = Path(r'/capstor/scratch/cscs/damrein/outputs/plots/snapshots')

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

plot_density_slice(
    positions=pos_dis,
    boxsize=Boxsize,
    slice_axis=2,
    slice_center=None,
    slice_thickness=slice_thickness_val,
    grid=grid_val,
    output_dir=output_dir,
    input_file=None,
)