import numpy as np
import pickle as pkl
import os
import shutil
from sys import argv

def update_tipsy_dark_positions(filename, new_positions):
    """
    Update the 'pos' field of dark matter particles in a TIPSY file.

    Parameters:
    - filename: path to the TIPSY file
    - new_positions: ndarray of shape (nDark, 3) with updated positions
    """
    header_dtype = np.dtype([
        ('time',   '>f8'),
        ('nBodies','>u4'),
        ('nDim',   '>u4'),
        ('nSph',   '>u4'),
        ('nDark',  '>u4'),
        ('nStar',  '>u4'),
        ('pad',    '>u4'),
    ])
    
    dark_dtype = np.dtype([
        ('mass',   '>f4'),
        ('pos',    '>f4', (3,)),
        ('vel',    '>f4', (3,)),
        ('eps',    '>f4'),
        ('phi',    '>f4'),
    ])

    with open(filename, 'r+b') as f:
        # Read header
        header = np.fromfile(f, dtype=header_dtype, count=1)[0]
        nSph = header['nSph']
        nDark = header['nDark']

        assert new_positions.shape == (nDark, 3), "new_positions shape must be (nDark, 3)"

        # Skip gas particles
        offset = header_dtype.itemsize + nSph * 48  # each gas particle is 48 bytes (assuming standard tipsyGas layout)
        f.seek(offset)

        # Read existing dark matter particles
        dark = np.fromfile(f, dtype=dark_dtype, count=nDark)

        # Update positions
        dark['pos'] = new_positions

        # Seek back and overwrite dark matter block
        f.seek(offset)
        dark.tofile(f)

    print(f"Updated positions for {nDark} dark matter particles in '{filename}'.")

# --- configuration ---
snapshot_file = argv[1] if len(argv) > 1 else "/cluster/scratch/damrein/outputs/snapshots/final_snapshot_cpu_60125122.npz"
source_tipsy  = "/cluster/scratch/damrein/outputs/ICs/000001_copy7/CosmoML.00000" # dummmy tipsy file with correct header and gas block, but placeholder dark positions
output_tipsy  = "/cluster/scratch/damrein/outputs/snapshots/CosmoML.00000"
Lbox = 900

print(f"Loading snapshot: {snapshot_file}")

# --- load positions from .npz or .pkl ---
if snapshot_file.endswith('.npz'):
    with np.load(snapshot_file, allow_pickle=True) as data:
        key = next((k for k in ['dark_xyz', 'pos', data.files[0]] if k in data), None)
        if key is None:
            raise RuntimeError(f"No arrays found in {snapshot_file}")
        dark_xyz = data[key]
        if dark_xyz.ndim == 1 and set(['x','y','z']).issubset(data.files):
            dark_xyz = np.vstack((data['x'], data['y'], data['z'])).T
else:
    with open(snapshot_file, 'rb') as f:
        dark_xyz = pkl.load(f)

# --- prepare positions ---
dark_xyz = np.array(dark_xyz, dtype='f4')
# flatten grid layouts like (Nx, Ny, Nz, 3) -> (N, 3)
if dark_xyz.ndim != 2 and dark_xyz.shape[-1] == 3:
    dark_xyz = dark_xyz.reshape(-1, 3)
if dark_xyz.ndim != 2 or dark_xyz.shape[1] != 3:
    raise ValueError(f"Expected shape (N, 3), got {dark_xyz.shape}")
print(f"Loaded {dark_xyz.shape[0]} particles. x=[{dark_xyz[:,0].min():.3f}, {dark_xyz[:,0].max():.3f}]")

# convert to pkdgrav units: [0, Lbox] -> [-0.5, 0.5]
dark_xyz /= Lbox
dark_xyz -= 0.5
print("Positions converted to pkdgrav units.")

# --- copy placeholder tipsy and overwrite dark positions ---
shutil.copy(source_tipsy, output_tipsy)
print(f"Copied {source_tipsy} -> {output_tipsy}")

update_tipsy_dark_positions(output_tipsy, dark_xyz)
print("All done!")