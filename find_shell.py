#!/usr/bin/env python3
"""
find_shell.py

Find the shell file in a directory that best matches a target redshift.

Usage: python find_shell.py -z 0.5 -d ICs/000001_copy6 --tolerance 0.01

The script searches common file formats (.npz, .npy, .h5, .txt) and
also attempts to parse redshift from filenames.
"""
import os
import re
import sys
import argparse
from math import inf

try:
    import numpy as np
except Exception:
    print("Error: numpy is required", file=sys.stderr)
    raise

try:
    import h5py
except Exception:
    h5py = None

try:
    from astropy.io import fits as _fits
except Exception:
    _fits = None


FNAME_Z_RE = re.compile(r'(?:z|redshift|z=|z_)([0-9]+\.[0-9]+)')
INLINE_Z_RE = re.compile(r'redshift\W*[=:]\W*([0-9]+\.?[0-9]*)', re.I)


def parse_z_from_filename(name):
    m = FNAME_Z_RE.search(name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # try to find a plain decimal in the filename if it looks like a z
    m2 = re.search(r'([0-9]+\.[0-9]{3,})', name)
    if m2:
        return float(m2.group(1))
    return None


def z_from_np_file(path):
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    # npz returns NpzFile (dict-like); .npy returns ndarray
    if hasattr(data, 'files'):
        # named arrays inside
        for key in data.files:
            key_low = key.lower()
            if 'redshift' in key_low or key_low == 'z':
                try:
                    v = float(data[key])
                    return v
                except Exception:
                    pass
        # sometimes metadata stored as object
        for key in data.files:
            try:
                arr = data[key]
                if hasattr(arr, 'dtype') and arr.dtype == object:
                    txt = str(arr)
                    m = INLINE_Z_RE.search(txt)
                    if m:
                        return float(m.group(1))
            except Exception:
                pass
    else:
        # .npy array; nothing to glean usually
        return None
    return None


def z_from_h5(path):
    if h5py is None:
        return None
    try:
        with h5py.File(path, 'r') as f:
            # check attrs
            for k, v in f.attrs.items():
                kn = str(k).lower()
                if 'redshift' in kn or kn == 'z' or 'scale' in kn:
                    try:
                        return float(v)
                    except Exception:
                        pass
            # check datasets
            def recurse(g):
                for name, item in g.items():
                    if isinstance(item, h5py.Dataset):
                        nlow = name.lower()
                        if 'redshift' in nlow or nlow == 'z' or 'scale' in nlow:
                            try:
                                val = item[()]
                                return float(val)
                            except Exception:
                                pass
                    elif isinstance(item, h5py.Group):
                        v = recurse(item)
                        if v is not None:
                            return v
                return None
            v = recurse(f)
            return v
    except Exception:
        return None


def z_from_fits(path):
    if _fits is None:
        return None
    try:
        # astropy will handle gzipped files transparently
        with _fits.open(path, mode='readonly') as hdul:
            # check primary header and extensions
            for hdu in hdul:
                hdr = hdu.header
                # common header keywords that may hold redshift
                for key in ('REDSHIFT', 'REDSHFT', 'Z', 'ZRED', 'Z_REF', 'REDSHF'):
                    if key in hdr:
                        try:
                            return float(hdr[key])
                        except Exception:
                            pass
                # sometimes scale factor 'A' or 'SCALE' is present
                for key in ('SCALE', 'SCALE_FACTOR', 'A', 'AA'):
                    if key in hdr:
                        try:
                            a = float(hdr[key])
                            if a > 0:
                                return (1.0 / a) - 1.0
                        except Exception:
                            pass
    except Exception:
        return None
    return None


def z_from_text(path):
    try:
        with open(path, 'r', errors='ignore') as fh:
            txt = fh.read(4096)
            m = INLINE_Z_RE.search(txt)
            if m:
                return float(m.group(1))
    except Exception:
        return None
    return None


def find_best_match(base_dir, target_z, tolerance=0.01, exts=None):
    if exts is None:
        exts = ['.npz', '.npy', '.h5', '.hdf5', '.txt', '.dat', '.fits', '.fit', '.fits.gz', '.fit.gz', '.fz']
    candidates = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            fn_low = fn.lower()
            if not any(fn_low.endswith(e) for e in exts):
                continue
            path = os.path.join(root, fn)
            z = None
            # filename parse
            z = parse_z_from_filename(fn)
            # try file inspection for better info based on suffix
            if fn_low.endswith(('.npz', '.npy')):
                z2 = z_from_np_file(path)
                if z2 is not None:
                    z = z2
            elif fn_low.endswith(('.h5', '.hdf5')):
                z2 = z_from_h5(path)
                if z2 is not None:
                    z = z2
            elif fn_low.endswith(('.fits', '.fit', '.fits.gz', '.fit.gz', '.fz')):
                z2 = z_from_fits(path)
                if z2 is not None:
                    z = z2
            elif fn_low.endswith(('.txt', '.dat')):
                z2 = z_from_text(path)
                if z2 is not None:
                    z = z2

            if z is not None:
                candidates.append((path, float(z)))
            else:
                # keep filename-parsed candidates too
                fname_z = parse_z_from_filename(fn)
                if fname_z is not None:
                    candidates.append((path, float(fname_z)))

    if not candidates:
        return None, []

    # compute closest
    best = None
    best_diff = inf
    for p, z in candidates:
        d = abs(z - target_z)
        if d < best_diff:
            best_diff = d
            best = (p, z)

    # filter by tolerance
    matches = [(p, z, abs(z - target_z)) for (p, z) in candidates if abs(z - target_z) <= tolerance]
    if matches:
        # sort by distance
        matches.sort(key=lambda x: x[2])
        return (matches[0][0], matches[0][1]), matches

    return best, [(p, z, abs(z - target_z)) for (p, z) in candidates]


def main():
    p = argparse.ArgumentParser(description='Find shell matching a target redshift')
    p.add_argument('-d', '--dir', default='/cluster/scratch/damrein/outputs/ICs/000001_copy6', help='Base directory to search')
    p.add_argument('-z', '--redshift', required=True, type=float, help='Target redshift (float)')
    p.add_argument('-t', '--tolerance', type=float, default=0.01, help='Tolerance in redshift')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    base = args.dir
    if not os.path.exists(base):
        print(f"Error: directory {base} does not exist", file=sys.stderr)
        sys.exit(2)

    best, all_matches = find_best_match(base, args.redshift, tolerance=args.tolerance)
    if not best:
        print("No candidate shell files found.")
        sys.exit(1)

    best_path, best_z = best
    print(f"Best match: {best_path}  (z={best_z:.6f}, Δ={abs(best_z-args.redshift):.6f})")
    if args.verbose:
        print('\nAll candidates (path, z, Δ):')
        for pth, z, d in sorted(all_matches, key=lambda x: x[2]):
            print(f"{pth}  {z:.6f}  Δ={d:.6f}")

    # exit success if within tolerance
    if abs(best_z - args.redshift) <= args.tolerance:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()