"""Shared full-sky HEALPix helpers: the real angular power spectrum (od_cl) and a
gnomonic zoom crop for visual inspection -- used by every pipeline's
cl_shell*.png / example_full_sky.png (unet, transfer).

Unlike radial_power.py's flat-patch FFT (bounded by one small patch's own Nyquist
wavenumber), od_cl is the genuine spherical-harmonic transform over the WHOLE sky, so
it is the only honest way to see behavior out to high ell (see the full-sky section
of unet/apply_flow.py's docstring for why the flat-patch metric can't).
"""
from __future__ import annotations
import numpy as np
import healpy as hp


def od_cl(m: np.ndarray, lmax: int) -> np.ndarray:
    """Full-sky RING-ordered raw-counts map -> angular power spectrum of its
    overdensity, via the real spherical-harmonic transform (hp.anafast)."""
    return hp.anafast((m / np.nanmean(m) - 1.0).astype(np.float64), lmax=lmax)


def gnomonic_crop(m: np.ndarray, nside: int, lon: float, lat: float,
                  xsize: int = 200, reso_arcmin: float = 1.5) -> np.ndarray:
    """Extract a flat (xsize,xsize) tangent-plane zoom of a full-sky RING map, centered
    at (lon,lat) in degrees -- for visual "by eye" inspection only (not a metric)."""
    proj = hp.projector.GnomonicProj(rot=(lon, lat, 0.0), xsize=xsize, ysize=xsize,
                                     reso=reso_arcmin)
    return proj.projmap(m, lambda x, y, z, _ns=nside: hp.vec2pix(_ns, x, y, z, nest=False))


def zbin_shell_samples(n_shells: int, zbin_start: int = 0, n_zbins: int = 3,
                       n_per_zbin: int = 5) -> list[tuple[str, np.ndarray]]:
    """Split the usable shell range [zbin_start, n_shells-1] into n_zbins
    contiguous bins, and sample up to n_per_zbin evenly-spaced shell indices per
    bin (rounded, deduplicated) -- keeps the Cl-ratio-by-redshift-bin diagnostic
    (one full-sky tile-and-reconstruct pass PER sampled shell, the expensive part)
    bounded regardless of how many shells the dataset has. zbin_start lets the
    caller skip the sparsest low shells (see transforms.py's eps-clipping note) --
    their Cl ratio is dominated by the log1p floor, not the model. Returns
    [(bin_label, shells), ...], one entry per bin, in order."""
    edges = np.unique(np.round(np.linspace(zbin_start, n_shells - 1, n_zbins + 1)).astype(int))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        shells = np.unique(np.round(np.linspace(lo, hi, n_per_zbin)).astype(int))
        bins.append((f"shells {lo}-{hi}", shells))
    return bins
