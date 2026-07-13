"""Flat-patch 2D-FFT radial power spectrum -- the local, patch-level analogue of the
angular power spectrum C_ell, used by every pipeline's example_patches.png (the 4th
column: "power ratio to truth" computed on one small gnomonic patch, as opposed to
full_sky.od_cl's REAL C_ell over the whole reconstructed sphere).

Single canonical numpy implementation, shared by unet_flow_jbucko and transfer. A
batched torch version lives in unet_flow_jbucko/apply_flow.py ONLY for GPU throughput
when scoring hundreds of patches at once (averaged power_spectrum_ratio.png) -- for a
handful of example rows, this plain numpy version is simpler and fast enough.
"""
from __future__ import annotations
import numpy as np


def radial_power(img: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    """(H,W) real-valued patch -> (n_bins,) mean 2D-FFT power per radial wavenumber bin."""
    H, W = img.shape
    n_bins = n_bins or H // 2
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.rfftfreq(W) * W
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = np.clip((r / r.max() * (n_bins - 1)).astype(np.int64), 0, n_bins - 1).ravel()
    counts = np.clip(np.bincount(bins, minlength=n_bins), 1, None).astype(np.float64)
    f = np.fft.rfft2(img)
    power = (f.real ** 2 + f.imag ** 2).ravel()
    binned = np.bincount(bins, weights=power, minlength=n_bins)
    return binned / counts
