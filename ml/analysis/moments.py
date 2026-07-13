"""Per-map one-point statistics (mean, variance, skewness, excess kurtosis) -- the
one-point-PDF analogue of radial_power.py's two-point statistic. Shared by every
pipeline's moments-vs-shell-depth and histogram figures (unet_flow_jbucko, transfer).

Motivated by the positivity/one-point-PDF investigation: a Cl ratio (two-point,
phase-blind) can look perfect while the marginal pixel distribution is badly wrong --
CosmoGrid's faint shells are sparse, non-Gaussian COUNT fields (e.g. shell 3 is 91.8%
exact zeros), so matching Cl alone says nothing about whether the corrected field's
pixel-value distribution (mean/variance/skew/kurtosis, zero-inflation, tail) looks
right. These figures exist specifically to catch that, on RAW counts (not the
log1p-delta space the Cl/power-ratio figures use).
"""
from __future__ import annotations
import numpy as np


def moments(m: np.ndarray) -> dict[str, float]:
    """Raw-count (or any real-valued) map/patch/pixel array -> mean, variance,
    skewness, and excess kurtosis (Fisher convention, 0 for a Gaussian) of its pixel
    values. Pools whatever pixels the caller passes in (a single patch, several
    stacked patches, or a whole full-sky map) -- scope is entirely the caller's
    choice, same convention as radial_power.py/full_sky.od_cl."""
    x = np.asarray(m, dtype=np.float64).ravel()
    mean = float(x.mean())
    c = x - mean
    var = float(np.mean(c ** 2))
    std = np.sqrt(max(var, 1e-300))
    skew = float(np.mean(c ** 3) / std ** 3)
    excess_kurtosis = float(np.mean(c ** 4) / std ** 4 - 3.0)
    return {"mean": mean, "variance": var, "skewness": skew, "excess_kurtosis": excess_kurtosis}
