"""Shared log1p(overdensity) transform, used identically by every pipeline that
compares a low/corrected/high map or patch (unet, transfer).

One canonical formula, used two ways:
  * log1p_delta(m)              -- single map, eps floor from its OWN mean.
  * log1p_delta_pair(low, high) -- a (low, high) PAIR sharing one eps floor
    (min of the two means), matching unet/dataset.py's
    raw_to_log1p_delta_pair exactly -- needed when comparing two patches/maps that
    must use the SAME clip so a fair delta comparison is possible.

eps = 0.5/mean is a small nudge above -1 so log1p(-1)=-inf is never hit by a
zero-count pixel, EXCEPT for very sparse fields (mean << 1, e.g. shell 0-10 of the
lightcone) where eps blows up and the floor clips away real structure -- this is a
property of the data/transform, not a bug (see unet/apply_flow.py's
full-sky section docstring for the full diagnosis).
"""
from __future__ import annotations
import numpy as np


def log1p_delta(m: np.ndarray, eps_ref_mean: float | None = None) -> np.ndarray:
    """m (raw counts) -> log1p(overdensity). eps_ref_mean overrides which mean sets
    the eps=0.5/mean clipping floor (see log1p_delta_pair); default is m's own mean."""
    mean = float(np.mean(m))
    ref = eps_ref_mean if eps_ref_mean is not None else mean
    eps = 0.5 / max(ref, 1e-9)
    delta = m / max(mean, 1e-9) - 1.0
    return np.log1p(np.maximum(delta, -1.0 + eps))


def log1p_delta_pair(low: np.ndarray, high: np.ndarray):
    """(low, high) raw counts -> (low_log, high_log), sharing one eps floor from
    min(low_mean, high_mean) -- identical semantics to dataset.py's
    raw_to_log1p_delta_pair. Use this (not two independent log1p_delta calls) when
    the two fields must be judged on the same clip."""
    ref = min(float(np.mean(low)), float(np.mean(high)))
    return log1p_delta(low, ref), log1p_delta(high, ref)
