"""PatchDataset: loads (low, high) patch pairs produced by make_patch_dataset.py.

Returns *raw* counts, not the log1p(overdensity) representation the model
actually trains on - that transform is applied batched, on GPU, in the
training loop (see raw_to_log1p_delta_pair below) instead of per-sample in
__getitem__. Measured GPU utilization was near-0% with the transform done here
in NumPy inside worker processes (data-loading bound, not compute-bound);
workers now just read+collate raw arrays, which is much cheaper per-sample.

Train/val split is by cosmology (not by patch index) so validation cosmologies
are never seen in training - a random per-patch split would leak, since many
patches from the same cosmology/run look highly correlated at large scales.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# order must match make_patch_dataset.py's cosmo_params tuple
COSMO_FIELDS = ("omega_m", "omega_b", "ns", "sigma8", "w0", "h")

# the exact field order FlowUNet's cosmology+redshift conditioning expects (see
# flow_model.py). H0 -> h (only h=H0/100 is stored; same convention as the rest of
# this project). Omega_cdm isn't separately stored either -- only Om (total matter)
# and Ob (baryons) are -- so it is derived as Om - Ob.
COSMO_Z_FIELDS = ("h", "omega_cdm", "omega_b", "omega_m", "ns", "sigma8", "w0", "z")


def cosmo_z_vector(cosmo, z):
    """cosmo: (...,6) tensor/array in COSMO_FIELDS order (Om,Ob,ns,s8,w0,h). z: (...,)
    redshift. Returns (...,8) in COSMO_Z_FIELDS order: [h, Omega_cdm, Ob, Om, ns, s8,
    w0, z]. The ONE place this ordering is defined -- train_flow.py and apply_flow.py
    both call this instead of re-deriving the field order themselves, so training and
    evaluation can never silently drift apart."""
    om, ob, ns, s8, w0, h = (cosmo[..., i] for i in range(6))
    omega_cdm = om - ob
    if torch.is_tensor(cosmo):
        return torch.stack([h, omega_cdm, ob, om, ns, s8, w0, z], dim=-1)
    return np.stack([h, omega_cdm, ob, om, ns, s8, w0, z], axis=-1)


def raw_to_log1p_delta_pair(low_raw: torch.Tensor, high_raw: torch.Tensor):
    """Batched GPU version of the per-sample transform previously done in
    __getitem__. low_raw/high_raw: (B, 1, H, W) raw counts. Returns
    (low_log, high_log), same log1p(overdensity) representation as before,
    identical semantics to dataset.py's old per-sample log1p_delta()."""
    low_mean = low_raw.mean(dim=(2, 3), keepdim=True)
    high_mean = high_raw.mean(dim=(2, 3), keepdim=True)
    eps = 0.5 / torch.minimum(low_mean, high_mean)

    low_delta = low_raw / low_mean - 1.0
    high_delta = high_raw / high_mean - 1.0

    low_log = torch.log1p(torch.maximum(low_delta, -1.0 + eps))
    high_log = torch.log1p(torch.maximum(high_delta, -1.0 + eps))
    return low_log, high_log


def raw_to_delta_pair(low_raw: torch.Tensor, high_raw: torch.Tensor):
    """LINEAR overdensity delta = n/<n> - 1 for both maps -- the space the SCIENCE
    metric lives in (analysis.full_sky.od_cl computes the angular power spectrum of
    exactly this field). Ported from diffusion/dataset.py's identical fix (2026-07-18
    finding, see that module's docstring): log1p compresses density peaks, so
    MEASURED in log space DISCO already looks ~correct (low/high power ratio
    0.93-1.05) while linear delta shows the real deficit (down to 0.62 at high k) --
    a model trained on the log-space residual optimizes a statistic that's nearly
    already right and barely moves the one actually evaluated. Also the precondition
    the high-pass residual formulation (flow_model.residual_target/compose_corrected)
    depends on: "large scales are already correct so pin them" was only verified in
    THIS space, not log1p's compressed one."""
    low_mean = low_raw.mean(dim=(2, 3), keepdim=True)
    high_mean = high_raw.mean(dim=(2, 3), keepdim=True)
    return low_raw / low_mean - 1.0, high_raw / high_mean - 1.0


def transform_pair(low_raw: torch.Tensor, high_raw: torch.Tensor, space: str):
    """(low, high) raw counts -> the pair of fields the model works in. THE single
    dispatch point for `space`, so train and apply cannot drift."""
    if space == "delta":
        return raw_to_delta_pair(low_raw, high_raw)
    if space == "log1p":
        return raw_to_log1p_delta_pair(low_raw, high_raw)
    raise ValueError(f"unknown space {space!r} (expected 'delta' or 'log1p')")


def low_to_field(low_raw: torch.Tensor, space: str):
    """Full-sky/inference counterpart of transform_pair when only the LOW map exists.
    Returns (field, low_mean) so the caller can invert with field_to_counts."""
    low_mean = low_raw.mean(dim=(2, 3), keepdim=True)
    if space == "delta":
        return low_raw / low_mean - 1.0, low_mean
    if space == "log1p":
        eps = 0.5 / low_mean
        return torch.log1p(torch.maximum(low_raw / low_mean - 1.0, -1.0 + eps)), low_mean
    raise ValueError(f"unknown space {space!r} (expected 'delta' or 'log1p')")


def field_to_counts(field: torch.Tensor, low_mean: torch.Tensor, space: str):
    """Inverse of low_to_field: model-space field -> raw counts, using the LOW map's
    mean (the only mean available at inference)."""
    delta = field if space == "delta" else torch.expm1(field)
    return (1.0 + delta) * low_mean


class PatchDataset(Dataset):
    def __init__(self, patch_dir: str | Path, indices: np.ndarray):
        d = Path(patch_dir)
        # mmap - patches are only paged in as __getitem__ touches them
        self.low = np.load(d / "low.npy", mmap_mode="r")
        self.high = np.load(d / "high.npy", mmap_mode="r")
        self.meta = np.load(d / "metadata.npy")
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        m = self.meta[idx]
        cosmo = np.array([m[f] for f in COSMO_FIELDS], dtype=np.float32)

        return {
            "low": torch.from_numpy(np.array(self.low[idx], dtype=np.float32)).unsqueeze(0),
            "high": torch.from_numpy(np.array(self.high[idx], dtype=np.float32)).unsqueeze(0),
            "reso_arcmin": float(m["reso_arcmin"]),
            "cosmo": torch.from_numpy(cosmo),               # (Om,Ob,ns,s8,w0,h) -- see
                                                             # cosmo_z_vector() for how
                                                             # FlowUNet actually consumes it
            "z": float(0.5 * (m["lower_z"] + m["upper_z"])),  # shell midpoint redshift
            "shell_idx": int(m["shell_idx"]),
            "shell_com": float(m["shell_com"]),  # comoving distance (Mpc/h) - drives the HPF cutoff
            "idx": idx,
        }


def split_by_cosmo(patch_dir: str | Path, val_frac: float = 0.15, seed: int = 0):
    """Returns (train_indices, val_indices) into low.npy/high.npy, splitting whole
    cosmologies between train and val so no cosmology appears in both.

    The LAST lightcone shell is excluded from BOTH splits (2026-07-22): DISCO's low
    map there carries only 16-65% of CosmoGrid's true counts (a truncated shell at
    the lightcone/box edge -- the same data-quality finding that already excludes it
    from the full-sky eval, see apply_*'s n_shells_total -= 1). Trained on anyway,
    those pairs teach "subtract high-ell power at z~3.4" (the count deficit reads as
    excess shot noise in delta space), and the z-conditioning cannot separate shell
    68 (z 3.37-3.50) from shell 67 (z 3.24-3.46) -- measured result: corrected/true
    Cl ~ 0.41-0.53 at shell 67 for ALL 30 held-out cosmologies (the panel-3
    percentile lobe in the hpc0.05_hpt0.12 e200 run), on a shell whose input was
    fine (low/true ~ 0.998)."""
    meta = np.load(Path(patch_dir) / "metadata.npy")
    keep = meta["shell_idx"] < meta["shell_idx"].max()
    cosmos = np.unique(meta["cosmo"])
    rng = np.random.default_rng(seed)
    rng.shuffle(cosmos)
    n_val = max(1, int(round(len(cosmos) * val_frac)))
    val_cosmos = set(cosmos[:n_val])
    is_val = np.isin(meta["cosmo"], list(val_cosmos))
    val_idx = np.where(is_val & keep)[0]
    train_idx = np.where(~is_val & keep)[0]
    return train_idx, val_idx, sorted(val_cosmos)
