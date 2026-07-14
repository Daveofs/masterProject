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
    cosmologies between train and val so no cosmology appears in both."""
    meta = np.load(Path(patch_dir) / "metadata.npy")
    cosmos = np.unique(meta["cosmo"])
    rng = np.random.default_rng(seed)
    rng.shuffle(cosmos)
    n_val = max(1, int(round(len(cosmos) * val_frac)))
    val_cosmos = set(cosmos[:n_val])
    is_val = np.isin(meta["cosmo"], list(val_cosmos))
    val_idx = np.where(is_val)[0]
    train_idx = np.where(~is_val)[0]
    return train_idx, val_idx, sorted(val_cosmos)
