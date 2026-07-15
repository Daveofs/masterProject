"""SpherePatchDataset: (low, high) HEALPix-superpixel patch pairs from
make_patch_dataset.py. Direct analogue of unet/dataset.py.

Returns *raw* counts plus each patch's shell-global means, NOT the arcsinh
signal the model trains on -- that transform is applied batched, on GPU, in the
training loop (raw_to_signal_pair below) rather than per-sample in __getitem__,
for the same reason unet/dataset.py gives: doing it per-sample in NumPy inside
worker processes is data-loading-bound, not compute-bound.

Train/val split is BY COSMOLOGY (split_by_cosmo), not by patch index: patches
from the same cosmology are highly correlated at large scales, so a random
per-patch split would leak the validation cosmologies into training.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def signal_forward_torch(delta: torch.Tensor, scale: float, softening: float = 1.0,
                         clip: float = 8.0) -> torch.Tensor:
    """delta -> normalized network signal y = clip(arcsinh(delta/soft)/scale).

    Torch/GPU/batched mirror of sphere_flow.signal_forward (numpy). Same formula
    and same default clip -- they must agree or the model is trained on one
    representation and applied on another. sphere_flow.signal_inverse is the
    exact inverse and is what apply_sphere_flow uses to get back to a map.
    """
    y = torch.asinh(delta / softening) / scale
    if clip:
        y = torch.clamp(y, -clip, clip)
    return y


def raw_to_signal_pair(low_raw: torch.Tensor, high_raw: torch.Tensor,
                       low_shell_mean: torch.Tensor, high_shell_mean: torch.Tensor,
                       sig_scale: float, softening: float = 1.0):
    """Batched GPU transform: raw patch counts -> (cond, x1) arcsinh signals.

    low_raw/high_raw: (B, M) raw counts. low/high_shell_mean: (B,) the mean of
    the WHOLE shell each patch came from -- NOT the patch's own mean. This
    matches sphere_flow.to_overdensity (which means over the full shell) and
    apply_sphere_flow.correct_shell (which divides by the full input map's
    mean); normalizing by a patch-local mean instead would train the model on a
    different field than it is applied to.
    """
    lo_m = low_shell_mean[:, None].to(low_raw.dtype)
    hi_m = high_shell_mean[:, None].to(high_raw.dtype)
    d_low = low_raw / lo_m - 1.0
    d_high = high_raw / hi_m - 1.0
    return (signal_forward_torch(d_low, sig_scale, softening),
            signal_forward_torch(d_high, sig_scale, softening))


class SpherePatchDataset(Dataset):
    def __init__(self, patch_dir: str | Path, indices: np.ndarray):
        d = Path(patch_dir)
        # mmap -- patches are paged in only as __getitem__ touches them
        self.low = np.load(d / "low.npy", mmap_mode="r")
        self.high = np.load(d / "high.npy", mmap_mode="r")
        self.meta = np.load(d / "metadata.npy")
        self.cosmo = np.load(d / "cosmo.npy")
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        m = self.meta[idx]
        # Normalized shell index -- tells the model which redshift/shot-noise
        # regime it is in (one model serves faint and dense shells).
        shell_norm = np.float32(int(m["shell_idx"]) / max(int(m["n_shells"]) - 1, 1))
        return {
            "low": torch.from_numpy(np.array(self.low[idx], dtype=np.float32)),
            "high": torch.from_numpy(np.array(self.high[idx], dtype=np.float32)),
            "low_shell_mean": torch.tensor(float(m["low_shell_mean"]), dtype=torch.float32),
            "high_shell_mean": torch.tensor(float(m["high_shell_mean"]), dtype=torch.float32),
            "cosmo": torch.from_numpy(np.asarray(self.cosmo[idx], dtype=np.float32)),
            "shell_norm": torch.tensor(shell_norm, dtype=torch.float32),
            "shell_idx": int(m["shell_idx"]),
            "idx": idx,
        }


def split_by_cosmo(patch_dir: str | Path, val_frac: float = 0.15, seed: int = 0,
                   val_cosmos: list[str] | None = None):
    """(train_idx, val_idx, val_cosmos) -- whole cosmologies go to train or val,
    never both.

    DELIBERATE DUPLICATE of unet/dataset.py's split_by_cosmo (not an import):
    ml/'s pipeline directories are kept independently runnable, so a rename in
    one must not silently break another (see also make_patch_dataset.cosmo_vector).
    Same default val_frac=0.15 as unet and as transfer_function.split_val_cosmos.

    val_cosmos: pin the held-out set explicitly instead of drawing it here (used
    to reproduce an exact split, or to match another pipeline's held-out set).
    """
    meta = np.load(Path(patch_dir) / "metadata.npy")
    all_cosmos = np.unique(meta["cosmo"])
    if val_cosmos is None:
        cosmos = all_cosmos.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(cosmos)
        n_val = max(1, int(round(len(cosmos) * val_frac)))
        val_cosmos = sorted(cosmos[:n_val])
    else:
        missing = sorted(set(val_cosmos) - set(all_cosmos.tolist()))
        if missing:
            raise SystemExit(f"--test-cosmos not present in the patch dataset: {missing}")
    is_val = np.isin(meta["cosmo"], list(val_cosmos))
    return np.where(~is_val)[0], np.where(is_val)[0], sorted(val_cosmos)


def estimate_sig_scale(patch_dir: str | Path, indices: np.ndarray, n_sample: int = 512,
                       softening: float = 1.0, seed: int = 0) -> float:
    """sig_scale = std(arcsinh(delta_low)) over a sample of TRAIN patches.

    The arcsinh signal is divided by this so flow-matching targets are O(1).
    Estimated from the train split only (never validation), and on the same
    shell-global-mean overdensity the model actually sees.
    """
    d = Path(patch_dir)
    low = np.load(d / "low.npy", mmap_mode="r")
    meta = np.load(d / "metadata.npy")
    rng = np.random.default_rng(seed)
    pick = rng.choice(indices, size=min(n_sample, len(indices)), replace=False)
    vals = []
    for i in pick:
        i = int(i)
        m = float(meta[i]["low_shell_mean"]) or 1.0
        delta = np.asarray(low[i], dtype=np.float64) / m - 1.0
        vals.append(np.arcsinh(delta / softening))
    return float(np.std(np.concatenate(vals)) + 1e-12)
