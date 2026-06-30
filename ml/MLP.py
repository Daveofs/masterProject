"""HEALPix Patch-Based Flow Matching Model for pixel-space small-scale correction.

Motivation
----------
The SH-space MLP (MLP.py) operates on global alm coefficients and struggles to
correct small-scale power because:
  1. High-ell alms mix contributions from the whole sky — a local patch error
     contaminates hundreds of alm coefficients simultaneously.
  2. The per-ell architecture has no spatial locality inductive bias.

This model instead operates in pixel space, extracting local HEALPix patches
(a central pixel + its nested neighbors up to some depth), running a shared
MLP over each patch, and predicting the velocity field pixel-by-pixel.

Key design choices
------------------
- Uses hp.get_all_neighbours() recursively to build a spatially contiguous
  patch around each pixel. For nside=2048, a depth-2 neighborhood contains
  ~49 pixels (~0.07 deg^2) — capturing sub-arcminute structure.
- Processes patches in large batches (100k+) to saturate GPU memory bandwidth.
- Conditioning (cosmo params + shell index) is appended to every patch vector,
  same interface as MLP.py.
- Weight sharing: the same MLP is applied to ALL patches on the sky, giving
  full translation-equivariance on the sphere (modulo HEALPix discretization).

Memory scaling
--------------
nside=2048: Npix = 50,331,648 pixels/shell
  Patch size p=7 (depth 1):  input dim = 7 + 7 + C,   ~350 MB/shell in float32
  Patch size p=49 (depth 2): input dim = 49 + 49 + C, ~2.4 GB/shell  (process in chunks)
  Patch size p=7, chunk=1M pixels: ~28 MB working memory — fully feasible.

nside=8192 (0.5B pixels): Npix = 805,306,368
  Process in chunks of 1M pixels, each chunk ~28 MB — feasible with chunked inference.

Usage
-----
  model = PatchMLP(patch_size=7, cond_dim=12, hidden=256)
  # x shape: (Npix,) float32 density map
  # corrected = apply_patch_flow(model, x_low, x_high, cond, nside, steps=25)
"""

from __future__ import annotations
import contextlib
import math
from typing import Optional

import numpy as np
import healpy as hp
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Patch index builder
# ---------------------------------------------------------------------------

def build_patch_indices(nside: int, depth: int = 1) -> np.ndarray:
    """
    Build a (Npix, patch_size) integer array: for each pixel, the indices of
    its local neighborhood (itself + neighbors up to `depth` hops).

    Uses HEALPix ring ordering. Boundary pixels (neighbors == -1) are clamped
    to the pixel itself (zero-padding equivalent on the sphere).

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    depth : int
        Neighborhood depth.
        depth=1 → 1 + 8 = 9 pixels (including self and 8 neighbors, clamped for boundary)
        depth=2 → up to 25 pixels
        In practice fewer unique neighbors due to HEALPix topology.

    Returns
    -------
    patch_idx : np.ndarray of shape (Npix, patch_size), dtype=int32
    """
    Npix = hp.nside2npix(nside)
    print(f"[PatchIndex] Building depth-{depth} patches for nside={nside} (Npix={Npix:,})...")

    # Start with center pixels
    current_ring = {i: {i} for i in range(Npix)}  # pixel -> set of patch members

    all_indices_per_pixel: list[set] = [set() for _ in range(Npix)]
    for i in range(Npix):
        all_indices_per_pixel[i].add(i)

    frontier = list(range(Npix))
    for d in range(depth):
        print(f"  Depth {d+1}/{depth}...")
        # For each pixel in frontier, fetch its neighbors
        # hp.get_all_neighbours returns shape (8,) per pixel
        # We process in bulk
        new_frontier_sets = [set() for _ in range(Npix)]
        # batch neighbor lookup
        chunk = 100_000
        for start in range(0, Npix, chunk):
            end = min(start + chunk, Npix)
            pix_chunk = np.arange(start, end)
            nbrs = hp.get_all_neighbours(nside, pix_chunk)  # (8, chunk)
            for local_i, pix in enumerate(pix_chunk):
                new_nbrs = nbrs[:, local_i]
                valid = new_nbrs[new_nbrs >= 0]
                new_members = valid[~np.isin(valid, list(all_indices_per_pixel[pix]))]
                all_indices_per_pixel[pix].update(new_members.tolist())
                new_frontier_sets[pix].update(new_members.tolist())

    # Find max patch size (should be uniform except near poles)
    patch_sizes = [len(s) for s in all_indices_per_pixel]
    max_patch = max(patch_sizes)
    print(f"  Patch sizes: min={min(patch_sizes)}, max={max_patch}, "
          f"median={int(np.median(patch_sizes))}")

    # Pack into array, clamping boundary pixels to center pixel
    patch_idx = np.zeros((Npix, max_patch), dtype=np.int32)
    for i, nbr_set in enumerate(all_indices_per_pixel):
        nbr_list = sorted(nbr_set)
        patch_idx[i, :len(nbr_list)] = nbr_list
        # Fill remaining with self (clamp/pad)
        patch_idx[i, len(nbr_list):] = i

    print(f"  Done. patch_idx shape: {patch_idx.shape}")
    return patch_idx


def build_patch_indices_fast(nside: int, depth: int = 1) -> np.ndarray:
    """
    Faster version using vectorized BFS over HEALPix neighbor graph.
    Returns (Npix, fixed_patch_size) index array.

    For depth=1: patch_size = 9  (self + 8 neighbors, duplicates=self for missing)
    For depth=2: patch_size = 25 (approximate — BFS up to 2 hops)
    """
    Npix = hp.nside2npix(nside)

    # Depth-1: just use direct neighbor lookup
    if depth == 1:
        all_pix = np.arange(Npix)
        nbrs = hp.get_all_neighbours(nside, all_pix)  # (8, Npix)
        # Replace -1 (missing neighbor) with the pixel itself (self-padding)
        for d in range(8):
            mask = nbrs[d] < 0
            nbrs[d, mask] = all_pix[mask]
        # patch = [self, n0, n1, ..., n7]  shape (Npix, 9)
        patch_idx = np.concatenate(
            [all_pix[:, None], nbrs.T], axis=1
        ).astype(np.int32)
        return patch_idx

    # Depth >= 2: iterative BFS
    # Start from depth-1
    patch_idx_d1 = build_patch_indices_fast(nside, depth=1)  # (Npix, 9)

    # For each pixel, collect unique neighbors of its depth-1 set
    patch_sets = [set(row.tolist()) for row in patch_idx_d1]
    for _ in range(depth - 1):
        new_sets = []
        chunk = 50_000
        all_pix_arr = np.arange(Npix)
        for start in range(0, Npix, chunk):
            end = min(start + chunk, Npix)
            pix_chunk = all_pix_arr[start:end]
            nbrs = hp.get_all_neighbours(nside, pix_chunk)  # (8, chunk_size)
            for local_i, pix in enumerate(pix_chunk):
                new_set = set(patch_sets[pix])
                for d in range(8):
                    n = nbrs[d, local_i]
                    if n >= 0:
                        new_set.add(int(n))
                new_sets.append(new_set)
        patch_sets[start:end] = new_sets  # type: ignore

    max_patch = max(len(s) for s in patch_sets)
    patch_idx = np.zeros((Npix, max_patch), dtype=np.int32)
    for i, s in enumerate(patch_sets):
        lst = sorted(s)
        patch_idx[i, :len(lst)] = lst
        patch_idx[i, len(lst):] = i
    return patch_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PatchMLP(nn.Module):
    """
    Shared-weight patch MLP for HEALPix flow matching in pixel space.

    For each pixel i, the input is:
        [x_low[patch_i],        # patch_size floats: low-res density values
         x_curr[patch_i],       # patch_size floats: current state (interpolated)
         t,                     # scalar: flow time in [0, 1]
         cond]                  # C floats: cosmo params + shell index

    Output: scalar velocity field correction at pixel i.

    The model is applied identically to every pixel (weight sharing), so it
    generalizes across sky positions and can be applied in chunks without
    reloading weights.

    Parameters
    ----------
    patch_size : int
        Number of pixels in each local patch (including center).
    cond_dim : int
        Dimension of conditioning vector (cosmo params + 1 shell index scalar).
    hidden : int
        Hidden layer width.
    n_layers : int
        Depth of MLP (default: 4 gives a good balance for small-scale detail).
    """

    def __init__(
        self,
        patch_size: int = 9,
        cond_dim: int = 12,
        hidden: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cond_dim = cond_dim

        # Input: (low_patch, curr_patch, t, cond)
        # We feed both the original low-res patch AND the current ODE state
        # so the model can learn residual corrections on top of the current trajectory
        dim_in = 2 * patch_size + 1 + cond_dim

        layers: list[nn.Module] = [nn.Linear(dim_in, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]  # predict scalar velocity per pixel

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_low_patches: torch.Tensor,   # (B, patch_size)
        x_curr_patches: torch.Tensor,  # (B, patch_size)
        t: torch.Tensor,               # (B,) or (B, 1)
        cond: torch.Tensor,            # (B, cond_dim)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_low_patches : (B, patch_size) — low-res input density patches
        x_curr_patches : (B, patch_size) — current ODE state patches
        t : (B,) — flow time
        cond : (B, cond_dim) — conditioning

        Returns
        -------
        velocity : (B,) — scalar velocity at the center pixel of each patch
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (B, 1)
        inp = torch.cat([x_low_patches, x_curr_patches, t, cond], dim=-1)
        return self.net(inp).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Dataset utility: precompute .npy patch-index files per nside
# ---------------------------------------------------------------------------

def get_or_build_patch_idx(
    nside: int,
    depth: int = 1,
    cache_dir: str = "/tmp/healpy_patch_cache",
) -> np.ndarray:
    """
    Load patch indices from disk cache or build and cache them.

    For nside=2048 depth=1: ~200 MB on disk, ~200 MB RAM.
    For nside=2048 depth=2: ~1.1 GB on disk.

    The cache is worth building once — it's reused across all training epochs.
    """
    import os
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"patch_idx_nside{nside}_depth{depth}.npy")
    if os.path.exists(cache_path):
        print(f"[PatchIndex] Loading cached patch indices from {cache_path}")
        return np.load(cache_path)
    patch_idx = build_patch_indices_fast(nside, depth=depth)
    np.save(cache_path, patch_idx)
    print(f"[PatchIndex] Saved patch indices to {cache_path}")
    return patch_idx


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def patch_flow_loss(model, x0, x1, cond, patch_idx_t, t, sigma, chunk_size, device,
                    use_amp: bool = False):
    npix = x0.shape[0]

    with torch.no_grad():
        noise = torch.randn_like(x0)
        xt = (1 - t) * x0 + t * x1 + sigma * noise
        target = x1 - x0  # velocity field

    chunk_starts = list(range(0, npix, chunk_size))
    n_chunks = len(chunk_starts)
    total_loss = 0.0  # plain float, NOT a tensor

    # In DDP, no_sync() suppresses the allreduce on every intermediate backward.
    # Only the final chunk triggers the gradient sync — reduces network traffic
    # from O(n_chunks) allreduces to O(1) per shell.
    no_sync_ctx = model.no_sync if hasattr(model, 'no_sync') else contextlib.nullcontext

    for ci, i in enumerate(chunk_starts):
        end = min(i + chunk_size, npix)
        idx = patch_idx_t[i:end]                          # (C, patch_size)

        patches_x0 = x0[idx]                              # (C, patch_size) — low-res input
        patches_xt = xt[idx]                              # (C, patch_size) — current ODE state
        target_chunk = target[i:end]                      # (C,)

        t_vec = torch.full((end - i, 1), t, device=device)
        cond_expand = cond.unsqueeze(0).expand(end - i, -1)

        # BF16 autocast: halves activation memory → allows larger chunk_size.
        # No GradScaler needed — BF16 has the same exponent range as FP32.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(patches_x0, patches_xt, t_vec, cond_expand)

        chunk_loss = ((pred.float() - target_chunk) ** 2).mean() / n_chunks

        is_last = (ci == n_chunks - 1)
        ctx = contextlib.nullcontext() if is_last else no_sync_ctx()
        with ctx:
            chunk_loss.backward()
        total_loss += chunk_loss.item()

    return total_loss  # return float for logging



# ---------------------------------------------------------------------------
# Inference: ODE integration in pixel space
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_patch_flow(
    model: PatchMLP,
    x_low: np.ndarray,          # (Npix,) float32 input (low-res) shell
    cond: torch.Tensor,         # (C,) conditioning
    patch_idx: np.ndarray,      # (Npix, patch_size) int32
    steps: int = 25,
    chunk_size: int = 1_000_000,
    device: torch.device = torch.device("cpu"),
    diagnostic: bool = False,
) -> np.ndarray:
    """
    Euler integration of the learned velocity field in pixel space.

    Parameters
    ----------
    x_low : (Npix,) float32 — low-res input density map
    cond : (C,) — conditioning vector (already normalized)
    patch_idx : (Npix, patch_size) int32 — precomputed patch indices
    steps : int — Euler integration steps
    chunk_size : int — pixels per GPU batch

    Returns
    -------
    x_corrected : (Npix,) float32 — corrected density map
    """
    Npix = x_low.shape[0]
    patch_idx_t = torch.from_numpy(patch_idx).long().to(device)

    x0_t = torch.from_numpy(x_low.astype(np.float32)).to(device)
    x_curr = x0_t.clone()

    dt = 1.0 / steps
    cond_exp = cond.unsqueeze(0)  # (1, C)

    for step in range(steps):
        t_val = step * dt
        v_field = torch.zeros(Npix, dtype=torch.float32, device=device)

        for start in range(0, Npix, chunk_size):
            end = min(start + chunk_size, Npix)
            B = end - start

            patch_rows = patch_idx_t[start:end]          # (B, patch_size)
            x0_patches   = x0_t[patch_rows]              # (B, patch_size)
            xcurr_patches = x_curr[patch_rows]           # (B, patch_size)

            t_chunk = torch.full((B,), t_val, dtype=torch.float32, device=device)
            cond_chunk = cond_exp.expand(B, -1)

            v_chunk = model(x0_patches, xcurr_patches, t_chunk, cond_chunk)
            v_field[start:end] = v_chunk

        if diagnostic and step % 5 == 0:
            print(f"  Step {step:02d}/{steps} | "
                  f"x_curr mean={x_curr.mean():.4f} std={x_curr.std():.4f} | "
                  f"v mean={v_field.mean():.4f} std={v_field.std():.4f}")

        x_curr = x_curr + v_field * dt

    return x_curr.cpu().numpy()


# ---------------------------------------------------------------------------
# ShellPixelDataset: pixel-space analogue of ShellAlmDataset
# ---------------------------------------------------------------------------

class ShellPixelDataset(torch.utils.data.Dataset):
    """
    Dataset that streams HEALPix shells from .npz files in pixel space.

    Each item is (x0_shell, x1_shell, cond) where x0/x1 are full pixel maps
    (Npix,) float32. The DataLoader is responsible for computing patches on-the-fly
    during the forward pass, which avoids storing all patches in RAM.

    This dataset is designed for shells_nside=2048.npz (low-res) and
    compressed_shells.npz (high-res reference).

    Parameters
    ----------
    data_dir : Path
        Root directory with cosmo_* subdirectories.
    nside_target : int
        Target nside for the output maps (used to downgrade high-res reference
        if needed).
    """

    def __init__(
        self,
        data_dir,
        nside_target: int = 2048,
        verbose: bool = True,
    ):
        from pathlib import Path
        import yaml

        data_dir = Path(data_dir)
        self.nside_target = nside_target

        self.low_paths: list[tuple[Path, int]] = []  # (npz_path, shell_idx)
        self.high_paths: list[tuple[Path, int]] = []
        self.cond_list: list[np.ndarray] = []

        subdirs = sorted(d for d in data_dir.iterdir()
                         if d.is_dir() and d.name.startswith("cosmo_"))
        if not subdirs:
            subdirs = [data_dir]

        def load_clean_params(p):
            params = yaml.safe_load(p.read_text())
            valid_keys = sorted(k for k, v in params.items()
                                if _try_float(v) is not None)
            return np.array([float(params[k]) for k in valid_keys], dtype=np.float32)

        def _try_float(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        total = 0
        for sd in subdirs:
            run_dirs = [r for r in sorted(sd.iterdir())
                        if r.is_dir() and r.name.startswith("run_")]
            leaf_dirs = run_dirs if run_dirs else [sd]
            for ld in leaf_dirs:
                params_yml = ld / "params.yml" if (ld / "params.yml").exists() \
                             else ld.parent / "params.yml"
                low_npz = ld / "shells_nside=2048.npz"
                high_npz = ld / "compressed_shells.npz"
                if not (params_yml.exists() and low_npz.exists() and high_npz.exists()):
                    continue
                cosmo_vec = load_clean_params(params_yml)

                # Peek at shell count
                low_data = np.load(low_npz, mmap_mode='r')
                high_data = np.load(high_npz, mmap_mode='r')
                n_shells = min(low_data["shells"].shape[0], high_data["shells"].shape[0])

                for i in range(n_shells):
                    self.low_paths.append((low_npz, i))
                    self.high_paths.append((high_npz, i))
                    self.cond_list.append(cosmo_vec)
                    total += 1

        assert total > 0, "No shells found!"
        if verbose:
            print(f"[ShellPixelDataset] {total} shells indexed across {len(subdirs)} cosmologies")

        # Normalize conditioning
        cond_arr = np.stack(self.cond_list)
        self.cond_mean = cond_arr.mean(0)
        self.cond_std = np.clip(cond_arr.std(0), 1e-8, None)
        self.cond_norm = ((cond_arr - self.cond_mean) / self.cond_std).astype(np.float32)

        # Shell index normalization
        self.max_shell_idx = float(max(
            [idx for _, idx in self.low_paths] + [1]
        ))

        # Per-worker mmap cache; populated lazily after DataLoader fork.
        self._npz_cache: dict = {}

    def _load_npz(self, path):
        # No mmap_mode: compressed npz files can't be mmapped anyway, and keeping
        # a ZipFile handle open under concurrent Lustre reads causes CRC errors.
        key = str(path)
        if key not in self._npz_cache:
            self._npz_cache[key] = np.load(str(path))
        return self._npz_cache[key]

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_npz, shell_i = self.low_paths[idx]
        high_npz, _ = self.high_paths[idx]

        low_data = self._load_npz(low_npz)
        high_data = self._load_npz(high_npz)

        x0 = low_data["shells"][shell_i].astype(np.float32)
        x1 = high_data["shells"][shell_i].astype(np.float32)

        # Downgrade x1 if it's at higher resolution
        nside_x1 = hp.npix2nside(x1.shape[0])
        if nside_x1 != self.nside_target:
            x1 = hp.ud_grade(x1, self.nside_target).astype(np.float32)

        cond_vec = self.cond_norm[idx]
        shell_idx_norm = np.float32(shell_i / self.max_shell_idx)
        cond = np.concatenate([cond_vec, [shell_idx_norm]])

        return torch.from_numpy(x0), torch.from_numpy(x1), torch.from_numpy(cond)