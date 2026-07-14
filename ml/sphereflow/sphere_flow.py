"""Generative flow matching with DeepSphere spherical convolutions (PyTorch).

Why this exists
---------------
A deterministic L2 map->map regressor (e.g. deepsphere/cgcnn with loss='l2')
provably SMOOTHS: the MSE-optimal output is the conditional mean, which suppresses
the stochastic small-scale power we want to restore. Small-scale structure in the
high-res shells is not a deterministic function of the low-res input, so it must be
*generated* with the right statistics, not regressed.

This module keeps DeepSphere's sphere-aware architecture (Chebyshev graph
convolutions on the HEALPix graph) but trains it as a **conditional flow-matching**
generator:

    x1 = delta_high        (per-shell overdensity of the high-res target)
    x0 ~ N(0, I)           (noise)
    cond = delta_low       (the low-res map, as a conditioning channel)
    xt = (1 - t) x0 + t x1
    target velocity  v* = x1 - x0
    train  v_theta(xt, t, cond, cosmo)  to match v*   (MSE on the velocity)

At inference we sample x0 ~ N(0, I) and integrate dx/dt = v_theta from t=0..1 to
draw a high-res realization; different noise -> different small-scale detail with
the learned statistics. Conditioning on delta_low ties the large scales to the
input.

The Chebyshev conv is a faithful PyTorch port of deepsphere.models.cgcnn.chebyshev5
(same lmax rescale to [-scale, scale], same Chebyshev recurrence). The HEALPix
graph Laplacian is built by deepsphere.utils (pure numpy/scipy — no TensorFlow).

High nside: the sphere is split into 12*order^2 patches of (nside/order)^2 pixels
(NESTED contiguous superpixels); the Laplacian is built once on one patch and
shared. This keeps nside=2048 tractable and preserves small scales inside patches.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import healpy as hp
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch import nn


# ---------------------------------------------------------------------------
# HEALPix graph Laplacian (self-contained numpy/scipy; same recipe as DeepSphere)
# ---------------------------------------------------------------------------

def n_patches(order: int) -> int:
    return 12 * order * order


def _healpix_weightmatrix(nside: int, npix_patch: int, nest: bool = True,
                          dtype=np.float32) -> sp.csr_matrix:
    """Gaussian-weighted adjacency over the first ``npix_patch`` NESTED pixels
    (a contiguous HEALPix superpixel). Same construction as
    deepsphere.utils.healpix_weightmatrix (fast/consecutive-index path)."""
    indexes = np.arange(npix_patch)
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.asarray(np.vstack([x, y, z]).T, dtype=dtype)
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)  # (8, npix)
    col = neighbors.T.reshape(npix_patch * 8)
    row = np.repeat(indexes, 8)
    keep = (col < npix_patch) & (col >= 0)          # drop cross-patch / missing
    col, row = col[keep], row[keep]
    dist = np.sum((coords[row] - coords[col]) ** 2, axis=1)
    weights = np.exp(-dist / (2 * np.mean(dist)))
    return sp.csr_matrix((weights, (row, col)), shape=(npix_patch, npix_patch), dtype=dtype)


def patch_npix(nside: int, order: int) -> int:
    """Pixels per graph: full sphere for order<=1, else (nside/order)^2 per patch."""
    return hp.nside2npix(nside) if order <= 1 else hp.nside2npix(nside) // n_patches(order)


def healpix_laplacian(nside: int, order: int = 1, nest: bool = True) -> sp.csr_matrix:
    """Normalized HEALPix graph Laplacian. order<=1 -> full sphere; order>1 ->
    Laplacian of one patch of (nside/order)^2 pixels (shared by all patches)."""
    W = _healpix_weightmatrix(nside, patch_npix(nside, order), nest=nest)
    d = np.ravel(W.sum(1))
    d12 = np.power(d, -0.5, where=d > 0)
    D12 = sp.diags(d12, 0, dtype=W.dtype).tocsc()
    return (sp.identity(W.shape[0], dtype=W.dtype) - D12 * W * D12).tocsr()


def prepare_laplacian(L: sp.spmatrix, lmax: Optional[float] = None,
                      scale: float = 0.75) -> torch.Tensor:
    """Rescale L to [-scale, scale] (as chebyshev5 does) and return a torch sparse
    COO tensor for torch.sparse.mm."""
    L = sp.csr_matrix(L, copy=True)
    if lmax is None:
        lmax = 1.02 * eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    M = L.shape[0]
    L = L * (2.0 * scale / lmax) - sp.identity(M, format="csr", dtype=L.dtype) * scale
    L = L.tocoo()
    idx = torch.from_numpy(np.vstack([L.row, L.col]).astype(np.int64))
    val = torch.from_numpy(L.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, (M, M)).coalesce()


def laplacian_to_gather(L: sp.spmatrix, lmax: Optional[float] = None,
                        scale: float = 0.75):
    """Rescaled Laplacian as (idx, w) gather tensors for a FAST dense conv.

    The HEALPix graph Laplacian has <=9 nonzeros per row (8 neighbors + self), so
    ``L @ x`` can be computed as a dense neighbor gather + weighted sum:

        (L x)[i] = sum_k  w[i, k] * x[idx[i, k]]

    Unlike torch.sparse.mm this is a coalesced dense op: it runs under bf16
    autocast and is ~an order of magnitude faster on GPU. Rows with fewer
    neighbors are padded with (idx=0, w=0).

    Returns
    -------
    idx : torch.LongTensor (M, nmax)   neighbor indices per node
    w   : torch.FloatTensor (M, nmax)  rescaled Laplacian weights per neighbor
    """
    L = sp.csr_matrix(L, copy=True)
    if lmax is None:
        lmax = 1.02 * eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    M = L.shape[0]
    L = (L * (2.0 * scale / lmax)
         - sp.identity(M, format="csr", dtype=L.dtype) * scale).tocsr()
    counts = np.diff(L.indptr)
    nmax = int(counts.max())
    idx = np.zeros((M, nmax), dtype=np.int64)
    w = np.zeros((M, nmax), dtype=np.float32)
    rows = np.repeat(np.arange(M), counts)
    pos = np.arange(L.nnz) - np.repeat(L.indptr[:-1], counts)
    idx[rows, pos] = L.indices
    w[rows, pos] = L.data
    return torch.from_numpy(idx), torch.from_numpy(w)


# ---------------------------------------------------------------------------
# Chebyshev graph convolution (port of deepsphere chebyshev5)
# ---------------------------------------------------------------------------

class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution of order K on a fixed graph.

    The Laplacian is applied via a dense neighbor GATHER (see laplacian_to_gather)
    instead of torch.sparse.mm: same math, ~10x faster on GPU, and it supports
    bf16 autocast (torch.sparse.mm has no bf16 CUDA kernel).
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super().__init__()
        self.in_channels, self.out_channels, self.K = in_channels, out_channels, K
        self.weight = nn.Parameter(torch.empty(in_channels * K, out_channels))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(in_channels * (K + 0.5) / 2))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, M, Fin) ; idx: (M, n) neighbor indices ; w: (M, n) weights
        B, M, Fin = x.shape
        K = self.K
        x0 = x.permute(1, 0, 2).reshape(M, B * Fin)          # (M, B*Fin)
        wq = w.to(x0.dtype).unsqueeze(-1)                     # (M, n, 1)

        def lap(y):                                           # y: (M, B*Fin)
            return (y[idx] * wq).sum(1)                       # gather -> (M, B*Fin)

        stack = [x0]
        if K > 1:
            x1 = lap(x0)
            stack.append(x1)
            for _ in range(2, K):
                x2 = 2 * lap(x1) - x0
                stack.append(x2)
                x0, x1 = x1, x2
        x = torch.stack(stack, 0)                             # (K, M, B*Fin)
        x = x.reshape(K, M, B, Fin).permute(2, 1, 3, 0)       # (B, M, Fin, K)
        x = x.reshape(B * M, Fin * K) @ self.weight.to(x.dtype)
        x = x.reshape(B, M, self.out_channels)
        return x + self.bias.to(x.dtype) if self.bias is not None else x


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of scalar flow time t in [0, 1]."""
    if t.dim() == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device, dtype=torch.float32)
                      / max(half - 1, 1))
    a = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(a), torch.cos(a)], -1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], -1)
    return emb


# ---------------------------------------------------------------------------
# Flow-matching velocity network (DeepSphere-conv backbone + FiLM conditioning)
# ---------------------------------------------------------------------------

class SphereFlowNet(nn.Module):
    """Velocity field v_theta(xt, t, cond) on the HEALPix graph.

    Input channels: [xt, delta_low]. Time + cosmo conditioning are injected per
    graph-conv layer via FiLM (feature-wise affine modulation). Output: 1-channel
    velocity per pixel. A residual (skip) path stabilizes the deep graph stack.
    """

    def __init__(self, laplacian: sp.spmatrix, cond_dim: int = 12, hidden: int = 64,
                 n_layers: int = 6, K: int = 5, time_embed: int = 64):
        super().__init__()
        # Gather-form Laplacian (dense tensors -> DDP/bf16 friendly buffers).
        idx, w = laplacian_to_gather(laplacian)
        self.register_buffer("L_idx", idx)
        self.register_buffer("L_w", w)
        self.npix = idx.shape[0]
        self.hidden = hidden

        self.in_proj = ChebConv(2, hidden, K)            # [xt, cond_map] -> hidden
        self.blocks = nn.ModuleList([ChebConv(hidden, hidden, K)
                                     for _ in range(n_layers)])
        self.out_proj = ChebConv(hidden, 1, K)

        # Conditioning MLP -> per-block (scale, shift) FiLM parameters.
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_embed + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * hidden * n_layers))
        self.time_embed = time_embed
        self.n_layers = n_layers
        self.act = nn.SiLU()

    def forward(self, xt: torch.Tensor, t: torch.Tensor, delta_low: torch.Tensor,
                cosmo: torch.Tensor) -> torch.Tensor:
        # xt, delta_low: (B, M) ; t: (B,) ; cosmo: (B, cond_dim)
        if xt.dim() == 1:
            xt, delta_low = xt[None], delta_low[None]
        if cosmo.dim() == 1:
            cosmo = cosmo[None]
        if t.dim() == 0:
            t = t[None]
        B = xt.shape[0]
        idx, w = self.L_idx, self.L_w

        h = torch.stack([xt, delta_low], -1)             # (B, M, 2)
        h = self.act(self.in_proj(h, idx, w))

        c = self.cond_mlp(torch.cat([sinusoidal_embedding(t, self.time_embed),
                                     cosmo], -1))
        c = c.view(B, self.n_layers, 2, self.hidden)

        for i, block in enumerate(self.blocks):
            y = block(h, idx, w)
            scale, shift = c[:, i, 0, :].unsqueeze(1), c[:, i, 1, :].unsqueeze(1)
            h = h + self.act(y * (1 + scale) + shift)    # residual + FiLM
        return self.out_proj(h, idx, w).squeeze(-1)      # (B, M)


# ---------------------------------------------------------------------------
# Flow-matching loss and ODE sampler
# ---------------------------------------------------------------------------

def flow_matching_loss(net: SphereFlowNet, x1: torch.Tensor, delta_low: torch.Tensor,
                       cosmo: torch.Tensor) -> torch.Tensor:
    """Conditional rectified-flow loss. x1 = delta_high (B, M); condition delta_low."""
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    v_target = x1 - x0
    v_pred = net(xt, t, delta_low, cosmo)
    return ((v_pred - v_target) ** 2).mean()


@torch.no_grad()
def sample_ode(net: SphereFlowNet, delta_low: torch.Tensor, cosmo: torch.Tensor,
               steps: int = 50, x0: Optional[torch.Tensor] = None,
               amp: bool = False) -> torch.Tensor:
    """Integrate dx/dt = v_theta from noise to a delta_high realization (Euler).

    amp=True runs each net() call under bf16 autocast (same dtype/pattern as
    flow_matching_loss's training-time autocast) -- the gather-based ChebConv
    supports it natively and this workload is memory-bandwidth-bound (see
    laplacian_to_gather's docstring), so bf16 roughly halves bytes moved per
    step. The Euler accumulator x stays fp32: net(...) returns bf16 under
    autocast, but `x (fp32) + bf16_tensor * dt` type-promotes to fp32
    automatically, so precision doesn't degrade across the 50 additive steps."""
    if delta_low.dim() == 1:
        delta_low = delta_low[None]
    if cosmo.dim() == 1:
        cosmo = cosmo[None]
    x = torch.randn_like(delta_low) if x0 is None else x0
    dt = 1.0 / steps
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
        for s in range(steps):
            t = torch.full((x.shape[0],), s * dt, device=x.device)
            x = x + net(x, t, delta_low, cosmo) * dt
    return x


# ---------------------------------------------------------------------------
# Patch + overdensity helpers (shared conventions with the correction pipeline)
# ---------------------------------------------------------------------------

def map_to_patches(maps: np.ndarray, order: int) -> np.ndarray:
    if order <= 1:
        return maps
    N, npix = maps.shape
    p = n_patches(order)
    return maps.reshape(N, p, npix // p).reshape(N * p, npix // p)


def patches_to_maps(patches: np.ndarray, order: int, n_maps: int) -> np.ndarray:
    if order <= 1:
        return patches
    p = n_patches(order)
    return patches.reshape(n_maps, p, patches.shape[1]).reshape(n_maps, -1)


def to_overdensity(maps: np.ndarray):
    """Per-shell overdensity delta = rho/mean - 1 and the per-shell means."""
    m = maps.mean(axis=-1, keepdims=True)
    m = np.where(m == 0, 1.0, m)
    return (maps / m - 1.0).astype(np.float32), m.squeeze(-1)


# Variance-stabilizing transform for the heavy-tailed density field.
# Shot-noise-dominated faint shells have overdensity up to ~1e3; arcsinh
# linearizes small delta and log-compresses the tail (and handles delta=-1),
# which is essential to keep flow-matching targets O(1) and avoid NaN.

def signal_forward(delta: np.ndarray, scale: float, softening: float = 1.0,
                   clip: float = 8.0) -> np.ndarray:
    """delta -> normalized network signal  y = clip(arcsinh(delta/softening) / scale).

    The clip bounds rare shot-noise spikes in faint shells (which otherwise dominate
    the flow-matching loss); real structure stays well within +-clip.
    """
    y = np.arcsinh(delta / softening) / scale
    if clip:
        y = np.clip(y, -clip, clip)
    return y.astype(np.float32)


def signal_inverse(y, scale: float, softening: float = 1.0):
    """Network signal y -> overdensity delta = softening * sinh(y * scale)."""
    lib = torch if isinstance(y, torch.Tensor) else np
    return softening * lib.sinh(y * scale)
