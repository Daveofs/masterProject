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
from concurrent.futures import ThreadPoolExecutor
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
    # `where=` without `out=` leaves the masked-out entries UNINITIALIZED (numpy
    # warns about exactly this): an isolated node (d==0) would get whatever was in
    # memory, and any non-zero garbage there propagates into the Laplacian. Seed
    # the output with zeros so d==0 rows stay 0 (no self-normalization), which is
    # the intended meaning.
    d12 = np.zeros_like(d)
    np.power(d, -0.5, where=d > 0, out=d12)
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


# ---------------------------------------------------------------------------
# OVERLAPPING patch geometry (2026-07-20): rotate an ARBITRARY sky direction onto
# "canonical patch 0" (the same npix_patch NESTED pixels _healpix_weightmatrix
# already builds the graph Laplacian for), instead of only ever reading the
# n_patches(order) DISJOINT, quad-tree-aligned NESTED blocks map_to_patches above
# reshapes out. This is what makes overlapping, taper-blended full-sky
# reconstruction possible for the graph-conv model (same spirit as
# analysis/patch_tiling.py's gnomonic overlap, adapted to the HEALPix graph): a
# patch can now be centered at ANY (lon, lat, psi), not just one of the
# n_patches(order) fixed centers, while its INTERNAL pixel adjacency is always
# (up to HEALPix's own pixelization irregularity under rotation -- see
# duplicate-pixel note below) the SAME topology the model's Laplacian was built
# for -- so NO changes are needed to healpix_laplacian/ChebConv/SphereFlowNet.
#
# THIS IS A TRAINING-TIME CHANGE, NOT JUST AN INFERENCE ONE: a checkpoint trained
# only on the OLD disjoint quad-tree blocks has only ever seen that one exact
# alignment, so applying it to an arbitrarily-rotated patch at inference is an
# extrapolation with unknown behaviour right at the patch boundary (precisely
# where the overlap/blend is supposed to help). make_patch_dataset.py therefore
# draws patches at RANDOM (lon, lat, psi) at TRAINING time too (mirroring how
# unet/diffusion's make_patch_dataset.py already draws random gnomonic centers +
# psi), so the model is trained on the SAME distribution of local topologies
# (any rotation) it will see reconstructed at inference.
#
# Geometry, validated 2026-07-20 (round-trip self-match 1.0, target-center
# recovery to 1e-5 deg, psi actually rotates the patch, duplicate-pixel rate
# ~4% away from poles / ~18% near them -- comparable to gnomonic tiling's own
# known ~17% duplicate rate, see diffusion/apply_diffusion.py):
#   R0 = Rotator(rot=(lon0, lat0, 0))         # canonical patch-0 center's rotator
#   v_local  = R0(canonical_directions)        # global(canonical) -> local ref frame
#   Rt_inv   = Rotator(rot=(lon, lat, psi)).get_inverse()
#   v_target = Rt_inv(v_local)                 # local ref frame -> global(target)
# (healpy's Rotator(rot=(lon,lat,psi)) convention: FORWARD maps a GLOBAL direction
# to the LOCAL frame where the requested (lon,lat) sits at the local X-axis
# [1,0,0] -- NOT the pole -- so going canonical-global -> local -> target-global
# needs R0 forward then Rt INVERSE, not the other way around.)
# ---------------------------------------------------------------------------

_CANON_CACHE: dict[tuple[int, int], tuple] = {}  # (nside, order) -> (canon_vec, lon0, lat0, R0)


def _canonical_patch(nside: int, order: int):
    """(canon_vec (npix_patch,3), lon0, lat0, R0) for canonical patch 0 -- cached,
    depends only on (nside, order)."""
    key = (nside, order)
    if key in _CANON_CACHE:
        return _CANON_CACHE[key]
    p = patch_npix(nside, order)
    x, y, z = hp.pix2vec(nside, np.arange(p), nest=True)
    canon_vec = np.vstack([x, y, z]).T.astype(np.float64)
    center = canon_vec.mean(axis=0); center /= np.linalg.norm(center)
    lon0, lat0 = hp.vec2ang(center[None, :], lonlat=True)
    lon0, lat0 = float(lon0[0]), float(lat0[0])
    R0 = hp.Rotator(rot=(lon0, lat0, 0.0), deg=True)
    _CANON_CACHE[key] = (canon_vec, lon0, lat0, R0)
    return _CANON_CACHE[key]


def rotated_patch_ids(nside: int, order: int, lon: float, lat: float,
                      psi: float = 0.0) -> np.ndarray:
    """(npix_patch,) NESTED pixel ids on the TRUE sky for a patch centered at
    (lon, lat) with in-plane rotation psi (degrees) -- the overlap-capable
    replacement for map_to_patches' fixed disjoint blocks. ONE-OFF (uncached
    Rotator per call): used by make_patch_dataset.py, where every draw is a
    fresh random center. For the reconstruction sweep (many shells/cosmologies
    reusing the SAME grid of centers), use healpix_overlap_index_maps instead,
    which caches this per center."""
    canon_vec, _lon0, _lat0, R0 = _canonical_patch(nside, order)
    v = canon_vec.T
    v_local = np.asarray(R0(v[0], v[1], v[2]))
    Rt_inv = hp.Rotator(rot=(lon, lat, psi), deg=True).get_inverse()
    v_t = np.asarray(Rt_inv(v_local[0], v_local[1], v_local[2]))
    return hp.vec2pix(nside, v_t[0], v_t[1], v_t[2], nest=True).astype(np.int64)


def patch_angular_taper(nside: int, order: int, taper_power: float = 1.0) -> np.ndarray:
    """(npix_patch,) blend weight, 1.0 at the canonical patch's own center falling
    to 0 at its angular edge (raised-cosine over angular distance from center) --
    the HEALPix-graph analogue of analysis.patch_tiling.cosine_taper, using
    angular separation instead of flat (x,y) distance. Same for EVERY rotated
    copy of the patch by construction (rotation preserves angular distances), so
    this is computed ONCE from the canonical geometry and reused for every
    center. taper_power sharpens toward nearest-patch-wins (see
    analysis.patch_tiling.tile_and_predict's taper_power docstring) -- relevant
    if/when this model's sampling is stochastic per patch."""
    canon_vec, _lon0, _lat0, _R0 = _canonical_patch(nside, order)
    center = canon_vec.mean(axis=0); center /= np.linalg.norm(center)
    cos_ang = np.clip(canon_vec @ center, -1.0, 1.0)
    ang = np.arccos(cos_ang)
    edge = ang.max()
    if edge <= 0:
        return np.ones(len(canon_vec), dtype=np.float64)
    t = np.clip(ang / edge, 0.0, 1.0)
    taper = 0.5 * (1 + np.cos(np.pi * t))          # 1 at center, 0 at edge
    return taper ** taper_power


_OVERLAP_IDX_CACHE: dict[tuple, np.ndarray] = {}   # (nside, order, nside_centers) -> (n_centers, npix_patch) int64
_OVERLAP_CENTERS_CACHE: dict[tuple, np.ndarray] = {}  # same key -> (n_centers, 2) lon,lat deg


def healpix_overlap_index_maps(nside: int, order: int, nside_centers: int,
                               n_workers: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(idx, centers_lonlat): idx is (n_centers, npix_patch) int64 true-sky NESTED
    pixel ids -- one row per overlapping patch, centers on a hp.nside2npix
    (nside_centers)-direction grid. Cached (built once per (nside, order,
    nside_centers), reused across every shell/cosmology) -- same role as
    analysis.patch_tiling.gnomonic_index_maps, adapted to rotated HEALPix
    patches instead of gnomonic projection. nside_centers should be finer than
    `order` (more centers than n_patches(order)) for genuine overlap; see
    auto_overlap_nside_centers."""
    key = (nside, order, nside_centers)
    if key in _OVERLAP_IDX_CACHE:
        return _OVERLAP_IDX_CACHE[key], _OVERLAP_CENTERS_CACHE[key]

    n_centers = hp.nside2npix(nside_centers)
    lon, lat = hp.pix2ang(nside_centers, np.arange(n_centers), nest=False, lonlat=True)
    print(f"[sphere_flow] building overlap index cache: {n_centers} patches "
          f"(nside={nside}, order={order}) -- once, then reused for every "
          f"shell/cosmology", flush=True)

    def build(c: int) -> np.ndarray:
        return rotated_patch_ids(nside, order, float(lon[c]), float(lat[c]))

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        idx = np.stack(list(ex.map(build, range(n_centers))))
    print(f"[sphere_flow] overlap index cache ready ({idx.nbytes / 1e9:.2f} GB)", flush=True)

    centers = np.stack([lon, lat], axis=1)
    _OVERLAP_IDX_CACHE[key] = idx
    _OVERLAP_CENTERS_CACHE[key] = centers
    return idx, centers


def auto_overlap_nside_centers(order: int, target_ratio: float = 4.0) -> int:
    """Pick a center-grid nside so patch-diameter / center-spacing ~= target_ratio
    -- the HEALPix-graph analogue of analysis.patch_tiling.auto_nside_centers.
    `order` itself defines the DISJOINT grid (n_patches(order) centers, spacing
    == patch diameter, ratio 1); this scales up to get real overlap."""
    for nc in [order, 2 * order, 4 * order, 8 * order, 16 * order]:
        if nc / order >= target_ratio:
            return nc
    return 16 * order


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
