"""Full-sky reconstruction FROM a patch-based model: tile the sphere with overlapping
gnomonic patches, run a model on each, blend back. Needed ONLY by pipelines whose
correction is computed per-patch (e.g. unet's flow model) -- NOT by
pipelines that already operate on the whole sky directly (e.g. transfer's
transfer-function+Poisson correction, which needs none of this).

This is deliberately torch-free at import time except inside tile_and_predict's model
call, so the CALLER'S predict function decides what "run the model on a batch of
patches" means (a flow ODE integration, a single forward pass, etc).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import healpy as hp

# (nside, nside_centers, patch_size) -> (n_centers, patch_size*patch_size) int32 array
# of HEALPix pixel indices, one row per gnomonic tile. See gnomonic_index_maps.
_IDX_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
# same key -> (npix,) float64 blend-weight map. Depends only on the taper + geometry,
# NOT on the map values, so like _IDX_CACHE it is identical for every shell/cosmology.
_WEIGHT_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def gnomonic_index_maps(nside: int, nside_centers: int, patch_size: int,
                        n_workers: int = 16) -> np.ndarray:
    """(n_centers, patch_size**2) int32 HEALPix indices: which sphere pixel each
    tile pixel reads from. Depends ONLY on the geometry (nside, center grid, patch
    size) -- NOT on the map values -- so it is identical for every shell and every
    cosmology, and is built once and cached here.

    This is the whole ballgame for full-sky reconstruction speed. The obvious
    implementation (what this replaced) called hp.projector.GnomonicProj.projmap
    twice per tile, per shell: once for the data and once on np.arange(npix) to
    recover the indices. That second call allocates and projects a fresh
    50M-element (400MB at nside=2048) index array for EVERY tile -- ~37ms/tile,
    x12288 tiles x ~60 shell reconstructions = hours of pure CPU projection,
    dwarfing the GPU work it was feeding. Building the indices directly
    (xy2vec -> vec2pix, verified to reproduce projmap's result EXACTLY) and reusing
    them across shells turns each tile extraction into a fancy-index (~0.14ms),
    and the one-time build parallelizes across threads (healpy's vec2pix is C and
    releases the GIL).

    Costs ~3.2GB of RAM at nside=2048/patch=256/12288 centers -- deliberate: the
    alternative is spending that same memory bandwidth recomputing the identical
    projection tens of thousands of times."""
    key = (nside, nside_centers, patch_size)
    if key in _IDX_CACHE:
        return _IDX_CACHE[key]

    reso_arcmin = hp.nside2resol(nside, arcmin=True)
    lon, lat = hp.pix2ang(nside_centers, np.arange(hp.nside2npix(nside_centers)),
                          nest=False, lonlat=True)
    n_centers = len(lon)

    def build(c: int) -> np.ndarray:
        proj = hp.projector.GnomonicProj(rot=(lon[c], lat[c], 0.0), xsize=patch_size,
                                         ysize=patch_size, reso=reso_arcmin) # rotation
        v = proj.xy2vec(proj.ij2xy())
        return hp.vec2pix(nside, v[0], v[1], v[2], nest=False).ravel().astype(np.int32)

    print(f"[patch_tiling] building gnomonic index cache: {n_centers} tiles "
          f"(nside={nside}, patch={patch_size}) -- once, then reused for every "
          f"shell/cosmology", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        idx = np.stack(list(ex.map(build, range(n_centers))))
    print(f"[patch_tiling] index cache ready ({idx.nbytes / 1e9:.2f} GB)", flush=True)

    _IDX_CACHE[key] = idx
    return idx


def auto_nside_centers(nside: int, patch_size: int, target_ratio: float = 4.0) -> int:
    """Pick a center-grid nside so patch FOV / center-spacing ~= target_ratio (a
    known-working overlap density gives ~13x area overlap, enough for the cosine
    taper to hide seams). A FIXED nside_centers tuned at one data nside silently
    undercovers the sphere at a different nside (patch FOV shrinks/grows with the
    data's nside while a fixed center grid's spacing does not) -- this scales the
    center grid proportionally instead."""
    fov_deg = patch_size * hp.nside2resol(nside, arcmin=True) / 60
    for nc in [8, 16, 32, 64, 128, 256, 512]:
        spacing_deg = hp.nside2resol(nc, arcmin=True) / 60
        if fov_deg / spacing_deg >= target_ratio:
            return nc
    return 512


def cosine_taper(patch_size: int) -> np.ndarray:
    """Smooth radial weight, 1 at center falling to 0 at the patch edge, so
    overlapping patches blend without visible seams at their boundaries."""
    x = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    return (0.5 * (1 + np.cos(np.pi * np.clip(r, 0, 1)))).astype(np.float32)


def tile_and_predict(predict_batch: Callable[[np.ndarray], np.ndarray], low_shell: np.ndarray,
                     nside_centers: int, patch_size: int, batch_size: int = 32,
                     n_workers: int = 16, log_every: int = 16,
                     pass_indices: bool = False, taper_power: float = 1.0):
    """Tile the sphere, call predict_batch on each mini-batch of low patches, blend
    back. predict_batch: (B,patch_size,patch_size) low-count patches -> (B,H,W)
    corrected-count patches (the model-specific part; e.g. a flow ODE integration).
    Returns (pred_map with NaN where uncovered, covered boolean mask).

    Tile extraction is a fancy-index into the cached geometry (gnomonic_index_maps),
    not a per-tile healpy re-projection -- see that function for why.

    pass_indices (default False, so every existing caller is unaffected): when True,
    predict_batch is called as predict_batch(low_batch, batch_idx) with batch_idx the
    (B, patch_size*patch_size) int32 array of HEALPix pixel indices each tile pixel
    reads from. A STOCHASTIC per-tile model needs this: each sky pixel is covered by
    ~N overlapping tiles and this function AVERAGES them, so if every tile draws its
    own independent randomness the tiles disagree and the average destroys most of the
    generated structure (measured: only ~41% of the injected amplitude, ~17% of the
    power, survives at ~16x overlap -- vs ~97% for a deterministic per-tile model).
    Knowing its sphere indices lets a caller crop a SINGLE global noise field so
    overlapping tiles share the same realization and therefore agree. See
    diffusion/apply_diffusion.py's full-sky predict_batch for the reference use.

    taper_power (default 1.0 = existing behaviour for every caller): raises the
    cosine blend weight to this power BEFORE normalizing, so large powers make the
    nearest tile own each pixel (soft Voronoi) instead of ~16 tiles averaging it --
    the Ronneberger overlap-tile idea: overlap provides CONTEXT, but each output
    pixel comes from ~one prediction. This matters ONLY for STOCHASTIC per-tile
    models (diffusion): a weighted mean of N_eff independent samples keeps the
    conditional-mean part of the prediction but shrinks the sample-to-sample
    (stochastic) part by 1/sqrt(N_eff), where N_eff = (sum w)^2 / sum w^2. At the
    default overlap, p=1 gives N_eff~6 (stochastic amplitude retention ~0.41);
    p=8 gives retention ~0.9+. Deterministic models (unet's flow) are unaffected by
    p since their overlapping predictions already agree -- keep p=1 there, the extra
    averaging is free seam smoothing."""
    nside = hp.npix2nside(len(low_shell))
    npix = hp.nside2npix(nside)
    taper = cosine_taper(patch_size).ravel().astype(np.float64) ** taper_power

    idx = gnomonic_index_maps(nside, nside_centers, patch_size, n_workers)  # (n_centers, ps*ps)
    n_centers = idx.shape[0]

    # The blend weights depend only on the taper and the (cached) geometry, so they
    # are the SAME for every shell and cosmology -- accumulate them once, not once
    # per shell (this was half of all the np.add.at work).
    key = (nside, nside_centers, patch_size, round(float(taper_power), 4))
    weight = _WEIGHT_CACHE.get(key)
    build_weight = weight is None
    if build_weight:
        weight = np.zeros(npix, dtype=np.float64)

    accum_pred = np.zeros(npix, dtype=np.float64)

    for start in range(0, n_centers, batch_size):
        end = min(start + batch_size, n_centers)
        batch_idx = idx[start:end]                                    # (B, ps*ps)
        low_batch = low_shell[batch_idx].astype(np.float32).reshape(
            end - start, patch_size, patch_size)

        pred_counts = predict_batch(low_batch, batch_idx) if pass_indices else predict_batch(low_batch)

        for k in range(end - start):
            idx_map = batch_idx[k]
            np.add.at(accum_pred, idx_map, pred_counts[k].ravel() * taper)  # Σ  pred·taper
            if build_weight:
                np.add.at(weight, idx_map, taper)                           # Σ  taper
        if (start // batch_size) % log_every == 0 or end == n_centers:
            print(f"[patch_tiling]   {end}/{n_centers} patches", flush=True)

    if build_weight:
        _WEIGHT_CACHE[key] = weight

    covered = weight > 0
    pred_map = np.full(npix, np.nan)
    pred_map[covered] = accum_pred[covered] / weight[covered]
    print(f"[patch_tiling]   sky coverage: {100 * covered.mean():.2f}%", flush=True)
    return pred_map, covered


def reconstruct_shell(predict_batch: Callable[[np.ndarray], np.ndarray], low_shell: np.ndarray,
                      nside_centers: int, patch_size: int, batch_size: int = 32,
                      min_coverage: float = 0.995, pass_indices: bool = False,
                      fill_map: np.ndarray | None = None, taper_power: float = 1.0):
    """Tile+predict one shell, gap-fill, and hard-fail on genuine undercoverage.
    Returns pred_filled (the corrected full-sky map).

    pass_indices is forwarded to tile_and_predict -- see there (stochastic per-tile
    models must share randomness across overlapping tiles or the blend averages it away).

    fill_map (default None -> low_shell): what the residual uncovered pixels are filled
    with. This MUST be in the same space predict_batch returns. It exists because a
    caller may blend in a transformed space -- e.g. diffusion/apply_diffusion.py returns
    log(counts) so the blend is a geometric mean -- in which case filling with raw
    low_shell counts would inject values that are nonsense in that space (a count of
    ~500 read as a log-count becomes e^500 = inf on the way back, silently NaN-ing the
    entire Cl via anafast). This is not hypothetical: nside=2048 leaves ~2824 gap
    pixels per shell (0.006%).

    A gap filled with raw DISCO is not a model failure but a TILING failure -- it
    would inject fake boundary discontinuities into the Cl and could easily be
    mistaken for genuine high-ell model behavior, so this fails loudly rather than
    silently reporting a misleading plot.
    """
    pred_map, covered = tile_and_predict(predict_batch, low_shell, nside_centers,
                                         patch_size, batch_size, pass_indices=pass_indices,
                                         taper_power=taper_power)
    cov_frac = covered.mean()
    if cov_frac < min_coverage:
        raise RuntimeError(
            f"only {100*cov_frac:.1f}% sky coverage (nside_centers={nside_centers}) -- "
            f"results would be contaminated by uncovered-pixel fallback, not a genuine "
            f"measurement. Increase nside_centers.")
    # >=min_coverage covered, but not necessarily exactly 100% -- hp.anafast has no NaN
    # handling, so even a few hundred leftover NaN pixels silently NaN out the ENTIRE
    # Cl array, not just those pixels. Fill the negligible remainder with DISCO.
    n_gap = int((~covered).sum())
    fill = low_shell if fill_map is None else fill_map
    if n_gap:
        print(f"[patch_tiling]   filling {n_gap} residual gap pixels "
              f"({100 * n_gap / len(covered):.3f}%) with DISCO"
              f"{'' if fill_map is None else ' (caller-supplied fill_map)'}", flush=True)
    return np.where(covered, pred_map, fill)
