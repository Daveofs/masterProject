"""Full-sky reconstruction FROM a patch-based model: tile the sphere with overlapping
gnomonic patches, run a model on each, blend back. Needed ONLY by pipelines whose
correction is computed per-patch (e.g. unet_flow_jbucko's flow model) -- NOT by
pipelines that already operate on the whole sky directly (e.g. transfer's
transfer-function+Poisson correction, which needs none of this).

This is deliberately torch-free at import time except inside tile_and_predict's model
call, so the CALLER'S predict function decides what "run the model on a batch of
patches" means (a flow ODE integration, a single forward pass, etc).
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import healpy as hp


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
                     nside_centers: int, patch_size: int, batch_size: int = 32):
    """Tile the sphere, call predict_batch on each mini-batch of low patches, blend
    back. predict_batch: (B,patch_size,patch_size) low-count patches -> (B,H,W)
    corrected-count patches (the model-specific part; e.g. a flow ODE integration).
    Returns (pred_map with NaN where uncovered, covered boolean mask)."""
    nside = hp.npix2nside(len(low_shell))
    npix = hp.nside2npix(nside)
    reso_arcmin = hp.nside2resol(nside, arcmin=True)
    taper = cosine_taper(patch_size)

    centers_lon, centers_lat = hp.pix2ang(nside_centers, np.arange(hp.nside2npix(nside_centers)),
                                          nest=False, lonlat=True)
    n_centers = len(centers_lon)

    accum_pred = np.zeros(npix, dtype=np.float64)
    weight = np.zeros(npix, dtype=np.float64)

    def vec2pix(x, y, z, _ns=nside):
        return hp.vec2pix(_ns, x, y, z, nest=False)

    for start in range(0, n_centers, batch_size):
        end = min(start + batch_size, n_centers)
        low_batch, idx_maps = [], []
        for c in range(start, end):
            proj = hp.projector.GnomonicProj(rot=(centers_lon[c], centers_lat[c], 0.0),
                                             xsize=patch_size, ysize=patch_size, reso=reso_arcmin)
            low_batch.append(proj.projmap(low_shell, vec2pix).astype(np.float32))
            idx_maps.append(proj.projmap(np.arange(npix), vec2pix).astype(np.int64))

        pred_counts = predict_batch(np.stack(low_batch))

        for k, c in enumerate(range(start, end)):
            idx_map = idx_maps[k].ravel(); w = taper.ravel()
            np.add.at(accum_pred, idx_map, pred_counts[k].ravel() * w)
            np.add.at(weight, idx_map, w)
        print(f"[patch_tiling]   {end}/{n_centers} patches", flush=True)

    covered = weight > 0
    pred_map = np.full(npix, np.nan)
    pred_map[covered] = accum_pred[covered] / weight[covered]
    print(f"[patch_tiling]   sky coverage: {100 * covered.mean():.2f}%", flush=True)
    return pred_map, covered


def reconstruct_shell(predict_batch: Callable[[np.ndarray], np.ndarray], low_shell: np.ndarray,
                      nside_centers: int, patch_size: int, batch_size: int = 32,
                      min_coverage: float = 0.995):
    """Tile+predict one shell, gap-fill, and hard-fail on genuine undercoverage.
    Returns pred_filled (the corrected full-sky map).

    A gap filled with raw DISCO is not a model failure but a TILING failure -- it
    would inject fake boundary discontinuities into the Cl and could easily be
    mistaken for genuine high-ell model behavior, so this fails loudly rather than
    silently reporting a misleading plot.
    """
    pred_map, covered = tile_and_predict(predict_batch, low_shell, nside_centers,
                                         patch_size, batch_size)
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
    if n_gap:
        print(f"[patch_tiling]   filling {n_gap} residual gap pixels "
              f"({100 * n_gap / len(covered):.3f}%) with DISCO", flush=True)
    return np.where(covered, pred_map, low_shell)
