"""One shared recipe for the example-patch figure, so all three pipelines show
THE SAME patches.

Why this exists: the three pipelines used to pick their example patches in two
incompatible ways. transfer/apply_transfer.py drew a random held-out cosmology, a
random HEALPix centre and a random rotation, then cut a gnomonic patch straight out
of the full-sky maps. unet/apply_flow.py and diffusion/apply_diffusion.py instead
indexed precomputed tiles out of the patch dataset. Same nominal shells, different
cosmologies, different sky positions, different field of view -- so the figures could
not be compared side by side, which is the one thing an example-patch figure is for.

`patch_plan` centralises the draws. Given the same seed, shell list and cosmology
names, every pipeline gets the identical (cosmology, centre pixel, rotation) per row,
so the corresponding rows of the three figures show the same piece of sky at the same
orientation and the only difference left is the correction itself.

The RNG draw ORDER reproduces transfer/apply_transfer.py's original loop exactly
(cosmology, then centre pixel, then rotation, per row), so the transfer figure is
unchanged by adopting this -- only flow and diffusion move.

Cosmologies are addressed BY NAME from a sorted list rather than by position in each
pipeline's own held-out array: the pipelines discover their held-out set differently
(transfer from --run-dirs, the other two from the patch dataset's split), and an index
would silently select a different cosmology in each.
"""
from __future__ import annotations

import numpy as np
import healpy as hp


def patch_plan(seed: int, patch_shells, n_per_shell: int, cosmo_names, nside: int):
    """Deterministic per-row draw: [(shell, cosmo_name, center_ipix, psi_deg), ...].

    cosmo_names is sorted internally, so callers may pass any order.
    """
    names = sorted(str(c) for c in cosmo_names)
    if not names:
        raise ValueError("patch_plan: no cosmologies given")
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    plan = []
    for s in [s for s in patch_shells for _ in range(n_per_shell)]:
        name = names[int(rng.integers(0, len(names)))]
        center_ipix = int(rng.integers(0, npix))
        psi = float(rng.uniform(0, 360))
        plan.append((int(s), name, center_ipix, psi))
    return plan


def extract_patch(shell_map: np.ndarray, nside: int, center_ipix: int, psi: float,
                  patch_size: int, reso_arcmin: float) -> np.ndarray:
    """Gnomonic-project one patch, matching make_patch_dataset.py exactly.

    Lifted here from transfer/apply_transfer.py so the flow and diffusion pipelines
    can cut patches the same way without importing across pipeline directories
    (analysis/ is the shared module; the pipeline dirs stay decoupled).
    """
    lon, lat = hp.pix2ang(nside, int(center_ipix), nest=False, lonlat=True)
    proj = hp.projector.GnomonicProj(rot=(lon, lat, psi), xsize=patch_size,
                                     ysize=patch_size, reso=reso_arcmin)
    vec2pix = lambda x, y, z, _ns=nside: hp.vec2pix(_ns, x, y, z, nest=False)
    return proj.projmap(shell_map, vec2pix)
