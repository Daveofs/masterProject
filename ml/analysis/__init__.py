"""Shared analysis/plotting tools used by every correction pipeline in this project
(unet's flow model, transfer's transfer-function+Poisson pipeline, ...).

Kept pipeline-agnostic on purpose: these modules know how to transform, tile, measure
and PLOT a low/corrected/high comparison, but nothing about how "corrected" was
produced -- that keeps every pipeline's diagnostic figures visually identical by
construction, not by convention, and means a fix here fixes every pipeline at once.

  transforms.py     log1p(overdensity), shared clip convention
  radial_power.py   flat-patch 2D-FFT power spectrum (example_patches.png 4th column)
  full_sky.py       real angular C_ell (hp.anafast) + gnomonic zoom crop
  patch_tiling.py   sphere tiling/blending for PATCH-based models only (not needed by
                    pipelines whose correction is already computed full-sky)
  moments.py        one-point pixel statistics (mean/variance/skewness/excess
                    kurtosis), the marginal-PDF check a Cl ratio can't provide
  plotting.py       the actual figure builders (example_patches, example_full_sky,
                    per-shell Cl+ratio, pctile-band ratio, histogram grid,
                    moments-vs-shell-depth)
"""
