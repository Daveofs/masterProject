#!/usr/bin/env python3
"""Identity test of the patch-tiling machinery: how much Cl error does tiling
ITSELF introduce, with no model involved at all?

Feeds the TRUE high shell through the full tile-and-blend pipeline
(patch_tiling.reconstruct_shell) with predict_batch = identity, then measures
Cl(reconstructed) / Cl(true). Since the "model" returns its input exactly, ANY
deviation from 1.0 is pure tiling systematic: the nearest-neighbor gnomonic
resample (sphere -> flat tile -> sphere via vec2pix) plus the multi-tile
weighted average, which acts like a low-pass near the patch pixel Nyquist
(ell ~ pi / pixel_scale ~ 1570 at nside=512).

Why this matters: the corrected/true Cl-ratio panels (e.g. unet/diffusion
cl_ratio_by_zbin_grid.png) droop at ell > ~1000, and without this test that
droop is indistinguishable from a model deficiency. If the identity ratio shows
the same droop, the tiling owns it -- it is then a known transfer function of
the reconstruction, not something more epochs/capacity/hp tuning can fix.

Runs per (shell, taper_power) combination: taper_power=1 is unet's blend,
32 is diffusion's near-Voronoi blend -- if their identity curves differ, part of
the unet-vs-diffusion high-ell difference is blend choice, not model quality.

CPU-only (healpy + numpy), a few seconds per shell -- no GPU, no checkpoint.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.patch_tiling import auto_nside_centers, reconstruct_shell  # noqa: E402
from analysis.full_sky import od_cl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/capstor/scratch/cscs/damrein/grid")
    p.add_argument("--cosmo", default=None,
                   help="cosmology dir (default: first one with a prepared "
                        "high_shells_nside=<nside>.npy stack)")
    p.add_argument("--run", default="run_0")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--nside-centers", type=int, default=None,
                   help="tile-center grid (default: auto_nside_centers, the same "
                        "choice apply_flow.py/apply_diffusion.py make)")
    p.add_argument("--shells", type=int, nargs="+", default=[5, 15, 30, 50, 67],
                   help="shell indices spanning the usable redshift range")
    p.add_argument("--taper-powers", type=float, nargs="+", default=[1.0, 32.0],
                   help="blend exponents to test (1=unet's mean blend, "
                        "32=diffusion's near-Voronoi blend)")
    p.add_argument("--lmax", type=int, default=1500)
    p.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/tiling_identity")
    args = p.parse_args()

    root = Path(args.data_root)
    if args.cosmo is None:
        stack = f"high_shells_nside={args.nside}.npy"
        args.cosmo = next(c for c in sorted(os.listdir(root))
                          if (root / c / args.run / stack).exists())
    run_dir = root / args.cosmo / args.run
    high_all = np.load(run_dir / f"high_shells_nside={args.nside}.npy", mmap_mode="r")

    nside_centers = args.nside_centers or auto_nside_centers(args.nside, args.patch_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[identity] cosmo={args.cosmo} nside={args.nside} patch={args.patch_size} "
          f"nside_centers={nside_centers} ({12 * nside_centers**2} tiles) "
          f"shells={args.shells} taper_powers={args.taper_powers}", flush=True)

    def identity(low_batch):
        return low_batch

    ratios = {}  # (shell, taper_power) -> Cl ratio array
    for shell in args.shells:
        true_map = np.asarray(high_all[shell], dtype=np.float64)
        cl_true = od_cl(true_map, args.lmax)
        for tp in args.taper_powers:
            rec = reconstruct_shell(identity, true_map, nside_centers,
                                    args.patch_size, taper_power=tp)
            cl_rec = od_cl(rec, args.lmax)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios[(shell, tp)] = cl_rec / cl_true
            ell = np.arange(len(cl_true))
            sel = lambda lo, hi: slice(max(lo, 2), min(hi, args.lmax))  # noqa: E731
            print(f"[identity] shell {shell} taper_power={tp}: ratio "
                  f"ell 100-300 = {np.nanmean(ratios[(shell, tp)][sel(100, 300)]):.4f}, "
                  f"ell 500-1000 = {np.nanmean(ratios[(shell, tp)][sel(500, 1000)]):.4f}, "
                  f"ell 1000-{args.lmax} = {np.nanmean(ratios[(shell, tp)][sel(1000, args.lmax)]):.4f}",
                  flush=True)

    np.savez(out_dir / "identity_ratios.npz",
             lmax=args.lmax, cosmo=args.cosmo, nside=args.nside,
             patch_size=args.patch_size, nside_centers=nside_centers,
             shells=np.array(args.shells), taper_powers=np.array(args.taper_powers),
             **{f"ratio_s{s}_tp{tp:g}": r for (s, tp), r in ratios.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ell = np.arange(args.lmax + 1)
    cmap = plt.get_cmap("viridis")
    styles = {tp: ls for tp, ls in zip(args.taper_powers, ["-", "--", ":", "-."])}
    for i, shell in enumerate(args.shells):
        color = cmap(i / max(len(args.shells) - 1, 1))
        for tp in args.taper_powers:
            r = ratios[(shell, tp)]
            # bin in ell for readability (same smoothing spirit as the pctile plots)
            nbin = 40
            edges = np.unique(np.geomspace(10, args.lmax, nbin).astype(int))
            bl = 0.5 * (edges[:-1] + edges[1:])
            br = [np.nanmean(r[a:b]) for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(bl, br, styles[tp], color=color,
                    label=f"shell {shell}, taper^{tp:g}" if tp == args.taper_powers[0]
                    or i == 0 else None)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{\rm tiled}/C_\ell^{\rm true}$")
    ax.set_title(f"Tiling identity test (predict = identity): pure reconstruction systematic\n"
                 f"{args.cosmo}, nside={args.nside}, patch={args.patch_size}, "
                 f"nside_centers={nside_centers}; solid taper^{args.taper_powers[0]:g}"
                 + (f", dashed taper^{args.taper_powers[1]:g}" if len(args.taper_powers) > 1 else ""))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_png = out_dir / "identity_cl_ratio.png"
    fig.savefig(out_png, dpi=150)
    print(f"[identity] -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
