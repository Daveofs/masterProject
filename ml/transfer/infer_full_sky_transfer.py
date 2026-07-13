#!/usr/bin/env python3
"""Real full-sky angular C_ell (+ log1p-overdensity example grid) for the transfer-
function+Poisson pipeline, using the SAME shared tools (../analysis/) as
unet_flow_jbucko/infer_full_sky.py -- so the two pipelines' full-sky diagnostics are
directly comparable, not just visually similar.

Unlike unet_flow_jbucko's flow model, the transfer-function+Poisson correction is
already computed on the WHOLE sky directly (poisson_resample.py's output IS a full
HEALPix map stack, not small patches needing tiling/blending) -- so this script skips
analysis.patch_tiling entirely and just loads + measures + plots.

  python infer_full_sky_transfer.py --run-dir <grid>/cosmo_X/run_0 \
      --counts <poisson_corrected.npz> --shell-indices 0 34 68 \
      --example-shells 5 10 30 60 --out-dir <out>
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.full_sky import od_cl, gnomonic_crop                       # noqa: E402
from analysis.transforms import log1p_delta                              # noqa: E402
from analysis.plotting import (plot_cl_shell, plot_example_full_sky_grid,  # noqa: E402
                               plot_moments_vs_shell, plot_histogram_grid)
from analysis.moments import moments                                     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="cosmo_X/run_0 (has low_shells_nside=*.npy)")
    p.add_argument("--counts", required=True, help="transfer-function+Poisson output npz (shells key)")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--info-npz", default="compressed_shells.npz", help="true CosmoGrid shells (relative to --run-dir)")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[0, 34, 68],
                   help="shells to render as individual cl_shell*.png (2-panel Cl+ratio); "
                        "pass with no values to skip these and only build example_full_sky.png")
    p.add_argument("--example-shells", type=int, nargs="+", default=None,
                   help="if given, ALSO render example_full_sky.png: one row per shell "
                        "(gnomonic zoom of low/corrected/high + real Cl ratio-to-truth)")
    p.add_argument("--example-rot", type=float, nargs=2, default=[45.0, 45.0],
                   help="(lon, lat) deg center of the zoom crop in example_full_sky.png")
    p.add_argument("--example-reso", type=float, default=1.5, help="arcmin/pixel of the zoom crop")
    p.add_argument("--example-xsize", type=int, default=200, help="pixel size of the zoom crop")
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    run = Path(args.run_dir)
    nside = args.nside
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    low_all = np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r")
    corr_all = np.load(args.counts, mmap_mode="r")["shells"]
    high_all = np.load(run / args.info_npz, mmap_mode="r")["shells"]
    print(f"[infer_full_sky_transfer] {run} nside={nside} | corrected={args.counts}", flush=True)

    lmax = min(args.lmax, 3 * nside - 1)
    ells = np.arange(lmax + 1)
    for s in args.shell_indices:
        low_shell = np.asarray(low_all[s], np.float32)
        corr_shell = np.asarray(corr_all[s], np.float32)
        high_shell = np.asarray(high_all[s], np.float32)
        cl_lo = od_cl(low_shell, lmax); cl_c = od_cl(corr_shell, lmax); cl_hi = od_cl(high_shell, lmax)
        plot_cl_shell(s, ells, cl_lo, cl_c, cl_hi, out_dir / f"cl_shell{s:03d}.png")

    # example_full_sky.png: same layout as unet_flow_jbucko's, for direct comparison.
    if args.example_shells:
        print(f"[infer_full_sky_transfer] building example_full_sky.png for shells "
              f"{args.example_shells}", flush=True)
        lon, lat = args.example_rot
        crop = lambda m: gnomonic_crop(m, nside, lon, lat, args.example_xsize, args.example_reso)

        rows = []
        mom_low, mom_corr, mom_high, hist_rows = [], [], [], []
        for s in args.example_shells:
            low_shell = np.asarray(low_all[s], np.float32)
            corr_shell = np.asarray(corr_all[s], np.float32)
            high_shell = np.asarray(high_all[s], np.float32)
            cl_lo = od_cl(low_shell, lmax); cl_c = od_cl(corr_shell, lmax); cl_hi = od_cl(high_shell, lmax)
            rows.append((f"shell {s}", crop(log1p_delta(low_shell)), crop(log1p_delta(corr_shell)),
                        crop(log1p_delta(high_shell)), ells, cl_lo, cl_c, cl_hi))

            # full-sky one-point PDF (raw counts, all pixels) -- the marginal-
            # distribution check a Cl ratio (two-point, phase-blind) can't provide
            # (see the transfer-function positivity/one-point-PDF investigation).
            mom_low.append(moments(low_shell)); mom_high.append(moments(high_shell))
            mom_corr.append(moments(corr_shell))
            hist_rows.append((f"shell {s}", low_shell.ravel(), corr_shell.ravel(), high_shell.ravel()))

        plot_example_full_sky_grid(
            rows, out_dir / "example_full_sky.png", corrected_label="corrected (transfer+Poisson)",
            suptitle=f"transfer-function+Poisson full-sky, log1p overdensity (gnomonic zoom @ "
                    f"lon={lon:g},lat={lat:g}) + real angular Cl ratio")

        plot_moments_vs_shell(
            args.example_shells, {"low": mom_low, "high (true)": mom_high,
                                  "corrected (transfer+Poisson)": mom_corr},
            out_dir / "moments_vs_shell.png",
            suptitle=f"moments vs. shell depth -- full-sky (raw counts)\n{run.parent.name}/{run.name}")
        plot_histogram_grid(
            hist_rows, out_dir / "example_histograms.png",
            corrected_label="corrected (transfer+Poisson)",
            suptitle=f"full-sky raw pixel-count histogram per shell\n{run.parent.name}/{run.name}")

    print(f"[infer_full_sky_transfer] figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
