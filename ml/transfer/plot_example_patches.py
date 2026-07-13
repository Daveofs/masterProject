#!/usr/bin/env python3
"""Example-patch + power-ratio plot for the transfer-function+Poisson pipeline, in
EXACTLY the same visual format as unet_flow_jbucko/apply_flow.py's example_patches.png
(gnomonic patch triptych + per-patch radial-power-ratio 4th column), so the two
pipelines' outputs can be viewed side by side.

The transform, power-spectrum metric, and figure layout are IMPORTED from
../analysis/ (shared with unet_flow_jbucko and every other pipeline), not
reimplemented here -- a fix or format change there now applies to both pipelines at
once instead of drifting apart. This script only supplies the transfer-function-
specific piece: extracting a gnomonic patch from its raw counts arrays.

"corrected" here = the FINAL Poisson count map (poisson_resample.py output), the
direct analogue of jbucko's flow-corrected patch: both are the pipeline's end
deliverable, not an intermediate diagnostic.

Usage
-----
  python plot_example_patches.py --run-dir <grid>/cosmo_X/run_0 \
      --counts <poisson_counts.npz> --shells 10 30 50 --n-per-shell 2 \
      --patch-size 256 --nside 2048 --seed 0 --out example_patches.png
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.transforms import log1p_delta_pair  # noqa: E402
from analysis.radial_power import radial_power  # noqa: E402
from analysis.plotting import plot_example_patch_grid, plot_pctile_band_ratio  # noqa: E402


def extract_patch(shell_map: np.ndarray, nside: int, center_ipix: int, psi: float,
                  patch_size: int, reso_arcmin: float) -> np.ndarray:
    """Gnomonic-project one patch, matching make_patch_dataset.py exactly."""
    lon, lat = hp.pix2ang(nside, int(center_ipix), nest=False, lonlat=True)
    proj = hp.projector.GnomonicProj(rot=(lon, lat, psi), xsize=patch_size,
                                     ysize=patch_size, reso=reso_arcmin)
    vec2pix = lambda x, y, z, _ns=nside: hp.vec2pix(_ns, x, y, z, nest=False)
    return proj.projmap(shell_map, vec2pix)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="cosmo_X/run_0 (has low_shells_nside=*.npy)")
    p.add_argument("--counts", required=True, help="poisson_resample.py output npz (shells key)")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--shells", type=int, nargs="+", default=[5, 10, 15, 30, 50])
    p.add_argument("--n-per-shell", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--info-npz", default="compressed_shells.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--n-pctile-patches", type=int, default=200,
                   help="random patches to sample (positions pooled across --shells) "
                        "for the pctile-band power-ratio plot, the aggregate/"
                        "uncertainty-aware analogue of unet_flow_jbucko/apply_flow.py's "
                        "power_ratio_pctile_band.png; 0 to skip")
    args = p.parse_args()

    run = Path(args.run_dir)
    nside = args.nside
    reso_arcmin = hp.nside2resol(nside, arcmin=True)
    npix = hp.nside2npix(nside)

    low_full = np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r")
    counts = np.load(args.counts, mmap_mode="r")["shells"]
    true = np.load(run / args.info_npz, mmap_mode="r")["shells"]

    rng = np.random.default_rng(args.seed)
    shell_rows = [s for s in args.shells for _ in range(args.n_per_shell)]

    rows = []
    for s in shell_rows:
        center_ipix = int(rng.integers(0, npix))
        psi = float(rng.uniform(0, 360))
        low_p = extract_patch(np.asarray(low_full[s], dtype=np.float64), nside,
                              center_ipix, psi, args.patch_size, reso_arcmin)
        corr_p = extract_patch(np.asarray(counts[s], dtype=np.float64), nside,
                               center_ipix, psi, args.patch_size, reso_arcmin)
        high_p = extract_patch(np.asarray(true[s], dtype=np.float64), nside,
                               center_ipix, psi, args.patch_size, reso_arcmin)
        low_log, high_log = log1p_delta_pair(low_p, high_p)
        corr_log, _ = log1p_delta_pair(corr_p, high_p)
        rows.append((f"shell {s}", low_log, corr_log, high_log))

    # validated ON: the cosmology the transfer function / emulator NEVER saw during
    # fit/train (--test-cosmo there) -- state it explicitly so this plot is legible
    # on its own, without having to cross-reference which run produced --run-dir.
    cosmo_label = f"{run.parent.name}/{run.name}"
    plot_example_patch_grid(
        rows, args.out, corrected_label="corrected (transfer+Poisson)",
        suptitle=f"transfer-function + Poisson: validated on {cosmo_label} "
                 "(held out of fit/train) -- example patches (log1p overdensity) "
                 "+ per-patch power ratio\n(same layout/transform as "
                 "unet_flow_jbucko/apply_flow.py's example_patches.png)")

    # --- pctile-band power-ratio plot: many random patches (positions pooled across
    # --shells, not just the --n-per-shell visual examples above) so a systematic bias
    # is distinguishable from per-patch noise -- same statistic and shared plotting
    # code as unet_flow_jbucko/apply_flow.py's power_ratio_pctile_band.png. ---
    if args.n_pctile_patches > 0:
        print(f"[plot_example_patches] sampling {args.n_pctile_patches} random patches "
              f"across shells {args.shells} for the pctile-band power-ratio plot", flush=True)
        lo_stack, co_stack = [], []
        for _ in range(args.n_pctile_patches):
            s = int(rng.choice(args.shells))
            center_ipix = int(rng.integers(0, npix))
            psi = float(rng.uniform(0, 360))
            low_p = extract_patch(np.asarray(low_full[s], dtype=np.float64), nside,
                                  center_ipix, psi, args.patch_size, reso_arcmin)
            corr_p = extract_patch(np.asarray(counts[s], dtype=np.float64), nside,
                                   center_ipix, psi, args.patch_size, reso_arcmin)
            high_p = extract_patch(np.asarray(true[s], dtype=np.float64), nside,
                                   center_ipix, psi, args.patch_size, reso_arcmin)
            low_log, high_log = log1p_delta_pair(low_p, high_p)
            corr_log, _ = log1p_delta_pair(corr_p, high_p)
            pr_low, pr_corr, pr_high = radial_power(low_log), radial_power(corr_log), radial_power(high_log)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo_stack.append(pr_low / pr_high); co_stack.append(pr_corr / pr_high)

        k = np.arange(len(lo_stack[0]))
        out_dir = Path(args.out).parent
        plot_pctile_band_ratio(
            k, {"low / high (baseline, no model)": np.array(lo_stack),
                "corrected (transfer+Poisson) / high": np.array(co_stack)},
            out_dir / "patch_power_ratio_pctile_band.png", xlabel="radial wavenumber bin",
            ylim=(0.4, 1.6),
            title=f"power ratio: transfer+Poisson vs baseline, validated on {cosmo_label} "
                  f"({args.n_pctile_patches} patches across shells {args.shells}, "
                  "16-84th pctile band)")


if __name__ == "__main__":
    main()
