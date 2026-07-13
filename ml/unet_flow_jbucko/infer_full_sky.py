#!/usr/bin/env python3
"""Full-sky reconstruction + real angular C_ell for the embedded jbucko flow model.

apply_flow.py's power-spectrum-ratio panel is computed on flat 256x256 gnomonic
patches via a 2D FFT -- it structurally cannot show anything past that patch's own
Nyquist wavenumber (bin ~130 of ~130), which is nowhere near the full ell~3000 range
the other pipelines' Cl-ratio plots cover. To see the same "how does it behave at very
high ell" question for the flow model, the correction has to be reconstructed on the
FULL SPHERE and passed through healpy.anafast, the actual spherical-harmonic
transform -- not approximated by a flat patch.

Tiling/blending and all plotting live in ../analysis/ (patch_tiling.py, full_sky.py,
transforms.py, plotting.py) -- shared with every other correction pipeline in this
project (e.g. transfer/) so the diagnostic figures are visually identical by
construction. This script only supplies the model-specific piece: how to turn one
batch of low-count patches into corrected-count patches (predict_batch below, a flow
ODE integration via the embedded FlowUNet + sample_ode).

Evaluated on a cosmology this checkpoint's split_by_cosmo held OUT of training (default:
the first such cosmology), analogous to --test-cosmo in the other pipelines.

  python infer_full_sky.py --data-root <prepared_dir> --model <run>/best.pt \
      --patch-dir <patch_dir> --out-dir <run>/eval
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_model import FlowUNet, sample_ode          # noqa: E402
from dataset import split_by_cosmo, COSMO_FIELDS, cosmo_z_vector  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.patch_tiling import auto_nside_centers, reconstruct_shell  # noqa: E402
from analysis.full_sky import od_cl, gnomonic_crop                       # noqa: E402
from analysis.transforms import log1p_delta                              # noqa: E402
from analysis.plotting import (plot_cl_shell, plot_example_full_sky_grid,  # noqa: E402
                               plot_moments_vs_shell, plot_histogram_grid)
from analysis.moments import moments                                     # noqa: E402


def make_predict_batch(net, dev, n_ode_steps, cosmo_z: np.ndarray | None = None):
    """The model-specific half of patch_tiling.reconstruct_shell: (B,H,W) low-count
    patches -> (B,H,W) corrected-count patches, via a flow ODE integration.

    cosmo_z: (8,) vector for THIS shell/cosmology (see dataset.cosmo_z_vector),
    constant across every patch of one shell -- broadcast to the batch inside.
    None if net was built with use_cosmo_cond=False."""
    cosmo_z_t = None if cosmo_z is None else torch.from_numpy(cosmo_z).to(dev)

    def predict_batch(low_batch: np.ndarray) -> np.ndarray:
        low_t = torch.from_numpy(low_batch).unsqueeze(1).to(dev)
        low_mean = low_t.mean(dim=(2, 3), keepdim=True)
        eps = 0.5 / low_mean
        low_log = torch.log1p(torch.maximum(low_t / low_mean - 1.0, -1.0 + eps))
        cz = None if cosmo_z_t is None else cosmo_z_t.unsqueeze(0).expand(low_t.shape[0], -1).to(low_log.dtype)
        with torch.no_grad():
            pred_log = sample_ode(net, low_log, n_steps=n_ode_steps, cosmo_z=cz)
        pred_delta = torch.expm1(pred_log)
        return ((1.0 + pred_delta) * low_mean).squeeze(1).cpu().numpy()
    return predict_batch


def lookup_cosmo_z(meta: np.ndarray, cosmo_name: str, shell_idx: int) -> np.ndarray:
    """(8,) cosmo_z vector for one shell/cosmology, looked up from the patch
    dataset's own metadata (authoritative, same source train_flow.py used) --
    cosmo params are constant per cosmology, redshift constant per shell_idx (z
    bounds come from CosmoGrid's fixed shell binning, not from the cosmology)."""
    rows_cosmo = meta[meta["cosmo"] == cosmo_name]
    if len(rows_cosmo) == 0:
        raise ValueError(f"lookup_cosmo_z: no patch-dir metadata row for cosmology "
                         f"{cosmo_name!r} -- can't recover its (Om,Ob,ns,s8,w0,h)")
    cosmo_vec = np.array([[rows_cosmo[0][f] for f in COSMO_FIELDS]], dtype=np.float32)  # (1,6)
    rows_shell = meta[meta["shell_idx"] == shell_idx]
    if len(rows_shell) == 0:
        raise ValueError(f"lookup_cosmo_z: no patch-dir metadata row for shell_idx="
                         f"{shell_idx} (any cosmology) -- can't recover its redshift; "
                         f"pick a --shell-indices/--example-shells value the training "
                         f"patch dataset actually sampled")
    z = np.array([0.5 * (rows_shell[0]["lower_z"] + rows_shell[0]["upper_z"])], dtype=np.float32)  # (1,)
    return cosmo_z_vector(cosmo_vec, z)[0]  # (8,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="prepare_maps.py output (low/high nside stacks)")
    p.add_argument("--model", required=True)
    p.add_argument("--patch-dir", required=True, help="only used to recover the held-out cosmology split")
    p.add_argument("--cosmo", default=None, help="default: first held-out (val) cosmology")
    p.add_argument("--run", default="run_0")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[0, 34, 68],
                   help="shells to render as individual cl_shell*.png (2-panel Cl+ratio); "
                        "pass with no values to skip these and only build example_full_sky.png")
    p.add_argument("--nside-centers", type=int, default=None,
                   help="default: auto-scaled from the data's nside so patch overlap "
                        "is consistent (see analysis.patch_tiling.auto_nside_centers)")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--n-ode-steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--example-shells", type=int, nargs="+", default=None,
                   help="if given, ALSO render example_full_sky.png: one row per shell "
                        "(gnomonic zoom of low/corrected/high + real Cl ratio-to-truth)")
    p.add_argument("--example-rot", type=float, nargs=2, default=[45.0, 45.0],
                   help="(lon, lat) deg center of the zoom crop in example_full_sky.png")
    p.add_argument("--example-reso", type=float, default=1.5, help="arcmin/pixel of the zoom crop")
    p.add_argument("--example-xsize", type=int, default=200, help="pixel size of the zoom crop")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location=dev)
    cfg = ckpt.get("args", {})
    # older checkpoints predate this flag entirely (no cosmo_mlp in their state_dict),
    # so the fallback must be False, not the FlowUNet default of True.
    use_cosmo_cond = bool(cfg.get("use_cosmo_cond", False))
    net = FlowUNet(in_channels=1, out_channels=1,
                   base_channels=int(cfg.get("base_channels", 32)),
                   time_emb_dim=int(cfg.get("time_emb_dim", 128)),
                   use_cosmo_cond=use_cosmo_cond).to(dev)
    net.load_state_dict(ckpt["model"]); net.eval()
    print(f"[infer_full_sky] checkpoint use_cosmo_cond={use_cosmo_cond}", flush=True)

    if args.cosmo is None:
        _, _, val_cosmos = split_by_cosmo(args.patch_dir, args.val_frac, args.seed)
        cosmo = val_cosmos[0]
        print(f"[infer_full_sky] --cosmo not given, using held-out cosmology {cosmo} "
              f"(full held-out set: {val_cosmos})", flush=True)
    else:
        cosmo = args.cosmo

    # nside MUST match what this checkpoint was actually trained on -- prepare_maps.py
    # has been run at multiple nsides for different pipelines, so a run_dir can contain
    # low_shells_nside=2048.npy AND low_shells_nside=512.npy side by side. Glob-sorting
    # those and taking the first match silently picked whichever nside sorted first
    # alphabetically ("2048" < "512"), NOT the nside this checkpoint saw in training --
    # feeding a 512-trained model 2048 patches (4x finer angular scale/pixel than
    # training) would silently produce a meaningless evaluation. The patch dataset's own
    # metadata is authoritative for what this checkpoint actually trained on.
    meta = np.load(Path(args.patch_dir) / "metadata.npy")
    meta_nside = meta["nside_source"]
    nside = int(np.unique(meta_nside)[0])
    if len(np.unique(meta_nside)) > 1:
        raise RuntimeError(f"--patch-dir mixes multiple source nsides: {np.unique(meta_nside)}")

    run_dir = Path(args.data_root) / cosmo / args.run
    low_all = np.load(run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
    high_all = np.load(run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
    nside_centers = args.nside_centers or auto_nside_centers(nside, args.patch_size)
    print(f"[infer_full_sky] {cosmo}/{args.run} nside={nside} | "
          f"nside_centers={nside_centers} ({12*nside_centers**2:,} centers) "
          f"patch_size={args.patch_size} ODE steps={args.n_ode_steps}", flush=True)

    def reconstruct(low_shell, s):
        cosmo_z = lookup_cosmo_z(meta, cosmo, s) if use_cosmo_cond else None
        predict_batch = make_predict_batch(net, dev, args.n_ode_steps, cosmo_z=cosmo_z)
        return reconstruct_shell(predict_batch, low_shell, nside_centers,
                                 args.patch_size, args.batch_size)

    lmax = min(args.lmax, 3 * nside - 1)
    ells = np.arange(lmax + 1)
    for s in args.shell_indices:
        low_shell = np.asarray(low_all[s], np.float32)
        high_shell = np.asarray(high_all[s], np.float32)
        print(f"[infer_full_sky] shell {s}: tiling + predicting...", flush=True)
        pred_filled = reconstruct(low_shell, s)
        cl_lo, cl_c, cl_hi = od_cl(low_shell, lmax), od_cl(pred_filled, lmax), od_cl(high_shell, lmax)
        plot_cl_shell(s, ells, cl_lo, cl_c, cl_hi, out_dir / f"cl_shell{s:03d}.png")

    # example_full_sky.png: one row per shell (gnomonic zoom of the full-sky
    # reconstruction, log1p overdensity to match example_patches.png's units) + the
    # real angular Cl ratio-to-truth.
    if args.example_shells:
        print(f"[infer_full_sky] building example_full_sky.png for shells "
              f"{args.example_shells}", flush=True)
        lon, lat = args.example_rot
        crop = lambda m: gnomonic_crop(m, nside, lon, lat, args.example_xsize, args.example_reso)

        rows = []
        mom_low, mom_pred, mom_high, hist_rows = [], [], [], []
        for s in args.example_shells:
            low_shell = np.asarray(low_all[s], np.float32)
            high_shell = np.asarray(high_all[s], np.float32)
            print(f"[infer_full_sky] shell {s}: tiling + predicting (example row)...", flush=True)
            pred_filled = reconstruct(low_shell, s)
            cl_lo, cl_c, cl_hi = od_cl(low_shell, lmax), od_cl(pred_filled, lmax), od_cl(high_shell, lmax)
            rows.append((f"shell {s}", crop(log1p_delta(low_shell)), crop(log1p_delta(pred_filled)),
                        crop(log1p_delta(high_shell)), ells, cl_lo, cl_c, cl_hi))

            # full-sky one-point PDF (raw counts, all pixels) -- the marginal-
            # distribution check a Cl ratio (two-point, phase-blind) can't provide.
            mom_low.append(moments(low_shell)); mom_high.append(moments(high_shell))
            mom_pred.append(moments(pred_filled))
            hist_rows.append((f"shell {s}", low_shell.ravel(), pred_filled.ravel(), high_shell.ravel()))

        plot_example_full_sky_grid(
            rows, out_dir / "example_full_sky.png", corrected_label="flow-corrected",
            suptitle=f"full-sky reconstruction, log1p overdensity (gnomonic zoom @ "
                    f"lon={lon:g},lat={lat:g}) + real angular Cl ratio")

        plot_moments_vs_shell(
            args.example_shells, {"low": mom_low, "high (true)": mom_high, "flow pred": mom_pred},
            out_dir / "moments_vs_shell.png",
            suptitle=f"moments vs. shell depth -- full-sky reconstruction (raw counts)\n{cosmo}/{args.run}")
        plot_histogram_grid(
            hist_rows, out_dir / "example_histograms.png", corrected_label="flow-corrected",
            suptitle=f"full-sky raw pixel-count histogram per shell\n{cosmo}/{args.run}")

    print(f"[infer_full_sky] figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
