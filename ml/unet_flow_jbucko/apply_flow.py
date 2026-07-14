#!/usr/bin/env python3
"""Apply a trained jbucko flow model to HELD-OUT test patches and evaluate.

Embeds his code directly: FlowUNet + sample_ode from flow_model.py, and the patch
loading / split / transform from dataset.py. Nothing is reimplemented here.

Test data = the validation (held-out COSMOLOGY) split of the patch dataset -- cosmologies
the model never saw in training (dataset.split_by_cosmo). For each patch we integrate the
flow ODE from x0 = low (log1p-delta) to a predicted high, then compare corrected vs the
true high by:
  * the 2D-FFT radial power spectrum ratio (the flat-patch analogue of the C_ell ratio --
    the statistic we care about: does the correction restore small-scale power?),
  * example low / corrected / high patch triptychs (judge by eye).

Because a flow SAMPLE (not a conditional mean) carries the full high-field variance, the
corrected power should track the truth at small scales where a deterministic regressor
would sag -- that is the whole reason for this approach.

  python apply_flow.py --patch-dir <dir> --model <run>/best.pt --out-dir <run>/eval
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# embed his files: this script lives alongside flow_model.py / dataset.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_model import FlowUNet, sample_ode              # noqa: E402
from dataset import (PatchDataset, split_by_cosmo, raw_to_log1p_delta_pair,  # noqa: E402
                     cosmo_z_vector, COSMO_FIELDS)

# every plotting/tiling/full-sky-power routine comes from ../analysis -- nothing
# reimplemented here (this is the ONLY diagnostics script for the flow model now;
# infer_full_sky.py was merged in and deleted, see the full-sky section below).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.plotting import (plot_example_patch_grid, plot_pctile_band_ratio,  # noqa: E402
                               plot_histogram_grid, plot_moments_vs_shell,
                               plot_cl_shell, plot_cl_ratio_pctile_grid,
                               plot_kappa_cl_multi_cosmo, plot_kappa_moments_scatter)
from analysis.moments import moments                      # noqa: E402
from analysis.patch_tiling import auto_nside_centers, reconstruct_shell  # noqa: E402
from analysis.full_sky import od_cl, zbin_shell_samples    # noqa: E402
from analysis import weak_lensing                          # noqa: E402


def stack_cosmo_z(ds: PatchDataset, dev, dtype) -> torch.Tensor:
    """Stack (cosmo, z) across every sample in ds, on dev, into the (N,8) cosmo_z
    tensor FlowUNet expects (see dataset.cosmo_z_vector) -- the eval-time analogue
    of train_flow.py's per-batch cosmo_z construction."""
    cosmo = torch.stack([ds[i]["cosmo"] for i in range(len(ds))]).to(dev, dtype)
    z = torch.tensor([ds[i]["z"] for i in range(len(ds))], device=dev, dtype=dtype)
    return cosmo_z_vector(cosmo, z)


def radial_power_batch(imgs: torch.Tensor, n_bins: int = None):
    """(B,1,H,W) -> (B,n_bins) PER-SAMPLE 2D-FFT power per radial wavenumber bin.

    Batched-on-GPU torch version, used for the aggregate power_spectrum_ratio.png
    (mean over samples) AND the pctile-band plot (needs the per-sample spread, not
    just the mean) -- a deliberate, performance-only duplication of
    analysis.radial_power's single-image numpy version, which the example_patches.png
    rows below use instead (a handful of images, GPU batching buys nothing there)."""
    B, _, H, W = imgs.shape
    dev = imgs.device
    n_bins = n_bins or H // 2
    fy = torch.fft.fftfreq(H, device=dev) * H
    fx = torch.fft.rfftfreq(W, device=dev) * W
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = torch.clamp((r / r.max() * (n_bins - 1)).long(), 0, n_bins - 1).view(-1)
    counts = torch.bincount(bins, minlength=n_bins).clamp(min=1).float()
    f = torch.fft.rfft2(imgs.squeeze(1))
    power = (f.real ** 2 + f.imag ** 2).view(B, -1)              # (B, H*(W//2+1))
    binned = torch.zeros(B, n_bins, device=dev).scatter_add_(1, bins.expand(B, -1), power)
    return (binned / counts).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True)
    p.add_argument("--model", required=True, help="checkpoint (best.pt / last.pt)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=512, help="held-out patches to evaluate")
    p.add_argument("--eval-batch", type=int, default=64,
                   help="mini-batch size for the ODE sampling pass (memory control)")
    p.add_argument("--example-shells", type=int, nargs="+", default=[5, 20, 40, 60],
                   help="shell indices to show as rows in example_patches.png (one "
                        "held-out patch per shell, picked via patch metadata)")
    p.add_argument("--steps", type=int, default=8, help="Euler ODE steps")
    p.add_argument("--n-stat-patches", type=int, default=64,
                   help="held-out patches PER --example-shells shell to pool for "
                        "moments_vs_shell.png / example_histograms.png -- the "
                        "one-point PDF (esp. skew/kurtosis of a sparse count field) "
                        "is far noisier per-patch than the power ratio, so this "
                        "needs many patches, unlike the single visual example patch")
    # --- optional full-sky reconstruction + REAL angular Cl (skipped unless
    # --data-root is given): the power-ratio panels above are a flat 2D-FFT on one
    # --patch-dir patch, bounded by that patch's own Nyquist wavenumber -- nowhere
    # near the full ell~3000 range a real spherical-harmonic transform covers. This
    # tiles the WHOLE sphere (analysis.patch_tiling) and runs healpy.anafast
    # (analysis.full_sky.od_cl) to see the real high-ell behavior. ---
    p.add_argument("--data-root", default=None,
                   help="prepare_maps.py output (full-sky low/high shell stacks). "
                        "If given, ALSO reconstructs the whole sphere and computes "
                        "the real full-sky Cl -> cl_shell*.png, example_full_sky.png, "
                        "fullsky_moments_vs_shell.png, fullsky_example_histograms.png")
    p.add_argument("--cosmo", default=None,
                   help="cosmology for the full-sky reconstruction; default: first "
                        "held-out (val) cosmology")
    p.add_argument("--run", default="run_0")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[0, 34, 68],
                   help="shells to render as individual cl_shell*.png (2-panel "
                        "Cl+ratio); pass with no values to skip these")
    p.add_argument("--nside-centers", type=int, default=None,
                   help="default: auto-scaled from the data's nside so patch overlap "
                        "is consistent (see analysis.patch_tiling.auto_nside_centers)")
    p.add_argument("--fullsky-patch-size", type=int, default=256,
                   help="gnomonic tile size for full-sky reconstruction -- should "
                        "match the patch size this checkpoint was trained on")
    p.add_argument("--lmax", type=int, default=3000)
    # --- example_full_sky.png: Cl-ratio-by-redshift-bin pctile grid (rows = held-out
    # cosmologies, columns = redshift/shell bins) -- no images (see example_patches.png
    # for those), just the aggregate two-point check. Each cell needs
    # --n-shells-per-zbin full-sky reconstructions, so rows*cols*shells-per-zbin is
    # the real cost knob -- these defaults are deliberately small; widen once the
    # per-shell cost on this setup is known (same philosophy as --shell-indices above). ---
    p.add_argument("--zbin-start", type=int, default=9,
                   help="first shell in the Cl-ratio-by-redshift-bin grid -- lower "
                        "shells are typically too sparse for a meaningful Cl ratio "
                        "(see analysis.transforms's eps-clipping note)")
    p.add_argument("--n-zbins", type=int, default=3,
                   help="number of redshift/shell-index bins (columns) spanning "
                        "[--zbin-start, last shell]")
    p.add_argument("--n-shells-per-zbin", type=int, default=5,
                   help="shells sampled (evenly spaced) per bin, each fully "
                        "reconstructed+Cl'd -- more = smoother percentile band, more compute")
    p.add_argument("--max-cosmologies", type=int, default=3,
                   help="held-out cosmologies to include as rows (each row costs "
                        "--n-zbins * --n-shells-per-zbin full-sky reconstructions)")
    # --- weak-lensing kappa map diagnostic (analysis.weak_lensing): reduces the
    # WHOLE usable lightcone [--kappa-zi, --kappa-zf] of low/corrected/high into one
    # kappa map each via UFalcon, for EVERY held-out cosmology (not capped by
    # --max-cosmologies -- one kappa map per cosmology is comparatively cheap once
    # its shells exist, but "corrected" needs every usable shell reconstructed via
    # full-sky tiling, not just a --n-shells-per-zbin sample, so this is still the
    # most expensive optional section here). Off by default. ---
    p.add_argument("--kappa", action="store_true",
                   help="build weak-lensing kappa maps (low/corrected/high) for "
                        "EVERY held-out cosmology and compute their Cl + moments. "
                        "Requires --data-root. Expensive: see module docstring.")
    p.add_argument("--kappa-nz",
                   default="/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin1.txt",
                   help="n(z) redshift distribution passed to UFalcon's "
                        "construct_kappa_map -- named explicitly in the kappa "
                        "plots' suptitle so the choice is never ambiguous")
    p.add_argument("--kappa-nside", type=int, default=128,
                   help="output kappa map nside (independent of --fullsky-patch-size)")
    p.add_argument("--kappa-zi", type=float, default=0.0)
    p.add_argument("--kappa-zf", type=float, default=1.05)
    p.add_argument("--kappa-lmax", type=int, default=350,
                   help="angular power spectrum lmax for the kappa maps "
                        "(--kappa-nside supports up to ~3*nside-1)")
    p.add_argument("--kappa-max-cosmologies", type=int, default=3,
                   help="held-out cosmologies to build kappa maps for. This is THE "
                        "cost knob of the whole script: each one needs every usable "
                        "shell (~47 of 69 for zf=1.05) fully reconstructed by tiling "
                        "the sphere and integrating the flow ODE per patch. Since "
                        "the tiling geometry got cached (analysis.patch_tiling), the "
                        "GPU ODE integration is what dominates -- so bound this "
                        "rather than the (now cheap) tiling. 0 = all held-out.")
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
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"[eval] checkpoint use_cosmo_cond={use_cosmo_cond}", flush=True)

    # held-out cosmologies = our test data
    _, val_idx, val_cosmos = split_by_cosmo(args.patch_dir, args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed)

    # one held-out patch per requested shell, via the patch dataset's own metadata
    # (shell_idx field) -- for the example_patches.png rows specifically. Also
    # reused below by the full-sky section (cosmo params + shell redshift lookup).
    meta = np.load(Path(args.patch_dir) / "metadata.npy")
    val_shell_idx = meta["shell_idx"][val_idx]
    example_pick, example_shells = [], []
    for s in args.example_shells:
        cand = val_idx[val_shell_idx == s]
        if len(cand) == 0:
            print(f"[eval] WARNING: no held-out patch for shell {s}, skipping", flush=True)
            continue
        example_pick.append(int(rng.choice(cand)))
        example_shells.append(s)

    pick = val_idx[rng.permutation(len(val_idx))[:args.n_eval]]
    ds = PatchDataset(args.patch_dir, pick)
    print(f"[eval] {len(val_idx)} held-out patches from {len(val_cosmos)} cosmologies "
          f"{val_cosmos}; evaluating {len(pick)} | ODE steps={args.steps}", flush=True)

    low_all = torch.stack([ds[i]["low"] for i in range(len(ds))])       # (N,1,H,W) raw, CPU
    high_all = torch.stack([ds[i]["high"] for i in range(len(ds))])
    cosmo_z_all = stack_cosmo_z(ds, dev, low_all.dtype)                 # (N,8)
    # ODE-sample in mini-batches: the full n_eval stack through an 8-level UNet at once
    # is needlessly memory-heavy (and fragile on a shared/partially-occupied GPU) --
    # only the final comparison needs everything gathered, not the forward pass.
    mb = args.eval_batch
    low_log_parts, high_log_parts, corr_log_parts = [], [], []
    for b in range(0, low_all.shape[0], mb):
        lo = low_all[b:b + mb].to(dev); hi = high_all[b:b + mb].to(dev)
        lo_log, hi_log = raw_to_log1p_delta_pair(lo, hi)
        with torch.no_grad():
            co_log = sample_ode(net, lo_log, n_steps=args.steps, cosmo_z=cosmo_z_all[b:b + mb])
        low_log_parts.append(lo_log); high_log_parts.append(hi_log); corr_log_parts.append(co_log)
    low_log = torch.cat(low_log_parts); high_log = torch.cat(high_log_parts)
    corr_log = torch.cat(corr_log_parts)

    mse_low = torch.mean((low_log - high_log) ** 2).item()
    mse_corr = torch.mean((corr_log - high_log) ** 2).item()

    pr_low_stack = radial_power_batch(low_log)
    pr_corr_stack = radial_power_batch(corr_log)
    pr_high_stack = radial_power_batch(high_log)
    pr_low, pr_corr, pr_high = pr_low_stack.mean(0), pr_corr_stack.mean(0), pr_high_stack.mean(0)
    k = np.arange(len(pr_high))
    with np.errstate(divide="ignore", invalid="ignore"):
        lo_r_stack, co_r_stack = pr_low_stack / pr_high_stack, pr_corr_stack / pr_high_stack
    lo_r, co_r = pr_low / pr_high, pr_corr / pr_high

    def band(r, a, b):
        return float(np.nanmean(r[a:b]))
    nb = len(k)
    print(f"[eval] MSE(log-delta) low={mse_low:.4e} corrected={mse_corr:.4e} "
          f"({100*(1-mse_corr/max(mse_low,1e-12)):+.0f}%)", flush=True)
    print(f"[eval] power ratio to high | low  (mid,high-k)=({band(lo_r,nb//4,nb//2):.3f},"
          f"{band(lo_r,nb//2,nb):.3f}) | corrected=({band(co_r,nb//4,nb//2):.3f},"
          f"{band(co_r,nb//2,nb):.3f})   [target 1.0]", flush=True)

    # --- power spectrum ratio plot ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].loglog(k[1:], pr_low[1:], ":", color="seagreen", label="low (DISCO)")
    ax[0].loglog(k[1:], pr_corr[1:], "-", color="steelblue", label="flow-corrected")
    ax[0].loglog(k[1:], pr_high[1:], "--", color="tomato", label="high (CosmoGrid)")
    ax[0].set_xlabel("radial wavenumber bin"); ax[0].set_ylabel("power")
    ax[0].legend(fontsize=8); ax[0].set_title("2D-FFT radial power (mean over held-out patches)")
    ax[1].semilogx(k[1:], lo_r[1:], ":", color="seagreen", label="low/high")
    ax[1].semilogx(k[1:], co_r[1:], "-", color="steelblue", label="corrected/high")
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0.4, 1.6)
    ax[1].set_xlabel("radial wavenumber bin"); ax[1].set_ylabel("power ratio to high")
    ax[1].legend(fontsize=8); ax[1].set_title("power ratio (the target metric)")
    fig.tight_layout(); fig.savefig(out_dir / "power_spectrum_ratio.png", dpi=150)
    plt.close(fig)

    # --- pctile-band ratio plot: same statistic as above, but showing the per-patch
    # spread (not just the mean) so a systematic bias is distinguishable from noise ---
    plot_pctile_band_ratio(
        k, {"low / high (baseline, no model)": lo_r_stack, "flow pred / high": co_r_stack},
        out_dir / "power_ratio_pctile_band.png", xlabel="radial wavenumber bin",
        ylim=(0.4, 1.6),
        title=f"power ratio: flow vs baseline ({len(pick)} val patches, 16-84th pctile band)")

    # --- example triptychs + per-patch power ratio (4th column), one row per shell ---
    # shared with every other pipeline (analysis.plotting.plot_example_patch_grid) so
    # the figure is visually identical by construction, not by convention.
    ns = len(example_pick)
    ex_ds = PatchDataset(args.patch_dir, np.array(example_pick, dtype=np.int64))
    ex_low = torch.stack([ex_ds[i]["low"] for i in range(ns)]).to(dev)
    ex_high = torch.stack([ex_ds[i]["high"] for i in range(ns)]).to(dev)
    ex_cosmo_z = stack_cosmo_z(ex_ds, dev, ex_low.dtype)
    ex_low_log, ex_high_log = raw_to_log1p_delta_pair(ex_low, ex_high)
    with torch.no_grad():
        ex_corr_log = sample_ode(net, ex_low_log, n_steps=args.steps, cosmo_z=ex_cosmo_z)

    rows = [(f"shell {example_shells[i]}", ex_low_log[i, 0].cpu().numpy(),
            ex_corr_log[i, 0].cpu().numpy(), ex_high_log[i, 0].cpu().numpy())
           for i in range(ns)]
    plot_example_patch_grid(rows, out_dir / "example_patches.png",
                            corrected_label="flow-corrected",
                            suptitle="held-out test patches (log1p overdensity) + "
                                     "per-patch power ratio")
    

    # --- per-shell moments + histograms (raw counts) -- pools ALL held-out patches of
    # each --example-shells shell (capped by --n-stat-patches), not just the single
    # visual example patch above, since the one-point PDF is much noisier per-patch. ---
    n_stat = args.n_stat_patches
    moment_shells, mom_low, mom_corr, mom_high, hist_rows = [], [], [], [], []
    for s in example_shells:
        cand = val_idx[val_shell_idx == s]
        cand = cand[rng.permutation(len(cand))[:n_stat]]
        sd = PatchDataset(args.patch_dir, cand)
        s_low = torch.stack([sd[i]["low"] for i in range(len(sd))]).to(dev)
        s_high = torch.stack([sd[i]["high"] for i in range(len(sd))]).to(dev)
        s_cosmo_z = stack_cosmo_z(sd, dev, s_low.dtype)
        s_low_mean = s_low.mean(dim=(2, 3), keepdim=True)
        s_low_log, _ = raw_to_log1p_delta_pair(s_low, s_high)
        with torch.no_grad():
            s_corr_log = sample_ode(net, s_low_log, n_steps=args.steps, cosmo_z=s_cosmo_z)
        s_corr_raw = (1.0 + torch.expm1(s_corr_log)) * s_low_mean

        low_np, high_np, corr_np = s_low.cpu().numpy(), s_high.cpu().numpy(), s_corr_raw.cpu().numpy()
        moment_shells.append(s)
        mom_low.append(moments(low_np)); mom_corr.append(moments(corr_np)); mom_high.append(moments(high_np))
        hist_rows.append((f"shell {s}", low_np.ravel(), corr_np.ravel(), high_np.ravel()))
        print(f"[eval] shell {s} moments (n={len(sd)} patches): low var={mom_low[-1]['variance']:.3g} "
              f"flow-pred var={mom_corr[-1]['variance']:.3g} high var={mom_high[-1]['variance']:.3g}",
              flush=True)

    plot_moments_vs_shell(
        moment_shells, {"low": mom_low, "high (true)": mom_high, "flow pred": mom_corr},
        out_dir / "moments_vs_shell.png",
        suptitle=f"moments vs. shell depth ({n_stat} held-out patches/shell, raw counts)")
    plot_histogram_grid(
        hist_rows, out_dir / "example_histograms.png", corrected_label="flow-corrected",
        suptitle=f"held-out patches, raw pixel-count histogram per shell ({n_stat} patches/shell)")

    # ============= optional: full-sky reconstruction + REAL angular Cl =============
    # Skipped entirely unless --data-root is given. Tiles the WHOLE sphere (not one
    # flat patch) via analysis.patch_tiling and runs the real spherical-harmonic
    # transform (analysis.full_sky.od_cl) -- the genuine "how does it behave at very
    # high ell" answer the flat 2D-FFT power ratio above structurally cannot give
    # (bounded by that patch's own Nyquist wavenumber).
    if args.data_root:
        nside = int(np.unique(meta["nside_source"])[0])
        if len(np.unique(meta["nside_source"])) > 1:
            raise RuntimeError(f"--patch-dir mixes multiple source nsides: {np.unique(meta['nside_source'])}")
        nside_centers = args.nside_centers or auto_nside_centers(nside, args.fullsky_patch_size)
        lmax = min(args.lmax, 3 * nside - 1)
        ells = np.arange(lmax + 1)

        def lookup_cosmo_z_fs(cosmo_name: str, shell_idx: int) -> np.ndarray:
            """(8,) cosmo_z vector for one full-sky shell/cosmology -- cosmo params
            constant per cosmology, redshift constant per shell_idx, both looked up
            from the patch dataset's own metadata (authoritative, same source
            train_flow.py used)."""
            rows_cosmo = meta[meta["cosmo"] == cosmo_name]
            if len(rows_cosmo) == 0:
                raise ValueError(f"lookup_cosmo_z_fs: no patch-dir metadata row for "
                                 f"cosmology {cosmo_name!r}")
            cosmo_vec = np.array([[rows_cosmo[0][f] for f in COSMO_FIELDS]], dtype=np.float32)
            rows_shell = meta[meta["shell_idx"] == shell_idx]
            if len(rows_shell) == 0:
                raise ValueError(f"lookup_cosmo_z_fs: no patch-dir metadata row for "
                                 f"shell_idx={shell_idx} (any cosmology)")
            z = np.array([0.5 * (rows_shell[0]["lower_z"] + rows_shell[0]["upper_z"])], dtype=np.float32)
            return cosmo_z_vector(cosmo_vec, z)[0]

        def make_predict_batch_fs(cosmo_z_vec: np.ndarray | None):
            cosmo_z_t = None if cosmo_z_vec is None else torch.from_numpy(cosmo_z_vec).to(dev)

            def predict_batch(low_batch: np.ndarray) -> np.ndarray:
                low_t = torch.from_numpy(low_batch).unsqueeze(1).to(dev)
                low_mean = low_t.mean(dim=(2, 3), keepdim=True)
                eps = 0.5 / low_mean
                low_log = torch.log1p(torch.maximum(low_t / low_mean - 1.0, -1.0 + eps))
                cz = (None if cosmo_z_t is None else
                     cosmo_z_t.unsqueeze(0).expand(low_t.shape[0], -1).to(low_log.dtype))
                with torch.no_grad():
                    pred_log = sample_ode(net, low_log, n_steps=args.steps, cosmo_z=cz)
                pred_delta = torch.expm1(pred_log)
                return ((1.0 + pred_delta) * low_mean).squeeze(1).cpu().numpy()
            return predict_batch

        def reconstruct(low_shell, cosmo_name, shell_idx):
            cosmo_z_vec = lookup_cosmo_z_fs(cosmo_name, shell_idx) if use_cosmo_cond else None
            predict_batch = make_predict_batch_fs(cosmo_z_vec)
            return reconstruct_shell(predict_batch, low_shell, nside_centers,
                                     args.fullsky_patch_size, args.eval_batch)

        # --- single-cosmology diagnostics: standalone cl_shell*.png + full-sky
        # moments/histograms (--cosmo, default: first held-out cosmology) ---
        fs_cosmo = args.cosmo
        if fs_cosmo is None:
            fs_cosmo = val_cosmos[0]
            print(f"[eval] --cosmo not given for full-sky reconstruction, using "
                  f"held-out cosmology {fs_cosmo} (full held-out set: {val_cosmos})", flush=True)

        run_dir = Path(args.data_root) / fs_cosmo / args.run
        low_full_all = np.load(run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
        high_full_all = np.load(run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
        print(f"[eval] full-sky: {fs_cosmo}/{args.run} nside={nside} "
              f"nside_centers={nside_centers} ({12*nside_centers**2:,} centers) "
              f"patch_size={args.fullsky_patch_size}", flush=True)

        for s in args.shell_indices:
            low_shell = np.asarray(low_full_all[s], np.float32)
            high_shell = np.asarray(high_full_all[s], np.float32)
            print(f"[eval] full-sky shell {s}: tiling + predicting...", flush=True)
            pred_filled = reconstruct(low_shell, fs_cosmo, s)
            cl_lo, cl_c, cl_hi = od_cl(low_shell, lmax), od_cl(pred_filled, lmax), od_cl(high_shell, lmax)
            plot_cl_shell(s, ells, cl_lo, cl_c, cl_hi, out_dir / f"cl_shell{s:03d}.png")

        if example_shells:
            fs_mom_low, fs_mom_pred, fs_mom_high, fs_hist_rows = [], [], [], []
            for s in example_shells:
                low_shell = np.asarray(low_full_all[s], np.float32)
                high_shell = np.asarray(high_full_all[s], np.float32)
                print(f"[eval] full-sky shell {s}: tiling + predicting (moments/hist)...", flush=True)
                pred_filled = reconstruct(low_shell, fs_cosmo, s)
                fs_mom_low.append(moments(low_shell)); fs_mom_high.append(moments(high_shell))
                fs_mom_pred.append(moments(pred_filled))
                fs_hist_rows.append((f"shell {s}", low_shell.ravel(), pred_filled.ravel(), high_shell.ravel()))
            plot_moments_vs_shell(
                example_shells, {"low": fs_mom_low, "high (true)": fs_mom_high, "flow pred": fs_mom_pred},
                out_dir / "fullsky_moments_vs_shell.png",
                suptitle=f"moments vs. shell depth -- full-sky reconstruction (raw counts)\n"
                        f"{fs_cosmo}/{args.run}")
            plot_histogram_grid(
                fs_hist_rows, out_dir / "fullsky_example_histograms.png", corrected_label="flow-corrected",
                suptitle=f"full-sky raw pixel-count histogram per shell\n{fs_cosmo}/{args.run}")

        # --- example_full_sky.png: Cl-ratio-by-redshift-bin pctile grid. One row per
        # held-out cosmology (up to --max-cosmologies), one column per redshift/shell
        # bin (analysis.full_sky.zbin_shell_samples) -- percentile band across
        # --n-shells-per-zbin sampled shells per bin. No images (already in
        # example_patches.png); this is purely the aggregate two-point check. ---
        n_shells_total = low_full_all.shape[0]
        zbins = zbin_shell_samples(n_shells_total, args.zbin_start, args.n_zbins,
                                   args.n_shells_per_zbin)
        grid_cosmos = list(val_cosmos[:args.max_cosmologies])
        print(f"[eval] full-sky: Cl-ratio-by-redshift-bin grid for {len(grid_cosmos)} "
              f"cosmologies {grid_cosmos} x {len(zbins)} bins {[b[0] for b in zbins]}", flush=True)

        grid = []
        for c in grid_cosmos:
            c_run_dir = Path(args.data_root) / c / args.run
            c_low_all = np.load(c_run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
            c_high_all = np.load(c_run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
            panels = []
            for bin_label, shells in zbins:
                lo_stack, co_stack = [], []
                for s in shells:
                    low_shell = np.asarray(c_low_all[int(s)], np.float32)
                    high_shell = np.asarray(c_high_all[int(s)], np.float32)
                    print(f"[eval] full-sky {c} shell {s} ({bin_label}): "
                          f"tiling + predicting...", flush=True)
                    pred_filled = reconstruct(low_shell, c, int(s))
                    cl_lo = od_cl(low_shell, lmax); cl_c = od_cl(pred_filled, lmax)
                    cl_hi = od_cl(high_shell, lmax)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lo_stack.append(cl_lo / cl_hi); co_stack.append(cl_c / cl_hi)
                panels.append((bin_label, shells, ells, np.array(lo_stack), np.array(co_stack)))
            grid.append((f"{c}/{args.run}", panels))

        plot_cl_ratio_pctile_grid(
            grid, out_dir / "example_full_sky.png",
            suptitle="Full-sky Cl ratio by redshift bin (flow)")

        # --- weak-lensing kappa map diagnostic (analysis.weak_lensing): reduces
        # the WHOLE usable lightcone into ONE kappa map each for low/corrected/high,
        # for EVERY held-out cosmology -- the comparison this whole pipeline is
        # ultimately in service of (kappa maps, not individual shells). Each
        # cosmology's OWN params.yml is loaded fresh (H0, Om, Ob, O_nu, ... --
        # "further values" beyond the 6 COSMO_FIELDS this script otherwise uses)
        # since weak_lensing_ufalcon.py's hardcoded example cosmology was checked
        # and does NOT match the lightcone it actually loads. ---
        if args.kappa:
            kappa_cosmos = (list(val_cosmos) if args.kappa_max_cosmologies <= 0
                           else list(val_cosmos[:args.kappa_max_cosmologies]))
            print(f"[eval] kappa: building kappa maps for {len(kappa_cosmos)} of "
                  f"{len(val_cosmos)} held-out cosmologies {kappa_cosmos} "
                  f"(zi={args.kappa_zi}, zf={args.kappa_zf}, "
                  f"nside={args.kappa_nside}), n(z)={args.kappa_nz}", flush=True)
            kappa_cosmo_labels = []
            kappa_cl_low, kappa_cl_corr, kappa_cl_high = [], [], []
            kappa_mom_low, kappa_mom_corr, kappa_mom_high = [], [], []
            for c in kappa_cosmos:
                c_run_dir = Path(args.data_root) / c / args.run
                cosmo_params = weak_lensing.load_cosmo_yaml(c_run_dir)
                shell_info = np.load(c_run_dir / "compressed_shells.npz",
                                     allow_pickle=True, mmap_mode="r")["shell_info"]
                lower_z_all = shell_info["lower_z"]; upper_z_all = shell_info["upper_z"]
                usable = np.where(weak_lensing.usable_shell_mask(
                    lower_z_all, upper_z_all, args.kappa_zi, args.kappa_zf))[0]
                lower_z, upper_z = lower_z_all[usable], upper_z_all[usable]

                c_low_all = np.load(c_run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
                c_high_all = np.load(c_run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
                low_shells = np.stack([np.asarray(c_low_all[int(s)], np.float64) for s in usable])
                high_shells = np.stack([np.asarray(c_high_all[int(s)], np.float64) for s in usable])
                print(f"[eval] kappa {c}: reconstructing {len(usable)} usable shells "
                      f"(z in [{lower_z.min():.3f},{upper_z.max():.3f}])...", flush=True)
                corr_shells = np.stack([
                    reconstruct(np.asarray(c_low_all[int(s)], np.float32), c, int(s)).astype(np.float64)
                    for s in usable])

                kw = dict(nside=args.kappa_nside, zi=args.kappa_zi, zf=args.kappa_zf)
                kappa_low = weak_lensing.kappa_map(low_shells, lower_z, upper_z, cosmo_params, args.kappa_nz, **kw)
                kappa_corr = weak_lensing.kappa_map(corr_shells, lower_z, upper_z, cosmo_params, args.kappa_nz, **kw)
                kappa_high = weak_lensing.kappa_map(high_shells, lower_z, upper_z, cosmo_params, args.kappa_nz, **kw)

                kappa_cosmo_labels.append(f"{c}/{args.run}")
                kappa_cl_low.append(weak_lensing.kappa_cl(kappa_low, args.kappa_lmax))
                kappa_cl_corr.append(weak_lensing.kappa_cl(kappa_corr, args.kappa_lmax))
                kappa_cl_high.append(weak_lensing.kappa_cl(kappa_high, args.kappa_lmax))
                kappa_mom_low.append(moments(kappa_low))
                kappa_mom_corr.append(moments(kappa_corr))
                kappa_mom_high.append(moments(kappa_high))
                print(f"[eval] kappa {c}: done", flush=True)

            kappa_ells = np.arange(args.kappa_lmax + 1)
            kappa_suptitle_common = (f"{len(kappa_cosmo_labels)} held-out cosmologies (flow) | "
                                     f"n(z)={Path(args.kappa_nz).name} | "
                                     f"z in [{args.kappa_zi:g},{args.kappa_zf:g}]")
            plot_kappa_cl_multi_cosmo(
                kappa_cosmo_labels, kappa_ells, kappa_cl_low, kappa_cl_corr, kappa_cl_high,
                out_dir / "kappa_cl_all_cosmologies.png", corrected_label="flow-corrected",
                suptitle=f"weak-lensing kappa Cl, {kappa_suptitle_common}")
            plot_kappa_moments_scatter(
                kappa_cosmo_labels, kappa_mom_low, kappa_mom_corr, kappa_mom_high,
                out_dir / "kappa_moments_scatter.png", corrected_label="flow-corrected",
                suptitle=f"weak-lensing kappa map moments, {kappa_suptitle_common}")
    # ============= end optional full-sky section =============

    print(f"[eval] figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
