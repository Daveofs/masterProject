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
                     cosmo_z_vector)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.plotting import (plot_example_patch_grid, plot_pctile_band_ratio,  # noqa: E402
                               plot_histogram_grid, plot_moments_vs_shell)
from analysis.moments import moments                      # noqa: E402


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
    # (shell_idx field) -- for the example_patches.png rows specifically.
    meta_shell_idx = np.load(Path(args.patch_dir) / "metadata.npy")["shell_idx"]
    val_shell_idx = meta_shell_idx[val_idx]
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

    print(f"[eval] figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
