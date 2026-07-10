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
from dataset import PatchDataset, split_by_cosmo, raw_to_log1p_delta_pair  # noqa: E402


def radial_power(imgs: torch.Tensor, n_bins: int = None):
    """(B,1,H,W) -> (n_bins,) mean 2D-FFT power per radial wavenumber bin (avg over B)."""
    B, _, H, W = imgs.shape
    n_bins = n_bins or H // 2
    fy = torch.fft.fftfreq(H) * H
    fx = torch.fft.rfftfreq(W) * W
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = torch.clamp((r / r.max() * (n_bins - 1)).long(), 0, n_bins - 1).view(-1)
    counts = torch.bincount(bins, minlength=n_bins).clamp(min=1).float()
    f = torch.fft.rfft2(imgs.squeeze(1))
    power = (f.real ** 2 + f.imag ** 2).view(B, -1).mean(0)     # (H*(W//2+1),)
    binned = torch.zeros(n_bins).scatter_add_(0, bins, power)
    return (binned / counts).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True)
    p.add_argument("--model", required=True, help="checkpoint (best.pt / last.pt)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=512, help="held-out patches to evaluate")
    p.add_argument("--n-show", type=int, default=4, help="example triptychs to render")
    p.add_argument("--steps", type=int, default=8, help="Euler ODE steps")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location=dev)
    cfg = ckpt.get("args", {})
    net = FlowUNet(in_channels=1, out_channels=1,
                   base_channels=int(cfg.get("base_channels", 32)),
                   time_emb_dim=int(cfg.get("time_emb_dim", 128))).to(dev)
    net.load_state_dict(ckpt["model"])
    net.eval()

    # held-out cosmologies = our test data
    _, val_idx, val_cosmos = split_by_cosmo(args.patch_dir, args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed)
    pick = val_idx[rng.permutation(len(val_idx))[:args.n_eval]]
    ds = PatchDataset(args.patch_dir, pick)
    print(f"[eval] {len(val_idx)} held-out patches from {len(val_cosmos)} cosmologies "
          f"{val_cosmos}; evaluating {len(pick)} | ODE steps={args.steps}", flush=True)

    low = torch.stack([ds[i]["low"] for i in range(len(ds))]).to(dev)   # (N,1,H,W) raw
    high = torch.stack([ds[i]["high"] for i in range(len(ds))]).to(dev)
    low_log, high_log = raw_to_log1p_delta_pair(low, high)             # model space
    with torch.no_grad():
        corr_log = sample_ode(net, low_log, n_steps=args.steps)

    mse_low = torch.mean((low_log - high_log) ** 2).item()
    mse_corr = torch.mean((corr_log - high_log) ** 2).item()

    pr_low = radial_power(low_log); pr_corr = radial_power(corr_log); pr_high = radial_power(high_log)
    k = np.arange(len(pr_high))
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

    # --- example triptychs ---
    ns = min(args.n_show, low_log.shape[0])
    vmin = float(high_log[:ns].min()); vmax = float(high_log[:ns].max())
    fig, axes = plt.subplots(ns, 3, figsize=(9, 3 * ns))
    axes = np.atleast_2d(axes)
    for i in range(ns):
        for j, (img, ttl) in enumerate([(low_log, "low (DISCO)"),
                                        (corr_log, "flow-corrected"),
                                        (high_log, "high (CosmoGrid)")]):
            a = axes[i, j]
            a.imshow(img[i, 0].cpu().numpy(), vmin=vmin, vmax=vmax, cmap="viridis")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(ttl, fontsize=10)
    fig.suptitle("held-out test patches (log1p overdensity)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_dir / "example_patches.png", dpi=140)
    plt.close(fig)
    print(f"[eval] figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
