#!/usr/bin/env python3
"""One figure comparing all three correction pipelines on the weak-lensing kappa
map moments -- the summary plot requested in the thesis review.

The construction, exactly as specified:

  for each held-out cosmology c and each moment m in (variance, skewness,
  excess kurtosis):
      ratio_m(c) = m(kappa_model(c)) / m(kappa_high(c))
  then plot  mean_c[ratio_m]  with an errorbar of  std_c[ratio_m]

so each (method, moment) pair collapses to ONE point with an errorbar, and a
perfect method sits at 1.0 with a short bar. The uncorrected DISCO lightcone is
shown the same way as the baseline: the distance between the baseline point and
1.0 is the gap each method is trying to close.

Note on interpretation: the errorbar is the cosmology-to-cosmology SCATTER of the
ratio (a consistency measure), NOT an uncertainty on the mean. A method can sit on
1.0 with a wide bar (right on average, unreliable per cosmology) or slightly off
1.0 with a tight bar (biased but predictable); those are different failure modes
and the plot is meant to distinguish them.

Inputs are the kappa_moments_<tag>.npz files written by each pipeline's --kappa
block (analysis/weak_lensing.save_kappa_moment_summary). They must exist first:
re-run each pipeline's diagnostics once (run_diagnostics_only.sh) if they do not.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_method_comparison.py \
        --npz <transfer_eval>/kappa_moments_bin1.npz \
              <flow_eval>/kappa_moments_bin1.npz \
              <diffusion_eval>/kappa_moments_bin1.npz \
        --out-dir /capstor/scratch/cscs/damrein/outputs/plots/comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# thesis typography, matching vis/plot_cl_ratio.py and ml/analysis/plotting.py
_AXIS_FONTSIZE = 16
_TICK_FONTSIZE = 14
_LEGEND_FONTSIZE = 13

MUTED = "#52514e"
# one colour per pipeline, matching the thesis TikZ palette (report preamble:
# ctrans / cunet / cdiff) so the methods keep the same identity everywhere
METHOD_COLORS = {
    "transfer": "#3F63A6",
    "unet-flow": "#28866A",
    "diffusion": "#7A51C6",
}
BASELINE_COLOR = "#B85F34"

MOMENT_LABELS = {
    "variance": r"variance  $\sigma^2$",
    "skewness": r"skewness  $S_3$",
    "excess_kurtosis": r"excess kurtosis  $K_4$",
}


def load_summary(path: Path):
    with np.load(path, allow_pickle=False) as d:
        return {
            "cosmo_labels": [str(c) for c in d["cosmo_labels"]],
            "moment_keys": [str(k) for k in d["moment_keys"]],
            "mom_low": d["mom_low"], "mom_corr": d["mom_corr"],
            "mom_high": d["mom_high"],
            "method_label": str(d["method_label"]), "tag": str(d["tag"]),
        }


def ratio_stats(num: np.ndarray, den: np.ndarray):
    """mean and std over cosmologies of num/den, per moment column."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    out_m, out_s = [], []
    for j in range(r.shape[1]):
        col = r[:, j]
        col = col[np.isfinite(col)]
        out_m.append(float(np.mean(col)) if col.size else np.nan)
        out_s.append(float(np.std(col)) if col.size else np.nan)
    return np.array(out_m), np.array(out_s)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", nargs="+", required=True,
                   help="kappa_moments_<tag>.npz, one per pipeline (same tag/bin)")
    p.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/comparison")
    p.add_argument("--out-name", default=None)
    args = p.parse_args()

    summaries = [load_summary(Path(f)) for f in args.npz]
    tags = {s["tag"] for s in summaries}
    if len(tags) > 1:
        raise SystemExit(f"inputs mix different n(z) bins {tags}; pass one bin at a time")
    tag = tags.pop()
    keys = summaries[0]["moment_keys"]
    for s in summaries:
        if s["moment_keys"] != keys:
            raise SystemExit("inputs disagree on moment ordering")
    n_cos = {len(s["cosmo_labels"]) for s in summaries}

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(keys), dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=1)

    # The uncorrected baseline is identical data in every file (same low maps), so
    # it is drawn ONCE, from the first summary, rather than three times over.
    base = summaries[0]
    b_mean, b_std = ratio_stats(base["mom_low"], base["mom_high"])
    n_series = len(summaries) + 1
    width = 0.62
    offsets = np.linspace(-width / 2, width / 2, n_series)

    ax.errorbar(x + offsets[0], b_mean, yerr=b_std, fmt="o", ms=9, capsize=5,
                lw=1.6, color=BASELINE_COLOR, label="uncorrected (DISCO)", zorder=3)

    for i, s in enumerate(summaries, start=1):
        m, sd = ratio_stats(s["mom_corr"], s["mom_high"])
        name = s["method_label"]
        ax.errorbar(x + offsets[i], m, yerr=sd, fmt="s", ms=9, capsize=5, lw=1.6,
                    color=METHOD_COLORS.get(name, None), label=name, zorder=3)
        print(f"[comparison] {name:10s} ({tag}): " +
              "  ".join(f"{k}={mm:.3f}+-{ss:.3f}" for k, mm, ss in zip(keys, m, sd)))
    print(f"[comparison] {'uncorrected':10s} ({tag}): " +
          "  ".join(f"{k}={mm:.3f}+-{ss:.3f}" for k, mm, ss in zip(keys, b_mean, b_std)))

    ax.set_xticks(x)
    ax.set_xticklabels([MOMENT_LABELS.get(k, k) for k in keys], fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel(r"model / truth   (mean $\pm$ std over cosmologies)",
                  fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)
    ax.grid(True, axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=_LEGEND_FONTSIZE, ncol=2, loc="best")

    name = args.out_name or f"method_comparison_kappa_moments_{tag}.png"
    out = out_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[comparison] {len(summaries)} methods x {sorted(n_cos)} cosmologies -> {out}")


if __name__ == "__main__":
    main()
