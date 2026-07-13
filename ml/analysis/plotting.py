"""Shared figure builders for every pipeline's example/diagnostic plots
(unet_flow_jbucko, transfer). These take ALREADY-COMPUTED arrays -- how a pipeline
gets its "corrected" data (running a generative model, applying a transfer function,
...) is entirely the caller's business; these functions only know how to draw the
comparison, so the visual format stays identical across pipelines by construction
instead of by convention.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .radial_power import radial_power


def plot_example_patch_grid(rows, out_path, corrected_label="corrected", suptitle=None):
    """rows: list of (row_label, low_log, corr_log, high_log) -- 2D log1p-delta
    patches (see transforms.log1p_delta_pair). One row per shell/example. 4 columns:
    low/corrected/high images + per-patch radial-power-ratio (the flat-patch analogue
    of the C_ell ratio, bounded by that one patch's own Nyquist wavenumber)."""
    ns = len(rows)
    fig, axes = plt.subplots(ns, 4, figsize=(13, 3 * ns),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    axes = np.atleast_2d(axes)
    for i, (label, low_log, corr_log, high_log) in enumerate(rows):
        vmin, vmax = float(high_log.min()), float(high_log.max())
        for j, (img, ttl) in enumerate([(low_log, "low (DISCO)"), (corr_log, corrected_label),
                                        (high_log, "high (CosmoGrid)")]):
            a = axes[i, j]
            a.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(ttl, fontsize=10)
            if j == 0:
                a.set_ylabel(label, fontsize=9)

        pr_low = radial_power(low_log); pr_corr = radial_power(corr_log); pr_high = radial_power(high_log)
        k = np.arange(len(pr_high))
        ai = axes[i, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ai.semilogx(k[1:], (pr_low / pr_high)[1:], ":", color="seagreen", label="low/high")
            ai.semilogx(k[1:], (pr_corr / pr_high)[1:], "-", color="steelblue", label="corrected/high")
        ai.axhline(1, color="k", lw=0.8); ai.set_ylim(0.4, 1.6)
        ai.tick_params(labelsize=8)
        if i == 0:
            ai.set_title("power ratio to truth", fontsize=10)
            ai.legend(fontsize=7, loc="lower left")
        if i == ns - 1:
            ai.set_xlabel("radial wavenumber bin", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] {ns} example patches -> {out_path}", flush=True)
    return out_path


def plot_example_full_sky_grid(rows, out_path, corrected_label="corrected", suptitle=None):
    """rows: list of (row_label, low_crop, corr_crop, high_crop, ells, cl_lo, cl_c, cl_hi)
    -- crops are log1p-delta gnomonic zooms (full_sky.gnomonic_crop of
    transforms.log1p_delta), cl_* are REAL angular power spectra over the WHOLE
    reconstructed sky (full_sky.od_cl on the raw-count maps), not the flat-patch
    approximation. One row per shell. 4th column is the genuine C_ell ratio."""
    ns = len(rows)
    fig, axes = plt.subplots(ns, 4, figsize=(13, 3 * ns),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    axes = np.atleast_2d(axes)
    for i, (label, low_crop, corr_crop, high_crop, ells, cl_lo, cl_c, cl_hi) in enumerate(rows):
        vmin = float(np.nanmin(high_crop)); vmax = float(np.nanmax(high_crop))
        for j, (img, ttl) in enumerate([(low_crop, "low (DISCO)"), (corr_crop, corrected_label),
                                        (high_crop, "high (CosmoGrid)")]):
            a = axes[i, j]
            a.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(ttl, fontsize=10)
            if j == 0:
                a.set_ylabel(label, fontsize=9)

        ai = axes[i, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ai.semilogx(ells[1:], (cl_lo / cl_hi)[1:], ":", color="seagreen", label="low/high")
            ai.semilogx(ells[1:], (cl_c / cl_hi)[1:], "-", color="steelblue", label="corrected/high")
        ai.axhline(1, color="k", lw=0.8); ai.set_ylim(0.0, 1.6)
        ai.tick_params(labelsize=8)
        if i == 0:
            ai.set_title(r"real $C_\ell$ ratio to truth", fontsize=10)
            ai.legend(fontsize=7, loc="lower left")
        if i == ns - 1:
            ai.set_xlabel(r"$\ell$", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] example full-sky grid -> {out_path}", flush=True)
    return out_path


def plot_pctile_band_ratio(x, ratio_stacks: dict, out_path, xlabel=r"$\ell$",
                           ylabel=None, pctile=(16, 84), ylim=None, title=None,
                           suptitle=None):
    """Aggregate, uncertainty-aware ratio-to-truth plot: a single ratio curve (e.g.
    plot_cl_shell's ratio panel) can't distinguish "systematically off" from "noisy
    but unbiased" -- pooling many samples (val patches, Poisson draws, ...) and
    showing the spread answers that.

    x: (Nx,) shared axis (radial wavenumber bin, or ell). ratio_stacks: dict of
    label -> (n_samples, Nx) array of per-sample ratio-to-truth (e.g. one row per
    held-out patch's radial-power ratio, or one row per realization's Cl ratio).
    Draws median line + shaded [pctile[0], pctile[1]] band per label, in insertion
    order (first entry gets the baseline/no-model styling, rest get the model
    colors) -- matches every pipeline's "low/high (baseline)" vs "prediction/high"
    convention."""
    fig, ax = plt.subplots(figsize=(9, 6))
    lo_pct, hi_pct = pctile
    colors = ["gray", "steelblue", "tomato", "seagreen"]
    n_samples = None
    for (label, stack), color in zip(ratio_stacks.items(), colors):
        stack = np.asarray(stack, dtype=np.float64)
        n_samples = stack.shape[0]
        med = np.nanmedian(stack, axis=0)
        p_lo = np.nanpercentile(stack, lo_pct, axis=0)
        p_hi = np.nanpercentile(stack, hi_pct, axis=0)
        ax.semilogx(x, med, "-o", ms=3, color=color, label=label)
        ax.fill_between(x, p_lo, p_hi, color=color, alpha=0.2)

    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or (r"$C_\ell/C_\ell^{true}$" if "ell" in xlabel.lower() else "ratio"))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title or f"ratio to truth ({n_samples} samples, "
                          f"{lo_pct}-{hi_pct}th pctile band)")
    ax.legend(fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] pctile-band ratio ({n_samples} samples) -> {out_path}", flush=True)
    return out_path


def plot_histogram_grid(rows, out_path, corrected_label="corrected", n_bins=60,
                        xlabel="pixel value (raw counts)", suptitle=None):
    """rows: list of (row_label, low_vals, corr_vals, high_vals) -- FLAT 1D arrays of
    raw pixel values (the caller decides scope: one patch, many pooled patches, or a
    whole full-sky map). One row per shell/example; log-y, shared bin edges per row
    spanning all three arrays so zero-spikes/tails are directly comparable -- the
    one-point-PDF analogue of plot_cl_shell's two-point check (see moments.py)."""
    ns = len(rows)
    fig, axes = plt.subplots(ns, 1, figsize=(8, 3 * ns), squeeze=False)
    for i, (label, low_v, corr_v, high_v) in enumerate(rows):
        ax = axes[i, 0]
        lo = float(min(np.min(low_v), np.min(corr_v), np.min(high_v)))
        hi = float(max(np.max(low_v), np.max(corr_v), np.max(high_v)))
        bins = np.linspace(lo, hi, n_bins + 1) if hi > lo else n_bins
        ax.hist(low_v, bins=bins, histtype="step", color="seagreen", lw=1.3,
               density=True, label="low (DISCO)")
        ax.hist(corr_v, bins=bins, histtype="step", color="steelblue", lw=1.3,
               density=True, label=corrected_label)
        ax.hist(high_v, bins=bins, histtype="step", color="tomato", lw=1.3,
               density=True, label="high (CosmoGrid)")
        ax.set_yscale("log"); ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)
        if i == ns - 1:
            ax.set_xlabel(xlabel)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] histogram grid ({ns} rows) -> {out_path}", flush=True)
    return out_path


def plot_moments_vs_shell(shell_idx, series: dict, out_path, suptitle=None):
    """3-panel (variance / skewness / excess kurtosis vs shell index) figure. series:
    dict of label -> list of moments.moments() dicts, one per entry of shell_idx, in
    insertion order (convention: "low", "high (true)", then the model's prediction) --
    catches one-point-PDF drift that a Cl-ratio plot (phase-blind, two-point only)
    cannot see (see moments.py)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [("variance", "variance"), ("skewness", "skewness"),
             ("excess_kurtosis", "excess kurtosis")]
    colors = ["darkorange", "seagreen", "steelblue", "tomato"]
    markers = ["o", "s", "^", "d"]
    for ax, (key, title) in zip(axes, panels):
        for (label, moms), color, marker in zip(series.items(), colors, markers):
            ax.plot(shell_idx, [m[key] for m in moms], "-", marker=marker,
                   color=color, label=label)
        ax.set_xlabel("shell index (depth)"); ax.set_title(title)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] moments vs shell depth -> {out_path}", flush=True)
    return out_path


def plot_train_val_loss(x, train_vals, val_vals, out_path, xlabel="epoch",
                        ylabel="loss", formula=None, train_label="train",
                        val_label="validation (held-out)"):
    """Shared train/validation loss curve -- ONE canonical figure, used identically
    by unet_flow_jbucko (plot_flow_loss.py, per-epoch flow-matching MSE) and
    transfer (transfer_function.py train(), per-iteration MLP squared-error) so the
    two pipelines' training diagnostics are visually and structurally identical, not
    just similarly styled.

    Both curves must be an actual LOSS (lower = better) on the SAME formula/units for
    train and val -- e.g. NOT one loss + one R^2 score (R^2 increases as it improves,
    so a "both curves decreasing" comparison across pipelines requires both sides to
    report a genuine held-out LOSS, not a score of a different sign convention)."""
    x = np.asarray(x); train_vals = np.asarray(train_vals); val_vals = np.asarray(val_vals)
    best_i = int(np.argmin(val_vals))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, train_vals, "-o", ms=3, color="steelblue", label=train_label)
    ax.plot(x, val_vals, "-o", ms=3, color="tomato", label=val_label)
    ax.axvline(x[best_i], color="0.6", ls=":", lw=1.0)
    ax.scatter([x[best_i]], [val_vals[best_i]], color="tomato", zorder=5,
               label=f"best val {val_vals[best_i]:.4g} @ {xlabel} {x[best_i]}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if formula:
        ax.set_title(formula, fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plotting] {len(x)} {xlabel}s | best val {val_vals[best_i]:.5g} @ "
          f"{xlabel} {x[best_i]} -> {out_path}", flush=True)
    return out_path


def plot_cl_shell(s, ells, cl_lo, cl_c, cl_hi, out_path):
    """Standard 2-panel (Cl log-log + ratio) figure for one shell."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].loglog(ells, cl_lo, "-", color="seagreen", label="DISCO (low)")
    ax[0].loglog(ells, cl_c, "-", color="steelblue", label="corrected")
    ax[0].loglog(ells, cl_hi, "--", color="tomato", label="CosmoGrid (high)")
    ax[0].set_ylabel(r"$C_\ell$"); ax[0].legend(); ax[0].set_title(f"shell {s}")
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].semilogx(ells, cl_lo / cl_hi, ":", color="seagreen", label="low/high")
        ax[1].semilogx(ells, cl_c / cl_hi, "-", color="steelblue", label="corrected/high")
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0.4, 1.6)
    ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel("ratio"); ax[1].legend()
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] shell {s} -> {out_path}", flush=True)
    return out_path
