"""Shared figure builders for every pipeline's example/diagnostic plots
(unet, transfer). These take ALREADY-COMPUTED arrays -- how a pipeline
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
        fig.suptitle(suptitle, fontsize=11, wrap=True)
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
        fig.suptitle(suptitle, fontsize=11, wrap=True)
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
                          f"{lo_pct}-{hi_pct}th pctile band)", fontsize=10, wrap=True)
    ax.legend(fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, wrap=True)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] pctile-band ratio ({n_samples} samples) -> {out_path}", flush=True)
    return out_path


def plot_cl_ratio_pctile_grid(grid, out_path, pctile=(16, 84), suptitle=None,
                              corrected_label="flow / true (after)",
                              n_ell_bins=40, ell_min=10):
    """grid: list of (row_label, panels) -- one row per held-out cosmology. panels:
    list of (bin_label, shells, ells, lo_stack, co_stack) -- one column per
    redshift/shell bin; lo_stack/co_stack: (n_shells_in_bin, n_ell) per-shell
    Cl-ratio-to-truth arrays.

    THESIS-STYLE presentation: ONE figure, ONE ROW of redshift-bin panels, each
    POOLING ALL held-out cosmologies. Within a panel, draws a median line + shaded
    [pctile] band on a log-spaced ell grid (analogous to plot_pctile_band_ratio).
    The band carries the full cosmology-to-cosmology + shell-to-shell scatter."""
    n_cols = max(len(panels) for _, panels in grid)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.2), squeeze=False)
    lo_pct, hi_pct = pctile
    n_cosmo = len(grid)

    for j in range(n_cols):
        ax = axes[0, j]
        # Pool this redshift bin's per-shell ratio stacks across ALL cosmology rows.
        bin_label = None; shells_ref = None; ells = None
        lo_parts, co_parts = [], []
        for _row_label, panels in grid:
            if j >= len(panels):
                continue
            bin_label, shells_ref, ells, lo_stack, co_stack = panels[j]
            lo_parts.append(np.asarray(lo_stack, dtype=np.float64))
            co_parts.append(np.asarray(co_stack, dtype=np.float64))
        if not lo_parts:
            ax.axis("off"); continue

        lo_pool = np.concatenate(lo_parts, axis=0)  # (n_cosmo*n_shells_in_bin, n_ell)
        co_pool = np.concatenate(co_parts, axis=0)
        ells = np.asarray(ells)

        # Build log-spaced bin edges and centers
        edges = np.geomspace(max(ell_min, 2), ells.max(), n_ell_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])

        for stack, color, label in [
                (lo_pool, "gray", "low / true (before)"),
                (co_pool, "steelblue", corrected_label)]:
            xs, med, p_lo_arr, p_hi_arr = [], [], [], []
            for k in range(n_ell_bins):
                sel = (ells >= edges[k]) & (ells < edges[k + 1])
                vals = stack[:, sel].ravel()
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                xs.append(centers[k])
                med.append(np.median(vals))
                p_lo_arr.append(np.percentile(vals, lo_pct))
                p_hi_arr.append(np.percentile(vals, hi_pct))

            xs = np.asarray(xs)
            med = np.asarray(med)
            p_lo_arr = np.asarray(p_lo_arr)
            p_hi_arr = np.asarray(p_hi_arr)

            ax.semilogx(xs, med, "-o", ms=3, color=color, label=label, lw=1.3)
            ax.fill_between(xs, p_lo_arr, p_hi_arr, color=color, alpha=0.2)

        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_title(f"{bin_label}\n(pooled: {n_cosmo} cosmologies x "
                     f"{len(shells_ref)} shells)", fontsize=9, wrap=True)
        ax.set_xlabel(r"$\ell$", fontsize=9)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.set_ylabel(r"$C_\ell/C_\ell^{\rm true}$", fontsize=10)
            ax.legend(fontsize=8, loc="lower left")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, wrap=True)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] Cl-ratio pctile band panels ({n_cols} zbins, pooled over "
          f"{n_cosmo} cosmologies) -> {out_path}", flush=True)
    return out_path



def plot_kappa_cl_grid(cosmo_labels, ells, cl_low_list, cl_corr_list, cl_high_list,
                       out_path, corrected_label="corrected", suptitle=None):
    """Per-cosmology kappa Cl -- ONE ROW PER HELD-OUT COSMOLOGY (left: raw Cl loglog,
    right: ratio to truth). Used by ALL THREE pipelines (transfer, sphereflow, unet)
    as kappa_cl_per_cosmology.png. It replaced a single-overlaid-axes version
    (kappa_cl_all_cosmologies.png, removed 2026-07-14): piling every cosmology's
    low/corrected/high lines onto one axes was unreadable past a couple of
    cosmologies. Same data, faceted, so each cosmology can be judged on its own; the
    aggregate/spread question is answered by kappa_cl_pctile_band.png instead."""
    n = len(cosmo_labels)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)
    x = ells[1:]
    for i, label in enumerate(cosmo_labels):
        cl_lo, cl_c, cl_hi = cl_low_list[i], cl_corr_list[i], cl_high_list[i]
        a0, a1 = axes[i, 0], axes[i, 1]
        a0.loglog(x, cl_lo[1:], ":", color="seagreen", lw=1.2, label="low (DISCO)")
        a0.loglog(x, cl_c[1:], "-", color="steelblue", lw=1.3, label=corrected_label)
        a0.loglog(x, cl_hi[1:], "--", color="tomato", lw=1.2, label="high (CosmoGrid)")
        a0.set_ylabel(f"{label}\n" + r"$C_\ell^{\kappa\kappa}$", fontsize=8)
        a0.tick_params(labelsize=7)
        with np.errstate(divide="ignore", invalid="ignore"):
            a1.semilogx(x, (cl_lo / cl_hi)[1:], ":", color="seagreen", lw=1.2, label="low/high")
            a1.semilogx(x, (cl_c / cl_hi)[1:], "-", color="steelblue", lw=1.3,
                       label="corrected/high")
        a1.axhline(1.0, color="k", ls="--", lw=0.8)
        a1.set_ylim(0.4, 1.6); a1.set_ylabel(r"$C_\ell/C_\ell^{true}$", fontsize=8)
        a1.tick_params(labelsize=7)
        if i == 0:
            a0.legend(fontsize=7, loc="lower left"); a1.legend(fontsize=7, loc="lower left")
        if i == n - 1:
            a0.set_xlabel(r"$\ell$", fontsize=8); a1.set_xlabel(r"$\ell$", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, wrap=True)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] kappa Cl grid ({n} cosmologies) -> {out_path}", flush=True)
    return out_path


def plot_kappa_moments_scatter(cosmo_labels, moms_low, moms_corr, moms_high, out_path,
                               corrected_label="corrected", suptitle=None):
    """Scatter (points, no connecting line) version of plot_moments_vs_shell: there
    is only ONE kappa map per cosmology (not one per shell depth), so the x-axis is
    categorical (cosmology), not a continuous depth a line plot would imply.
    moms_*: list of moments.moments() dicts, one per cosmo_labels entry, low/
    corrected/high aligned."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [("variance", "variance"), ("skewness", "skewness"),
             ("excess_kurtosis", "excess kurtosis")]
    x = np.arange(len(cosmo_labels))
    series = [("low", moms_low, "darkorange", "o"), (corrected_label, moms_corr, "steelblue", "^"),
             ("high (true)", moms_high, "tomato", "s")]
    for ax, (key, title) in zip(axes, panels):
        for name, moms, color, marker in series:
            ax.scatter(x, [m[key] for m in moms], color=color, marker=marker, label=name, s=40)
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(cosmo_labels, rotation=45, ha="right", fontsize=7)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, wrap=True)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] kappa moments scatter, {len(cosmo_labels)} cosmologies -> {out_path}", flush=True)
    return out_path


def plot_histogram_grid(rows, out_path, corrected_label="corrected", n_bins=60,
                        xlabel="pixel value (raw counts)", suptitle=None):
    """rows: list of (row_label, low_vals, corr_vals, high_vals) -- FLAT 1D arrays of
    raw pixel values (the caller decides scope: one patch, many pooled patches, or a
    whole full-sky map). One row per shell/example; log-y, shared bin edges per row
    spanning all three arrays so zero-spikes/tails are directly comparable -- the
    one-point-PDF analogue of plot_cl_shell's two-point check (see moments.py)."""
    ns = len(rows)
    # 2-column layout once the shell list gets long (the denser thesis default is
    # ~13 shells; a single column would be a ~40-inch-tall figure).
    n_cols = 2 if ns > 6 else 1
    n_rows_fig = (ns + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(8 * n_cols, 3 * n_rows_fig),
                             squeeze=False)
    flat = axes.ravel()
    for i, (label, low_v, corr_v, high_v) in enumerate(rows):
        ax = flat[i]
        lo = float(min(np.min(low_v), np.min(corr_v), np.min(high_v)))
        hi = float(max(np.max(low_v), np.max(corr_v), np.max(high_v)))
        bins = np.linspace(lo, hi, n_bins + 1) if hi > lo else n_bins
        # DISTINCT line styles + draw order, so a curve that overlaps another
        # exactly (low often sits under corrected/high on shells the correction
        # barely touches) stays visible: high solid at the back, corrected
        # dash-dot, low DASHED and drawn LAST (topmost).
        ax.hist(high_v, bins=bins, histtype="step", color="tomato", lw=2.4,
               ls="solid", density=True, label="high (CosmoGrid)", zorder=1)
        ax.hist(corr_v, bins=bins, histtype="step", color="steelblue", lw=1.6,
               ls="dashdot", density=True, label=corrected_label, zorder=2)
        ax.hist(low_v, bins=bins, histtype="step", color="seagreen", lw=1.4,
               ls="dashed", density=True, label="low (DISCO)", zorder=3)
        ax.set_yscale("log"); ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)
        if i >= ns - n_cols:
            ax.set_xlabel(xlabel)
    for k in range(ns, len(flat)):
        flat[k].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, wrap=True)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] histogram grid ({ns} rows) -> {out_path}", flush=True)
    return out_path


def plot_moments_vs_shell(shell_idx, series: dict, out_path, suptitle=None, note=None,
                          pctile=(16, 84)):
    """3-panel (variance / skewness / excess kurtosis vs shell index) figure. series:
    dict of label -> list of length len(shell_idx), each entry itself a list of
    per-SAMPLE moments.moments() dicts (one sample per held-out cosmology for a
    full-sky diagnostic, or one sample per held-out patch for a patch-pooled one) --
    catches one-point-PDF drift that a Cl-ratio plot (phase-blind, two-point only)
    cannot see (see moments.py).

    PCTILE-BAND presentation (2026-07-16, matches plot_cl_ratio_pctile_grid): per
    shell, the marker is the MEDIAN moment value across samples and the errorbar is
    the [pctile] spread -- so the cosmology-to-cosmology (or patch-to-patch) spread
    is visible directly, instead of being hidden by pooling every sample into one
    number per shell before computing the moment (the old behaviour). Series are
    nudged apart horizontally so overlapping errorbars stay distinguishable. Pass
    the per-cosmology parameters via `note` (rendered under the panels) so an
    outlier cosmology -- e.g. cosmo_000003's sigma8=1.15 -- is visible on the
    figure itself instead of needing a params.yml lookup."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [("variance", "variance"), ("skewness", "skewness"),
             ("excess_kurtosis", "excess kurtosis")]
    colors = ["darkorange", "seagreen", "steelblue", "tomato"]
    markers = ["o", "s", "^", "d"]
    lo_pct, hi_pct = pctile
    x = np.asarray(shell_idx, dtype=np.float64)
    n_series = len(series)
    # spread series within +-0.3 shell-index units around each x -- comfortably
    # inside the minimum spacing of every --*-shells default list used (>=2).
    offsets = np.linspace(-0.3, 0.3, n_series) if n_series > 1 else np.zeros(1)
    n_samples = 0
    for ax, (key, title) in zip(axes, panels):
        for (label, moms_per_shell), color, marker, off in zip(
                series.items(), colors, markers, offsets):
            med, elo, ehi = [], [], []
            for shell_moms in moms_per_shell:
                vals = np.asarray([m[key] for m in shell_moms], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                n_samples = max(n_samples, vals.size)
                if vals.size == 0:
                    med.append(np.nan); elo.append(0.0); ehi.append(0.0)
                    continue
                m = float(np.median(vals))
                med.append(m)
                elo.append(m - float(np.percentile(vals, lo_pct)))
                ehi.append(float(np.percentile(vals, hi_pct)) - m)
            ax.errorbar(x + off, med, yerr=[elo, ehi], fmt=marker, ms=4.5, lw=1.1,
                       capsize=2.5, color=color, label=label)
        ax.set_xlabel("shell index (depth)"); ax.set_title(title)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, wrap=True)
    if note:
        n_lines = note.count("\n") + 1
        pad = min(0.32, 0.028 * n_lines + 0.03)
        fig.text(0.01, 0.01, note, fontsize=7.5, family="monospace", va="bottom")
        fig.tight_layout(rect=(0, pad, 1, 1))
    else:
        fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] moments vs shell depth (up to {n_samples} samples/shell, "
          f"{lo_pct}-{hi_pct}th pctile band) -> {out_path}", flush=True)
    return out_path


# params.yml key -> axis label, in the CosmoGridV1 paper's corner-plot ordering
# (Omega_m, sigma_8, w_0, H_0, n_s, Omega_b).
_PARAM_LABELS = {"Om": r"$\Omega_m$", "s8": r"$\sigma_8$", "w0": r"$w_0$",
                 "H0": r"$H_0$", "ns": r"$n_s$", "Ob": r"$\Omega_b$"}


def plot_cosmo_param_matrix(pool: dict, held: dict, out_path,
                            params=("Om", "s8", "w0", "H0", "ns", "Ob"),
                            pool_label="training pool",
                            held_label="held-out (validation)", suptitle=None):
    """Where do the VALIDATION cosmologies sit in parameter space, and what are
    their values? pool/held: dict cosmo_name -> dict of parameter values (from each
    run's params.yml). Left: corner scatter of every pairwise parameter plane --
    the full pool in gray, the held-out set highlighted (labelled with the cosmo
    number in the first panel, e.g. the classic s8-Om plane). Right: a monospace
    table of every held-out cosmology's parameters, so an outlier like
    cosmo_000003 (sigma8=1.15) is identifiable at a glance."""
    k = len(params) - 1
    fig = plt.figure(figsize=(2.9 * k + 5.2, max(2.7 * k, 0.22 * (len(held) + 4))))
    gs = fig.add_gridspec(k, k + 2)
    pool_v = {q: np.array([c[q] for c in pool.values()], dtype=float) for q in params}
    held_names = list(held.keys())
    held_v = {q: np.array([held[n][q] for n in held_names], dtype=float) for q in params}
    for i in range(k):
        for j in range(i + 1):
            ax = fig.add_subplot(gs[i, j])
            xq, yq = params[j], params[i + 1]
            ax.scatter(pool_v[xq], pool_v[yq], s=11, color="0.78", label=pool_label)
            ax.scatter(held_v[xq], held_v[yq], s=26, color="steelblue",
                       edgecolor="k", linewidth=0.4, zorder=3, label=held_label)
            if i == 0 and j == 0:
                for nme, xv, yv in zip(held_names, held_v[xq], held_v[yq]):
                    ax.annotate(nme.replace("cosmo_", "").lstrip("0") or "0",
                                (xv, yv), fontsize=5.5, color="steelblue",
                                xytext=(2, 2), textcoords="offset points")
                ax.legend(fontsize=7, loc="best")
            (ax.set_ylabel(_PARAM_LABELS.get(yq, yq), fontsize=11)
             if j == 0 else ax.set_yticklabels([]))
            (ax.set_xlabel(_PARAM_LABELS.get(xq, xq), fontsize=11)
             if i == k - 1 else ax.set_xticklabels([]))
            ax.tick_params(labelsize=7)
    axt = fig.add_subplot(gs[:, k:])
    axt.axis("off")
    hdr = f"{'held-out cosmology':<19}" + "".join(f"{q:>8}" for q in params)
    lines = [hdr, "-" * len(hdr)]
    for nme in held_names:
        lines.append(f"{nme:<19}" + "".join(f"{held[nme][q]:>8.4g}" for q in params))
    axt.text(0.02, 1.0, "\n".join(lines), family="monospace", fontsize=7.5,
             va="top", ha="left", transform=axt.transAxes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, wrap=True)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] cosmo parameter matrix ({len(held)} held-out of {len(pool)}) "
          f"-> {out_path}", flush=True)
    return out_path


def plot_train_val_loss(x, train_vals, val_vals, out_path, xlabel="epoch",
                        ylabel="loss", formula=None, train_label="train",
                        val_label="validation (held-out)", skip_first=0):
    """Shared train/validation loss curve -- ONE canonical figure, used identically
    by unet (plot_flow_loss.py, per-epoch flow-matching MSE) and
    transfer (transfer_function.py train(), per-iteration MLP squared-error) so the
    two pipelines' training diagnostics are visually and structurally identical, not
    just similarly styled.

    Both curves must be an actual LOSS (lower = better) on the SAME formula/units for
    train and val -- e.g. NOT one loss + one R^2 score (R^2 increases as it improves,
    so a "both curves decreasing" comparison across pipelines requires both sides to
    report a genuine held-out LOSS, not a score of a different sign convention).

    skip_first: omit the first N points from the AXES (they still exist in the log).
    The first epoch starts from random weights, so its loss can sit orders of
    magnitude above the plateau and compress every later point into a flat band even
    on the log scale. The omitted values are ANNOTATED on the figure (not silently
    dropped) so the plot stays honest about what it isn't showing. Default 0 = old
    behavior; callers opt in."""
    x = np.asarray(x); train_vals = np.asarray(train_vals); val_vals = np.asarray(val_vals)
    note = None
    if skip_first > 0 and len(x) > skip_first + 1:
        note = (f"first {skip_first} {xlabel}(s) omitted from axes: "
                f"train {', '.join(f'{v:.3g}' for v in train_vals[:skip_first])} / "
                f"val {', '.join(f'{v:.3g}' for v in val_vals[:skip_first])}")
        x, train_vals, val_vals = x[skip_first:], train_vals[skip_first:], val_vals[skip_first:]
    best_i = int(np.argmin(val_vals))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, train_vals, "-o", ms=3, color="steelblue", label=train_label)
    ax.plot(x, val_vals, "-o", ms=3, color="tomato", label=val_label)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, color="0.4")
    ax.axvline(x[best_i], color="0.6", ls=":", lw=1.0)
    ax.scatter([x[best_i]], [val_vals[best_i]], color="tomato", zorder=5,
               label=f"best val {val_vals[best_i]:.4g} @ {xlabel} {x[best_i]}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if formula:
        ax.set_title(formula, fontsize=10, wrap=True)
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
