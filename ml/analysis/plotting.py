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

from matplotlib.ticker import FuncFormatter, MaxNLocator

from .radial_power import radial_power


def _sci_tick(v, _pos=None):
    """Tick label as a x 10^b in mathtext (see plot_kappa_moments_scatter)."""
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    return rf"${m:.1f}\times10^{{{e}}}$"

# ---------------------------------------------------------------------------
# Typography. These figures go into the thesis at roughly \textwidth, so every
# label has to survive that downscaling -- hence sizes well above matplotlib's
# defaults. Axes TITLES are deliberately not drawn: the shell index, redshift
# range, cosmology and pooling are stated in the LaTeX caption instead, which
# keeps that information in one place and out of the rasterised image. The
# `suptitle` arguments below are kept in the signatures (many callers pass them)
# but are ignored for the same reason.
# ---------------------------------------------------------------------------
FS_AXIS = 15      # axis labels
FS_TICK = 13      # tick labels
FS_LEGEND = 12    # legend entries
FS_PANEL = 13     # in-panel annotations (e.g. the z-range of a shell bin)
FS_ROWLAB = 13    # per-row identifying labels on faceted grids


def centre_crop(img, n):
    """Central (n, n) crop of a 2D array (returned unchanged if n is None or too
    large). Used to bring the nside=512 patch figures onto the SAME sky area as
    the nside=2048 ones -- see plot_example_patch_grid's `crop`."""
    if n is None:
        return img
    h, w = img.shape[-2:]
    if n >= min(h, w):
        return img
    i0, j0 = (h - n) // 2, (w - n) // 2
    return img[..., i0:i0 + n, j0:j0 + n]


def plot_example_patch_grid(rows, out_path, corrected_label="corrected", suptitle=None,
                            crop=None):
    """rows: list of (row_label, low_img, corr_img, high_img) -- 2D patches in
    whatever overdensity space the CALLER chose to display (NOT necessarily log1p):
    unet/apply_flow.py and diffusion/apply_diffusion.py pass their model's own
    --space (default 'delta', the linear overdensity analysis.full_sky.od_cl
    measures, since 2026-07-18); transfer/apply_transfer.py and
    sphereflow/apply_sphere_flow.py pass genuine log1p (via transforms.log1p_delta_pair)
    purely for display contrast. This function only draws the comparison -- it has
    no opinion on which space rows arrives in, so the caller's own suptitle should
    name the space (as the two model pipelines' now do). One row per shell/example.
    4 columns: low/corrected/high images + per-patch radial-power-ratio (the
    flat-patch analogue of the C_ell ratio, bounded by that one patch's own Nyquist
    wavenumber).

    `crop` centre-crops the three image columns to (crop, crop) pixels BEFORE
    display, leaving the power-ratio column (computed on the full patch) alone.
    Its purpose is field of view, not zoom: a 256-pixel patch spans
    256*nside2resol(nside), i.e. 29.3 deg at nside=512 but only 7.33 deg at
    nside=2048. Over 29 deg the square display grid beats against HEALPix's
    diamond lattice and the patch acquires long streaky moire fringes -- visible
    identically in the low, corrected AND high columns, so an artefact of
    displaying that much sky, not of any model. Cropping an nside=512 patch to 64
    pixels restores exactly the 7.33 deg of the nside=2048 figures, removing the
    fringes and making the two directly comparable; the coarser pixels that remain
    are the genuine resolution difference between the runs.
    """
    ns = len(rows)
    fig, axes = plt.subplots(ns, 4, figsize=(13, 3 * ns),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    axes = np.atleast_2d(axes)
    for i, (label, low_img_full, corr_img_full, high_img_full) in enumerate(rows):
        low_img = centre_crop(low_img_full, crop)
        corr_img = centre_crop(corr_img_full, crop)
        high_img = centre_crop(high_img_full, crop)
        vmin, vmax = float(high_img.min()), float(high_img.max())
        # Smooth the RENDERING of small (cropped, nside=512) patches: at 64 px
        # blown up to a full panel, nearest-neighbour shows every pixel as a hard
        # block, which reads as a defect rather than as the coarser resolution it
        # actually is. Bilinear is a display choice only -- no data is altered,
        # and no detail is implied beyond the pixel scale, which the caption
        # states. Large patches (the nside=2048 figures) keep nearest, so their
        # appearance is unchanged. Lanczos was tried and rejected: it rings.
        interp = "bilinear" if min(low_img.shape[-2:]) < 128 else "nearest"
        for j, (img, ttl) in enumerate([(low_img, "low (DISCO)"), (corr_img, corrected_label),
                                        (high_img, "high (CosmoGrid)")]):
            a = axes[i, j]
            a.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis", interpolation=interp)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(ttl, fontsize=FS_AXIS)
            if j == 0:
                a.set_ylabel(label, fontsize=FS_ROWLAB)

        pr_low = radial_power(low_img); pr_corr = radial_power(corr_img); pr_high = radial_power(high_img)
        k = np.arange(len(pr_high))
        ai = axes[i, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ai.semilogx(k[1:], (pr_low / pr_high)[1:], ":", color="seagreen", label="low/high")
            ai.semilogx(k[1:], (pr_corr / pr_high)[1:], "-", color="steelblue", label="corrected/high")
        ai.axhline(1, color="k", lw=0.8); ai.set_ylim(0.4, 1.6)
        ai.tick_params(labelsize=FS_TICK)
        if i == 0:
            ai.set_title("power ratio to truth", fontsize=FS_AXIS)
            ai.legend(fontsize=FS_LEGEND, loc="lower left")
        if i == ns - 1:
            ai.set_xlabel("radial wavenumber bin", fontsize=FS_AXIS)

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
                a.set_title(ttl, fontsize=FS_AXIS)
            if j == 0:
                a.set_ylabel(label, fontsize=FS_ROWLAB)

        ai = axes[i, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ai.semilogx(ells[1:], (cl_lo / cl_hi)[1:], ":", color="seagreen", label="low/high")
            ai.semilogx(ells[1:], (cl_c / cl_hi)[1:], "-", color="steelblue", label="corrected/high")
        ai.axhline(1, color="k", lw=0.8); ai.set_ylim(0.0, 1.6)
        ai.tick_params(labelsize=FS_TICK)
        if i == 0:
            ai.set_title(r"real $C_\ell$ ratio to truth", fontsize=FS_AXIS)
            ai.legend(fontsize=FS_LEGEND, loc="lower left")
        if i == ns - 1:
            ai.set_xlabel(r"$\ell$", fontsize=FS_AXIS)

    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] example full-sky grid -> {out_path}", flush=True)
    return out_path


def plot_pctile_band_ratio(x, ratio_stacks: dict, out_path, xlabel=r"$\ell$",
                           ylabel=None, pctile=(16, 84), ylim=None, title=None,
                           suptitle=None, smooth_window=21):
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
    convention.

    smooth_window (2026-07-16): the median/percentile curves are boxcar-smoothed in
    log-value space over `smooth_window` consecutive x-index points BEFORE
    plotting -- the SAME recipe transfer_function.smooth_cl uses for T(ell), applied
    here to the display curves rather than to the raw per-sample stack. With few
    samples (e.g. a handful of held-out cosmologies for a kappa Cl ratio) the
    per-index median/percentile is itself noisy point-to-point, which made this plot
    look far more jagged than cl_ratio_by_zbin_grid.png's binned/pooled curves even
    though both answer the same "how far from 1.0" question. 1 disables (raw
    per-index curve)."""
    from scipy.ndimage import uniform_filter1d

    def _smooth(y):
        if smooth_window <= 1:
            return y
        with np.errstate(divide="ignore", invalid="ignore"):
            log_y = np.log10(np.clip(y, 1e-30, None))
        return 10.0 ** uniform_filter1d(log_y, size=smooth_window, mode="nearest")

    fig, ax = plt.subplots(figsize=(9, 6))
    lo_pct, hi_pct = pctile
    colors = ["gray", "steelblue", "tomato", "seagreen"]
    n_samples = None
    for (label, stack), color in zip(ratio_stacks.items(), colors):
        stack = np.asarray(stack, dtype=np.float64)
        n_samples = stack.shape[0]
        med = _smooth(np.nanmedian(stack, axis=0))
        p_lo = _smooth(np.nanpercentile(stack, lo_pct, axis=0))
        p_hi = _smooth(np.nanpercentile(stack, hi_pct, axis=0))
        ax.semilogx(x, med, "-", lw=1.6, color=color, label=label)
        ax.fill_between(x, p_lo, p_hi, color=color, alpha=0.2)

    ax.axhline(1.0, color="k", ls="--", lw=1)
    # The kappa Cl panels are reproduced at full text width in the thesis, so they
    # carry a larger type scale than the faceted grids: at FS_AXIS the axis labels
    # were noticeably smaller than the surrounding body text.
    ax.set_xlabel(xlabel, fontsize=FS_AXIS + 5)
    ax.set_ylabel(ylabel or (r"$C_\ell/C_\ell^{true}$" if "ell" in xlabel.lower() else "ratio"),
                  fontsize=FS_AXIS + 5)
    ax.tick_params(labelsize=FS_TICK + 4)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        # Data-driven window. A fixed (0.4, 1.6) leaves the entire upper half of
        # these panels empty -- these ratios approach unity from BELOW (the fast
        # solver loses power, it does not gain any), so the symmetric window wastes
        # roughly half the figure. Span the drawn curves and bands, always keeping
        # 1.0 in view, and pad by 6%.
        finite = np.concatenate([np.asarray(v, dtype=np.float64).ravel()
                                 for v in ratio_stacks.values()])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lo_v = min(float(np.nanpercentile(finite, 0.5)), 1.0)
            hi_v = max(float(np.nanpercentile(finite, 99.5)), 1.0)
            pad = max(hi_v - lo_v, 1e-3) * 0.06
            ax.set_ylim(lo_v - pad, hi_v + pad)
    ax.legend(fontsize=FS_LEGEND + 4)
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"[plotting] pctile-band ratio ({n_samples} samples, smooth_window="
          f"{smooth_window}) -> {out_path}", flush=True)
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

        # Legend labels are deliberately SHORT. The y-axis already reads
        # C_l / C_l^true, so the "/ true (before|after)" half of the caller's
        # label is redundant -- and with it, "corrected (transfer (no-clip)) /
        # true (after)" is wide enough that the legend box overflowed the panel
        # into its neighbour. Strip that suffix and keep only the method name.
        short_corr = corrected_label.split(" / ")[0].strip() or "corrected"
        for stack, color, label in [
                (lo_pool, "gray", "low (DISCO)"),
                (co_pool, "steelblue", short_corr)]:
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
        # The redshift range goes INSIDE the panel rather than into a title: with
        # one panel per z-bin, that range is the only thing distinguishing them,
        # so it has to travel with the panel even when the figure is read on its
        # own. bin_label already carries it in "z in (lo, hi)" form.
        ax.text(0.03, 0.04, bin_label, transform=ax.transAxes,
                fontsize=FS_PANEL, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
        ax.set_xlabel(r"$\ell$", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        if j == 0:
            ax.set_ylabel(r"$C_\ell/C_\ell^{\rm true}$", fontsize=FS_AXIS)
            # Stacked directly ABOVE the z-range box in the same bottom-left
            # corner. Both curves approach 1 from below and the interesting part
            # of the panel is the upper half, so anchoring the legend up there
            # (matplotlib's "best"/upper-left) put it straight on top of them.
            leg = ax.legend(fontsize=FS_LEGEND, loc="lower left",
                            bbox_to_anchor=(0.03, 0.14), borderaxespad=0.0)
            # hard guard: if the (possibly caller-supplied) labels still make the
            # box wider than the panel, shrink the text until it fits rather than
            # letting it spill over the neighbouring panel
            fig.canvas.draw()
            for _ in range(6):
                lw = leg.get_window_extent().width
                aw = ax.get_window_extent().width
                if lw <= 0.94 * aw:
                    break
                for t in leg.get_texts():
                    t.set_fontsize(t.get_fontsize() * 0.9)
                fig.canvas.draw()

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
        a0.set_ylabel(f"{label}\n" + r"$C_\ell^{\kappa\kappa}$", fontsize=FS_ROWLAB)
        a0.tick_params(labelsize=FS_TICK)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_lo = (cl_lo / cl_hi)[1:]
            r_c = (cl_c / cl_hi)[1:]
            a1.semilogx(x, r_lo, ":", color="seagreen", lw=1.2, label="low/high")
            a1.semilogx(x, r_c, "-", color="steelblue", lw=1.3, label="corrected/high")
        a1.axhline(1.0, color="k", ls="--", lw=0.8)
        # Data-driven y-range instead of a fixed (0.4, 1.6): the corrected curve
        # typically lives within a few percent of unity, so the fixed window spent
        # most of the panel on empty space. Bracket what is actually drawn (both
        # curves), keep 1.0 in view, and pad by 8%.
        finite = np.concatenate([r_lo[np.isfinite(r_lo)], r_c[np.isfinite(r_c)]])
        if finite.size:
            lo_v = min(float(np.percentile(finite, 0.5)), 1.0)
            hi_v = max(float(np.percentile(finite, 99.5)), 1.0)
            pad = max(hi_v - lo_v, 1e-3) * 0.08
            a1.set_ylim(lo_v - pad, hi_v + pad)
        a1.set_ylabel(r"$C_\ell/C_\ell^{\rm true}$", fontsize=FS_AXIS)
        a1.tick_params(labelsize=FS_TICK)
        if i == 0:
            a0.legend(fontsize=FS_LEGEND, loc="lower left")
            a1.legend(fontsize=FS_LEGEND, loc="lower left")
        if i == n - 1:
            a0.set_xlabel(r"$\ell$", fontsize=FS_AXIS)
            a1.set_xlabel(r"$\ell$", fontsize=FS_AXIS)
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    panels = [("variance", r"variance  $\langle\kappa^2\rangle$"),
              ("skewness", r"skewness  $S_3$"),
              ("excess_kurtosis", r"excess kurtosis  $K_4$")]
    series = [("low (DISCO)", moms_low, "darkorange", "o"),
              (corrected_label, moms_corr, "steelblue", "^")]
    for ax, (key, axis_label) in zip(axes, panels):
        true_vals = np.asarray([m[key] for m in moms_high], dtype=np.float64)
        # Small quantities (kappa variance is ~1e-4..1e-8 depending on the source
        # bin) get SCIENTIFIC TICK LABELS, written out per tick as a x 10^b. The
        # alternative matplotlib reaches for by default -- a bare "1e-8" floated
        # in the corner as an axis offset -- is easy to miss entirely and easier
        # to misread as a data point. Quantities already of order unity (skewness,
        # excess kurtosis) keep plain ticks; nothing is rescaled either way, so
        # the numbers on the axes are always the real values.
        finite_all = np.concatenate([
            true_vals,
            *[np.asarray([m[key] for m in mm], dtype=np.float64) for _, mm, _, _ in series]])
        finite_all = finite_all[np.isfinite(finite_all) & (finite_all != 0)]
        exp10 = 0
        if finite_all.size:
            exp10 = int(np.floor(np.log10(np.max(np.abs(finite_all)))))
        use_sci = abs(exp10) >= 3
        scale, unit = 1.0, ""
        # Plotted AGAINST THE TRUE VALUE rather than against a categorical
        # cosmology axis: with one point per held-out cosmology, the old version
        # spent its x-axis on ten rotated "cosmo_000176"-style labels that carried
        # no quantitative meaning. Here x is the reference moment itself, so the
        # dashed 1:1 line IS perfect agreement and each point's vertical distance
        # from it is that cosmology's error -- the same data, made readable.
        for name, moms, color, marker in series:
            vals = np.asarray([m[key] for m in moms], dtype=np.float64) * scale
            ax.scatter(true_vals * scale, vals, color=color, marker=marker, label=name,
                       s=55, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
        both = np.concatenate([true_vals * scale,
                               *[np.asarray([m[key] for m in mm], dtype=np.float64) * scale
                                 for _, mm, _, _ in series]])
        both = both[np.isfinite(both)]
        if both.size:
            lo_v, hi_v = float(both.min()), float(both.max())
            pad = max(hi_v - lo_v, 1e-12) * 0.08
            lims = (lo_v - pad, hi_v + pad)
            ax.plot(lims, lims, "--", color="0.35", lw=1.0, zorder=2,
                    label="perfect agreement" if ax is axes[0] else None)
            ax.set_xlim(*lims); ax.set_ylim(*lims)
        ax.set_xlabel(f"{axis_label}{unit}   CosmoGridV1", fontsize=FS_AXIS)
        ax.set_ylabel(f"{axis_label}{unit}   model", fontsize=FS_AXIS)
        if use_sci:
            fmt = FuncFormatter(_sci_tick)
            ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(20); lbl.set_horizontalalignment("right")
        else:
            ax.ticklabel_format(style="plain", axis="both", useOffset=False)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(fontsize=FS_LEGEND, loc="upper left")
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
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

    PCTILE-BAND presentation (2026-07-16, matches plot_pctile_band_ratio): per
    shell, a MEDIAN line is drawn across shells with a continuous [pctile] shaded
    band per series -- so the cosmology-to-cosmology (or patch-to-patch) spread is
    visible as a band, instead of being hidden by pooling every sample into one
    number per shell before computing the moment (the old behaviour), or shown as
    disconnected per-shell errorbars (the 2026-07-16 intermediate version). Pass
    the per-cosmology parameters via `note` (rendered under the panels) so an
    outlier cosmology -- e.g. cosmo_000003's sigma8=1.15 -- is visible on the
    figure itself instead of needing a params.yml lookup."""
    # Two rows: the moment itself (log y) over a ratio-to-truth row. Log y because
    # the moments span orders of magnitude across shell depth, which on a linear
    # axis compresses every shell except the sparsest into an unreadable band; the
    # ratio row then shows the AGREEMENT, which absolute curves at that dynamic
    # range cannot resolve. The reference series (the one whose label mentions
    # "high"/"true") is the ratio denominator.
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1.0]})
    panels = [("variance", r"variance  $\sigma^2$"), ("skewness", r"skewness  $S_3$"),
             ("excess_kurtosis", r"excess kurtosis  $K_4$")]
    colors = ["darkorange", "seagreen", "steelblue", "tomato"]
    markers = ["o", "s", "^", "d"]
    lo_pct, hi_pct = pctile
    x = np.asarray(shell_idx, dtype=np.float64)
    n_samples = 0

    def _is_ref(lbl):
        low = lbl.lower()
        return ("high" in low) or ("true" in low)

    ref_key = next((lbl for lbl in series if _is_ref(lbl)), None)

    for col, (key, axis_label) in enumerate(panels):
        ax, axr = axes[0, col], axes[1, col]
        med_by_label = {}
        for (label, moms_per_shell), color, marker in zip(
                series.items(), colors, markers):
            med, p_lo, p_hi = [], [], []
            for shell_moms in moms_per_shell:
                vals = np.asarray([m[key] for m in shell_moms], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                n_samples = max(n_samples, vals.size)
                if vals.size == 0:
                    med.append(np.nan); p_lo.append(np.nan); p_hi.append(np.nan)
                    continue
                med.append(float(np.median(vals)))
                p_lo.append(float(np.percentile(vals, lo_pct)))
                p_hi.append(float(np.percentile(vals, hi_pct)))
            med = np.asarray(med); p_lo = np.asarray(p_lo); p_hi = np.asarray(p_hi)
            med_by_label[label] = (med, color, marker)
            ax.plot(x, med, "-", marker=marker, ms=4.5, lw=1.4, color=color, label=label)
            ax.fill_between(x, p_lo, p_hi, color=color, alpha=0.2)

        # log y only where the quantity is positive throughout: skewness and
        # excess kurtosis legitimately cross zero, and a symlog/linear axis is the
        # honest choice there rather than silently dropping the negative points.
        all_med = np.concatenate([m for m, _, _ in med_by_label.values()])
        all_med = all_med[np.isfinite(all_med)]
        if all_med.size and all_med.min() > 0:
            ax.set_yscale("log")
        ax.set_ylabel(axis_label, fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        if col == 0:
            ax.legend(fontsize=FS_LEGEND)

        # ---- ratio row ----
        if ref_key is not None:
            ref_med = med_by_label[ref_key][0]
            for label, (med, color, marker) in med_by_label.items():
                if label == ref_key:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = med / ref_med
                axr.plot(x, ratio, "-", marker=marker, ms=4.0, lw=1.3, color=color,
                         label=f"{label} / {ref_key}")
            axr.axhline(1.0, color="k", ls="--", lw=0.8)
            axr.set_ylabel("ratio to truth", fontsize=FS_AXIS)
        axr.set_xlabel("shell index (depth)", fontsize=FS_AXIS)
        axr.tick_params(labelsize=FS_TICK)

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
    run's params.yml). Corner scatter of every pairwise parameter plane -- the
    full pool in gray, the held-out set highlighted (labelled with the cosmo
    number in the first panel, e.g. the classic s8-Om plane)."""
    k = len(params) - 1
    fig = plt.figure(figsize=(2.9 * k, max(2.7 * k, 0.22 * (len(held) + 4))))
    gs = fig.add_gridspec(k, k)
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
    fig.tight_layout()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400); plt.close(fig)
    print(f"[plotting] cosmo parameter matrix ({len(held)} held-out of {len(pool)}) "
          f"-> {out_path}", flush=True)
    return out_path


def _rolling_mean(v: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean, shrinking window at the edges (no NaN padding, no
    look-ahead bias beyond what a centered window already implies) -- window=1
    returns v unchanged."""
    if window <= 1:
        return v
    half = window // 2
    out = np.empty_like(v, dtype=np.float64)
    for i in range(len(v)):
        lo, hi = max(0, i - half), min(len(v), i + half + 1)
        out[i] = v[lo:hi].mean()
    return out


def plot_train_val_loss(x, train_vals, val_vals, out_path, xlabel="epoch",
                        ylabel="loss", formula=None, train_label="train",
                        val_label="validation (held-out)", skip_first=0,
                        smooth_window=1):
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
    behavior; callers opt in.

    smooth_window (default 1 = off, existing callers unaffected): plots a centered
    rolling-mean curve INSTEAD of the raw per-epoch points. For diffusion's EDM loss
    this is cosmetic, not a training fix -- 2026-07-23 found that cutting the LR 3x
    (job diffusion_..._hpc0.05_hpt0.30, lr 3e-5->1e-5) left the sawtooth amplitude
    essentially unchanged, which rules out optimization noise: the per-epoch mean is
    a small-sample MC estimate of <lambda(sigma)*MSE> over a randomly-drawn sigma
    batch each epoch (ln-sigma ~ N(P_mean,P_std^2), lambda(sigma) spans orders of
    magnitude), so it is measurement variance in the metric, not the model's
    trajectory -- no LR can smooth that away, only averaging over more epochs can.
    The best-val marker still uses the RAW (unsmoothed) values, so early-stopping
    checkpoint selection is reported honestly even when the displayed curve is
    smoothed."""
    x = np.asarray(x); train_vals = np.asarray(train_vals); val_vals = np.asarray(val_vals)
    note = None
    if skip_first > 0 and len(x) > skip_first + 1:
        note = (f"first {skip_first} {xlabel}(s) omitted from axes: "
                f"train {', '.join(f'{v:.3g}' for v in train_vals[:skip_first])} / "
                f"val {', '.join(f'{v:.3g}' for v in val_vals[:skip_first])}")
        x, train_vals, val_vals = x[skip_first:], train_vals[skip_first:], val_vals[skip_first:]
    best_i = int(np.argmin(val_vals))  # off the RAW values, before any smoothing

    if smooth_window > 1:
        note = (note + " | " if note else "") + f"{smooth_window}-{xlabel} rolling mean"
        train_plot = _rolling_mean(train_vals, smooth_window)
        val_plot = _rolling_mean(val_vals, smooth_window)
    else:
        train_plot, val_plot = train_vals, val_vals

    fig, ax = plt.subplots(figsize=(9, 5))
    # train drawn wider + dashed, val drawn on top solid + thinner + semi-
    # transparent: whenever the two curves track closely (common once
    # converged -- e.g. transfer's MLP emulator, where val is plotted second
    # and used to fully occlude train underneath it since both lines were
    # solid/opaque/same width), the dashed train line still pokes out from
    # under val instead of disappearing entirely -- robust to how close the
    # VALUES are, unlike an alpha-only fix. val's alpha=0.45 is additive on
    # top of that (not a substitute for it): the dash pattern guarantees
    # train is visible even under EXACT overlap, alpha just makes the overlap
    # itself blend rather than read as a single flat line. (0.75 was tried
    # first and was too subtle to actually notice -- 0.45 is a real, visible
    # change, checked against a real plot, not just "should be different".)
    ax.plot(x, train_plot, "--o", ms=3, lw=2.2, color="steelblue", label=train_label)
    ax.plot(x, val_plot, "-o", ms=3, lw=1.3, alpha=0.45, color="tomato", label=val_label)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, color="0.4")
    ax.axvline(x[best_i], color="0.6", ls=":", lw=1.0)
    # Marker sits ON the drawn (possibly smoothed) curve for visual consistency;
    # the label text still reports the true RAW best value/epoch used for
    # checkpoint selection, so smoothing never misrepresents what was picked.
    ax.scatter([x[best_i]], [val_plot[best_i]], color="tomato", zorder=5,
               label=f"best val {val_vals[best_i]:.4g} @ {xlabel} {x[best_i]}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
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
    ax[0].set_ylabel(r"$C_\ell$"); ax[0].legend()
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
