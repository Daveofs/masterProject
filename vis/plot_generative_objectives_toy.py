#!/usr/bin/env python3
"""Toy illustration: why per-pixel MSE regression, flow matching, and
diffusion behave differently on the same one-to-many correction problem.

Not derived from simulation output -- a synthetic stand-in built to make the
argument of Sec.~\\ref{sec:correction-pipelines} ("A regression trained on
paired maps with a per-pixel squared error converges to the conditional
mean...") visible rather than only stated. One (nearly-fixed) low-fidelity
input is made consistent with TWO very different, equally plausible
small-scale outcomes (a bimodal target x1) -- standing in for "the missing
high-frequency content could genuinely look like this, or like that."

Panel (a): the MSE-optimal single prediction is the mean of the samples,
which lands in the valley between the two modes -- a point with almost no
support in the true distribution, and zero predicted variance against a
genuinely spread-out target.

Panel (b): flow matching never regresses x1 from x0 directly. It regresses
the LOCAL velocity along the straight-line path between real (x0, x1) pairs,
at every intermediate t. The background field is the Nadaraya-Watson kernel
estimate of the MARGINAL vector field u_t(x) = E[x1-x0 | x_t=x], built
directly from the individual conditional paths (thin grey lines). Integrating
that marginal field via a plain ODE (thick colored lines) from three
nearly-identical starting points does NOT land on the mean -- it resolves
onto the two true modes, because conditional uncertainty shrinks to nothing
as t -> 1 even though the field was estimated by local averaging throughout.

Panel (c): the diffusion picture, deliberately NOT drawn as a vector field --
diffusion in this thesis (Sec.~\\ref{subsec:pipe-diffusion}) is an iterative
denoiser, not a transport map integrated once. What is shown is the analytic
relationship every diffusion model is trained against, x_t = x1 + sigma(t)
* eps, reversed from noisy (t=0) to clean (t=1). Each violin is the shape of
the DISTRIBUTION of x at that noise level: one broad, featureless hump at
sigma_max, visibly splitting into the true two-mode shape as sigma shrinks.
No ODE, no field -- the resolution happens in the shape of the density
itself, sample by sample.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_generative_objectives_toy.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_AXIS, FS_LEGEND = 13, 9.5
NOTE_KW = dict(fontsize=9.5, color="#555555",
               bbox=dict(fc="white", ec="#C9CEDC", lw=0.8,
                         boxstyle="round,pad=0.34", alpha=0.95))
C_A, C_B = "#B85F34", "#3F63A6"
C_MODE1, C_MODE2, C_MID = "#28866A", "#7A51C6", "#B85F34"
C_VIOLIN, C_VIOLIN_EDGE = "#9B7FC2", "#6E4FA0"


def marginal_velocity(t, x, x0, x1, u_true, bw=0.35):
    """Nadaraya-Watson estimate of u_t(x) = E[x1-x0 | x_t=x] from the
    individual conditional straight-line paths pos_i(t) = (1-t) x0_i + t x1_i."""
    pos = (1 - t) * x0 + t * x1
    w = np.exp(-0.5 * ((x - pos) / bw) ** 2)
    s = w.sum()
    return float((w * u_true).sum() / s) if s > 1e-12 else float(u_true.mean())


def panel_mse(ax, x1):
    """(a) Per-pixel squared error collapses the target distribution to its
    mean -- a point with almost no support in the true (bimodal) density."""
    ax.hist(x1, bins=32, density=True, color=C_B, alpha=0.35, edgecolor="white",
            lw=0.6, label=r"true $p(x_1\,|\,x_0)$: samples")
    mean_pred = x1.mean()
    ax.axvline(mean_pred, color=C_A, lw=2.0, ls="--", alpha=0.55, zorder=1,
               label=fr"MSE-optimal $f^*(x_0)=\mathbb{{E}}[x_1|x_0]={mean_pred:.2f}$")
    ax.scatter([mean_pred], [0], color=C_A, s=90, zorder=5, marker="D",
               label="MSE prediction (wrong)")
    # no leader line: it sat directly on top of the dashed MSE line and read as
    # a second, solid orange line
    ax.text(mean_pred, 0.30, "lands in a valley\nof near-zero true density",
            ha="center", va="bottom", fontsize=10.5, color=C_A, zorder=6)
    # the two TRUE possible answers -- either is right, their average is not --
    # placed clearly BELOW the histogram baseline so the markers don't sit
    # inside the bars' own footprint at y=0
    for mode_x, col in ((-2.0, C_MODE1), (2.0, C_MODE2)):
        ax.scatter([mode_x], [-0.028], color=col, s=90, zorder=5, marker="D",
                   edgecolor="white", lw=0.8, clip_on=False)
    ax.text(-2.0, -0.075, "a true answer", color=C_MODE1, fontsize=9, ha="center", va="top")
    ax.text(2.0, -0.075, "another true answer", color=C_MODE2, fontsize=9, ha="center", va="top")
    ax.set_xlabel(r"small-scale value $x_1$", fontsize=FS_AXIS)
    ax.set_ylabel("probability density", fontsize=FS_AXIS)
    ax.set_title("(a)  per-pixel MSE $\\Rightarrow$ one smoothed point,\nzero predicted variance",
                 fontsize=FS_AXIS, loc="left")
    ax.set_xlim(-4, 4); ax.set_ylim(-0.10, 0.55)
    ax.legend(fontsize=FS_LEGEND, loc="upper left", framealpha=0.97)
    ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)


def panel_flow(ax, rng, x1, N):
    """(b) Flow matching regresses a LOCAL velocity along the interpolation
    path between real (x0, x1) pairs. Integrating the marginal field -- itself
    a local average, just like the MSE case -- resolves onto the two true
    modes rather than their average, because conditional uncertainty shrinks
    to nothing as t -> 1."""
    x0 = rng.normal(0.0, 0.05, N)          # real inputs are never bit-identical
    u_true = x1 - x0

    sub = rng.choice(N, 40, replace=False)
    for i in sub:
        ax.plot([0, 1], [x0[i], x1[i]], color="#B9BFD0", lw=0.7, alpha=0.8, zorder=1)

    tg = np.linspace(0.04, 0.97, 16)
    xg = np.linspace(-3.4, 3.4, 22)
    TT, XX = np.meshgrid(tg, xg)
    UU = np.array([[marginal_velocity(t, x, x0, x1, u_true) for t in tg] for x in xg])
    dt_arrow = 0.05
    ax.quiver(TT, XX, np.full_like(UU, dt_arrow), UU * dt_arrow, color="#8E96AC",
              angles="xy", scale_units="xy", scale=1, width=0.0028, alpha=0.85, zorder=2)

    starts = [-0.15, -0.03, 0.12]
    colors = [C_MODE1, C_MID, C_MODE2]
    n_steps = 60
    for x0_s, col in zip(starts, colors):
        ts = np.linspace(0, 1, n_steps + 1)
        xs = [x0_s]
        x = x0_s
        for k in range(n_steps):
            t = ts[k]; dt = ts[k + 1] - t
            x = x + marginal_velocity(t, x, x0, x1, u_true) * dt
            xs.append(x)
        ax.plot(ts, xs, color=col, lw=2.6, zorder=4)
        ax.scatter([0], [x0_s], color=col, s=45, zorder=5, edgecolor="white", lw=0.8)
        ax.scatter([1], [xs[-1]], color=col, s=95, zorder=5, marker="D", edgecolor="white", lw=0.8)

    ax.axvline(1.0, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax.text(0.02, -4.15, r"$t=0$: nearly the" + "\nsame input $x_0$",
            ha="left", va="bottom", zorder=7, **NOTE_KW)
    ax.text(0.99, -4.15, r"$t=1$: resolves onto" + "\nthe true modes",
            ha="right", va="bottom", zorder=7, **NOTE_KW)
    ax.set_xlabel(r"flow time $t$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_t = (1-t)x_0 + t\,x_1$", fontsize=FS_AXIS)
    ax.set_title("(b)  flow matching $\\Rightarrow$ transports the\nwhole distribution, mode by mode",
                 fontsize=FS_AXIS, loc="left")
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-4.35, 3.75)
    ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)


def panel_diffusion(ax, rng, x1, N):
    """(c) Deliberately NOT a vector field. Diffusion is shown as the analytic
    corruption relationship x_t = x1 + sigma(t)*eps, reversed from noisy to
    clean: a sequence of DISTRIBUTIONS (violins) whose shape itself resolves
    from one broad hump into the true two-mode shape as sigma shrinks."""
    sigma_max = 3.0
    def sigma(t):
        return sigma_max * (1 - t) ** 1.3

    t_positions = np.linspace(0.0, 1.0, 7)
    eps = rng.standard_normal(N)           # one fixed noise draw per sample,
                                           # reused (rescaled) at every level --
                                           # this is what "denoising" reverses.
    datasets = [x1 + sigma(t) * eps for t in t_positions]
    vp = ax.violinplot(datasets, positions=t_positions, widths=0.115,
                       showmeans=False, showextrema=False, showmedians=False)
    for body in vp["bodies"]:
        body.set_facecolor(C_VIOLIN); body.set_edgecolor(C_VIOLIN_EDGE)
        body.set_alpha(0.55); body.set_linewidth(0.8)

    # a couple of individual samples' reverse-denoising path, traced through
    # the SAME analytic relationship -- not a field, just less corruption.
    # Picked near each mode centre with a modest own noise draw, so the paths
    # stay legible instead of a rare extreme point shooting outside the violins.
    def typical_example(mode_centre, eps_cap=1.0):
        near = np.where((np.abs(x1 - mode_centre) < 0.2) & (np.abs(eps) < eps_cap))[0]
        if len(near) == 0:
            near = np.where(np.abs(x1 - mode_centre) < 0.2)[0]
        return near[0]

    # a third, orange example chosen to CROSS: its true value belongs to mode A,
    # but once fully noised it sits well inside mode B's territory. It is the
    # clearest statement of the panel's point -- the noisy start carries no
    # information about which mode is correct, and the denoising still resolves
    # it to the right one. Picked to be the crossing sample furthest from the
    # green endpoint, so the two mode-A markers stay visually distinct.
    idx_green = typical_example(-2.0)
    x_t0_all = x1 + sigma(0.0) * eps
    crossing = np.where((x1 < 0) & (x_t0_all > 2.0))[0]
    if len(crossing):
        idx_mid = int(crossing[np.argmax(np.abs(x1[crossing] - x1[idx_green]))])
    else:
        idx_mid = int(np.argmin(np.abs(x_t0_all)))

    examples = [(idx_green, C_MODE1), (idx_mid, C_MID),
               (typical_example(2.0), C_MODE2)]
    for idx, col in examples:
        path_x = [x1[idx] + sigma(t) * eps[idx] for t in t_positions]
        ax.plot(t_positions, path_x, "-o", color=col, lw=1.8, ms=4.2, zorder=5,
                mec="white", mew=0.6)
        ax.scatter([1.0], [path_x[-1]], color=col, s=95, marker="D", zorder=6,
                   edgecolor="white", lw=0.8)

    ax.axvline(1.0, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax.text(-0.07, -5.35, "$t=0$: one broad noisy\nhump, no info about $x_0$",
            ha="left", va="bottom", zorder=7, **NOTE_KW)
    ax.text(1.07, -5.35, "$t=1$: the DISTRIBUTION\nitself resolves into $p(x_1)$",
            ha="right", va="bottom", zorder=7, **NOTE_KW)
    ax.set_xlabel(r"diffusion step (noise level $\sigma$ decreasing)", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x$", fontsize=FS_AXIS)
    ax.set_title("(c)  diffusion $\\Rightarrow$ denoise the whole\ndistribution, step by step (no field)",
                 fontsize=FS_AXIS, loc="left")
    ax.set_xlim(-0.09, 1.09); ax.set_ylim(-5.55, 4.9)
    ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-samples", type=int, default=260)
    ap.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/toy")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    # bimodal target distribution shared by all three panels: two equally
    # plausible small-scale realisations consistent with the same coarse input
    x1 = np.concatenate([rng.normal(-2.0, 0.42, a.n_samples // 2),
                         rng.normal(2.0, 0.42, a.n_samples // 2)])

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18.0, 5.0))
    panel_mse(axA, x1)
    panel_flow(axB, rng, x1, a.n_samples)
    panel_diffusion(axC, rng, x1, a.n_samples)
    fig.tight_layout()

    out = out_dir / "generative_objectives_toy.png"
    fig.savefig(out, dpi=190, bbox_inches="tight"); plt.close(fig)
    print(f"[toy] -> {out}")


if __name__ == "__main__":
    main()
