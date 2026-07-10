#!/usr/bin/env python3
"""Lognormal-intensity + Poisson resampling: turn the Cl-optimal (but continuous,
negative-valued) transfer-corrected field into a valid COUNT map.

Why this exists
---------------
`transfer_function.py apply --no-clip` gives a field whose Cl matches CosmoGrid to
~1% at all ell and whose mean is exact, but it is a continuous OVERDENSITY field:
on faint shells ~60% of pixels are negative and 0% are exactly zero, whereas the
true CosmoGrid shell is a sparse COUNT field (shell 3: 91.8% exact zeros). No
pixel-wise fix (clip / additive shift / log-density) can give Cl + positivity +
the right one-point pdf at once -- all three were measured and each destroys the
small-scale Cl (clip: 0.74, debias: 0.61, log-density: 0.41 at ell 800-1500, vs
0.93 for --no-clip). A field with nbar~0.1 counts/pixel and CosmoGrid's small-
scale power MUST go negative if it is continuous. The resolution is to stop
treating the output as a continuous field and re-discretize it:

    counts ~ Poisson(lambda),   lambda = lognormal intensity, lambda > 0

Then Cl_counts = Cl_lambda + shot(nbar), so we build lambda to carry exactly the
SHOT-DECONVOLVED target clustering, and Poisson sampling puts the shot noise back:

    Cl_lambda(ell) = Cl_high(ell) - nbar * Omega_pix          (shot subtraction)
    xi_lambda(theta) = Legendre(Cl_lambda)
    xi_g(theta) = ln(1 + xi_lambda(theta) / nbar^2)           (lognormal transform)
    Cl_g(ell) = Legendre^-1(xi_g)
    lambda = nbar * exp(g - sigma_g^2/2),   g Gaussian with Cl_g and DISCO's phases

Getting `g` right is the whole difficulty. The lognormal transform assumes g is a
GAUSSIAN RANDOM FIELD. Two attempts that FAIL (both measured, don't retry):
  * g from DISCO's log-density alms directly: log1p of sparse counts is mostly
    zeros with spikes, wildly non-Gaussian -> exp(g) explodes, lambda_max ~1e6.
  * rank-order Gaussianize the source, then impose Cl_g with a harmonic filter:
    fixes the MARGINAL (skew -0.002, kurt 0.005) but not the FIELD. The sparse
    source is nearly white, so the filter must boost ell<100 by ~10-12x, which
    resurrects a fat tail (skew 0.61, kurt 9.8, max/std 16.5 vs 5.7 Gaussian) and
    lambda_max ~1e9.
What works: build g as a real GRF -- DISCO's structure only where it is smooth and
approximately Gaussian (low ell, from the SMOOTHED corrected map's log-density),
plus an independent Gaussian realization above ell_c. Randomizing the small-scale
phases costs nothing: the fitted r(ell) shows low/high are decorrelated there
(r=0.56 at ell=1000, 0.057 at ell=2500) and 86% of sigma_g^2 sits below ell=1000.

CALIBRATING g'S POWER (2026-07-10, second pass): a single global amplitude (found
by bisecting realized var(lambda) to the target) works but leaves a SYSTEMATIC
(not noise -- reproducible across seeds to <1%) shape bias, because one scalar
can only fix total power, not the ell-dependent shape: shell 10 sat flat at
0.90-0.96 across ALL bands. On top of that, the hard ell_c cutoff between "DISCO
phases" and "random Gaussian" is itself a bias source: even on shell 30 (dense,
no lognormal difficulty at all) it left a flat 0.95 that a smooth cosine taper
alone (no other change) fixed to 0.99-1.01. So: (a) taper the ell_c transition
over +-`taper` instead of a hard step, (b) replace the single amplitude with a
PER-ELL multiplicative correction on cl_g, updated by DAMPED, AVERAGED fixed-point
iteration: cl_g *= ratio**damp where ratio = smoothed(Cl_high)/smoothed(Cl_counts),
Cl_counts averaged over `n_avg` independent Poisson draws per iteration (a single
draw is dominated by shot noise at low nbar and the correction chases that noise
-> diverges, measured: undamped single-draw feedback oscillated between 0.4x and
1.8x over 4 iterations and never settled). damp=0.4, n_avg=4, n_iter=5 converges
cleanly on every shell tried (3, 10, 30) to Cl_counts/Cl_high within ~3.5% at every
band on a FINAL held-out draw (i.e. not one of the draws used to fit the
correction). This is `n_iter*n_avg`-times the cost of one draw -- budget for it.

Status (measured on cosmo_000122, ell_c=300, taper=100, damp=0.4, n_avg=4, n_iter=5):
  shell  mean   %zero (true)     std (true)      Cl_counts/Cl_high (final draw)
    3   0.999   91.87 (91.81)   0.437 (0.442)    1.02 1.03 0.99 0.97 0.97
   10   0.999   56.00 (51.04)   2.576 (2.755)    1.01 1.00 1.00 1.00 1.00
Non-negative integer counts, exact mean, sparsity right, Cl within a few percent
at all ell bands tried. Remaining known gap: shell 3's max=98 vs true 56 (rare
outlier pixels still a bit hot -- the lognormal tail is intrinsically hard to
control exactly at sigma_g^2=2.23). Not yet re-validated on shell 30 with the
iteration (taper alone already gave 0.99-1.01 there in one pass, so it may not
need it) or on the full 69-shell range -- start there before trusting shells
outside {3, 10, 30} blindly.

Usage
-----
  python poisson_resample.py --corrected <noclip.npz> --run-dir <grid>/cosmo_X/run_0 \
      --lmax 3000 --nside 2048 --out counts.npz [--shells 3 10 30] [--seed 0]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import healpy as hp


def _alm(vec, n_alm):
    return (vec[:n_alm] + 1j * vec[n_alm:]).astype(np.complex128)


# ---------------------------------------------------------------------------
# Exact Cl <-> xi Legendre transforms (Gauss-Legendre quadrature).
# Verified round-trip to 2.5e-8 relative. healpy has no built-in for this.
# ---------------------------------------------------------------------------

def cl2xi(cl: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """xi(mu) = sum_l (2l+1)/(4pi) * Cl * P_l(mu), via the standard recurrence."""
    xi = np.zeros_like(mu)
    p_prev2 = np.ones_like(mu)
    p_prev1 = mu.copy()
    xi += (1.0 / (4 * np.pi)) * cl[0] * p_prev2
    if cl.size > 1:
        xi += (3.0 / (4 * np.pi)) * cl[1] * p_prev1
    for ell in range(2, cl.size):
        p = ((2 * ell - 1) * mu * p_prev1 - (ell - 1) * p_prev2) / ell
        xi += (2 * ell + 1) / (4 * np.pi) * cl[ell] * p
        p_prev2, p_prev1 = p_prev1, p
    return xi


def xi2cl(xi: np.ndarray, mu: np.ndarray, w: np.ndarray, lmax: int) -> np.ndarray:
    """Cl = 2pi * integral xi(mu) P_l(mu) dmu, on the Gauss-Legendre grid (mu, w)."""
    cl = np.zeros(lmax + 1)
    p_prev2 = np.ones_like(mu)
    p_prev1 = mu.copy()
    cl[0] = 2 * np.pi * np.sum(w * xi * p_prev2)
    if lmax >= 1:
        cl[1] = 2 * np.pi * np.sum(w * xi * p_prev1)
    for ell in range(2, lmax + 1):
        p = ((2 * ell - 1) * mu * p_prev1 - (ell - 1) * p_prev2) / ell
        cl[ell] = 2 * np.pi * np.sum(w * xi * p)
        p_prev2, p_prev1 = p_prev1, p
    return cl


def smooth_cl(cl: np.ndarray, window: int = 81) -> np.ndarray:
    """Boxcar-smooth Cl in log10-ell-bin space (see transfer_function.py's version).
    Used to denoise the feedback ratio in the iteration below -- a single Poisson
    draw's per-ell Cl is dominated by shot noise, especially at low nbar, and an
    unsmoothed ratio makes the iteration chase that noise instead of the bias."""
    from scipy.ndimage import uniform_filter1d
    log_cl = np.log10(np.clip(cl, 1e-30, None))
    return 10.0 ** uniform_filter1d(log_cl, size=window, mode="nearest")


def _taper(ells: np.ndarray, ell_c: int, taper: int) -> np.ndarray:
    """1 for ell<<ell_c, 0 for ell>>ell_c, smooth cosine transition over +-taper.
    A HARD step here (win=1 below ell_c else 0) is itself a measurable Cl bias --
    on shell 30 (dense, no lognormal difficulty at all) it alone left a flat 0.95
    that this taper fixed to 0.99-1.01 with no other change."""
    lo, hi = ell_c - taper, ell_c + taper
    return 0.5 * (1 + np.cos(np.pi * np.clip((ells - lo) / (hi - lo), 0.0, 1.0)))


def _one_draw(rho_native, source_map, lbar, cl_g, lmax, nside, ells, win, rng):
    """Build one Gaussian g (DISCO structure below ell_c via `win`, independent
    Gaussian realization above) carrying power cl_g, exponentiate to a positive
    mass-conserving lambda, and draw one Poisson realization from it."""
    m_s = hp.alm2map(hp.almxfl(hp.map2alm(source_map, lmax=lmax, iter=0), win),
                     nside=nside, lmax=lmax)
    m_s = m_s - m_s.mean() + lbar
    g_low = np.log(np.clip(m_s, 1e-3 * lbar, None) / lbar)
    g_low -= g_low.mean()
    gl_alm = hp.map2alm(g_low, lmax=lmax, iter=0)
    cl_gl = hp.alm2cl(gl_alm, lmax=lmax)
    amp = np.zeros(lmax + 1)
    ok = cl_gl > 0
    amp[ok] = np.sqrt(cl_g[ok] * win[ok] ** 2 / cl_gl[ok])
    g = hp.alm2map(hp.almxfl(gl_alm, amp), nside=nside, lmax=lmax)

    cl_hi_part = cl_g * (1.0 - win ** 2)
    np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))   # hp.synalm uses the global RNG
    g = g + hp.alm2map(hp.synalm(cl_hi_part, lmax=lmax, new=True), nside=nside, lmax=lmax)

    sig2 = float(g.var())
    lam = lbar * np.exp(g - 0.5 * sig2)
    lam *= rho_native.sum() / lam.sum()               # exact mass conservation
    return rng.poisson(lam).astype(np.float64), lam


def resample_shell(rho_native, cl_high, source_map, lmax, nside, mu, w, rng,
                   ell_c=300, taper=100, n_avg=4, n_iter=5, damp=0.4, verbose=True):
    """One shell: shot-deconvolve -> lognormal target Cl_g -> iteratively calibrated
    Gaussian g -> lambda>0 -> Poisson counts. See module docstring for why a single
    global amplitude (first pass) left a systematic ~5-10% shape bias and why a
    hard ell_c cutoff is itself a bias source independent of the lognormal step.

    `g` must be a genuine GAUSSIAN RANDOM FIELD, not merely a field with a Gaussian
    marginal (rank-order Gaussianizing the source then filtering to Cl_g FAILS: the
    sparse source is nearly white, the filter boosts ell<100 by ~10-12x, resurrects
    a fat tail, lambda_max ~1e9). Build g as DISCO's structure (from the smoothed,
    hence ~Gaussian, log-density of the corrected map) below ell_c, tapered into an
    independent Gaussian realization above -- the fitted r(ell) shows low/high are
    decorrelated there anyway (r=0.056 at ell=2500), so this costs no real info.
    """
    npix = hp.nside2npix(nside)
    omega = 4 * np.pi / npix
    ells = np.arange(lmax + 1)
    lbar = float(rho_native.mean())

    # 1. Target INTENSITY power = observed count power minus the Poisson shot floor.
    #    Poisson sampling below puts that shot floor back, so counts land on cl_high.
    cl_lam = np.clip(cl_high - lbar * omega, 0.0, None)
    cl_lam[0] = 0.0                                   # monopole carried by lbar

    # 2. Exact lognormal transform in configuration space.
    xi_lam = cl2xi(cl_lam, mu)
    arg = 1.0 + xi_lam / lbar ** 2
    n_bad = int(np.sum(arg <= 0))
    xi_g = np.log(np.clip(arg, 1e-12, None))
    cl_g = np.clip(xi2cl(xi_g, mu, w, lmax), 0.0, None)
    cl_g[0] = 0.0

    win = _taper(ells, ell_c, taper)
    cl_h_s = smooth_cl(cl_high)

    # 3. Damped, averaged fixed-point iteration on a PER-ELL correction to cl_g.
    #    A single draw's Cl is shot-noise dominated at low nbar; averaging n_avg
    #    draws before computing the feedback ratio, and damping the update
    #    (corr *= ratio**damp, not the full ratio), is what keeps this from
    #    oscillating/diverging -- measured: undamped single-draw feedback swung
    #    between 0.4x and 1.8x of target over 4 iterations and never settled.
    corr = np.ones(lmax + 1)
    for it in range(n_iter):
        cl_g_iter = cl_g * corr
        cl_n_sum = None
        for _ in range(n_avg):
            n_it, _ = _one_draw(rho_native, source_map, lbar, cl_g_iter, lmax, nside,
                                ells, win, rng)
            cl_n_it = hp.anafast(n_it, lmax=lmax)
            cl_n_sum = cl_n_it.copy() if cl_n_sum is None else cl_n_sum + cl_n_it
        cl_n_s = smooth_cl(cl_n_sum / n_avg)
        ratio = np.clip(cl_h_s / np.clip(cl_n_s, 1e-30, None), 0.3, 3.0)
        corr = np.clip(corr * ratio ** damp, 0.02, 50.0)

    # 4. Final held-out draw with the converged correction (not one of the draws
    #    used to fit `corr`, so this is a fair check, not curve-fitting-on-itself).
    counts, lam = _one_draw(rho_native, source_map, lbar, cl_g * corr, lmax, nside,
                            ells, win, rng)
    if verbose:
        print(f"    lbar={lbar:.4f} lam:[{lam.min():.2e},{lam.max():.1f}] "
              f"corr:[{corr.min():.3f},{corr.max():.3f}]"
              + (f"  [WARN xi_g arg<=0 at {n_bad} angles]" if n_bad else ""), flush=True)
    return counts.astype(np.float32)


def main():
    from numpy.polynomial.legendre import leggauss
    p = argparse.ArgumentParser()
    p.add_argument("--corrected", required=True,
                   help="npz from `transfer_function.py apply --no-clip` (shells key).")
    p.add_argument("--run-dir", required=True, help="Run dir with high_alms + low shells.")
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ell-c", type=int, default=300,
                   help="Below this ell keep DISCO's phases; above it use an "
                        "independent Gaussian realization (r(ell) shows the phases "
                        "are noise up there anyway). 300 measured better than 800.")
    p.add_argument("--taper", type=int, default=100,
                   help="Cosine-taper width around ell_c. A hard cutoff (taper=0) "
                        "is itself a measurable Cl bias (flat ~0.95 on shell 30).")
    p.add_argument("--n-avg", type=int, default=4,
                   help="Poisson draws averaged per iteration to denoise the "
                        "feedback ratio (shot noise dominates a single draw).")
    p.add_argument("--n-iter", type=int, default=5,
                   help="Damped fixed-point iterations calibrating the per-ell "
                        "correction on cl_g. Cost scales as n_avg*n_iter draws/shell.")
    p.add_argument("--damp", type=float, default=0.4,
                   help="Exponent on the per-iteration correction (corr *= "
                        "ratio**damp). damp=1 (full step) measured to oscillate/"
                        "diverge; 0.4 converges cleanly within n_iter=5.")
    p.add_argument("--shells", type=int, nargs="*", default=None,
                   help="Subset of shell indices (default: all).")
    p.add_argument("--info-npz", default="compressed_shells.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lmax, nside = args.lmax, args.nside
    n_alm = (lmax + 1) * (lmax + 2) // 2
    run = Path(args.run_dir)
    mu, w = leggauss(2 * lmax + 64)
    rng = np.random.default_rng(args.seed)

    corrected = np.load(args.corrected, mmap_mode="r")["shells"]
    low_full = np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r")
    high_alms = np.load(run / f"high_alms_lmax{lmax}.npy", mmap_mode="r")

    n_shells = corrected.shape[0]
    idx = sorted(args.shells) if args.shells else list(range(n_shells))
    out = np.zeros((n_shells, hp.nside2npix(nside)), dtype=np.float32)

    for i in idx:
        print(f"  shell {i}/{n_shells}", flush=True)
        rho = np.asarray(low_full[i], dtype=np.float64)
        cl_h = hp.alm2cl(_alm(np.asarray(high_alms[i]), n_alm), lmax=lmax)
        src = np.asarray(corrected[i], dtype=np.float64)
        out[i] = resample_shell(rho, cl_h, src, lmax, nside, mu, w, rng,
                                ell_c=args.ell_c, taper=args.taper,
                                n_avg=args.n_avg, n_iter=args.n_iter, damp=args.damp)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    info = run / args.info_npz
    if info.exists():
        d = np.load(info, allow_pickle=True)
        extra = {k: d[k] for k in d.files if k != "shells"}
    # `shells_done` marks which shells were actually resampled. With a --shells
    # subset the rest stay all-zero, which downstream tools cannot distinguish
    # from a real (empty) shell -- vis/visualize.py rightly dies with
    # "Shell map has non-positive mean" on them. Record the mask and shout.
    done = np.zeros(n_shells, dtype=bool); done[idx] = True
    np.savez(outp, shells=out, shells_done=done, **extra)
    print(f"[poisson] saved count map to {outp}", flush=True)
    if not done.all():
        missing = np.flatnonzero(~done)
        print(f"[poisson] WARNING: {missing.size}/{n_shells} shells were NOT resampled "
              f"and are left as ZEROS: {missing.tolist()}\n"
              f"          Only shells {idx} are usable. Plotting/normalizing any other "
              f"shell will fail (mean=0). Re-run without --shells for a full product.",
              flush=True)


if __name__ == "__main__":
    main()
