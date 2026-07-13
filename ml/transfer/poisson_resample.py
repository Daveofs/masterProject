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


def _r_window(R: np.ndarray, window: int = 41) -> np.ndarray:
    """The per-ell DISCO/CosmoGrid phase-correlation R(ell) (from transfer_function.py
    fit/train, same R used by --stochastic there), lightly boxcar-smoothed, IS the
    correct mixing weight -- not a fixed ell_c cutoff.

    FIXED ell_c=300 (first pass) was wrong: measured R(ell) varies enormously by
    shell -- shell 3 (faint) drops to 0.56 by ell=1000, but shell 30/50 (dense)
    stay at R>=0.985 all the way to ell=2500. A fixed cutoff at 300 discarded
    PERFECTLY GOOD, ~100%-correlated DISCO structure on the dense shells for no
    reason, replacing real filaments with random noise -- visibly grainier
    "corrected" patches than either input, even though the aggregate Cl still came
    out close (power spectra don't see phases, so this damage was invisible to the
    band-averaged Cl checks and only showed up by eye). R directly controls the
    win/(1-win^2) split already used by amp/cl_hi_part below, so win=R is not a
    heuristic -- it is exactly the standard Wiener/constrained-realization
    correlated-signal-plus-independent-noise decomposition (r*signal +
    sqrt(1-r^2)*noise preserves total power for any r(ell), keeping ALL the real
    structure the data actually supports, no more no less)."""
    from scipy.ndimage import uniform_filter1d
    r = np.clip(uniform_filter1d(np.clip(R, 0.0, 1.0), size=window, mode="nearest"), 0.0, 1.0)
    return r


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


def resample_shell(rho_native, cl_high, source_map, R, lmax, nside, mu, w, rng,
                   n_avg=4, n_iter=5, damp=0.4, verbose=True):
    """One shell: shot-deconvolve -> lognormal target Cl_g -> iteratively calibrated
    Gaussian g -> lambda>0 -> Poisson counts. See module docstring for why a single
    global amplitude (first pass) left a systematic ~5-10% shape bias, and why a
    FIXED ell_c cutoff (second pass) was itself a worse bias -- it discarded good
    DISCO phase structure on dense shells (R stays ~1 to ell=2500 there) and
    visibly degraded those images even though the aggregate Cl still looked fine.

    `g` must be a genuine GAUSSIAN RANDOM FIELD, not merely a field with a Gaussian
    marginal (rank-order Gaussianizing the source then filtering to Cl_g FAILS: the
    sparse source is nearly white, the filter boosts ell<100 by ~10-12x, resurrects
    a fat tail, lambda_max ~1e9). Build g as DISCO's structure (from the smoothed,
    hence ~Gaussian, log-density of the corrected map), mixed with an independent
    Gaussian realization using the ACTUAL per-ell phase correlation R(ell) as the
    mixing weight (see _r_window) -- not a fixed cutoff -- so every shell keeps
    exactly as much real structure as the data supports.
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

    win = _r_window(R)
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


# ---------------------------------------------------------------------------
# Shell-level PARALLELISM. Different shells are completely independent (no data
# dependency), but resample_shell's own OMP threading (map2alm/alm2map/anafast at
# nside=2048, lmax=3000) saturates well before using many-dozens of cores: measured
# the SAME ~2h wall-clock for n_avg=4/n_iter=5 across 69 shells whether the node had
# 64 or 288 cpus allocated -- the extra cores were simply never used, since a single
# shell's SHT calls don't scale that far no matter how high OMP_NUM_THREADS is set.
# Real speedup needs TASK parallelism across shells (like preprocess_alms.py's
# --num-workers), combined with a SMALLER per-worker thread count so many shells run
# concurrently instead of one shell hogging every core.
#
# Each worker gets only cheap-to-pickle inputs (a shell index, small (lmax+1,) T/R
# arrays, file PATHS) and mmap-reads its own shell's slice of the on-disk alm/shell
# files directly -- deliberately NOT the large (npix,) precomputed arrays (~400MB
# each at nside=2048), which would make inter-process pickling itself a bottleneck.
# ---------------------------------------------------------------------------

def _worker_init(threads_per_worker: int):
    """ProcessPoolExecutor initializer: cap OMP threads per worker so n_workers run
    concurrently instead of each saturating the whole node. Must happen before any
    healpy/libsharp call in this (fresh, spawn-context) process reads the env var."""
    import os
    os.environ["OMP_NUM_THREADS"] = str(max(1, threads_per_worker))


def _resample_one_shell_task(task: dict):
    """One shell's FULL pipeline (transfer-function residual -> Poisson resample),
    run in a worker process. See module docstring above for why inputs are paths/
    small arrays, not the large per-shell maps themselves."""
    import healpy as hp
    i = task["i"]
    lmax, nside = task["lmax"], task["nside"]
    N_alm = (lmax + 1) * (lmax + 2) // 2
    ell = np.empty(N_alm, dtype=np.int64)
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            ell[m * (2 * lmax + 1 - m) // 2 + l] = l

    run = Path(task["run_dir"])
    low = np.load(run / task["low_alm_fname"], mmap_mode="r")
    low_full = np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r")
    high_alms = np.load(run / f"high_alms_lmax{lmax}.npy", mmap_mode="r")

    Ti, Ri = task["Ti"], task["Ri"]                    # already ell_min-masked by caller
    v = np.asarray(low[i], dtype=np.float64)
    tvec = (Ti - 1.0)[ell]
    alm_delta = v[:N_alm] * tvec + 1j * v[N_alm:] * tvec
    delta_map = hp.alm2map(alm_delta, nside=nside, lmax=lmax)
    rho_native = np.asarray(low_full[i], dtype=np.float64)
    m_unclipped = rho_native + delta_map

    cl_h_i = hp.alm2cl(_alm(np.asarray(high_alms[i]), N_alm), lmax=lmax)
    from numpy.polynomial.legendre import leggauss
    mu, w = leggauss(2 * lmax + 64)
    rng = np.random.default_rng(task["seed"] + i)
    counts = resample_shell(rho_native, cl_h_i, m_unclipped, Ri, lmax, nside, mu, w, rng,
                            n_avg=task["n_avg"], n_iter=task["n_iter"], damp=task["damp"],
                            verbose=task["verbose"])
    return i, counts


def resample_all_shells_parallel(run_dir, low_alm_fname, lmax, nside, T, R,
                                 ell_min_per_shell, n_avg=4, n_iter=5, damp=0.4,
                                 seed=0, n_workers=1, total_cpus=None, verbose=True):
    """Drop-in replacement for a sequential `for i: resample_shell(...)` loop.

    Two speed levers, in order of how much they're trusted:
      1. SKIP shells where ell_min_i >= lmax entirely (T forced to 1 at every mode
         -> the correction is IDENTICALLY ZERO -> m_unclipped == rho_native exactly
         -- already a valid non-negative integer count map). Running the full
         lognormal+Poisson machinery there wouldn't just waste ~90s/shell, it would
         actively REPLACE DISCO's real structure with fresh Poisson noise for a
         shell --ell-min-mpc says to leave untouched. Free (mathematically exact,
         not an approximation) and, with --ell-min-mpc 3, cuts real work by ~49%
         (34/69 shells on cosmo_000122). Always on.
      2. n_workers > 1: dispatch the REMAINING shells across a process pool so
         independent shells run concurrently. OFF by default (n_workers=1, plain
         sequential at whatever OMP_NUM_THREADS the caller set) -- measured 23
         workers x 12 threads to be dramatically SLOWER than sequential (memory-
         bandwidth contention from many concurrent large-array SHTs), and OMP
         threading alone saturates at ~128 threads (measured: 128->256 threads
         gave zero further speedup on a single shell). Only raise this after
         validating a specific worker count on THIS hardware -- it is not a safe
         default lever the way (1) is.

    T, R: (n_shells_T, lmax+1) arrays (transfer.npz's T/R, possibly fewer rows than
    n_shells -- the last row is reused, matching apply()'s existing convention).
    ell_min_per_shell: (n_shells,) int array (0 = no restriction).
    """
    import os
    import healpy as hp

    n_shells = len(ell_min_per_shell)
    npix = 12 * nside * nside
    total_cpus = total_cpus or os.cpu_count() or n_workers
    corrected = np.zeros((n_shells, npix), dtype=np.float32)

    tasks, skipped = [], []
    for i in range(n_shells):
        Ti = T[min(i, T.shape[0] - 1)].copy()
        Ri = R[min(i, R.shape[0] - 1)].copy() if R is not None else np.ones(lmax + 1)
        ell_min_i = int(ell_min_per_shell[i])
        if ell_min_i > 0:
            Ti[:ell_min_i] = 1.0
            Ri[:ell_min_i] = 1.0
        if ell_min_i >= lmax:
            skipped.append(i)   # T==1 everywhere -> handled below, no task needed
            continue
        tasks.append({"i": i, "run_dir": str(run_dir), "low_alm_fname": low_alm_fname,
                     "lmax": lmax, "nside": nside, "Ti": Ti, "Ri": Ri,
                     "n_avg": n_avg, "n_iter": n_iter, "damp": damp, "seed": seed,
                     "verbose": verbose and (len(tasks) % max(1, n_shells // 20) == 0)})

    if skipped:
        low_full = np.load(Path(run_dir) / f"low_shells_nside={nside}.npy", mmap_mode="r")
        for i in skipped:
            corrected[i] = np.asarray(low_full[i], dtype=np.float32)
        if verbose:
            print(f"[poisson-parallel] {len(skipped)}/{n_shells} shells have "
                  f"ell_min>=lmax (T==1 everywhere) -> passed through as DISCO's "
                  f"own counts, unmodified, no resample needed", flush=True)

    n_workers = max(1, min(n_workers, len(tasks))) if tasks else 1
    if verbose:
        print(f"[poisson-parallel] {len(tasks)} shells need resampling, "
              f"{n_workers} worker(s)" + (f" x {total_cpus // n_workers} OMP threads "
              f"each ({total_cpus} cpus total)" if n_workers > 1 else
              f" (sequential, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')})"),
              flush=True)

    if n_workers == 1:
        # Plain sequential -- see docstring for why this is the trusted default.
        for n_done, t in enumerate(tasks, 1):
            i, counts = _resample_one_shell_task(t)
            corrected[i] = counts
            if verbose and n_done % max(1, len(tasks) // 10) == 0:
                print(f"[poisson-parallel] {n_done}/{len(tasks)} shells done", flush=True)
        return corrected

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    threads_per_worker = max(1, total_cpus // n_workers)
    ctx = mp.get_context("spawn")   # fresh interpreter per worker -- avoids inheriting
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(threads_per_worker,)) as ex:
        futures = [ex.submit(_resample_one_shell_task, t) for t in tasks]
        n_done = 0
        for fut in as_completed(futures):
            i, counts = fut.result()
            corrected[i] = counts
            n_done += 1
            if verbose and n_done % max(1, len(tasks) // 10) == 0:
                print(f"[poisson-parallel] {n_done}/{len(tasks)} shells done", flush=True)
    return corrected


def main():
    from numpy.polynomial.legendre import leggauss
    p = argparse.ArgumentParser()
    p.add_argument("--corrected", required=True,
                   help="npz from `transfer_function.py apply --no-clip` (shells key).")
    p.add_argument("--transfer", required=True,
                   help="transfer.npz from transfer_function.py fit/emulate -- its R "
                        "array (per-shell, per-ell DISCO/CosmoGrid phase correlation) "
                        "is the mixing weight between DISCO's real structure and an "
                        "independent Gaussian realization (see _r_window). NOT a fixed "
                        "ell_c cutoff -- that discarded good structure on dense shells.")
    p.add_argument("--run-dir", required=True, help="Run dir with high_alms + low shells.")
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
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
    R_all = np.load(args.transfer)["R"]

    n_shells = corrected.shape[0]
    idx = sorted(args.shells) if args.shells else list(range(n_shells))
    out = np.zeros((n_shells, hp.nside2npix(nside)), dtype=np.float32)

    for i in idx:
        print(f"  shell {i}/{n_shells}", flush=True)
        rho = np.asarray(low_full[i], dtype=np.float64)
        cl_h = hp.alm2cl(_alm(np.asarray(high_alms[i]), n_alm), lmax=lmax)
        src = np.asarray(corrected[i], dtype=np.float64)
        R_i = R_all[min(i, R_all.shape[0] - 1)]
        out[i] = resample_shell(rho, cl_h, src, R_i, lmax, nside, mu, w, rng,
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
