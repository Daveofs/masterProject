#!/usr/bin/env python3
"""Per-ell transfer-function correction for DISCO-DJ -> CosmoGrid small scales.

Physics (measured via the per-ell cross-correlation r(ell)): for the shells that
carry signal, low and high SHARE PHASES (r~0.9-1.0 well into small scales); DISCO
just gets the small-scale AMPLITUDE wrong. So the Cl correction is deterministic:

    corrected_alm(ell, m) = low_alm(ell, m) * T(ell, shell),
    T(ell, shell) = sqrt( <Cl_high(ell, shell)> / <Cl_low(ell, shell)> )   (train avg)

This matches Cl_high exactly (by construction), preserves the (correct) phases, is
completely stable, and touches only the scales where T != 1 (small scales; T~1 at
large ell). The transfer function is ~a property of the sim-code pair, so a per-
shell (redshift) average over training cosmologies transfers to the test cosmology.

Two ways to get T(ell, shell):
  * fit      : single train-averaged T (deterministic; assumes T is cosmology-
               independent). Robust baseline.
  * train +  : a standard MLP *emulator* T = f(l, z, H0, O_cdm, Ob, Om, ns, s8,
    emulate    Cl_low) that interpolates T to a held-out cosmology. `emulate`
               writes a transfer.npz in the same schema `apply` consumes.

Usage
-----
  # A. Averaged transfer function (baseline):
  python transfer_function.py fit --data-dir <grid> --lmax 3000 \
      --test-cosmo cosmo_000001 --out transfer.npz
  # B. Emulated transfer function (cosmology-conditioned MLP):
  python transfer_function.py train --data-dir <grid> --lmax 3000 \
      --test-cosmo cosmo_000001 --out emulator.pkl
  python transfer_function.py emulate --emulator emulator.pkl \
      --run-dir <grid>/cosmo_000001/run_0 --out transfer.npz
  # Then apply either transfer.npz to a test run's low alms -> corrected shells:
  python transfer_function.py apply --transfer transfer.npz \
      --run-dir <grid>/cosmo_000001/run_0 --nside 2048 --ell-min 0 --out corrected.npz
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import healpy as hp

# Cosmological parameters the emulator conditions on. Only these keys are read
# from each run's params.yml (the user-requested subset of the CosmoGrid params).
COSMO_KEYS = ["H0", "O_cdm", "Ob", "Om", "ns", "s8"]


def _alm(vec, N_alm):
    return (vec[:N_alm] + 1j * vec[N_alm:]).astype(np.complex128)


def _debias_mean(m_unclipped: np.ndarray, tol: float = 1e-9, max_iter: int = 50) -> np.ndarray:
    """Undo the clip-at-0 mean bias with a pure ADDITIVE shift-then-reclip.

    Flooring negative pixels at 0 always raises the mean (it only ever removes
    negative mass, never adds it back) -- measured up to +25% on the faintest
    shells even in --log-density mode, because those shells are so sparse that
    most pixels sit exactly at the log1p(rho)=0 boundary already, so almost ANY
    correction pushes a large fraction of them slightly negative. An ADDITIVE
    shift (subtract a constant, reclip) only touches the ell=0/monopole term in
    the region where no pixel is reclipped by it, unlike a multiplicative /
    mass-conserving rescale -- tested and found to distort Cl at ALL ell by
    roughly scale^2, since it rescales the already-good small-scale structure
    too. Solve for the shift c with mean(clip(m_unclipped,0) - c, 0) ==
    mean(m_unclipped) (the unclipped mean tracks the true mean far better than
    the clipped one -- measured within 0.5-5% on shells where clipping fires on
    30-70% of pixels) via bisection (monotonic in c; no closed form since
    clipping is piecewise)."""
    m_clipped = np.clip(m_unclipped, 0.0, None)
    target = m_unclipped.mean()
    lo, hi = 0.0, float(m_clipped.max())
    if hi <= 0.0:
        return m_clipped
    for _ in range(max_iter):
        c = 0.5 * (lo + hi)
        if np.clip(m_clipped - c, 0.0, None).mean() > target:
            lo = c
        else:
            hi = c
        if hi - lo < tol:
            break
    return np.clip(m_clipped - 0.5 * (lo + hi), 0.0, None)


def alm_fname(kind: str, lmax: int, log_density: bool) -> str:
    """kind: 'low' or 'high'. log_density selects the log1p(rho) alm variant
    (written by preprocess_alms.py --log-density) instead of the raw-density one.
    See apply()'s --log-density branch for why: correcting log-density and
    reconstructing via expm1 is always >= -1 (density >= 0 with a tiny clip for
    fp noise) with NO large clipping bias, unlike correcting density directly and
    flooring negative excursions at 0 -- which, measured on real data, inflates
    the mean of shot-noise shells by ~20% and biases Cl by the same amount at
    ALL ell (proven: pre-clip corrected/high ~1.00, post-clip ~1.2)."""
    return f"{kind}{'_log' if log_density else ''}_alms_lmax{lmax}.npy"


def ell_of_flat_index(lmax: int) -> np.ndarray:
    ell = np.empty((lmax + 1) * (lmax + 2) // 2, dtype=np.int64)
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            ell[m * (2 * lmax + 1 - m) // 2 + l] = l
    return ell


# ---------------------------------------------------------------------------
# Emulator helpers: read cosmology / redshift and build the feature matrix
# ---------------------------------------------------------------------------

def load_cosmo(run_dir: Path) -> np.ndarray:
    """Read the requested cosmological parameters (COSMO_KEYS) from params.yml."""
    import yaml
    with open(run_dir / "params.yml") as f:
        p = yaml.safe_load(f)
    missing = [k for k in COSMO_KEYS if k not in p]
    if missing:
        raise KeyError(f"{run_dir/'params.yml'} missing params {missing}")
    return np.array([float(p[k]) for k in COSMO_KEYS], dtype=np.float64)


def shell_redshifts(run_dir: Path, n_shells: int, info_npz: str) -> np.ndarray:
    """Per-shell redshift z = mean(lower_z, upper_z) from the shell_info metadata.

    Falls back to the shell index if the metadata is unavailable, so the emulator
    still runs (z then just acts as an ordered shell label)."""
    info = run_dir / info_npz
    if info.exists():
        d = np.load(info, allow_pickle=True)
        if "shell_info" in d.files:
            si = d["shell_info"]
            z = 0.5 * (si["lower_z"].astype(np.float64) + si["upper_z"].astype(np.float64))
            if z.shape[0] >= n_shells:
                return z[:n_shells]
            # pad by extrapolating the last spacing (rare shell-count mismatch)
            return np.concatenate([z, z[-1] + (np.arange(n_shells - z.shape[0]) + 1)
                                   * (z[-1] - z[-2] if z.shape[0] > 1 else 1.0)])
    print(f"  [warn] no shell_info in {info}; using shell index as z", flush=True)
    return np.arange(n_shells, dtype=np.float64)


def ell_min_from_mpc_h(z: np.ndarray, cosmo: np.ndarray, scale_mpc_h: float) -> np.ndarray:
    """Per-shell ell below which T is forced to 1 (no correction), so a FIXED
    physical (comoving) scale in Mpc/h -- not a fixed ell -- is what's left
    untouched. A fixed ell_min corresponds to a DIFFERENT comoving scale at every
    shell (comoving distance grows with z), which is not what "only correct scales
    smaller than 3 Mpc/h" means physically.

    Flat-sky/Limber mapping ell(k, chi) ~= k*chi (LoVerde & Afshordi 2008), with
    k = 2*pi/L the 3D wavenumber for comoving scale L, and chi(z) the comoving
    distance -- both derived from the SAME cosmology.yml used elsewhere (COSMO_KEYS
    order: H0, O_cdm, Ob, Om, ns, s8), not a hardcoded fiducial value, so this
    tracks the actual test cosmology's expansion history.
    """
    from astropy.cosmology import FlatLambdaCDM
    H0, Om = float(cosmo[0]), float(cosmo[3])
    h = H0 / 100.0
    chi_mpc_h = FlatLambdaCDM(H0=H0, Om0=Om).comoving_distance(z).value * h  # Mpc/h
    ell_min = np.ceil(2.0 * np.pi * chi_mpc_h / scale_mpc_h).astype(np.int64)
    return np.clip(ell_min, 0, None)


def highpass_ell_ramp(ell: np.ndarray, ell_min: int, transition: float) -> np.ndarray:
    """Raised-cosine ramp w(ell) in [0,1]: 0 below ell_min, smoothly rising to 1
    over a transition band of width `transition` (in ell), 1 above ell_min+transition.

    Same shape (0.5*(1-cos(pi*t))) as the highpass masks the OTHER three
    correction pipelines converged on independently after all three showed a
    systematic kappa Cl bias traced to a HARD large-scale cutoff (see
    unet/flow_model.py's _highpass_mask, diffusion/model.py's _highpass_mask,
    sphereflow/sphere_flow.py's graph_highpass -- and the [[deepsphere-shell-
    correction]] memory) -- ported here, not imported (this project's ml/
    pipeline dirs never cross-import, see [[feedback-decoupled-pipeline-
    modules]]; transfer_function.py -> apply_transfer.py is a WITHIN-pipeline
    import, same as those three already do internally, so this one is fine).
    Applied here to apply()'s (Ti-1)/(Ri-1) correction deltas directly in ell
    space -- the EXACT equivalent of those pipelines' 2D-FFT/graph radial
    highpass on a flat-patch or graph signal, but exact rather than
    approximate, since T/R are already per-ell scalars with no spatial extent
    to Fourier-transform.

    transition<=0 reduces to a hard step (0 below ell_min, else 1) -- the
    ORIGINAL apply() behavior (a step function in harmonic space rings in real
    space at the cutoff scale, same Gibbs-phenomenon motivation the other three
    pipelines' docstrings give for smoothing theirs)."""
    if transition <= 0:
        return (ell >= ell_min).astype(np.float64)
    t = np.clip((ell.astype(np.float64) - ell_min) / transition, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def build_features(ell: np.ndarray, z: float, cosmo: np.ndarray,
                   cl_low: np.ndarray) -> np.ndarray:
    """Rows of [l, z, H0, O_cdm, Ob, Om, ns, s8, Cl_low] for one shell (per ell).

    l and Cl_low span many decades, so they enter in log10; the StandardScaler
    fitted at train time normalises everything else. cosmo is the COSMO_KEYS vector.
    """
    n = ell.shape[0]
    cl = np.log10(np.clip(cl_low, 1e-30, None))
    X = np.empty((n, 2 + len(COSMO_KEYS) + 1), dtype=np.float64)
    X[:, 0] = np.log10(ell.astype(np.float64) + 1.0)   # l
    X[:, 1] = z                                          # z
    X[:, 2:2 + len(COSMO_KEYS)] = cosmo                  # H0, O_cdm, Ob, Om, ns, s8
    X[:, -1] = cl                                        # Cl_low
    return X


def smooth_cl(cl: np.ndarray, window: int) -> np.ndarray:
    """Smooth a per-ell Cl in log10-space with a boxcar filter of `window` points.

    A single map's Cl(ell) has real sample-variance / shot-noise scatter at fixed
    ell; the physical transfer function T(ell) = sqrt(Cl_high/Cl_low) is expected
    to vary SMOOTHLY with ell. If we feed raw per-ell Cl as an input feature and
    train against the raw per-realization ratio as the target, the MLP partially
    fits that per-ell noise (can still show high validation R^2, dominated by
    easy low-ell samples) and reproduces/amplifies it at predict time on a new
    cosmology -> an oscillating T(ell). Smoothing before computing ratios/
    features removes the noise at its source instead of papering over it.
    """
    if window <= 1:
        return cl
    from scipy.ndimage import uniform_filter1d
    log_cl = np.log10(np.clip(cl, 1e-30, None))
    return 10.0 ** uniform_filter1d(log_cl, size=window, mode="nearest")


def _run_cls(run_dir: Path, lmax: int, log_density: bool = False):
    """Per-shell (Cl_low, Cl_high, Cl_cross) for a run from its preprocessed alms,
    in ONE pass over the alm arrays. `train()` used to call this for just
    (Cl_low, Cl_high) and then separately reload the SAME low/high alm files and
    recompute alm2cl(al)/alm2cl(ah) a second time just to also get the cross-
    spectrum -- doubling both the disk I/O and the SHT compute for no reason.
    Returning all three here in one loop eliminates that."""
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(run_dir / alm_fname("low", lmax, log_density), mmap_mode="r")
    high = np.load(run_dir / alm_fname("high", lmax, log_density), mmap_mode="r")
    # Drop the LAST lightcone shell (2026-07-22, matches unet/dataset.py's
    # split_by_cosmo and apply_*'s n_shells_total-=1): DISCO's low map there
    # carries only 16-65% of CosmoGrid's true counts (a truncated shell at the
    # lightcone/box edge), and z-conditioning can't separate it from the
    # second-to-last shell -- training on it was measured to teach a spurious
    # "subtract high-ell power at z~3.4" correction (corrected/true Cl ~0.41-0.53
    # at the SECOND-to-last shell for unet, across all 30 held-out cosmologies)
    # even though that shell's own input was fine. Same truncated-edge-shell data
    # feeds `train`/`fit` here, so the same exclusion applies.
    n = min(low.shape[0], high.shape[0]) - 1
    cl_low = np.empty((n, lmax + 1))
    cl_high = np.empty((n, lmax + 1))
    cl_cross = np.empty((n, lmax + 1))
    for i in range(n):
        al = _alm(np.asarray(low[i]), N_alm)
        ah = _alm(np.asarray(high[i]), N_alm)
        cl_low[i] = hp.alm2cl(al, lmax=lmax)
        cl_high[i] = hp.alm2cl(ah, lmax=lmax)
        cl_cross[i] = hp.alm2cl(al, ah, lmax=lmax)
    return cl_low, cl_high, cl_cross


def _gather_worker_init(threads_per_worker: int):
    """ProcessPoolExecutor initializer -- caps each worker's OpenMP threads so
    N_WORKERS x threads_per_worker stays within the node's cpu budget (matches
    poisson_resample.py's _worker_init pattern). Must run before this process's
    first healpy/OpenMP call, which it does since it's the pool's initializer."""
    os.environ["OMP_NUM_THREADS"] = str(max(1, threads_per_worker))


def _gather_fit_task(task):
    """Worker: per-run (Cl_low, Cl_high, Cl_cross), for `fit`'s parallel gather."""
    lo, hi, lmax, log_density = task
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(lo, mmap_mode="r")
    high = np.load(hi, mmap_mode="r")
    n = min(low.shape[0], high.shape[0]) - 1     # drop last shell -- see _run_cls
    cl_low = np.empty((n, lmax + 1)); cl_high = np.empty((n, lmax + 1))
    cl_cross = np.empty((n, lmax + 1))
    for i in range(n):
        al = _alm(np.asarray(low[i]), N_alm)
        ah = _alm(np.asarray(high[i]), N_alm)
        cl_low[i] = hp.alm2cl(al, lmax=lmax)
        cl_high[i] = hp.alm2cl(ah, lmax=lmax)
        cl_cross[i] = hp.alm2cl(al, ah, lmax=lmax)
    return str(lo.parent), cl_low, cl_high, cl_cross


def _gather_train_task(task):
    """Worker: everything `train()`'s per-run loop body needs, for `train`'s
    parallel gather -- the alm2cl calls (via _run_cls) are the expensive part;
    the rest (smoothing, build_features, sampling) is cheap vectorized numpy,
    bundled in here too so the main process only has to reduce/concatenate."""
    ld, lmax, log_density, info_npz, smooth_window, sample_frac, seed = task
    cosmo = load_cosmo(ld)
    cl_low, cl_high, cl_cross = _run_cls(ld, lmax, log_density)
    n_shells = cl_low.shape[0]
    z = shell_redshifts(ld, n_shells, info_npz)
    cl_low_s = np.stack([smooth_cl(c, smooth_window) for c in cl_low])
    cl_high_s = np.stack([smooth_cl(c, smooth_window) for c in cl_high])
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.sqrt(np.where(cl_low_s > 0, cl_high_s / cl_low_s, 1.0))
    T = np.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0)
    ell = np.arange(lmax + 1)
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for i in range(n_shells):
        X = build_features(ell, float(z[i]), cosmo, cl_low_s[i])
        y = T[i]
        if sample_frac < 1.0:
            keep = rng.random(ell.shape[0]) < sample_frac
            X, y = X[keep], y[keep]
        Xs.append(X); ys.append(y)
    return (str(ld), n_shells, cl_low, cl_high, cl_cross, np.concatenate(Xs), np.concatenate(ys),
           cosmo, z, cl_low_s)


def split_val_cosmos(data_dir: Path, val_frac: float = 0.15, seed: int = 0,
                     low_glob: str = "disco_sim/*/disco_shells_nside=2048.npz",
                     high_name: str = "compressed_shells.npz") -> list[str]:
    """Hold out a random FRACTION of whole cosmologies for validation, mirroring
    unet/dataset.py's split_by_cosmo (same default val_frac=0.15) --
    so both pipelines validate on a comparable multi-cosmology held-out set
    instead of a single fixed test cosmology.

    Only samples from cosmologies that actually HAVE both low (DISCO) and high
    (CosmoGrid) source data -- some cosmology directories exist with only the
    CosmoGrid side present (no disco_sim/ at all, e.g. cosmo_000054), which
    `fit`/`train`'s own run-discovery silently skips when building the TRAINING
    set (see _discover_runs's lo.exists()/hi.exists() check) but which crashes
    `emulate`/`apply_transfer.py` outright if such a cosmology is picked into the
    HELD-OUT set instead (they assume the run-dir they're given is valid -- no
    equivalent existence check). Filtering here, at selection time, is the single
    point that protects every downstream consumer of the held-out list."""
    data_dir = Path(data_dir)
    cosmos = sorted(d.name for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))

    def _has_data(name: str) -> bool:
        c = data_dir / name
        rs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")] or [c]
        return any((next(r.glob(low_glob), None) is not None) and (r / high_name).exists()
                  for r in rs)

    cosmos = [c for c in cosmos if _has_data(c)]
    rng = np.random.default_rng(seed)
    rng.shuffle(cosmos)
    n_val = max(1, int(round(len(cosmos) * val_frac)))
    return sorted(cosmos[:n_val])


# ---------------------------------------------------------------------------
# Fit: accumulate mean Cl_low / Cl_high per (shell, ell) over training runs
# ---------------------------------------------------------------------------

def _discover_fit_runs(data_dir: Path, lmax: int, test_cosmos: set[str],
                       include_test: bool, log_density: bool):
    """(lo, hi) alm path pairs for `fit`'s training set -- same exclusion logic as
    _discover_runs, kept separate since fit() works with alm file PATHS directly
    (no per-run object needed) whereas train()/_discover_runs returns run dirs."""
    cosmos = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    runs = []
    for c in cosmos:
        if c.name in test_cosmos and not include_test:
            continue
        rs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        for ld in (rs or [c]):
            lo = ld / alm_fname("low", lmax, log_density)
            hi = ld / alm_fname("high", lmax, log_density)
            if lo.exists() and hi.exists():
                runs.append((lo, hi))
    return runs


def _gather_fit_runs(runs, lmax: int, log_density: bool, gather_workers: int):
    """The accumulation half of `fit`: sum_low/sum_high/sum_cross/counts over
    `runs` (a LIST of (lo, hi) alm path pairs -- may be the FULL training set, or
    just one node's SHARD of it for multi-node gather -- see `gather-shard`/
    `gather-merge`). Split out of fit() so both the single-node path and the
    sharded multi-node path call the exact same code."""
    N_alm = (lmax + 1) * (lmax + 2) // 2
    sum_low = sum_high = sum_cross = None
    counts = None

    def _accumulate(cl_low, cl_high, cl_cross):
        nonlocal sum_low, sum_high, sum_cross, counts
        n = cl_low.shape[0]
        if sum_low is None:
            sum_low = np.zeros((n, lmax + 1))
            sum_high = np.zeros((n, lmax + 1))
            sum_cross = np.zeros((n, lmax + 1))
            counts = np.zeros(n)
        m = min(n, sum_low.shape[0])
        sum_low[:m] += cl_low[:m]; sum_high[:m] += cl_high[:m]
        sum_cross[:m] += cl_cross[:m]; counts[:m] += 1

    # --gather-workers > 1: dispatch each run's alm2cl work (I/O + SHT, the
    # expensive part) to a separate process -- these are INDEPENDENT runs
    # working on much smaller per-shell arrays than the full nside=2048 maps
    # poisson_resample.py's parallel path handles, so (unlike that path, which
    # was measured to be memory-bandwidth-bound and net-negative) this is
    # embarrassingly parallel and safe to scale up. Default 1 (sequential) --
    # opt in explicitly once validated on your hardware. For scaling BEYOND one
    # node, see gather-shard/gather-merge, which run this same function once per
    # node on a disjoint slice of `runs` and sum the results together.
    if gather_workers > 1:
        threads_per_worker = max(1, (os.cpu_count() or gather_workers) // gather_workers)
        tasks = [(lo, hi, lmax, log_density) for lo, hi in runs]
        with ProcessPoolExecutor(max_workers=gather_workers,
                                 initializer=_gather_worker_init,
                                 initargs=(threads_per_worker,)) as ex:
            futures = {ex.submit(_gather_fit_task, t): t for t in tasks}
            for fut in as_completed(futures):
                name, cl_low, cl_high, cl_cross = fut.result()
                _accumulate(cl_low, cl_high, cl_cross)
                print(f"  processed {name}", flush=True)
    else:
        for lo, hi in runs:
            low = np.load(lo, mmap_mode="r")
            high = np.load(hi, mmap_mode="r")
            n = min(low.shape[0], high.shape[0]) - 1     # drop last shell -- see _run_cls
            cl_low = np.empty((n, lmax + 1)); cl_high = np.empty((n, lmax + 1))
            cl_cross = np.empty((n, lmax + 1))
            for i in range(n):
                al, ah = _alm(np.asarray(low[i]), N_alm), _alm(np.asarray(high[i]), N_alm)
                cl_low[i] = hp.alm2cl(al, lmax=lmax)
                cl_high[i] = hp.alm2cl(ah, lmax=lmax)
                cl_cross[i] = hp.alm2cl(al, ah, lmax=lmax)
            _accumulate(cl_low, cl_high, cl_cross)
            print(f"  processed {lo.parent}", flush=True)

    return sum_low, sum_high, sum_cross, counts


def _finalize_fit(sum_low, sum_high, sum_cross, counts, lmax: int, log_density: bool,
                  test_cosmos: set[str], out):
    """T(ell,shell)/R(ell,shell) from fit()'s (possibly MERGED, see gather-merge)
    accumulators, and save transfer.npz. Split out of fit() so both the
    single-node and sharded-then-merged paths produce byte-identical output."""
    mean_low = sum_low / counts[:, None]
    mean_high = sum_high / counts[:, None]
    mean_cross = sum_cross / counts[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.sqrt(np.where(mean_low > 0, mean_high / mean_low, 1.0))
        # r(ell,shell): phase cross-correlation of low vs high. r->1 means low's
        # small-scale phases genuinely track high (T's power boost recovers real
        # structure); r->0 means that ell is shot-noise/decorrelation dominated
        # for BOTH maps (independent Poisson draws at the same mean density) and
        # boosting by T alone just amplifies DISCO's own uncorrelated noise to
        # match the CosmoGrid noise LEVEL, not its pattern -- visually this reads
        # as extra unstructured graininess. `apply` uses r*T (Wiener/MMSE gain)
        # by default so faint/decorrelated scales aren't blindly over-inflated.
        R = np.where((mean_low > 0) & (mean_high > 0),
                     mean_cross / np.sqrt(mean_low * mean_high), 0.0)
    T = np.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    R = np.clip(np.nan_to_num(R, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, T=T, R=R, lmax=lmax, log_density=log_density,
             mean_low=mean_low.astype(np.float32), mean_high=mean_high.astype(np.float32),
             test_cosmos=np.array(sorted(test_cosmos)))
    print(f"[fit] saved transfer function T{T.shape} (mean r={R.mean():.3f}, "
          f"log_density={log_density}) to {out}", flush=True)


def fit(args):
    data_dir = Path(args.data_dir)
    lmax = args.lmax

    # Hold out a SET of cosmologies (explicit --test-cosmos, or auto-selected via
    # --val-frac/--val-seed -- same whole-cosmology-split convention as
    # unet/dataset.py's split_by_cosmo) so validation covers MULTIPLE
    # cosmologies, not just one -- a single held-out cosmo can't distinguish a
    # genuinely generalizing T from one that got lucky on that particular cosmology.
    test_cosmos = set(args.test_cosmos) if args.test_cosmos \
        else set(split_val_cosmos(data_dir, args.val_frac, args.val_seed))

    runs = _discover_fit_runs(data_dir, lmax, test_cosmos, args.include_test, args.log_density)
    if not runs:
        raise RuntimeError(f"No low/high alm files (lmax={lmax}) found under {data_dir}")
    mode = "INCLUDING test (sanity check)" if args.include_test \
        else f"excluding {len(test_cosmos)} held-out cosmologies: {sorted(test_cosmos)}"
    print(f"[fit] {len(runs)} training runs ({mode})", flush=True)

    sum_low, sum_high, sum_cross, counts = _gather_fit_runs(
        runs, lmax, args.log_density, args.gather_workers)
    _finalize_fit(sum_low, sum_high, sum_cross, counts, lmax, args.log_density,
                  test_cosmos, args.out)


# ---------------------------------------------------------------------------
# Emulator: learn T(ell, shell) = f(l, z, H0, O_cdm, Ob, Om, ns, s8, Cl_low)
# ---------------------------------------------------------------------------
# Instead of a single train-averaged T (fit), train a standard MLP regressor that
# predicts the per-mode transfer function from the low-res Cl and the cosmology
# vector, so it *interpolates* T to a new (held-out) cosmology. Same target as fit
# (T = sqrt(Cl_high / Cl_low)); output of `emulate` is a transfer.npz in the exact
# schema `apply` / prepare_tcorr_dataset already consume.

def _discover_runs(data_dir: Path, lmax: int, test_cosmos: set[str], include_test: bool,
                   log_density: bool = False):
    cosmos = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    runs = []
    for c in cosmos:
        if c.name in test_cosmos and not include_test:
            continue
        rs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        for ld in (rs or [c]):
            if (ld / alm_fname("low", lmax, log_density)).exists() and \
               (ld / alm_fname("high", lmax, log_density)).exists():
                runs.append(ld)
    return runs


def _gather_train_runs(runs, lmax: int, log_density: bool, info_npz, smooth_window,
                       sample_frac: float, gather_workers: int):
    """The gather half of `train`: builds the concatenated (X, y) feature/target
    arrays plus the r(ell,shell) accumulators over `runs` (a LIST of run dirs --
    may be the FULL training set, or just one node's SHARD of it for multi-node
    gather -- see `gather-shard`/`gather-merge`). Split out of train() so both the
    single-node path and the sharded multi-node path call the exact same code.
    Returns (X, y, sum_cross, sum_low, sum_high, r_counts, last_cosmo, last_z,
    last_cl_low_s) -- the last three are just SOME real (cosmo, z, cl_low) triple
    (whichever run finishes last) for the cosmo-vector sanity check in
    _finalize_train."""
    ell = np.arange(lmax + 1)
    Xs, ys = [], []
    # r(ell,shell) (phase cross-correlation, see `fit`) is averaged over training
    # cosmologies rather than emulated: it's mostly a shot-noise/resolution
    # property of the shell (not strongly cosmology-dependent), and at apply time
    # for a held-out cosmology there is no ground-truth high map to compute it
    # from directly -- this train-set average is the best available estimate.
    sum_cross = sum_low = sum_high = None
    r_counts = None
    # Stashed from whichever run is accumulated LAST (order depends on completion
    # order under --gather-workers > 1) -- used below by the cosmo-vector sanity
    # check, which just needs SOME real (cosmo, z, cl_low) triple, not a specific one.
    last_cosmo = last_z = last_cl_low_s = None

    def _accumulate(name, n_shells, cl_low, cl_high, cl_cross, X, y, cosmo, z, cl_low_s):
        nonlocal sum_cross, sum_low, sum_high, r_counts, last_cosmo, last_z, last_cl_low_s
        if sum_cross is None:
            sum_cross = np.zeros((n_shells, lmax + 1))
            sum_low = np.zeros((n_shells, lmax + 1))
            sum_high = np.zeros((n_shells, lmax + 1))
            r_counts = np.zeros(n_shells)
        m = min(n_shells, sum_cross.shape[0])
        sum_cross[:m] += cl_cross[:m]; sum_low[:m] += cl_low[:m]
        sum_high[:m] += cl_high[:m]; r_counts[:m] += 1
        Xs.append(X); ys.append(y)
        last_cosmo, last_z, last_cl_low_s = cosmo, z, cl_low_s
        print(f"  gathered {name} ({n_shells} shells)", flush=True)

    # --gather-workers > 1: same rationale as fit()'s parallel path -- runs are
    # independent, and _gather_train_task bundles the ENTIRE per-run body
    # (alm2cl + smoothing + feature-building) so a worker returns everything
    # the main process needs to just accumulate/concatenate. Default 1
    # (sequential, exactly today's behavior). For scaling BEYOND one node, see
    # gather-shard/gather-merge, which run this same function once per node on a
    # disjoint slice of `runs` and merge the results together.
    if gather_workers > 1:
        threads_per_worker = max(1, (os.cpu_count() or gather_workers) // gather_workers)
        tasks = [(ld, lmax, log_density, info_npz, smooth_window,
                 sample_frac, i) for i, ld in enumerate(runs)]
        with ProcessPoolExecutor(max_workers=gather_workers,
                                 initializer=_gather_worker_init,
                                 initargs=(threads_per_worker,)) as ex:
            futures = {ex.submit(_gather_train_task, t): t for t in tasks}
            for fut in as_completed(futures):
                name, n_shells, cl_low, cl_high, cl_cross, X, y, cosmo, z, cl_low_s = fut.result()
                _accumulate(name, n_shells, cl_low, cl_high, cl_cross, X, y, cosmo, z, cl_low_s)
    else:
        rng = np.random.default_rng(0)
        for ld in runs:
            cosmo = load_cosmo(ld)
            cl_low, cl_high, cl_cross = _run_cls(ld, lmax, log_density)
            n_shells = cl_low.shape[0]
            z = shell_redshifts(ld, n_shells, info_npz)
            # Smooth BEFORE computing the ratio: T's target is otherwise a per-ell,
            # per-realization noisy ratio (see smooth_cl docstring for why this is
            # the actual source of the oscillating predictions).
            cl_low_s = np.stack([smooth_cl(c, smooth_window) for c in cl_low])
            cl_high_s = np.stack([smooth_cl(c, smooth_window) for c in cl_high])
            with np.errstate(divide="ignore", invalid="ignore"):
                T = np.sqrt(np.where(cl_low_s > 0, cl_high_s / cl_low_s, 1.0))
            T = np.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0)
            run_Xs, run_ys = [], []
            for i in range(n_shells):
                X = build_features(ell, float(z[i]), cosmo, cl_low_s[i])
                y = T[i]
                if sample_frac < 1.0:                     # subsample ell for tractability
                    keep = rng.random(ell.shape[0]) < sample_frac
                    X, y = X[keep], y[keep]
                run_Xs.append(X); run_ys.append(y)
            _accumulate(str(ld), n_shells, cl_low, cl_high, cl_cross,
                       np.concatenate(run_Xs), np.concatenate(run_ys), cosmo, z, cl_low_s)

    X = np.concatenate(Xs); y = np.concatenate(ys)
    return X, y, sum_cross, sum_low, sum_high, r_counts, last_cosmo, last_z, last_cl_low_s


def _finalize_train(X, y, sum_cross, sum_low, sum_high, r_counts, last_cosmo, last_z,
                    last_cl_low_s, lmax: int, log_density: bool, test_cosmos: set[str],
                    hidden_str: str, alpha: float, max_iter: int, smooth_window, out):
    """R(ell,shell) + MLP training + save, from train()'s (possibly MERGED, see
    gather-merge) gather outputs. Split out of train() so both the single-node
    and sharded-then-merged paths run the exact same finalize logic -- unlike
    _finalize_fit, this is NOT byte-identical between the two paths, since the
    manual train/val split + partial_fit loop below is inherently seeded by
    array ORDER, which differs once shards from different nodes are
    concatenated (see gather-merge's docstring for why this is acceptable)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib

    ell = np.arange(lmax + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_cross, mean_low_r, mean_high_r = (s / r_counts[:, None]
                                                for s in (sum_cross, sum_low, sum_high))
        R = np.where((mean_low_r > 0) & (mean_high_r > 0),
                     mean_cross / np.sqrt(mean_low_r * mean_high_r), 0.0)
    R = np.clip(np.nan_to_num(R, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    print(f"[train] mean r(ell,shell) across training set = {R.mean():.3f}", flush=True)
    print(f"[train] {X.shape[0]:,} samples x {X.shape[1]} features", flush=True)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    hidden = tuple(int(h) for h in hidden_str.split(","))

    # Manual train/val split + partial_fit loop, NOT model.fit(early_stopping=True):
    # sklearn's built-in early stopping only exposes validation_scores_ (R^2, a
    # SCORE that increases as it improves), while loss_curve_ (training) is a LOSS
    # that decreases -- opposite sign conventions, so they can never be shown as
    # "both curves decreasing" on the same axis. jbucko's flow loss_curve.png plots
    # train_loss and val_loss on the SAME formula/footing (see plot_flow_loss.py);
    # matching that structurally (not just visually) needs a genuine per-iteration
    # validation LOSS, which requires driving the iterations ourselves.
    rng_split = np.random.default_rng(0)
    n = Xs.shape[0]
    val_idx_mask = rng_split.random(n) < 0.1          # same 10% convention as before
    X_tr, y_tr = Xs[~val_idx_mask], y[~val_idx_mask]
    X_va, y_va = Xs[val_idx_mask], y[val_idx_mask]

    # Loss/val-loss are evaluated on a FIXED random SUBSAMPLE, not the full
    # (multi-million-row) train/val sets: predict() over the full training set
    # EVERY iteration (just to plot train_loss) was measured to make each
    # iteration take ~9 minutes on the real ~9M-sample dataset -- 200 iterations
    # would be tens of hours -- even though partial_fit() itself (the actual
    # training step) is fast. A fixed subsample gives the same genuine held-out
    # MSE semantics (matching the plotted formula) at a small, bounded,
    # iteration-count-independent cost.
    eval_rng = np.random.default_rng(1)
    n_eval = 50_000
    tr_eval_idx = eval_rng.choice(X_tr.shape[0], size=min(n_eval, X_tr.shape[0]), replace=False)
    va_eval_idx = eval_rng.choice(X_va.shape[0], size=min(n_eval, X_va.shape[0]), replace=False)
    X_tr_eval, y_tr_eval = X_tr[tr_eval_idx], y_tr[tr_eval_idx]
    X_va_eval, y_va_eval = X_va[va_eval_idx], y_va[va_eval_idx]

    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                         solver="adam", alpha=alpha, batch_size=4096,
                         learning_rate_init=1e-4, verbose=False, random_state=0)
    train_loss_hist, val_loss_hist = [], []
    best_val, best_weights, no_improve, patience = np.inf, None, 0, 10
    for it in range(max_iter):
        model.partial_fit(X_tr, y_tr)                  # one epoch (adam, batch_size=4096)
        train_loss = float(np.mean((model.predict(X_tr_eval) - y_tr_eval) ** 2)) / 2
        val_loss = float(np.mean((model.predict(X_va_eval) - y_va_eval) ** 2)) / 2
        train_loss_hist.append(train_loss); val_loss_hist.append(val_loss)
        if val_loss < best_val - 1e-9:
            best_val, best_weights, no_improve = val_loss, \
                ([c.copy() for c in model.coefs_], [b.copy() for b in model.intercepts_]), 0
        else:
            no_improve += 1
        if it % 10 == 0:
            print(f"  epoch {it}: train_loss={train_loss:.3e} val_loss={val_loss:.3e}", flush=True)
        if no_improve >= patience:
            print(f"[train] early stopping at epoch {it} "
                  f"(no val improvement for {patience} epochs)", flush=True)
            break
    if best_weights is not None:                       # restore best-val weights
        model.coefs_, model.intercepts_ = best_weights
    print(f"[train] final: train_loss={train_loss_hist[-1]:.3e} "
          f"best_val_loss={best_val:.3e}", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "lmax": lmax,
                 "smooth_window": smooth_window, "R": R,
                 "log_density": log_density, "test_cosmos": sorted(test_cosmos),
                 "cosmo_keys": COSMO_KEYS, "feature_order":
                 ["log10(l+1)", "z", *COSMO_KEYS, "log10(Cl_low)"]}, out)
    print(f"[train] saved emulator to {out}", flush=True)

    # ---- loss / validation-loss curve plot ----
    # Same shared figure (analysis.plot_train_val_loss) as jbucko's plot_flow_loss.py
    # -- both curves are plain squared error (no L2 term; alpha's weight decay is an
    # OPTIMIZER detail, not part of the reported/plotted loss, so train vs val is an
    # apples-to-apples comparison), so a gap between them is the generalization gap,
    # directly comparable to jbucko's train-vs-held-out-cosmology gap.
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from analysis.plotting import plot_train_val_loss
        loss_png = Path(out).with_suffix("").with_suffix(".loss.png")
        plot_train_val_loss(
            np.arange(len(train_loss_hist)), train_loss_hist, val_loss_hist, loss_png,
            xlabel="epoch", ylabel="MLP squared-error loss (T emulator)",
            val_label=f"validation (10% held-out samples, alpha={alpha:g})",
            formula=r"loss $=\frac{1}{2N}\sum_i(T_i-\hat T_i)^2$  "
                    "(L2 weight decay is an optimizer detail, not shown)")
    except Exception as e:
        print(f"[train] loss plot failed: {e}", flush=True)

    # ---- cosmo-vector sanity check: does the model actually respond to it? ----
    # Compare T predicted with SOME training run's real cosmo vector (whichever
    # was accumulated last -- order depends on completion order under
    # --gather-workers > 1, but any real run works equally well here) against T
    # predicted with the training-set MEAN cosmo vector (scaler.mean_ stores means
    # in original, unscaled units). If the model ignored cosmo entirely, these
    # would be identical.
    mean_cosmo = scaler.mean_[2:2 + len(COSMO_KEYS)]
    z_ref = float(last_z[len(last_z) // 2]); cl_ref = last_cl_low_s[len(last_z) // 2]
    X_real = build_features(ell, z_ref, last_cosmo, cl_ref)
    X_mean = build_features(ell, z_ref, mean_cosmo, cl_ref)
    T_real = model.predict(scaler.transform(X_real))
    T_mean = model.predict(scaler.transform(X_mean))
    diff = np.abs(T_real - T_mean)
    print(f"[train] cosmo-vec sanity: |T(real cosmo) - T(mean training cosmo)| "
          f"max={diff.max():.4f} mean={diff.mean():.4f} "
          f"(near-zero would mean the model ignores cosmo)", flush=True)


def train(args):
    data_dir = Path(args.data_dir)
    lmax = args.lmax
    # See fit()'s docstring comment: MULTIPLE held-out cosmologies (explicit
    # --test-cosmos, or auto-selected via --val-frac/--val-seed, same convention
    # as unet's split_by_cosmo), not just one.
    test_cosmos = set(args.test_cosmos) if args.test_cosmos \
        else set(split_val_cosmos(data_dir, args.val_frac, args.val_seed))
    runs = _discover_runs(data_dir, lmax, test_cosmos, args.include_test, args.log_density)
    if not runs:
        raise RuntimeError(f"No low/high alm files (lmax={lmax}) found under {data_dir}")
    mode = "INCLUDING test (sanity)" if args.include_test \
        else f"excluding {len(test_cosmos)} held-out cosmologies: {sorted(test_cosmos)}"
    print(f"[train] {len(runs)} training runs ({mode})", flush=True)

    X, y, sum_cross, sum_low, sum_high, r_counts, last_cosmo, last_z, last_cl_low_s = \
        _gather_train_runs(runs, lmax, args.log_density, args.info_npz, args.smooth_window,
                           args.sample_frac, args.gather_workers)
    _finalize_train(X, y, sum_cross, sum_low, sum_high, r_counts, last_cosmo, last_z,
                    last_cl_low_s, lmax, args.log_density, test_cosmos, args.hidden,
                    args.alpha, args.max_iter, args.smooth_window, args.out)


def _pad_rows(arr, n):
    """Zero-pad `arr` (shape (m, ...)) along axis 0 out to n rows. Used by
    gather-merge: a shard's accumulator shape is set by whichever run its OWN
    gather loop processes first, so two shards can disagree on n_shells if runs
    happen to have different shell counts (a pre-existing quirk of the single-node
    _accumulate closures too -- not new here). Padding with zero rows is safe
    because the matching `counts`/`r_counts` row is also 0 there, so it never
    contributes to the merged mean (mean = sum/counts, and _finalize_fit/
    _finalize_train already nan_to_num the resulting 0/0)."""
    if arr.shape[0] == n:
        return arr
    pad = np.zeros((n - arr.shape[0],) + arr.shape[1:], dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def gather_shard(args):
    """Multi-node stage-2 gather, ONE shard: run fit's or train's gather step
    over just this shard's slice of the (identical, deterministically-sorted)
    training-run list (`runs[shard_index::num_shards]`), and save the raw
    (unfinalized) accumulators to --out. Meant to be launched once per node
    (see run_transfer.sh) with a different --shard-index each; `gather-merge`
    then combines every shard's --out file into the final transfer.npz /
    emulator.pkl. Uses the SAME test_cosmos resolution (--test-cosmos, or
    --val-frac/--val-seed) as `fit`/`train` so every shard excludes the exact
    same held-out cosmologies."""
    import joblib
    data_dir = Path(args.data_dir)
    lmax = args.lmax
    test_cosmos = set(args.test_cosmos) if args.test_cosmos \
        else set(split_val_cosmos(data_dir, args.val_frac, args.val_seed))

    if args.method == "fit":
        runs = _discover_fit_runs(data_dir, lmax, test_cosmos, args.include_test, args.log_density)
    else:
        runs = _discover_runs(data_dir, lmax, test_cosmos, args.include_test, args.log_density)
    if not runs:
        raise RuntimeError(f"No runs found under {data_dir} for method={args.method}")

    shard_runs = runs[args.shard_index::args.num_shards]
    print(f"[gather-shard] method={args.method} shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard_runs)}/{len(runs)} runs", flush=True)
    if not shard_runs:
        raise RuntimeError(f"Shard {args.shard_index}/{args.num_shards} got 0 runs "
                           f"out of {len(runs)} total -- --num-shards too high?")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.method == "fit":
        sum_low, sum_high, sum_cross, counts = _gather_fit_runs(
            shard_runs, lmax, args.log_density, args.gather_workers)
        joblib.dump({"method": "fit", "lmax": lmax, "log_density": args.log_density,
                    "sum_low": sum_low, "sum_high": sum_high, "sum_cross": sum_cross,
                    "counts": counts, "test_cosmos": sorted(test_cosmos)}, args.out)
    else:
        X, y, sum_cross, sum_low, sum_high, r_counts, last_cosmo, last_z, last_cl_low_s = \
            _gather_train_runs(shard_runs, lmax, args.log_density, args.info_npz,
                               args.smooth_window, args.sample_frac, args.gather_workers)
        joblib.dump({"method": "train", "lmax": lmax, "log_density": args.log_density,
                    "smooth_window": args.smooth_window, "X": X, "y": y,
                    "sum_cross": sum_cross, "sum_low": sum_low, "sum_high": sum_high,
                    "r_counts": r_counts, "last_cosmo": last_cosmo, "last_z": last_z,
                    "last_cl_low_s": last_cl_low_s, "test_cosmos": sorted(test_cosmos)}, args.out)
    print(f"[gather-shard] saved shard to {args.out}", flush=True)


def gather_merge(args):
    """Combine every shard file from `gather-shard` (one per node) into the
    final transfer.npz (method=fit) or emulator.pkl (method=train). For fit,
    this is byte-identical to running `fit` single-node on the full run list
    (plain accumulator sums). For train, the MERGED (X, y) is handed to the
    exact same manual partial_fit/early-stopping loop as single-node `train` --
    NOT byte-identical (that loop's random train/val split and eval subsample
    are seeded by array ORDER, which differs once shards are concatenated), but
    trains on the same total data with the same procedure."""
    import joblib
    shards = [joblib.load(s) for s in args.shards]
    if len({s["method"] for s in shards}) != 1:
        raise RuntimeError(f"Mixed shard methods in --shards: "
                           f"{sorted({s['method'] for s in shards})}")
    method = shards[0]["method"]
    lmaxes = {s["lmax"] for s in shards}
    log_densities = {s["log_density"] for s in shards}
    if len(lmaxes) != 1 or len(log_densities) != 1:
        raise RuntimeError("Shards disagree on lmax/log_density -- were they all "
                           "produced by the same gather-shard invocation pattern?")
    lmax, log_density = lmaxes.pop(), log_densities.pop()
    test_cosmos = set(shards[0]["test_cosmos"])
    if any(set(s["test_cosmos"]) != test_cosmos for s in shards[1:]):
        raise RuntimeError("Shards disagree on held-out test_cosmos -- were they "
                           "produced with the same --test-cosmos/--val-frac/--val-seed?")

    if method == "fit":
        n = max(s["sum_low"].shape[0] for s in shards)
        sum_low = sum(_pad_rows(s["sum_low"], n) for s in shards)
        sum_high = sum(_pad_rows(s["sum_high"], n) for s in shards)
        sum_cross = sum(_pad_rows(s["sum_cross"], n) for s in shards)
        counts = sum(_pad_rows(s["counts"], n) for s in shards)
        _finalize_fit(sum_low, sum_high, sum_cross, counts, lmax, log_density,
                      test_cosmos, args.out)
    else:
        smooth_windows = {s["smooth_window"] for s in shards}
        if len(smooth_windows) != 1:
            raise RuntimeError("Shards disagree on --smooth-window -- were they all "
                               "gathered with the same value?")
        smooth_window = smooth_windows.pop()
        X = np.concatenate([s["X"] for s in shards])
        y = np.concatenate([s["y"] for s in shards])
        n = max(s["sum_low"].shape[0] for s in shards)
        sum_cross = sum(_pad_rows(s["sum_cross"], n) for s in shards)
        sum_low = sum(_pad_rows(s["sum_low"], n) for s in shards)
        sum_high = sum(_pad_rows(s["sum_high"], n) for s in shards)
        r_counts = sum(_pad_rows(s["r_counts"], n) for s in shards)
        last = shards[-1]        # any shard's real (cosmo, z, cl_low) triple works
        _finalize_train(X, y, sum_cross, sum_low, sum_high, r_counts,
                        last["last_cosmo"], last["last_z"], last["last_cl_low_s"],
                        lmax, log_density, test_cosmos, args.hidden,
                        args.alpha, args.max_iter, smooth_window, args.out)
    print(f"[gather-merge] merged {len(shards)} shards (method={method}) -> {args.out}", flush=True)


def emulate(args):
    """Predict T(shell, ell) for a run with the trained emulator -> transfer.npz."""
    import joblib
    bundle = joblib.load(args.emulator)
    model, scaler = bundle["model"], bundle["scaler"]
    lmax = int(bundle["lmax"])
    smooth_window = int(bundle.get("smooth_window", 1))
    log_density = bool(bundle.get("log_density", False))
    ell = np.arange(lmax + 1)

    run = Path(args.run_dir)
    cosmo = load_cosmo(run)
    print(f"  [cosmo] {run.name}: " +
          ", ".join(f"{k}={v:.4g}" for k, v in zip(COSMO_KEYS, cosmo)), flush=True)
    # Sanity check: does the model's output actually respond to THIS cosmo vector,
    # or would it predict about the same T regardless (i.e. ignoring cosmo)?
    mean_cosmo = scaler.mean_[2:2 + len(COSMO_KEYS)]
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(run / alm_fname("low", lmax, log_density), mmap_mode="r")
    n_shells = low.shape[0]
    z = shell_redshifts(run, n_shells, args.info_npz)

    # alm2cl (one SHT per shell) can't be batched -- each shell's alm is a genuinely
    # different array. But the MLP predict() that follows CAN: it was previously
    # called once PER SHELL (69 separate (lmax+1, 9) predict() calls), each paying
    # its own Python/BLAS call overhead on a matrix too small to amortize it
    # (measured: MLPRegressor throughput on matrices this size is dominated by
    # per-call overhead, not FLOPs -- see transfer_function.py train()'s
    # OMP_NUM_THREADS/batch-size docstring notes for the same effect during
    # training). Building ALL shells' features first and predicting ONCE on the
    # (n_shells*(lmax+1), 9) matrix removes 68 of those 69 redundant calls for
    # free -- identical output, just not re-paying fixed overhead 69 times.
    cl_low_all = np.empty((n_shells, lmax + 1))
    for i in range(n_shells):
        cl_low_all[i] = smooth_cl(hp.alm2cl(_alm(np.asarray(low[i]), N_alm), lmax=lmax),
                                  smooth_window)
        if i % 10 == 0:
            print(f"  gathered Cl_low for shell {i}/{n_shells}", flush=True)

    X_all = np.concatenate([build_features(ell, float(z[i]), cosmo, cl_low_all[i])
                            for i in range(n_shells)])
    Ti_all = model.predict(scaler.transform(X_all)).reshape(n_shells, lmax + 1)

    mid = n_shells // 2
    X_mean = build_features(ell, float(z[mid]), mean_cosmo, cl_low_all[mid])
    T_mean = model.predict(scaler.transform(X_mean))
    d = np.abs(Ti_all[mid] - T_mean)
    print(f"  [cosmo check] shell {mid}: |T(this cosmo) - T(mean training cosmo)| "
          f"max={d.max():.4f} mean={d.mean():.4f} "
          f"(near-zero would mean cosmo is being ignored)", flush=True)

    # Post-hoc smoothing of the OUTPUT is a safety net: even with a smoothed
    # target/feature, a generic MLP evaluated row-by-row over ell need not be
    # perfectly smooth in ell. Same window as used for the input features.
    if smooth_window > 1:
        T = np.stack([smooth_cl(np.clip(Ti_all[i], 1e-6, None), smooth_window)
                      for i in range(n_shells)]).astype(np.float32)
    else:
        T = Ti_all.astype(np.float32)
    # T is a physical amplitude ratio; clip away nonphysical negatives.
    T = np.clip(T, 0.0, None)
    # r(ell,shell) averaged over the training cosmologies (see `train`) -- used by
    # `apply` (Wiener gain r*T) so decorrelated/shot-noise scales aren't blindly
    # boosted to the full power-matching amplitude.
    R = bundle.get("R")
    if R is not None and R.shape[0] != n_shells:
        R = R[np.minimum(np.arange(n_shells), R.shape[0] - 1)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if R is not None:
        np.savez(args.out, T=T, R=R, lmax=lmax, log_density=log_density)
    else:
        np.savez(args.out, T=T, lmax=lmax, log_density=log_density)
    print(f"[emulate] saved emulated transfer function T{T.shape} to {args.out}\n"
          f"          -> feed to `transfer_function.py apply --transfer {args.out}`",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit")
    pf.add_argument("--data-dir", required=True)
    pf.add_argument("--lmax", type=int, default=3000)
    pf.add_argument("--test-cosmos", nargs="*", default=None,
                    help="Explicit held-out cosmology name(s). If omitted, "
                         "auto-selected via --val-frac/--val-seed (whole-cosmology "
                         "split, same convention as unet's "
                         "split_by_cosmo) so validation covers MULTIPLE "
                         "cosmologies, not just one.")
    pf.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of cosmologies to hold out when --test-cosmos "
                         "is not given explicitly (same default as jbucko).")
    pf.add_argument("--val-seed", type=int, default=0)
    pf.add_argument("--include-test", action="store_true",
                    help="SANITY CHECK: include the test cosmology in the fit (expect "
                         "corrected/high ~ 1 by construction). Default is leave-one-out.")
    pf.add_argument("--log-density", action="store_true",
                    help="Fit T/R on log1p(rho) alms (preprocess_alms.py --log-density) "
                         "instead of raw density alms. `apply` reconstructs via expm1, "
                         "which is always >= -1 (i.e. rho >= 0) for any correction size, "
                         "eliminating the clip-at-0 bias that inflates Cl on shot-noise "
                         "shells with plain density-space T (measured: post-clip "
                         "corrected/high ~1.2 vs ~1.0 pre-clip on a shell where 59%% of "
                         "pixels get clipped). Saved into the output transfer.npz so "
                         "`apply` auto-detects it -- no matching flag needed there.")
    pf.add_argument("--gather-workers", type=int, default=1,
                    help="Process-pool workers for gathering per-run Cls (the "
                         "expensive alm2cl/IO part). Independent runs, small "
                         "per-shell arrays -- safe to parallelize (unlike "
                         "poisson_resample.py's per-shell full-map path, which "
                         "was measured to be memory-bandwidth-bound). Default 1 "
                         "(sequential). Remember to also raise OMP_NUM_THREADS "
                         "for this stage -- it is NOT the file-wide pipeline "
                         "default of 8.")
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=fit)

    pt = sub.add_parser("train", help="Train an MLP emulator T=f(l,z,cosmo,Cl_low).")
    pt.add_argument("--data-dir", required=True)
    pt.add_argument("--lmax", type=int, default=3000)
    pt.add_argument("--test-cosmos", nargs="*", default=None,
                    help="Same as `fit --test-cosmos` -- explicit held-out "
                         "cosmology name(s), or auto-selected via "
                         "--val-frac/--val-seed if omitted.")
    pt.add_argument("--val-frac", type=float, default=0.15)
    pt.add_argument("--val-seed", type=int, default=0)
    pt.add_argument("--include-test", action="store_true",
                    help="SANITY CHECK: train on the test cosmology too (default LOO).")
    pt.add_argument("--log-density", action="store_true",
                    help="Same as `fit --log-density` -- train on log1p(rho) alms. "
                         "Saved into the emulator bundle so `emulate` auto-detects it.")
    pt.add_argument("--info-npz", default="compressed_shells.npz",
                    help="npz holding shell_info (per-shell redshifts).")
    pt.add_argument("--hidden", default="256,256,128",
                    help="Comma-separated MLP hidden-layer sizes.")
    pt.add_argument("--alpha", type=float, default=1e-4, help="L2 regularisation.")
    pt.add_argument("--smooth-window", type=int, default=21,
                    help="Boxcar window (in ell, log10-Cl space) used to smooth "
                         "Cl_low/Cl_high before computing the T target/feature — "
                         "fixes oscillating T from per-ell realization noise. "
                         "Saved in the emulator bundle so `emulate` matches "
                         "automatically. 1 disables smoothing.")
    pt.add_argument("--max-iter", type=int, default=200)
    pt.add_argument("--sample-frac", type=float, default=1.0,
                    help="Randomly keep this fraction of ell per shell (speed).")
    pt.add_argument("--gather-workers", type=int, default=1,
                    help="Same as `fit --gather-workers` -- process-pool workers "
                         "for the per-run Cl-gathering loop. Default 1 (sequential).")
    pt.add_argument("--out", required=True, help="Output emulator .pkl (joblib).")
    pt.set_defaults(func=train)

    pe = sub.add_parser("emulate", help="Predict T for a run -> transfer.npz (apply-ready).")
    pe.add_argument("--emulator", required=True, help="emulator .pkl from `train`.")
    pe.add_argument("--run-dir", required=True,
                    help="Run dir with low_alms_lmax{lmax}.npy + params.yml.")
    pe.add_argument("--info-npz", default="compressed_shells.npz")
    pe.add_argument("--out", required=True, help="transfer.npz (same schema as fit).")
    pe.set_defaults(func=emulate)

    # ---- multi-node stage-2 gather: run `gather-shard` once per node (each with a
    # different --shard-index, same --num-shards), then one `gather-merge` call
    # combining every shard's --out file into the final transfer.npz/emulator.pkl.
    # See run_transfer.sh for the actual srun/wait orchestration. Single-node `fit`/
    # `train` are unaffected -- this is an alternative entry point into the same
    # _gather_fit_runs/_gather_train_runs + _finalize_fit/_finalize_train code.
    ps = sub.add_parser("gather-shard",
                        help="Multi-node: gather ONE shard of fit's/train's training "
                             "runs (--shard-index/--num-shards) to a partial-"
                             "accumulator file. Combine shards with `gather-merge`.")
    ps.add_argument("--data-dir", required=True)
    ps.add_argument("--lmax", type=int, default=3000)
    ps.add_argument("--method", required=True, choices=["fit", "train"],
                    help="Which gather step to shard -- must match the eventual "
                         "`gather-merge --method`.")
    ps.add_argument("--test-cosmos", nargs="*", default=None,
                    help="Must match across every shard for a given merge -- "
                         "pass the SAME explicit list (or the same --val-frac/"
                         "--val-seed) to every gather-shard invocation.")
    ps.add_argument("--val-frac", type=float, default=0.15)
    ps.add_argument("--val-seed", type=int, default=0)
    ps.add_argument("--include-test", action="store_true")
    ps.add_argument("--log-density", action="store_true")
    ps.add_argument("--info-npz", default="compressed_shells.npz",
                    help="train method only.")
    ps.add_argument("--smooth-window", type=int, default=21, help="train method only.")
    ps.add_argument("--sample-frac", type=float, default=1.0, help="train method only.")
    ps.add_argument("--shard-index", type=int, required=True,
                    help="This shard's index in [0, num_shards) -- "
                         "runs[shard_index::num_shards] get processed here.")
    ps.add_argument("--num-shards", type=int, required=True,
                    help="Total number of shards (nodes) splitting the run list.")
    ps.add_argument("--gather-workers", type=int, default=1,
                    help="Within-node parallelism, same meaning as `fit`/`train "
                         "--gather-workers` -- combine with --num-shards for "
                         "two-level (multi-node x multi-process) parallelism.")
    ps.add_argument("--out", required=True, help="Partial-accumulator file (joblib).")
    ps.set_defaults(func=gather_shard)

    pm = sub.add_parser("gather-merge",
                        help="Multi-node: combine `gather-shard` --out files into "
                             "the final transfer.npz (method=fit) or emulator.pkl "
                             "(method=train).")
    pm.add_argument("--shards", nargs="+", required=True,
                    help="Every gather-shard --out file (one per node/shard).")
    pm.add_argument("--hidden", default="256,256,128", help="train method only.")
    pm.add_argument("--alpha", type=float, default=1e-4, help="train method only.")
    pm.add_argument("--max-iter", type=int, default=200, help="train method only.")
    pm.add_argument("--out", required=True,
                    help="Output transfer.npz (fit) or emulator .pkl (train).")
    pm.set_defaults(func=gather_merge)

    args = p.parse_args()
    args.func(args)
