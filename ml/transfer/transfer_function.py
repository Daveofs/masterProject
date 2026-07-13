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
from pathlib import Path

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
    """Per-shell (Cl_low, Cl_high) for a run from its preprocessed alms."""
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(run_dir / alm_fname("low", lmax, log_density), mmap_mode="r")
    high = np.load(run_dir / alm_fname("high", lmax, log_density), mmap_mode="r")
    n = min(low.shape[0], high.shape[0])
    cl_low = np.empty((n, lmax + 1))
    cl_high = np.empty((n, lmax + 1))
    for i in range(n):
        cl_low[i] = hp.alm2cl(_alm(np.asarray(low[i]), N_alm), lmax=lmax)
        cl_high[i] = hp.alm2cl(_alm(np.asarray(high[i]), N_alm), lmax=lmax)
    return cl_low, cl_high


# ---------------------------------------------------------------------------
# Fit: accumulate mean Cl_low / Cl_high per (shell, ell) over training runs
# ---------------------------------------------------------------------------

def fit(args):
    data_dir = Path(args.data_dir)
    lmax = args.lmax
    N_alm = (lmax + 1) * (lmax + 2) // 2

    cosmos = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    runs = []
    for c in cosmos:
        # Leave the test cosmology out (proper generalization test) unless
        # --include-test is set (SANITY CHECK: fit on it too, expect ~perfect).
        if args.test_cosmo and c.name == args.test_cosmo and not args.include_test:
            continue
        rs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        for ld in (rs or [c]):
            lo = ld / alm_fname("low", lmax, args.log_density)
            hi = ld / alm_fname("high", lmax, args.log_density)
            if lo.exists() and hi.exists():
                runs.append((lo, hi))
    if not runs:
        raise RuntimeError(f"No low/high alm files (lmax={lmax}) found under {data_dir}")
    mode = "INCLUDING test (sanity check)" if args.include_test \
        else f"excluding {args.test_cosmo}"
    print(f"[fit] {len(runs)} training runs ({mode})", flush=True)

    sum_low = sum_high = sum_cross = None
    counts = None
    for lo, hi in runs:
        low = np.load(lo, mmap_mode="r")
        high = np.load(hi, mmap_mode="r")
        n = min(low.shape[0], high.shape[0])
        if sum_low is None:
            sum_low = np.zeros((n, lmax + 1))
            sum_high = np.zeros((n, lmax + 1))
            sum_cross = np.zeros((n, lmax + 1))
            counts = np.zeros(n)
        for i in range(min(n, sum_low.shape[0])):
            al, ah = _alm(np.asarray(low[i]), N_alm), _alm(np.asarray(high[i]), N_alm)
            sum_low[i] += hp.alm2cl(al, lmax=lmax)
            sum_high[i] += hp.alm2cl(ah, lmax=lmax)
            sum_cross[i] += hp.alm2cl(al, ah, lmax=lmax)
            counts[i] += 1
        print(f"  processed {lo.parent}", flush=True)

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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T=T, R=R, lmax=lmax, log_density=args.log_density,
             mean_low=mean_low.astype(np.float32), mean_high=mean_high.astype(np.float32))
    print(f"[fit] saved transfer function T{T.shape} (mean r={R.mean():.3f}, "
          f"log_density={args.log_density}) to {args.out}", flush=True)


# ---------------------------------------------------------------------------
# Emulator: learn T(ell, shell) = f(l, z, H0, O_cdm, Ob, Om, ns, s8, Cl_low)
# ---------------------------------------------------------------------------
# Instead of a single train-averaged T (fit), train a standard MLP regressor that
# predicts the per-mode transfer function from the low-res Cl and the cosmology
# vector, so it *interpolates* T to a new (held-out) cosmology. Same target as fit
# (T = sqrt(Cl_high / Cl_low)); output of `emulate` is a transfer.npz in the exact
# schema `apply` / prepare_tcorr_dataset already consume.

def _discover_runs(data_dir: Path, lmax: int, test_cosmo: str, include_test: bool,
                   log_density: bool = False):
    cosmos = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    runs = []
    for c in cosmos:
        if test_cosmo and c.name == test_cosmo and not include_test:
            continue
        rs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        for ld in (rs or [c]):
            if (ld / alm_fname("low", lmax, log_density)).exists() and \
               (ld / alm_fname("high", lmax, log_density)).exists():
                runs.append(ld)
    return runs


def train(args):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib

    data_dir = Path(args.data_dir)
    lmax = args.lmax
    ell = np.arange(lmax + 1)
    runs = _discover_runs(data_dir, lmax, args.test_cosmo, args.include_test, args.log_density)
    if not runs:
        raise RuntimeError(f"No low/high alm files (lmax={lmax}) found under {data_dir}")
    mode = "INCLUDING test (sanity)" if args.include_test else f"excluding {args.test_cosmo}"
    print(f"[train] {len(runs)} training runs ({mode})", flush=True)

    rng = np.random.default_rng(0)
    Xs, ys = [], []
    # r(ell,shell) (phase cross-correlation, see `fit`) is averaged over training
    # cosmologies rather than emulated: it's mostly a shot-noise/resolution
    # property of the shell (not strongly cosmology-dependent), and at apply time
    # for a held-out cosmology there is no ground-truth high map to compute it
    # from directly -- this train-set average is the best available estimate.
    sum_cross = sum_low = sum_high = None
    r_counts = None
    for ld in runs:
        cosmo = load_cosmo(ld)
        cl_low, cl_high = _run_cls(ld, lmax, args.log_density)
        n_shells = cl_low.shape[0]
        z = shell_redshifts(ld, n_shells, args.info_npz)
        N_alm = (lmax + 1) * (lmax + 2) // 2
        low_alm = np.load(ld / alm_fname("low", lmax, args.log_density), mmap_mode="r")
        high_alm = np.load(ld / alm_fname("high", lmax, args.log_density), mmap_mode="r")
        if sum_cross is None:
            sum_cross = np.zeros((n_shells, lmax + 1))
            sum_low = np.zeros((n_shells, lmax + 1))
            sum_high = np.zeros((n_shells, lmax + 1))
            r_counts = np.zeros(n_shells)
        for i in range(min(n_shells, sum_cross.shape[0])):
            al = _alm(np.asarray(low_alm[i]), N_alm)
            ah = _alm(np.asarray(high_alm[i]), N_alm)
            sum_cross[i] += hp.alm2cl(al, ah, lmax=lmax)
            sum_low[i] += hp.alm2cl(al, lmax=lmax)
            sum_high[i] += hp.alm2cl(ah, lmax=lmax)
            r_counts[i] += 1
        # Smooth BEFORE computing the ratio: T's target is otherwise a per-ell,
        # per-realization noisy ratio (see smooth_cl docstring for why this is
        # the actual source of the oscillating predictions).
        cl_low_s = np.stack([smooth_cl(c, args.smooth_window) for c in cl_low])
        cl_high_s = np.stack([smooth_cl(c, args.smooth_window) for c in cl_high])
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.sqrt(np.where(cl_low_s > 0, cl_high_s / cl_low_s, 1.0))
        T = np.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0)
        for i in range(n_shells):
            X = build_features(ell, float(z[i]), cosmo, cl_low_s[i])
            y = T[i]
            if args.sample_frac < 1.0:                    # subsample ell for tractability
                keep = rng.random(ell.shape[0]) < args.sample_frac
                X, y = X[keep], y[keep]
            Xs.append(X); ys.append(y)
        print(f"  gathered {ld} ({n_shells} shells)", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_cross, mean_low_r, mean_high_r = (s / r_counts[:, None]
                                                for s in (sum_cross, sum_low, sum_high))
        R = np.where((mean_low_r > 0) & (mean_high_r > 0),
                     mean_cross / np.sqrt(mean_low_r * mean_high_r), 0.0)
    R = np.clip(np.nan_to_num(R, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    print(f"[train] mean r(ell,shell) across training set = {R.mean():.3f}", flush=True)

    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"[train] {X.shape[0]:,} samples x {X.shape[1]} features", flush=True)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    hidden = tuple(int(h) for h in args.hidden.split(","))

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

    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                         solver="adam", alpha=args.alpha, batch_size=4096,
                         learning_rate_init=1e-3, verbose=False, random_state=0)
    train_loss_hist, val_loss_hist = [], []
    best_val, best_weights, no_improve, patience = np.inf, None, 0, 10
    for it in range(args.max_iter):
        model.partial_fit(X_tr, y_tr)                  # one epoch (adam, batch_size=4096)
        train_loss = float(np.mean((model.predict(X_tr) - y_tr) ** 2)) / 2
        val_loss = float(np.mean((model.predict(X_va) - y_va) ** 2)) / 2
        train_loss_hist.append(train_loss); val_loss_hist.append(val_loss)
        if val_loss < best_val - 1e-9:
            best_val, best_weights, no_improve = val_loss, \
                ([c.copy() for c in model.coefs_], [b.copy() for b in model.intercepts_]), 0
        else:
            no_improve += 1
        if it % 10 == 0:
            print(f"  iter {it}: train_loss={train_loss:.3e} val_loss={val_loss:.3e}", flush=True)
        if no_improve >= patience:
            print(f"[train] early stopping at iter {it} "
                  f"(no val improvement for {patience} iters)", flush=True)
            break
    if best_weights is not None:                       # restore best-val weights
        model.coefs_, model.intercepts_ = best_weights
    print(f"[train] final: train_loss={train_loss_hist[-1]:.3e} "
          f"best_val_loss={best_val:.3e}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "lmax": lmax,
                 "smooth_window": args.smooth_window, "R": R,
                 "log_density": args.log_density,
                 "cosmo_keys": COSMO_KEYS, "feature_order":
                 ["log10(l+1)", "z", *COSMO_KEYS, "log10(Cl_low)"]}, args.out)
    print(f"[train] saved emulator to {args.out}", flush=True)

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
        loss_png = Path(args.out).with_suffix("").with_suffix(".loss.png")
        plot_train_val_loss(
            np.arange(len(train_loss_hist)), train_loss_hist, val_loss_hist, loss_png,
            xlabel="iteration", ylabel="MLP squared-error loss (T emulator)",
            val_label=f"validation (10% held-out samples, alpha={args.alpha:g})",
            formula=r"loss $=\frac{1}{2N}\sum_i(T_i-\hat T_i)^2$  "
                    "(L2 weight decay is an optimizer detail, not shown)")
    except Exception as e:
        print(f"[train] loss plot failed: {e}", flush=True)

    # ---- cosmo-vector sanity check: does the model actually respond to it? ----
    # Compare T predicted with the LAST training run's real cosmo vector against T
    # predicted with the training-set MEAN cosmo vector (scaler.mean_ stores means
    # in original, unscaled units). If the model ignored cosmo entirely, these
    # would be identical.
    mean_cosmo = scaler.mean_[2:2 + len(COSMO_KEYS)]
    z_ref, cl_ref = float(z[len(z) // 2]), cl_low_s[len(z) // 2]
    X_real = build_features(ell, z_ref, cosmo, cl_ref)
    X_mean = build_features(ell, z_ref, mean_cosmo, cl_ref)
    T_real = model.predict(scaler.transform(X_real))
    T_mean = model.predict(scaler.transform(X_mean))
    diff = np.abs(T_real - T_mean)
    print(f"[train] cosmo-vec sanity: |T(real cosmo) - T(mean training cosmo)| "
          f"max={diff.max():.4f} mean={diff.mean():.4f} "
          f"(near-zero would mean the model ignores cosmo)", flush=True)


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

    T = np.empty((n_shells, lmax + 1), dtype=np.float32)
    for i in range(n_shells):
        cl_low = smooth_cl(hp.alm2cl(_alm(np.asarray(low[i]), N_alm), lmax=lmax),
                          smooth_window)
        X = build_features(ell, float(z[i]), cosmo, cl_low)
        Ti = model.predict(scaler.transform(X))
        if i == n_shells // 2:
            X_mean = build_features(ell, float(z[i]), mean_cosmo, cl_low)
            T_mean = model.predict(scaler.transform(X_mean))
            d = np.abs(Ti - T_mean)
            print(f"  [cosmo check] shell {i}: |T(this cosmo) - T(mean training cosmo)| "
                  f"max={d.max():.4f} mean={d.mean():.4f} "
                  f"(near-zero would mean cosmo is being ignored)", flush=True)
        # Post-hoc smoothing of the OUTPUT is a safety net: even with a smoothed
        # target/feature, a generic MLP evaluated row-by-row over ell need not be
        # perfectly smooth in ell. Same window as used for the input features.
        T[i] = smooth_cl(np.clip(Ti, 1e-6, None), smooth_window).astype(np.float32) \
            if smooth_window > 1 else Ti.astype(np.float32)
        if i % 10 == 0:
            print(f"  emulated shell {i}/{n_shells}", flush=True)
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


# ---------------------------------------------------------------------------
# Apply: corrected_alm = low_alm * T(ell, shell); alm2map
# ---------------------------------------------------------------------------

def apply(args):
    tf = np.load(args.transfer)
    T = tf["T"]                          # (n_shells_train, lmax+1)
    # r(ell,shell): phase cross-correlation of low vs high (see `fit`/`train`).
    # NOTE: applying the Wiener gain r*T here (--wiener) is OFF by default because
    # it makes things WORSE for the stated goal (matching Cl). Where r is small
    # (faint shells, high ell), r*T << 1, so the residual scale (r*T - 1) is
    # strongly NEGATIVE -> it SUBTRACTS DISCO's existing power, driving Cl_corr
    # far BELOW even Cl_disco (measured: 0.019 vs disco 0.52 at ell>1500 on shell
    # 3) and making the map visibly lighter than DISCO. Plain full T instead
    # matches Cl_high ~0.9-1.0 across all ell (that IS the transfer function's
    # job). The grainy high-ell texture on faint shells is inherent -- those modes
    # are shot noise, so only the POWER is recoverable, not the true structure;
    # recovering structure is what the generative sphere-flow is for, not this.
    #
    # --stochastic (constrained realization) fixes the real-space overshoot seen
    # with plain full T: scaling EVERY mode (including r<1 ones) by T amplifies
    # DISCO's own single-realization noise by up to T, which still averages to
    # Cl_high over many maps but overshoots std/max for any ONE map (measured:
    # pixel-histogram std and max both exceed CosmoGrid's on shells 10/50/60, and
    # shell 3's corrected/high ratio scatters +-15-20% at high ell -- far more
    # than cosmic variance). Fix: split the correction into a phase-correlated
    # part (gain r*T, trustworthy since low's phases genuinely track high there)
    # and the REMAINING power T^2*(1-r^2)*Cl_low, filled with an INDEPENDENT
    # random realization instead of amplified DISCO noise. Still matches Cl_high
    # exactly in expectation: (r*T)^2 + T^2*(1-r^2) = T^2 = Cl_high/Cl_low.
    R = tf["R"] if "R" in tf.files else None
    if args.stochastic and R is None:
        raise RuntimeError("--stochastic needs R in the transfer file "
                            "(re-run `fit`/`train`, which always saves it).")
    lmax = int(tf["lmax"])
    # --log-density: T/R were fit on log1p(rho) alms (see alm_fname), so this run's
    # low alms and native map must be treated the same way -- auto-detected from
    # the transfer file so `apply` can't silently mismatch what `fit`/`emulate` used.
    log_density = bool(tf["log_density"]) if "log_density" in tf.files else False
    N_alm = (lmax + 1) * (lmax + 2) // 2
    ell = ell_of_flat_index(lmax)

    run = Path(args.run_dir)
    low = np.load(run / alm_fname("low", lmax, log_density))
    # Native full-resolution DISCO map (nside=2048 supports ell up to ~3*nside-1,
    # i.e. ~6143 -- roughly double lmax=3000). Reconstructing the corrected map
    # purely via alm2map(..., lmax=lmax) band-limits it to ell<=lmax and throws
    # away all genuine small-scale structure the native map already had above
    # that -- which made the "corrected" map look visibly WORSE (blurrier, peaks
    # flattened) than either input even though Cl(ell<=lmax) improved. Fix: add
    # the correction as a RESIDUAL on top of the native map, touching only the
    # ell<=lmax band (scaled by T-1) and leaving everything above lmax untouched.
    low_full = np.load(run / f"low_shells_nside={args.nside}.npy", mmap_mode="r")
    n_shells = low.shape[0]
    npix = hp.nside2npix(args.nside)
    corrected = np.zeros((n_shells, npix), dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    # PER-SHELL ell_min from a FIXED comoving (Mpc/h) scale -- see ell_min_from_mpc_h.
    # Confirmed against real Disco/CosmoGrid Cl-ratio diagnostics (2026-07-13): the
    # simulation-resolution deficit sets in at the SAME fixed comoving scale (~ the
    # L_box/N_pm PM grid cell) in every shell, but that fixed length maps to a
    # DIFFERENT ell depending on the shell's own comoving distance -- shell 3
    # (z~0.05) visibly deviates starting around ell~150-300; shell 65 (z~2.85)
    # shows NO deviation at all out to ell=3000, because the same 3 Mpc/h scale
    # only becomes resolvable well beyond lmax that far away. A SINGLE global ell
    # cannot reproduce this (tried and reverted: median-z reference gave ell_min
    # ~2939, correcting almost nothing anywhere; nearest-shell reference gave
    # ell_min~36, correcting almost everything everywhere) -- per-shell conversion
    # is the physically correct one.
    if args.ell_min_mpc > 0:
        z_shells = shell_redshifts(run, n_shells, args.info_npz)
        cosmo_vec = load_cosmo(run)
        ell_min_per_shell = ell_min_from_mpc_h(z_shells, cosmo_vec, args.ell_min_mpc)
        n_uncorrected = int(np.sum(ell_min_per_shell >= lmax))
        print(f"[apply] --ell-min-mpc {args.ell_min_mpc:g} -> per-shell ell_min "
              f"range [{ell_min_per_shell.min()}, {ell_min_per_shell.max()}] across "
              f"{n_shells} shells (z={z_shells.min():.3f}-{z_shells.max():.3f})"
              + (f"  [{n_uncorrected} distant shells get NO correction: even ell=lmax="
                 f"{lmax} resolves scales > {args.ell_min_mpc:g} Mpc/h there -- "
                 f"consistent with those shells showing no measurable Disco/CosmoGrid "
                 f"deviation within lmax]" if n_uncorrected else ""), flush=True)
    else:
        ell_min_per_shell = np.full(n_shells, args.ell_min, dtype=np.int64)

    if args.poisson:
        # Poisson-resample every shell in ONE call, dispatched across a process pool
        # (poisson_resample.resample_all_shells_parallel) -- NOT a sequential
        # per-shell Python loop. Measured: a sequential loop gets the SAME ~2h
        # wall-clock whether the node has 64 or 288 cpus allocated, because a single
        # shell's own OMP-threaded SHT calls saturate well before that many cores;
        # real speedup needs independent shells running CONCURRENTLY. Ignores
        # --no-clip/--no-debias-mean/--wiener/--stochastic (Poisson replaces all of
        # them -- see poisson_resample.py's docstring for why clipping/debiasing
        # can't give Cl + positivity + the one-point pdf at once).
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        import poisson_resample
        corrected = poisson_resample.resample_all_shells_parallel(
            run, alm_fname("low", lmax, log_density), lmax, args.nside, T, R,
            ell_min_per_shell, n_avg=args.poisson_n_avg, n_iter=args.poisson_n_iter,
            damp=args.poisson_damp, seed=args.seed, n_workers=args.poisson_workers)
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        info_npz = run / args.info_npz
        extra = {}
        if info_npz.exists():
            d = np.load(info_npz, allow_pickle=True)
            extra = {k: d[k] for k in d.files if k != "shells"}
        np.savez(out, shells=corrected, **extra)
        print(f"[apply] saved corrected+Poisson shells to {out}", flush=True)
        if args.plot_shells:
            _plot_cl(run, corrected, lmax, args.plot_shells, Path(args.plot_dir or out.parent))
        return

    for i in range(n_shells):
        Ti = T[min(i, T.shape[0] - 1)].copy()      # per-shell transfer
        Ri = R[min(i, R.shape[0] - 1)].copy() if R is not None else np.ones_like(Ti)
        if args.wiener and not args.stochastic:
            Ti = Ti * Ri                             # Wiener/MMSE gain r*T
        ell_min_i = int(ell_min_per_shell[i])
        if ell_min_i > 0:                            # leave large scales untouched
            Ti[:ell_min_i] = 1.0
            Ri[:ell_min_i] = 1.0
        v = low[i].astype(np.float64)
        if args.stochastic:
            gain = Ti * Ri                            # phase-correlated part only
            tvec = (gain - 1.0)[ell]
            alm_signal = v[:N_alm] * tvec + 1j * v[N_alm:] * tvec
            signal_map = hp.alm2map(alm_signal, nside=args.nside, lmax=lmax)
            # Missing power T^2*(1-r^2) at ell where r<1: DISCO's own alm carries
            # no usable phase info there, so fill with a FRESH random realization
            # (uncorrelated with low_alm) instead of amplifying DISCO's noise.
            cl_low_i = smooth_cl(hp.alm2cl(_alm(v, N_alm), lmax=lmax), args.smooth_window)
            cl_noise = np.clip(Ti ** 2 - gain ** 2, 0.0, None) * cl_low_i
            # healpy.synalm draws from numpy's global RNG (no seed kwarg) -> seed
            # it explicitly per shell so --seed makes the whole run reproducible.
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            alm_noise = hp.synalm(cl_noise, lmax=lmax, new=True)
            noise_map = hp.alm2map(alm_noise, nside=args.nside, lmax=lmax)
            delta_map = signal_map + noise_map
        else:
            tvec = (Ti - 1.0)[ell]                    # per-mode DELTA scale (N_alm,)
            alm_delta = (v[:N_alm] * tvec + 1j * v[N_alm:] * tvec)
            delta_map = hp.alm2map(alm_delta, nside=args.nside, lmax=lmax)
        rho_native = np.asarray(low_full[i], dtype=np.float64)
        if log_density:
            # Correction was fit/applied in log1p(rho) space, so reconstruct via
            # expm1: ALWAYS >= -1 (i.e. rho >= 0 up to fp noise) for ANY delta_map,
            # no matter how large the T boost -- unlike adding delta_map straight
            # to rho, which routinely drives ~half a faint shell's pixels below 0
            # and, once floored at 0, biases the WHOLE shell's Cl high (measured on
            # shell 3: pre-clip corrected/high ~1.00 at ell 1-300, post-clip ~1.2 --
            # the floor itself was the source of the "small-scale overshoot").
            s_native = np.log1p(rho_native)
            s_corrected = s_native + delta_map
            m_unclipped = np.expm1(s_corrected)
        else:
            # Density is a COUNT: it cannot be negative. Adding a linear harmonic
            # residual (Gaussian-like, can undershoot) to a faint shell -- most of
            # whose pixels are already empty (density 0) -- drives ~half the pixels
            # below 0, i.e. delta < -1, which is unphysical and makes log10(1.01+delta)
            # NaN when plotting. Floor at 0. NOTE: on faint/shot-noise shells this floor
            # fires for a large fraction of pixels and BIASES Cl across all ell (see
            # --log-density above, which avoids this by construction). --stochastic
            # reduces (but doesn't eliminate) how often this fires by not riding the
            # injected power on DISCO's own noise.
            m_unclipped = rho_native + delta_map

        if args.poisson:
            # Ri already has ell<ell_min forced to 1.0 above when --ell-min(-mpc) is
            # set, so resample_shell's win=R trusts DISCO's phases fully there (no
            # random-noise replacement) -- but note the AMPLITUDE at those ell still
            # gets recalibrated to shot-deconvolved Cl_high (resample_shell always
            # retargets Cl_high everywhere), not held byte-for-byte equal to DISCO's
            # own realization. In practice this is a small effect where T~1 was
            # already true (the whole premise of a small-scale-only correction).
            cl_h_i = hp.alm2cl(_alm(np.asarray(high_alms[i]), N_alm), lmax=lmax)
            corrected[i] = poisson_resample.resample_shell(
                rho_native, cl_h_i, m_unclipped, Ri, lmax, args.nside,
                pois_mu, pois_w, rng, n_avg=args.poisson_n_avg,
                n_iter=args.poisson_n_iter, damp=args.poisson_damp,
                verbose=(i % 10 == 0))
            if i % 10 == 0:
                print(f"  corrected+Poisson shell {i}/{n_shells}", flush=True)
            continue

        neg = np.mean(m_unclipped < 0.0)
        # Positivity vs Cl: measured on cosmo_000122 shell 3 (nbar=0.104, 59% of
        # pixels want to go negative), corrected/high Cl by ell band and mean ratio:
        #   no-clip   : mean 0.998 | 1.00 (l50-150) 0.99 (l400-800) 0.93 (l800-1500)
        #   clip@0    : mean 1.215 | 1.17            0.88            0.74
        #   debias    : mean 0.998 | 1.04            0.77            0.61
        # i.e. ANY pixel-wise positivity enforcement destroys small-scale power. A
        # Gaussian field with nbar~0.1 and enough small-scale power to match
        # CosmoGrid MUST go negative; CosmoGrid's own shell is not Gaussian, it's a
        # sparse COUNT field (91.8% zeros). Matching Cl AND positivity AND the
        # one-point pdf requires re-discretizing (lognormal intensity + Poisson
        # resample), not clipping. --no-clip therefore yields the Cl-optimal
        # OVERDENSITY field (rho can be < 0; not a count map) and is the right
        # output for Cl / weak-lensing work.
        if args.no_clip:
            m = m_unclipped
        elif args.no_debias_mean:
            m = np.clip(m_unclipped, 0.0, None)
        else:
            m = _debias_mean(m_unclipped)
        corrected[i] = m.astype(np.float32)
        if i % 10 == 0 or neg > 0.02:
            note = ""
            if neg > 0.02:
                note = (f"  [{neg:.1%} pixels < 0 -> shot-noise shell; "
                        + ("KEPT (--no-clip, Cl-optimal)]" if args.no_clip
                           else "floored at 0, suppresses small-scale Cl]"))
            print(f"  corrected shell {i}/{n_shells}{note}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    info_npz = run / args.info_npz              # metadata (shell_info) from high npz
    extra = {}
    if info_npz.exists():
        d = np.load(info_npz, allow_pickle=True)
        extra = {k: d[k] for k in d.files if k != "shells"}
    np.savez(out, shells=corrected, **extra)
    print(f"[apply] saved corrected shells to {out}", flush=True)

    if args.plot_shells:
        _plot_cl(run, corrected, lmax, args.plot_shells, Path(args.plot_dir or out.parent))


def _plot_cl(run, corrected, lmax, shells, out_dir):
    """Cl-ratio plots: low/high and corrected/high (low,high from preprocessed alms)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(run / f"low_alms_lmax{lmax}.npy", mmap_mode="r")
    high = np.load(run / f"high_alms_lmax{lmax}.npy", mmap_mode="r")
    out_dir.mkdir(parents=True, exist_ok=True)
    ells = np.arange(lmax + 1)

    def cl_alm(v):
        a = (v[:N_alm] + 1j * v[N_alm:]).astype(np.complex128)
        return hp.alm2cl(a, lmax=lmax)

    for s in shells:
        if s >= corrected.shape[0]:
            continue
        # All three as DENSITY Cl for consistency: low/high come from the density
        # alms (map2alm of the density map), so the corrected map's Cl must also be
        # density (NOT overdensity, which differs by mean^2 and caused a ~mean^2
        # offset in the ratio plot).
        cl_l = cl_alm(np.asarray(low[s])); cl_h = cl_alm(np.asarray(high[s]))
        cl_c = hp.anafast(corrected[s].astype(np.float64), lmax=lmax)
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax[0].loglog(ells, cl_l, label="DISCO (low)", color="seagreen")
        ax[0].loglog(ells, cl_c, label="corrected", color="steelblue")
        ax[0].loglog(ells, cl_h, "--", label="CosmoGrid (high)", color="tomato")
        ax[0].set_ylabel(r"$C_\ell$"); ax[0].legend(); ax[0].set_title(f"shell {s}")
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[1].semilogx(ells, cl_l / cl_h, ":", color="seagreen", label="low/high")
            ax[1].semilogx(ells, cl_c / cl_h, color="steelblue", label="corrected/high")
        ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0.5, 1.5)
        ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel("ratio"); ax[1].legend()
        fig.tight_layout(); fig.savefig(out_dir / f"cl_shell{s:03d}.png", dpi=150); plt.close(fig)
        print(f"  [plot] {out_dir / f'cl_shell{s:03d}.png'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit")
    pf.add_argument("--data-dir", required=True)
    pf.add_argument("--lmax", type=int, default=3000)
    pf.add_argument("--test-cosmo", default="")
    pf.add_argument("--include-test", action="store_true",
                    help="SANITY CHECK: include the test cosmology in the fit (expect "
                         "corrected/high ~ 1 by construction). Default is leave-one-out.")
    pf.add_argument("--log-density", action="store_true",
                    help="Fit T/R on log1p(rho) alms (preprocess_alms.py --log-density) "
                         "instead of raw density alms. `apply` reconstructs via expm1, "
                         "which is always >= -1 (i.e. rho >= 0) for any correction size, "
                         "eliminating the clip-at-0 bias that inflates Cl on shot-noise "
                         "shells with plain density-space T (measured: post-clip "
                         "corrected/high ~1.2 vs ~1.0 pre-clip on a shell where 59% of "
                         "pixels get clipped). Saved into the output transfer.npz so "
                         "`apply` auto-detects it -- no matching flag needed there.")
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=fit)

    pt = sub.add_parser("train", help="Train an MLP emulator T=f(l,z,cosmo,Cl_low).")
    pt.add_argument("--data-dir", required=True)
    pt.add_argument("--lmax", type=int, default=3000)
    pt.add_argument("--test-cosmo", default="")
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
    pt.add_argument("--out", required=True, help="Output emulator .pkl (joblib).")
    pt.set_defaults(func=train)

    pe = sub.add_parser("emulate", help="Predict T for a run -> transfer.npz (apply-ready).")
    pe.add_argument("--emulator", required=True, help="emulator .pkl from `train`.")
    pe.add_argument("--run-dir", required=True,
                    help="Run dir with low_alms_lmax{lmax}.npy + params.yml.")
    pe.add_argument("--info-npz", default="compressed_shells.npz")
    pe.add_argument("--out", required=True, help="transfer.npz (same schema as fit).")
    pe.set_defaults(func=emulate)

    pa = sub.add_parser("apply")
    pa.add_argument("--transfer", required=True)
    pa.add_argument("--run-dir", required=True,
                    help="Test run dir with low_alms_lmax{lmax}.npy (+ high_alms for plots).")
    pa.add_argument("--info-npz", default="compressed_shells.npz",
                    help="npz to copy non-shell metadata (shell_info) from.")
    pa.add_argument("--nside", type=int, default=2048)
    pa.add_argument("--wiener", action="store_true",
                     help="Apply the Wiener gain r*T instead of full T. OFF by "
                          "default: it SUBTRACTS power where r is small (faint "
                          "shells / high ell), pushing Cl below even DISCO and the "
                          "map lighter than DISCO. Full T (default) matches Cl_high. "
                          "Only meaningful if you deliberately want to suppress the "
                          "shot-noise-dominated high-ell modes at the cost of Cl.")
    pa.add_argument("--stochastic", action="store_true",
                     help="Constrained-realization gain instead of plain full T: "
                          "scale the phase-correlated part by r*T and fill the "
                          "remaining power T^2*(1-r^2) with an independent random "
                          "realization instead of amplifying DISCO's own noise. "
                          "Fixes the real-space overshoot (excess std/max vs "
                          "CosmoGrid in pixel histograms, noisy corrected/high Cl "
                          "ratio) seen with plain full T, while still matching "
                          "Cl_high exactly in expectation. Needs R (always saved "
                          "by `fit`/`train`). Recommended default going forward.")
    pa.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the --stochastic noise realization "
                         "(per-shell seeds derived from it; reproducible).")
    pa.add_argument("--smooth-window", type=int, default=21,
                    help="Boxcar window (log10-Cl space) for smoothing this run's "
                         "own Cl_low before using it as the --stochastic noise-"
                         "power target (avoids injecting per-mode sample-variance "
                         "noise on top of the random realization). 1 disables.")
    pa.add_argument("--ell-min", type=int, default=0,
                    help="Leave ell<ell_min untouched (T=1) — correct only small scales. "
                         "Overridden by --ell-min-mpc when that is > 0.")
    pa.add_argument("--ell-min-mpc", type=float, default=0.0,
                    help="Leave comoving scales LARGER than this (Mpc/h) untouched -- "
                         "i.e. only correct scales smaller than this physical size. "
                         "Converted to a PER-SHELL ell_min via each shell's own "
                         "redshift + the test cosmology's params.yml (see "
                         "ell_min_from_mpc_h), since a fixed ell corresponds to a "
                         "different physical scale at every shell. 0 disables "
                         "(falls back to the scalar --ell-min).")
    pa.add_argument("--no-clip", action="store_true",
                    help="Do NOT enforce rho>=0: emit the raw corrected field "
                         "rho_native + delta (an OVERDENSITY field, can go negative "
                         "on faint shot-noise shells). This is the Cl-OPTIMAL output "
                         "-- measured on shell 3: mean ratio 0.998 and corrected/high "
                         "= 1.00/0.99/0.93 at ell 50-150/400-800/800-1500, vs "
                         "1.215 and 1.17/0.88/0.74 with the default clip. Use for "
                         "Cl / weak-lensing work. It is NOT a valid count map; for "
                         "that you need lognormal-intensity + Poisson resampling "
                         "(clipping cannot give Cl + positivity + the right pdf).")
    pa.add_argument("--no-debias-mean", action="store_true",
                    help="Skip the post-clip mean debiasing (see _debias_mean). ON by "
                         "default: flooring negative pixels at 0 always raises the "
                         "mean (measured up to +25% on the faintest shells even with "
                         "--log-density). The debias applies an additive shift-then-"
                         "reclip that restores the mean to what the UNCLIPPED "
                         "reconstruction gave (which tracks the true mean much more "
                         "closely) while touching Cl far less than a multiplicative "
                         "rescale would (tested: rescale distorts Cl at ALL ell by "
                         "~scale^2). Pass this flag to get the old raw-clip behavior.")
    pa.add_argument("--plot-shells", type=int, nargs="*", default=[3, 30, 50])
    pa.add_argument("--plot-dir", default="")
    pa.add_argument("--poisson", action="store_true",
                    help="Poisson-resample every shell into valid non-negative integer "
                         "counts RIGHT AFTER the transfer-function correction (see "
                         "poisson_resample.py resample_all_shells_parallel) -- no "
                         "separate stage, no large intermediate --no-clip npz written "
                         "to disk. Ignores --no-clip/--no-debias-mean/--wiener/"
                         "--stochastic (Poisson replaces all of them). Shells are "
                         "processed in PARALLEL across a process pool (see "
                         "--poisson-workers), not sequentially.")
    pa.add_argument("--poisson-n-avg", type=int, default=4,
                    help="See poisson_resample.py --n-avg.")
    pa.add_argument("--poisson-n-iter", type=int, default=5,
                    help="See poisson_resample.py --n-iter.")
    pa.add_argument("--poisson-damp", type=float, default=0.4,
                    help="See poisson_resample.py --damp.")
    pa.add_argument("--poisson-workers", type=int, default=1,
                    help="Worker PROCESSES for --poisson, one independent shell per "
                         "task. Default 1 (plain sequential, trusted): measured 23 "
                         "workers x 12 OMP threads each to be dramatically SLOWER "
                         "than sequential (memory-bandwidth contention from many "
                         "concurrent large-array SHTs at nside=2048), and OMP "
                         "threading alone saturates around 128 threads (128->256 "
                         "measured zero further speedup on one shell) -- so more "
                         "cpus does not automatically mean faster here. Shells with "
                         "ell_min>=lmax (T==1 everywhere, e.g. distant shells under "
                         "--ell-min-mpc) are ALWAYS skipped for free regardless of "
                         "this setting (exact no-op, not resampled). Only raise "
                         "this after validating a specific worker count.")
    pa.add_argument("--out", required=True)
    pa.set_defaults(func=apply)

    args = p.parse_args()
    args.func(args)
