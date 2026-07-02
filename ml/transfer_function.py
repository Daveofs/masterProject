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


def _run_cls(run_dir: Path, lmax: int):
    """Per-shell (Cl_low, Cl_high) for a run from its preprocessed alms."""
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(run_dir / f"low_alms_lmax{lmax}.npy", mmap_mode="r")
    high = np.load(run_dir / f"high_alms_lmax{lmax}.npy", mmap_mode="r")
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
            lo = ld / f"low_alms_lmax{lmax}.npy"
            hi = ld / f"high_alms_lmax{lmax}.npy"
            if lo.exists() and hi.exists():
                runs.append((lo, hi))
    if not runs:
        raise RuntimeError(f"No low/high alm files (lmax={lmax}) found under {data_dir}")
    mode = "INCLUDING test (sanity check)" if args.include_test \
        else f"excluding {args.test_cosmo}"
    print(f"[fit] {len(runs)} training runs ({mode})", flush=True)

    sum_low = sum_high = None
    counts = None
    for lo, hi in runs:
        low = np.load(lo, mmap_mode="r")
        high = np.load(hi, mmap_mode="r")
        n = min(low.shape[0], high.shape[0])
        if sum_low is None:
            sum_low = np.zeros((n, lmax + 1))
            sum_high = np.zeros((n, lmax + 1))
            counts = np.zeros(n)
        for i in range(min(n, sum_low.shape[0])):
            sum_low[i] += hp.alm2cl(_alm(np.asarray(low[i]), N_alm), lmax=lmax)
            sum_high[i] += hp.alm2cl(_alm(np.asarray(high[i]), N_alm), lmax=lmax)
            counts[i] += 1
        print(f"  processed {lo.parent}", flush=True)

    mean_low = sum_low / counts[:, None]
    mean_high = sum_high / counts[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.sqrt(np.where(mean_low > 0, mean_high / mean_low, 1.0))
    T = np.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T=T, lmax=lmax, mean_low=mean_low.astype(np.float32),
             mean_high=mean_high.astype(np.float32))
    print(f"[fit] saved transfer function T{T.shape} to {args.out}", flush=True)


# ---------------------------------------------------------------------------
# Emulator: learn T(ell, shell) = f(l, z, H0, O_cdm, Ob, Om, ns, s8, Cl_low)
# ---------------------------------------------------------------------------
# Instead of a single train-averaged T (fit), train a standard MLP regressor that
# predicts the per-mode transfer function from the low-res Cl and the cosmology
# vector, so it *interpolates* T to a new (held-out) cosmology. Same target as fit
# (T = sqrt(Cl_high / Cl_low)); output of `emulate` is a transfer.npz in the exact
# schema `apply` / prepare_tcorr_dataset already consume.

def _discover_runs(data_dir: Path, lmax: int, test_cosmo: str, include_test: bool):
    cosmos = sorted(d for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    runs = []
    for c in cosmos:
        if test_cosmo and c.name == test_cosmo and not include_test:
            continue
        rs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        for ld in (rs or [c]):
            if (ld / f"low_alms_lmax{lmax}.npy").exists() and \
               (ld / f"high_alms_lmax{lmax}.npy").exists():
                runs.append(ld)
    return runs


def train(args):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib

    data_dir = Path(args.data_dir)
    lmax = args.lmax
    ell = np.arange(lmax + 1)
    runs = _discover_runs(data_dir, lmax, args.test_cosmo, args.include_test)
    if not runs:
        raise RuntimeError(f"No low/high alm files (lmax={lmax}) found under {data_dir}")
    mode = "INCLUDING test (sanity)" if args.include_test else f"excluding {args.test_cosmo}"
    print(f"[train] {len(runs)} training runs ({mode})", flush=True)

    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for ld in runs:
        cosmo = load_cosmo(ld)
        cl_low, cl_high = _run_cls(ld, lmax)
        n_shells = cl_low.shape[0]
        z = shell_redshifts(ld, n_shells, args.info_npz)
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.sqrt(np.where(cl_low > 0, cl_high / cl_low, 1.0))
        T = np.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0)
        for i in range(n_shells):
            X = build_features(ell, float(z[i]), cosmo, cl_low[i])
            y = T[i]
            if args.sample_frac < 1.0:                    # subsample ell for tractability
                keep = rng.random(ell.shape[0]) < args.sample_frac
                X, y = X[keep], y[keep]
            Xs.append(X); ys.append(y)
        print(f"  gathered {ld} ({n_shells} shells)", flush=True)

    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"[train] {X.shape[0]:,} samples x {X.shape[1]} features", flush=True)

    scaler = StandardScaler().fit(X)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                         solver="adam", alpha=args.alpha, batch_size=4096,
                         learning_rate_init=1e-3, max_iter=args.max_iter,
                         early_stopping=True, n_iter_no_change=10,
                         validation_fraction=0.1, verbose=True, random_state=0)
    model.fit(scaler.transform(X), y)
    print(f"[train] final loss={model.loss_:.3e} "
          f"(val best={getattr(model, 'best_validation_score_', float('nan')):.4f})",
          flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "lmax": lmax,
                 "cosmo_keys": COSMO_KEYS, "feature_order":
                 ["log10(l+1)", "z", *COSMO_KEYS, "log10(Cl_low)"]}, args.out)
    print(f"[train] saved emulator to {args.out}", flush=True)


def emulate(args):
    """Predict T(shell, ell) for a run with the trained emulator -> transfer.npz."""
    import joblib
    bundle = joblib.load(args.emulator)
    model, scaler = bundle["model"], bundle["scaler"]
    lmax = int(bundle["lmax"])
    ell = np.arange(lmax + 1)

    run = Path(args.run_dir)
    cosmo = load_cosmo(run)
    N_alm = (lmax + 1) * (lmax + 2) // 2
    low = np.load(run / f"low_alms_lmax{lmax}.npy", mmap_mode="r")
    n_shells = low.shape[0]
    z = shell_redshifts(run, n_shells, args.info_npz)

    T = np.empty((n_shells, lmax + 1), dtype=np.float32)
    for i in range(n_shells):
        cl_low = hp.alm2cl(_alm(np.asarray(low[i]), N_alm), lmax=lmax)
        X = build_features(ell, float(z[i]), cosmo, cl_low)
        T[i] = model.predict(scaler.transform(X)).astype(np.float32)
        if i % 10 == 0:
            print(f"  emulated shell {i}/{n_shells}", flush=True)
    # T is a physical amplitude ratio; clip away nonphysical negatives.
    T = np.clip(T, 0.0, None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T=T, lmax=lmax)
    print(f"[emulate] saved emulated transfer function T{T.shape} to {args.out}\n"
          f"          -> feed to `transfer_function.py apply --transfer {args.out}`",
          flush=True)


# ---------------------------------------------------------------------------
# Apply: corrected_alm = low_alm * T(ell, shell); alm2map
# ---------------------------------------------------------------------------

def apply(args):
    tf = np.load(args.transfer)
    T = tf["T"]                          # (n_shells_train, lmax+1)
    lmax = int(tf["lmax"])
    N_alm = (lmax + 1) * (lmax + 2) // 2
    ell = ell_of_flat_index(lmax)

    run = Path(args.run_dir)
    low = np.load(run / f"low_alms_lmax{lmax}.npy")
    n_shells = low.shape[0]
    npix = hp.nside2npix(args.nside)
    corrected = np.zeros((n_shells, npix), dtype=np.float32)

    for i in range(n_shells):
        Ti = T[min(i, T.shape[0] - 1)].copy()      # per-shell transfer
        if args.ell_min > 0:                         # leave large scales untouched
            Ti[: args.ell_min] = 1.0
        tvec = Ti[ell]                               # per-mode scale (N_alm,)
        v = low[i].astype(np.float64)
        alm = (v[:N_alm] * tvec + 1j * v[N_alm:] * tvec)
        corrected[i] = hp.alm2map(alm, nside=args.nside, lmax=lmax).astype(np.float32)
        if i % 10 == 0:
            print(f"  corrected shell {i}/{n_shells}", flush=True)

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
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=fit)

    pt = sub.add_parser("train", help="Train an MLP emulator T=f(l,z,cosmo,Cl_low).")
    pt.add_argument("--data-dir", required=True)
    pt.add_argument("--lmax", type=int, default=3000)
    pt.add_argument("--test-cosmo", default="")
    pt.add_argument("--include-test", action="store_true",
                    help="SANITY CHECK: train on the test cosmology too (default LOO).")
    pt.add_argument("--info-npz", default="compressed_shells.npz",
                    help="npz holding shell_info (per-shell redshifts).")
    pt.add_argument("--hidden", default="256,256,128",
                    help="Comma-separated MLP hidden-layer sizes.")
    pt.add_argument("--alpha", type=float, default=1e-4, help="L2 regularisation.")
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
    pa.add_argument("--ell-min", type=int, default=0,
                    help="Leave ell<ell_min untouched (T=1) — correct only small scales.")
    pa.add_argument("--plot-shells", type=int, nargs="*", default=[3, 30, 50])
    pa.add_argument("--plot-dir", default="")
    pa.add_argument("--out", required=True)
    pa.set_defaults(func=apply)

    args = p.parse_args()
    args.func(args)
