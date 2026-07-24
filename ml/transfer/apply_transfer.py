#!/usr/bin/env python3
"""Apply the fitted/emulated transfer function and run every shared analysis/
diagnostic -- all in ONE script/process. Moved out of
transfer_function.py (which keeps fit/train/emulate: building T) and out of the
formerly-separate plot_example_patches.py / infer_full_sky_transfer.py (now embedded
below as plot_patches/plot_full_sky). Analogous to unet/apply_flow.py,
which embeds prediction + evaluation + plotting in one file the same way.

"Connects cleanly": apply() returns the corrected array directly in memory, so
plot_patches/plot_full_sky use it straight away -- no writing the (13.9GB at
nside=2048) result to disk and reading it back just to plot it.

Validates on MULTIPLE held-out cosmologies at once (--run-dirs takes one or more),
mirroring unet/apply_flow.py's split_by_cosmo-based held-out set rather
than a single fixed test cosmology -- pctile-band power ratios and full-sky moments/
histograms pool patches/pixels across ALL of them; the visual example grid
(example_patches.png) draws each row from a random held-out cosmology, labeled with
the full set.

The Cl diagnostic is cl_ratio_by_zbin_grid.png: one row per held-out cosmology (up to
--max-cosmologies) x one column per redshift bin, with a percentile band -- the same
statistic + shared plotting code (analysis.plot_cl_ratio_pctile_grid +
zbin_shell_samples) as apply_flow.py's cl_ratio_by_zbin_grid.png. Our OWN
example_full_sky.png (gnomonic-zoom triptych + per-shell Cl) was removed by request:
it only ever showed one cosmology at one fixed sky position, and its Cl-ratio panel is
strictly subsumed by cl_ratio_by_zbin_grid.png.

Weak lensing (--kappa) reduces the whole lightcone to ONE kappa map per cosmology and
emits two views of its Cl -- faceted per-cosmology (kappa_cl_per_cosmology.png) and
median + 16-84 band ACROSS cosmologies (kappa_cl_pctile_band.png) -- plus
kappa_moments_scatter.png. NB --kappa-nside must be
large enough to resolve the ell range the transfer function actually acts on; see its
help text (the old nside=128 default was blind to the entire correction).

  python apply_transfer.py --transfer <transfer.npz> \
      --run-dirs <grid>/cosmo_A/run_0 <grid>/cosmo_B/run_0 <grid>/cosmo_C/run_0 \
      --nside 2048 --ell-min-mpc 3.0 --no-clip --out-counts-dir <counts_dir> \
      --out-dir <eval_dir> --patch-shells 5 10 15 30 50 --fullsky-shells 5 10 15 30 50

  # `emulate` method: one transfer.npz PER held-out cosmology, same order as --run-dirs:
  python apply_transfer.py --transfer t_A.npz t_B.npz t_C.npz \
      --run-dirs <grid>/cosmo_A/run_0 <grid>/cosmo_B/run_0 <grid>/cosmo_C/run_0 \
      --nside 2048 --out-dir <eval_dir>
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer_function import (_alm, ell_of_flat_index, alm_fname, smooth_cl,      # noqa: E402
                               ell_min_from_mpc_h, load_cosmo, shell_redshifts,     # noqa: E402
                               _debias_mean, highpass_ell_ramp)                     # noqa: E402
# poisson_resample.py (lognormal+Poisson re-discretization into valid counts) is no
# longer wired in here (2026-07-16): validated that for kappa specifically, the
# --no-clip continuous field is BOTH cheaper AND more accurate than the Poisson path
# (kappa Cl ratio to truth: no-clip ~0.94-1.01 vs poisson ~0.81-1.08 across 5 log-ell
# bands, on 2 held-out cosmologies, one of which also exposed a real Poisson tail-
# calibration bug). clip-at-0 was tested as a positivity-only alternative and is
# WORSE, not a middle ground (injects +14-23% spurious large-scale power from
# filling in spatially-correlated void regions; a mean-rescale "fix" overcorrects in
# the opposite direction to -20%, since the bias isn't a monopole issue). The module
# itself is left on disk (not deleted -- no git history here to recover it from) in
# case a FUTURE consumer needs genuine per-pixel count validity, which --no-clip
# does not provide (its negative-pixel fraction reaches ~49% on the faintest shells).

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.transforms import log1p_delta_pair                                   # noqa: E402
from analysis.radial_power import radial_power                                     # noqa: E402
from analysis.full_sky import od_cl, zbin_shell_samples                            # noqa: E402
from analysis.moments import moments                                               # noqa: E402
from analysis.plotting import (plot_example_patch_grid, plot_pctile_band_ratio,    # noqa: E402
                               plot_cl_shell,                                       # noqa: E402
                               plot_moments_vs_shell, plot_histogram_grid,          # noqa: E402
                               plot_cl_ratio_pctile_grid,                           # noqa: E402
                               plot_kappa_cl_grid, plot_kappa_moments_scatter)       # noqa: E402
from analysis import weak_lensing                                                  # noqa: E402


# ---------------------------------------------------------------------------
# apply(): the transfer-function correction. Originally moved from
# transfer_function.py's old apply(); the legacy _plot_cl/--plot-shells hook was
# dropped (superseded by plot_full_sky's real Cl-ratio panel below), and the
# --poisson path (lognormal+Poisson re-discretization) was removed entirely --
# see the import-section comment above for why.
# ---------------------------------------------------------------------------

def apply(args, run: Path, transfer_path, out_path=None):
    tf = np.load(transfer_path)
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

    # HIGH-PASS transition band (2026-07-21, ported from unet/diffusion/sphereflow's
    # highpass_residual formulation for consistency -- see highpass_ell_ramp's
    # docstring). --hp-transition in ell = args.hp_transition * lmax, the same
    # "fraction of Nyquist" convention those pipelines use for their
    # --hp-transition (lmax plays the role of Nyquist here: it IS the max
    # resolvable ell, exactly like patch/graph Nyquist there). --hp-transition 0
    # recovers the ORIGINAL hard ell_min_i step exactly.
    # NOTE (2026-07-21): an ell_min-relative version of this (transition width =
    # hp_transition * ell_min_i instead of hp_transition * lmax) was tried and
    # REVERTED -- it measurably improved the flat-patch power-ratio "washed out"
    # look but made kappa_cl/cl_ratio_by_zbin_grid WORSE (user-observed, real job
    # comparison), so it is not a strict improvement. Root cause of "washed out"
    # is still open -- possibly SHT/pixelization resolution (nside/lmax), not
    # this transition band at all; do not re-attempt the ell_min-relative
    # version without re-validating BOTH the patch view and kappa/cl_ratio.
    ell_arr = np.arange(lmax + 1, dtype=np.float64)
    hp_transition_ell = args.hp_transition * lmax

    for i in range(n_shells):
        Ti = T[min(i, T.shape[0] - 1)].copy()      # per-shell transfer
        Ri = R[min(i, R.shape[0] - 1)].copy() if R is not None else np.ones_like(Ti)
        if args.wiener and not args.stochastic:
            Ti = Ti * Ri                             # Wiener/MMSE gain r*T
        ell_min_i = int(ell_min_per_shell[i])
        if ell_min_i > 0:                            # leave large scales untouched
            # Smooth raised-cosine hand-over (not a hard truncation, see above):
            # w=0 (no correction, Ti/Ri->1) below ell_min_i, w=1 (full Ti/Ri)
            # above ell_min_i+hp_transition_ell.
            w = highpass_ell_ramp(ell_arr, ell_min_i, hp_transition_ell)
            Ti = 1.0 + w * (Ti - 1.0)
            Ri = 1.0 + w * (Ri - 1.0)
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
            # expm1: ALWAYS >= -1 (i.e. rho >= 0 up to fp noise) for ANY delta_map.
            s_native = np.log1p(rho_native)
            s_corrected = s_native + delta_map
            m_unclipped = np.expm1(s_corrected)
        else:
            # Density is a COUNT: it cannot be negative. Floor at 0 (default) --
            # measured to bias the mean/Cl on shot-noise shells (see _debias_mean),
            # which is why --no-clip (the Cl-optimal, KEPT-negative field) is the
            # pipeline's actual default for anything that doesn't need positivity
            # (e.g. kappa -- see module docstring for why Poisson resampling, the
            # only option that gave both positivity and correct Cl, was dropped).
            m_unclipped = rho_native + delta_map

        neg = np.mean(m_unclipped < 0.0)
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

    if out_path:
        out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
        info_npz = run / args.info_npz              # metadata (shell_info) from high npz
        extra = {}
        if info_npz.exists():
            d = np.load(info_npz, allow_pickle=True)
            extra = {k: d[k] for k in d.files if k != "shells"}
        np.savez(out, shells=corrected, **extra)
        print(f"[apply] saved corrected shells to {out}", flush=True)

    return corrected


# ---------------------------------------------------------------------------
# plot_patches: moved from plot_example_patches.py. Flat-patch triptych + 2D-FFT
# power ratio (bounded by that patch's own Nyquist ell) + pctile-band aggregate.
# ---------------------------------------------------------------------------

def extract_patch(shell_map: np.ndarray, nside: int, center_ipix: int, psi: float,
                  patch_size: int, reso_arcmin: float) -> np.ndarray:
    """Gnomonic-project one patch, matching make_patch_dataset.py exactly."""
    lon, lat = hp.pix2ang(nside, int(center_ipix), nest=False, lonlat=True)
    proj = hp.projector.GnomonicProj(rot=(lon, lat, psi), xsize=patch_size,
                                     ysize=patch_size, reso=reso_arcmin)
    vec2pix = lambda x, y, z, _ns=nside: hp.vec2pix(_ns, x, y, z, nest=False)
    return proj.projmap(shell_map, vec2pix)


def plot_patches(args, run_dirs: list[Path], corrected_by_run: dict):
    """Visual rows: ONE example patch per shell, drawn from a RANDOMLY-chosen
    held-out cosmology per row (pooled across all run_dirs, same as jbucko's
    apply_flow.py picking example_pick from its pooled val_idx rather than pinning
    to one cosmology). The pctile-band power-ratio pools random patches across ALL
    run_dirs too (all held-out cosmologies), same pooling jbucko does over its
    held-out patch set."""
    nside = args.nside
    reso_arcmin = hp.nside2resol(nside, arcmin=True)
    npix = hp.nside2npix(nside)
    method_label = "transfer (no-clip)" if args.no_clip else "transfer (clipped)"

    # low_full/true differ per cosmology -- load once per run, reused by both the
    # visual grid and the pooled pctile-band sampling below.
    arrays = {run: (np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r"),
                    np.load(run / args.info_npz, mmap_mode="r")["shells"])
              for run in run_dirs}

    rng = np.random.default_rng(args.seed)
    shell_rows = [s for s in args.patch_shells for _ in range(args.n_per_shell)]

    rows = []
    for s in shell_rows:
        run = run_dirs[int(rng.integers(0, len(run_dirs)))]
        low_full, true = arrays[run]
        corrected = corrected_by_run[run]
        center_ipix = int(rng.integers(0, npix))
        psi = float(rng.uniform(0, 360))
        low_p = extract_patch(np.asarray(low_full[s], dtype=np.float64), nside,
                              center_ipix, psi, args.patch_size, reso_arcmin)
        corr_p = extract_patch(np.asarray(corrected[s], dtype=np.float64), nside,
                               center_ipix, psi, args.patch_size, reso_arcmin)
        high_p = extract_patch(np.asarray(true[s], dtype=np.float64), nside,
                               center_ipix, psi, args.patch_size, reso_arcmin)
        low_log, high_log = log1p_delta_pair(low_p, high_p)
        corr_log, _ = log1p_delta_pair(corr_p, high_p)
        rows.append((f"shell {s} ({run.parent.name})", low_log, corr_log, high_log))

    # validated ON: cosmologies the transfer function / emulator NEVER saw during
    # fit/train (held out via --test-cosmos / --val-frac there) -- state the FULL
    # held-out set explicitly, since each row above may come from a different one
    # of them (labeled per-row) so this plot is legible on its own, without
    # cross-referencing which runs produced --run-dirs.
    all_cosmos = [f"{r.parent.name}/{r.name}" for r in run_dirs]
    out_dir = Path(args.out_dir)
    plot_example_patch_grid(
        rows, out_dir / "example_patches.png", corrected_label=f"corrected ({method_label})",
        suptitle=f"{method_label}: validated on {len(run_dirs)} held-out "
                 f"cosmologies: {all_cosmos} "
                 "(held out of fit/train) -- example patches (log1p overdensity) "
                 "+ per-patch power ratio\n(same layout/transform as "
                 "unet/apply_flow.py's example_patches.png)")

    # --- pctile-band power-ratio plot: many random patches, POOLED ACROSS ALL
    # held-out cosmologies (not just --n-per-shell visual examples above, and not
    # just run_dirs[0]) so a systematic bias is distinguishable from both per-patch
    # noise AND one-cosmology luck -- same statistic and shared plotting code as
    # unet/apply_flow.py's patch_power_ratio_pctile_band.png, which pools
    # its own multi-cosmology held-out patch set the same way. ---
    if args.n_pctile_patches > 0:
        print(f"[plot_patches] sampling {args.n_pctile_patches} random patches "
              f"across shells {args.patch_shells} and {len(run_dirs)} held-out "
              f"cosmologies for the pctile-band power-ratio plot", flush=True)
        lo_stack, co_stack = [], []
        for _ in range(args.n_pctile_patches):
            run = run_dirs[int(rng.integers(0, len(run_dirs)))]
            low_full, true = arrays[run]
            corrected = corrected_by_run[run]
            s = int(rng.choice(args.patch_shells))
            center_ipix = int(rng.integers(0, npix))
            psi = float(rng.uniform(0, 360))
            low_p = extract_patch(np.asarray(low_full[s], dtype=np.float64), nside,
                                  center_ipix, psi, args.patch_size, reso_arcmin)
            corr_p = extract_patch(np.asarray(corrected[s], dtype=np.float64), nside,
                                   center_ipix, psi, args.patch_size, reso_arcmin)
            high_p = extract_patch(np.asarray(true[s], dtype=np.float64), nside,
                                   center_ipix, psi, args.patch_size, reso_arcmin)
            low_log, high_log = log1p_delta_pair(low_p, high_p)
            corr_log, _ = log1p_delta_pair(corr_p, high_p)
            pr_low, pr_corr, pr_high = radial_power(low_log), radial_power(corr_log), radial_power(high_log)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo_stack.append(pr_low / pr_high); co_stack.append(pr_corr / pr_high)

        k = np.arange(len(lo_stack[0]))
        plot_pctile_band_ratio(
            k, {"low / high (baseline, no model)": np.array(lo_stack),
                f"corrected ({method_label}) / high": np.array(co_stack)},
            out_dir / "patch_power_ratio_pctile_band.png", xlabel="radial wavenumber bin",
            ylim=(0.4, 1.6),
            title=f"power ratio: {method_label} vs baseline, pooled over "
                  f"{len(run_dirs)} held-out cosmologies "
                  f"({args.n_pctile_patches} patches across shells {args.patch_shells}, "
                  "16-84th pctile band)")


# ---------------------------------------------------------------------------
# plot_full_sky: moved from infer_full_sky_transfer.py. Real angular Cl (not the
# flat-patch approximation) + gnomonic zoom + one-point-PDF (moments/histogram).
# ---------------------------------------------------------------------------

def plot_full_sky(args, run_dirs: list[Path], corrected_by_run: dict):
    """Full-sky ONE-POINT-PDF diagnostics (moments_vs_shell.png,
    example_histograms.png), POOLING raw pixels from ALL run_dirs per shell -- these
    are marginal-distribution statistics that only get more informative pooled over
    more of the held-out set, and they are the thing a Cl ratio (two-point,
    phase-blind) structurally cannot tell you.

    example_full_sky.png (gnomonic-zoom triptych + per-shell Cl ratio) was REMOVED by
    request: its Cl-ratio panel is strictly worse than cl_ratio_by_zbin_grid.png at
    the same job (which covers every held-out cosmology x redshift bin with a
    percentile band, instead of ONE cosmology's hand-picked shells), and its images
    only ever showed a single cosmology at one fixed sky position. --fullsky-shells
    now selects which shells the moments/histograms below are computed on."""
    nside = args.nside
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    method_label = "transfer (no-clip)" if args.no_clip else "transfer (clipped)"

    arrays = {run: (np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r"),
                    np.load(run / args.info_npz, mmap_mode="r")["shells"])
              for run in run_dirs}
    run0 = run_dirs[0]
    low_all0, high_all0 = arrays[run0]
    corrected0 = corrected_by_run[run0]
    all_cosmos = [f"{r.parent.name}/{r.name}" for r in run_dirs]

    lmax = min(args.lmax, 3 * nside - 1)
    ells = np.arange(lmax + 1)
    for s in args.fullsky_shell_indices:
        low_shell = np.asarray(low_all0[s], np.float32)
        corr_shell = np.asarray(corrected0[s], np.float32)
        high_shell = np.asarray(high_all0[s], np.float32)
        cl_lo = od_cl(low_shell, lmax); cl_c = od_cl(corr_shell, lmax); cl_hi = od_cl(high_shell, lmax)
        plot_cl_shell(s, ells, cl_lo, cl_c, cl_hi, out_dir / f"cl_shell{s:03d}.png")

    if args.fullsky_shells:
        print(f"[plot_full_sky] full-sky moments (median + 16-84th pctile band across "
              f"{len(run_dirs)} held-out cosmologies) + pooled histograms for shells "
              f"{args.fullsky_shells}", flush=True)

        mom_low, mom_corr, mom_high, hist_rows = [], [], [], []
        for s in args.fullsky_shells:
            # ONE sample per held-out cosmology (not pooled before moments()), so
            # plot_moments_vs_shell can draw the cosmology-to-cosmology spread as a
            # pctile band instead of hiding it in a single pooled number.
            low_per_cosmo = [np.asarray(arrays[r][0][s], np.float32) for r in run_dirs]
            high_per_cosmo = [np.asarray(arrays[r][1][s], np.float32) for r in run_dirs]
            corr_per_cosmo = [np.asarray(corrected_by_run[r][s], np.float32) for r in run_dirs]
            mom_low.append([moments(m) for m in low_per_cosmo])
            mom_high.append([moments(m) for m in high_per_cosmo])
            mom_corr.append([moments(m) for m in corr_per_cosmo])
            hist_rows.append((f"shell {s}",
                              np.concatenate([m.ravel() for m in low_per_cosmo]),
                              np.concatenate([m.ravel() for m in corr_per_cosmo]),
                              np.concatenate([m.ravel() for m in high_per_cosmo])))

        plot_moments_vs_shell(
            args.fullsky_shells, {"low": mom_low, "high (true)": mom_high,
                                  f"corrected ({method_label})": mom_corr},
            out_dir / "moments_vs_shell.png",
            suptitle=f"moments vs. shell depth -- full-sky (raw counts). Median + "
                     f"16-84th pctile band ACROSS {len(run_dirs)} held-out cosmologies "
                     f"(one sample per cosmology; see heldout_cosmo_params.png for "
                     f"their parameters).")
        plot_histogram_grid(
            hist_rows, out_dir / "example_histograms.png",
            corrected_label=f"corrected ({method_label})",
            suptitle=f"full-sky raw pixel-count histogram per shell, pooled over "
                     f"{len(run_dirs)} held-out cosmologies: {all_cosmos}")

    print(f"[plot_full_sky] figures -> {out_dir}", flush=True)


def plot_cl_zbin_grid(args, run_dirs: list[Path], corrected_by_run: dict):
    """Cl-ratio-by-redshift-bin pctile grid: one row per held-out cosmology (up to
    --max-cosmologies), one column per redshift/shell bin (zbin_shell_samples) --
    the SAME multi-cosmology two-point check unet/apply_flow.py's
    cl_ratio_by_zbin_grid.png uses (plot_cl_ratio_pctile_grid). THIS is the pipeline's
    Cl diagnostic: it is the genuine "more than one held-out cosmology" validation,
    which is why our own single-cosmology example_full_sky.png was dropped -- a
    systematic bias that happened to look fine on one hand-picked cosmology's
    shells would go undetected there. Cheap here (unlike jbucko, which must
    tile+integrate a flow ODE per patch to reconstruct each shell) because apply()
    already produced the WHOLE corrected shell in-memory -- no per-patch
    reconstruction needed, just od_cl on arrays we already have."""
    out_dir = Path(args.out_dir)
    nside = args.nside
    lmax = min(args.lmax, 3 * nside - 1)
    ells = np.arange(lmax + 1)

    run0 = run_dirs[0]
    n_shells_total = np.load(run0 / f"low_shells_nside={nside}.npy", mmap_mode="r").shape[0]
    # EXCLUDE the LAST lightcone shell (2026-07-20 data-quality finding): measured
    # across every grid AND cosmogridv1 cosmology checked, DISCO's low map at the
    # final shell (index n_shells_total-1, z~3.46-3.50 -- a narrow, truncated shell
    # at the lightcone/box edge) carries only 16-65% of CosmoGrid's true mean count,
    # vs 99.8-99.9% agreement on every other shell (0-67). A raw-count DEFICIT of
    # that size is a DISCO input artifact, not a correction-model failure -- no
    # transfer function or generative model can restore mass DISCO never had. Left
    # in, it single-handedly blew the old "shells 45-68" panel's pctile band out to
    # ~1.9 in every pipeline's cl_ratio_by_zbin_grid.png (now "shells 45-67").
    n_shells_total -= 1
    zbins = zbin_shell_samples(n_shells_total, args.zbin_start, args.n_zbins,
                               args.n_shells_per_zbin)
    grid_runs = run_dirs[:args.max_cosmologies]
    print(f"[plot_cl_zbin_grid] {len(grid_runs)} held-out cosmologies x "
          f"{len(zbins)} redshift bins {[b[0] for b in zbins]}", flush=True)

    grid = []
    for run in grid_runs:
        low_all = np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r")
        high_all = np.load(run / args.info_npz, mmap_mode="r")["shells"]
        corrected = corrected_by_run[run]
        panels = []
        for bin_label, shells in zbins:
            lo_stack, co_stack = [], []
            for s in shells:
                s = int(s)
                low_shell = np.asarray(low_all[s], np.float32)
                corr_shell = np.asarray(corrected[s], np.float32)
                high_shell = np.asarray(high_all[s], np.float32)
                cl_lo = od_cl(low_shell, lmax); cl_c = od_cl(corr_shell, lmax)
                cl_hi = od_cl(high_shell, lmax)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lo_stack.append(cl_lo / cl_hi); co_stack.append(cl_c / cl_hi)
            panels.append((bin_label, shells, ells, np.array(lo_stack), np.array(co_stack)))
        grid.append((f"{run.parent.name}/{run.name}", panels))

    method_label = "transfer (no-clip)" if args.no_clip else "transfer (clipped)"
    plot_cl_ratio_pctile_grid(
        grid, out_dir / "cl_ratio_by_zbin_grid.png",
        corrected_label=f"corrected ({method_label}) / true (after)",
        suptitle=f"Full-sky Cl ratio by redshift bin ({method_label})")


def _nz_tag(nz_path) -> str:
    """'bin4' from .../desy3_nz_metacal_bin4.txt (falls back to the file stem)."""
    import re
    m = re.search(r"bin\d+", Path(nz_path).stem)
    return m.group(0) if m else Path(nz_path).stem


def plot_kappa(args, run_dirs: list[Path], corrected_by_run: dict):
    """Weak-lensing kappa map diagnostic (analysis.weak_lensing): reduces the WHOLE
    usable lightcone [--kappa-zi, --kappa-zf] of low/corrected/high into ONE kappa
    map each via UFalcon, for EVERY held-out run-dir. NOT compute-cheap in the
    sense the COMPUTE itself is light (just filtering + a handful of
    construct_kappa_map calls per cosmology instead of tiling+ODE per shell), but
    it IS memory-heavy: this runs while `corrected_by_run` already holds every
    held-out cosmology's full array simultaneously (needed by the earlier
    patches/full_sky/zbin_grid stages too), and materializes its OWN per-cosmology
    low/high/corrected shell-stack copies on top of that -- OOM-killed a
    10-cosmology run at ~420GB (job 2884508, 2026-07-24) before those copies were
    switched to float32. --run-dirs is the only cap that actually limits this
    (run_transfer.sh's DIAG_MAX_COSMOLOGIES, not apply_transfer.py's own
    --max-cosmologies, which only caps cl_ratio_by_zbin_grid.png's rows). Each
    run's OWN params.yml is loaded fresh via weak_lensing.load_cosmo_yaml (H0, Om,
    Ob, O_nu, ... -- "further values" beyond transfer_function.py's own COSMO_KEYS
    subset), since weak_lensing_ufalcon.py's hardcoded example cosmology was
    checked and does NOT match the lightcone it actually loads."""
    out_dir = Path(args.out_dir)
    nside = args.nside
    method_label = "transfer (no-clip)" if args.no_clip else "transfer (clipped)"
    # One full diagnostic set PER n(z) BIN (2026-07-16; --kappa-nz takes several,
    # default DES-Y3 metacal bin1 + bin4): bin1 peaks at z~0.23, the low-z regime
    # where getting the correction right is hardest; bin4 peaks at z~0.98, needs
    # less correction but carries the most weight in cosmological analyses.
    # Shells are gathered ONCE per cosmology up to max(--kappa-zf); each bin's
    # kappa_map then integrates only its own [zi, zf] window (UFalcon skips
    # shells outside internally), so extra bins cost kappa_map calls only.
    nz_list = list(args.kappa_nz)
    zf_list = (list(args.kappa_zf) if len(args.kappa_zf) == len(nz_list)
               else [args.kappa_zf[0]] * len(nz_list))
    tags = [_nz_tag(nz) for nz in nz_list]
    zf_max = max(zf_list)
    print(f"[plot_kappa] building kappa maps for ALL {len(run_dirs)} held-out "
          f"cosmologies | n(z) bins: "
          + ", ".join(f"{t} (zf={zf:g})" for t, zf in zip(tags, zf_list))
          + f" | zi={args.kappa_zi}, nside={args.kappa_nside}", flush=True)

    cosmo_labels = []
    acc = {t: {k: [] for k in ("cl_low", "cl_corr", "cl_high",
                               "mom_low", "mom_corr", "mom_high")} for t in tags}
    for run in run_dirs:
        cosmo_params = weak_lensing.load_cosmo_yaml(run)
        shell_info = np.load(run / args.info_npz, allow_pickle=True)["shell_info"]
        lower_z_all = shell_info["lower_z"]; upper_z_all = shell_info["upper_z"]
        usable = np.where(weak_lensing.usable_shell_mask(
            lower_z_all, upper_z_all, args.kappa_zi, zf_max))[0]
        lower_z, upper_z = lower_z_all[usable], upper_z_all[usable]

        low_all = np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r")
        high_all = np.load(run / args.info_npz, mmap_mode="r")["shells"]
        corrected = corrected_by_run[run]
        # float32, not float64 (2026-07-24): these are materialized COPIES (up to
        # ~69 usable shells x nside=2048's 50M pixels each), not mmap views, and
        # this loop runs while `corrected_by_run` ALREADY holds every held-out
        # cosmology's full array simultaneously (needed by the earlier patches/
        # full_sky/zbin_grid stages too) -- three float64 shell stacks here added
        # up to ~83GB of PEAK EXTRA memory per cosmology on top of that baseline,
        # which OOM-killed a 10-cosmology run (job 2884508) during this exact
        # stage. float32 halves it; kappa_map/kappa_cl's own internals don't
        # require float64 (only kappa_cl's FINAL, much-smaller-array output is
        # explicitly cast to float64, unaffected by this).
        low_shells = np.stack([np.asarray(low_all[int(s)], np.float32) for s in usable])
        high_shells = np.stack([np.asarray(high_all[int(s)], np.float32) for s in usable])
        corr_shells = np.stack([np.asarray(corrected[int(s)], np.float32) for s in usable])
        print(f"[plot_kappa] {run.parent.name}: {len(usable)} usable shells "
              f"(z in [{lower_z.min():.3f},{upper_z.max():.3f}])", flush=True)

        cosmo_labels.append(f"{run.parent.name}/{run.name}")
        for nz, zf, tag in zip(nz_list, zf_list, tags):
            kw = dict(nside=args.kappa_nside, zi=args.kappa_zi, zf=zf)
            kappa_low = weak_lensing.kappa_map(low_shells, lower_z, upper_z, cosmo_params, nz, **kw)
            kappa_corr = weak_lensing.kappa_map(corr_shells, lower_z, upper_z, cosmo_params, nz, **kw)
            kappa_high = weak_lensing.kappa_map(high_shells, lower_z, upper_z, cosmo_params, nz, **kw)
            a = acc[tag]
            a["cl_low"].append(weak_lensing.kappa_cl(kappa_low, args.kappa_lmax))
            a["cl_corr"].append(weak_lensing.kappa_cl(kappa_corr, args.kappa_lmax))
            a["cl_high"].append(weak_lensing.kappa_cl(kappa_high, args.kappa_lmax))
            a["mom_low"].append(moments(kappa_low)); a["mom_corr"].append(moments(kappa_corr))
            a["mom_high"].append(moments(kappa_high))

    kappa_ells = np.arange(args.kappa_lmax + 1)
    # Two views of the SAME kappa Cl per n(z) bin, because each answers a different
    # question (unet/apply_flow.py and sphereflow/apply_sphere_flow.py emit both too):
    #  - _per_cosmology (faceted):   how does each cosmology behave on its own?
    #  - _pctile_band (median+band): the aggregate, with the cosmology-to-cosmology
    #    SPREAD made explicit. Filenames carry the bin tag (_bin1/_bin4/...).
    for nz, zf, tag in zip(nz_list, zf_list, tags):
        a = acc[tag]
        suptitle_common = (f"{len(cosmo_labels)} held-out cosmologies ({method_label}) | "
                          f"n(z)={Path(nz).name} | z in [{args.kappa_zi:g},{zf:g}]"
                          f" | kappa nside={args.kappa_nside}, lmax={args.kappa_lmax}")
        plot_kappa_cl_grid(
            cosmo_labels, kappa_ells, a["cl_low"], a["cl_corr"], a["cl_high"],
            out_dir / f"kappa_cl_per_cosmology_{tag}.png",
            corrected_label=f"corrected ({method_label})",
            suptitle=f"weak-lensing kappa Cl per cosmology, {suptitle_common}")

        with np.errstate(divide="ignore", invalid="ignore"):
            lo_stack = np.array([lo / hi for lo, hi in zip(a["cl_low"], a["cl_high"])])
            co_stack = np.array([co / hi for co, hi in zip(a["cl_corr"], a["cl_high"])])
        plot_pctile_band_ratio(
            kappa_ells[1:], {"low / high (baseline, no model)": lo_stack[:, 1:],
                            f"corrected ({method_label}) / high": co_stack[:, 1:]},
            out_dir / f"kappa_cl_pctile_band_{tag}.png", xlabel=r"$\ell$", ylim=(0.4, 1.6),
            title=f"weak-lensing kappa Cl ratio to truth ({tag}) -- median + 16-84th "
                  f"pctile band ACROSS {len(cosmo_labels)} held-out cosmologies")

        plot_kappa_moments_scatter(
            cosmo_labels, a["mom_low"], a["mom_corr"], a["mom_high"],
            out_dir / f"kappa_moments_scatter_{tag}.png",
            corrected_label=f"corrected ({method_label})",
            suptitle=f"weak-lensing kappa map moments, {suptitle_common}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    # --- apply() args ---
    p.add_argument("--transfer", required=True, nargs="+",
                   help="One transfer.npz, BROADCAST to every --run-dirs entry "
                        "(the `fit` method: one cosmology-independent T for all "
                        "held-out cosmologies) -- OR one transfer.npz PER "
                        "--run-dirs entry, same order (the `emulate` method: a "
                        "separate emulated T per held-out cosmology).")
    p.add_argument("--run-dirs", required=True, nargs="+",
                   help="One or more held-out-cosmology run dirs, each with "
                        "low_alms_lmax{lmax}.npy (+ high_alms) -- validating on "
                        "MULTIPLE held-out cosmologies (not just one) mirrors "
                        "unet's split_by_cosmo/apply_flow.py.")
    p.add_argument("--info-npz", default="compressed_shells.npz",
                   help="npz to copy non-shell metadata (shell_info) from; also the "
                        "true CosmoGrid shells used by the plotting stages below.")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--wiener", action="store_true",
                    help="Apply the Wiener gain r*T instead of full T. OFF by "
                         "default: it SUBTRACTS power where r is small (faint "
                         "shells / high ell), pushing Cl below even DISCO and the "
                         "map lighter than DISCO. Full T (default) matches Cl_high.")
    p.add_argument("--stochastic", action="store_true",
                    help="Constrained-realization gain instead of plain full T -- "
                         "see apply()'s docstring comment for the Cl_high = "
                         "(r*T)^2 + T^2*(1-r^2) derivation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smooth-window", type=int, default=21)
    p.add_argument("--ell-min", type=int, default=0,
                   help="Leave ell<ell_min untouched (T=1), smoothly ramped up over "
                        "--hp-transition above ell_min (see there). Overridden by "
                        "--ell-min-mpc when that is > 0.")
    p.add_argument("--ell-min-mpc", type=float, default=3.0,
                   help="Leave comoving scales LARGER than this (Mpc/h) untouched, "
                        "converted to a PER-SHELL ell_min via each shell's own "
                        "redshift (see ell_min_from_mpc_h). 0 disables.")
    p.add_argument("--hp-transition", type=float, default=0.10,
                   help="Width (as a fraction of lmax, same 'fraction of Nyquist' "
                        "convention as unet/diffusion/sphereflow's --hp-transition) "
                        "of the raised-cosine hand-over band above ell_min -- "
                        "smoothly ramps Ti/Ri from 1 (no correction) up to their "
                        "full values instead of a hard step, avoiding the "
                        "ringing/Gibbs artifact a step produces in real space (see "
                        "highpass_ell_ramp's docstring; ported for consistency with "
                        "the other 3 correction pipelines, which independently "
                        "converged on the same fix for a kappa Cl bias traced to "
                        "exactly this kind of hard cutoff). 0 recovers the original "
                        "hard ell_min step exactly.")
    p.add_argument("--no-clip", action="store_true",
                   help="Emit the raw Cl-optimal overdensity field instead of a "
                        "positivity-clipped one -- KEEPS negative pixels (up to ~49%% "
                        "on the faintest shells). Validated (2026-07-16) as both "
                        "cheaper AND more accurate than Poisson resampling for kappa "
                        "specifically (which doesn't need per-pixel positivity); "
                        "clip-at-0 was tested as an alternative and is WORSE, not a "
                        "middle ground (injects +14-23%% spurious large-scale power "
                        "from filling in spatially-correlated voids). Use this unless "
                        "you specifically need a valid (non-negative) count map.")
    p.add_argument("--no-debias-mean", action="store_true")
    p.add_argument("--out-counts-dir", default="",
                   help="Optional: also save each cosmology's corrected shells to "
                        "<dir>/<cosmo_name>_counts.npz (not required for the plots "
                        "below, which use the in-memory arrays directly).")

    # --- plot_patches args ---
    p.add_argument("--patch-shells", type=int, nargs="*", default=[5, 10, 15, 30, 50],
                   help="Shells for example_patches.png + pctile band. Empty to skip.")
    p.add_argument("--n-per-shell", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--n-pctile-patches", type=int, default=200)

    # --- plot_full_sky args (one-point PDF only; example_full_sky.png removed --
    # cl_ratio_by_zbin_grid.png is the Cl diagnostic now, see plot_full_sky) ---
    p.add_argument("--fullsky-shell-indices", type=int, nargs="*", default=[],
                   help="Shells for individual cl_shell*.png. Empty (default) skips "
                        "these -- cl_ratio_by_zbin_grid.png already covers the real "
                        "Cl ratio across every held-out cosmology and redshift bin.")
    p.add_argument("--fullsky-shells", type=int, nargs="*",
                   default=[3, 5, 8, 12, 16, 20, 25, 30, 36, 42, 50, 58, 66],
                   help="Shells for the full-sky moments/histogram (one-point PDF) "
                        "plots. Empty to skip. Densified 2026-07-16 (was 5 10 15 30 "
                        "50): with only 5 shells a spike at one shell is "
                        "indistinguishable from a trend.")
    p.add_argument("--lmax", type=int, default=3000)

    # --- plot_cl_zbin_grid args (multi-cosmology Cl-ratio-by-redshift-bin pctile
    # grid, mirrors unet/apply_flow.py's cl_ratio_by_zbin_grid.png) ---
    p.add_argument("--zbin-start", type=int, default=5,
                   help="first shell in the Cl-ratio-by-redshift-bin grid. 5 (was 0, "
                        "changed 2026-07-20): shell 0 shows weird behaviour in the "
                        "grid's first panel (user-observed, job 4247908) -- excluded. "
                        "Same default as unet/diffusion/sphereflow so all pipelines' "
                        "grids keep binning the SAME shells.")
    p.add_argument("--n-zbins", type=int, default=3,
                   help="Redshift/shell bins (grid columns). 0 skips this plot.")
    p.add_argument("--n-shells-per-zbin", type=int, default=5,
                   help="Shells sampled per bin, each fully Cl'd -- more = "
                        "smoother percentile band, more compute.")
    p.add_argument("--max-cosmologies", type=int, default=3,
                   help="Held-out cosmologies to include as grid rows (each row "
                        "costs --n-zbins * --n-shells-per-zbin od_cl calls).")

    # --- weak-lensing kappa map diagnostic (analysis.weak_lensing) -- see
    # apply_flow.py's mirror of this section for the full rationale. Off by
    # default; comparatively cheap here since `corrected` is already the full
    # in-memory array (no per-shell reconstruction needed, unlike jbucko). ---
    p.add_argument("--kappa", action="store_true",
                   help="build weak-lensing kappa maps (low/corrected/high) for "
                        "EVERY held-out cosmology (--run-dirs) and compute their "
                        "Cl + moments.")
    p.add_argument("--kappa-nz", nargs="+",
                   default=["/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin1.txt",
                            "/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin4.txt"],
                   help="one or more n(z) redshift distributions -- a FULL kappa "
                        "diagnostic set is produced per bin (files tagged _bin1/"
                        "_bin4/...). Default: DES-Y3 metacal bin1 (z~0.23 peak, the "
                        "low-z regime where the correction is hardest) AND bin4 "
                        "(z~0.98 peak, less correction needed but the most weight "
                        "in cosmological analyses).")
    p.add_argument("--kappa-nside", type=int, default=1024,
                   help="output kappa map nside (independent of --nside). 1024 (not "
                        "128) because the kappa map's own resolution CUTS OFF the "
                        "diagnostic: nside=128 can only represent ell<~383 (3*nside-1; "
                        "27.5 arcmin pixels), but the transfer function is ~1 there and "
                        "does essentially ALL of its work above it -- measured on real "
                        "transfer_cosmo_000003.npz, max|T-1| over ell<=350 is only "
                        "0.002-0.025 on shells 10-30 versus 0.15-1.10 over ell 351-3000. "
                        "At nside=128 the kappa comparison would show corrected ~ low ~ "
                        "high and say nothing about whether the correction works. "
                        "nside=1024 reaches ell~3071, covering the corrected band.")
    p.add_argument("--kappa-zi", type=float, default=0.0)
    p.add_argument("--kappa-zf", type=float, nargs="+", default=[1.05, 1.85],
                   help="integration upper redshift PER --kappa-nz entry (single "
                        "value broadcasts). Defaults hold >=95%% of each bin's "
                        "n(z): 1.05 for bin1, 1.85 (~99%%) for bin4.")
    p.add_argument("--kappa-lmax", type=int, default=2048,
                   help="angular power spectrum lmax for the kappa maps (--kappa-nside "
                        "supports up to ~3*nside-1, so keep this below that). Must reach "
                        "well past ell~350 or the correction is invisible -- see "
                        "--kappa-nside.")

    p.add_argument("--reuse-counts", default="",
                   help="Directory of <cosmo>_counts.npz from a PREVIOUS run (i.e. a "
                        "previous --out-counts-dir): load those corrected shells "
                        "instead of recomputing apply(). For iterating on the "
                        "DIAGNOSTIC PLOTS only -- the correction is deterministic "
                        "given (transfer, --seed, --ell-min-mpc, --no-clip), so "
                        "reusing them is exact ONLY if those are unchanged. Any "
                        "cosmology missing from the directory is recomputed normally.")
    p.add_argument("--out-dir", required=True, help="Where all plots are written.")
    args = p.parse_args()

    run_dirs = [Path(r) for r in args.run_dirs]
    if len(args.transfer) == 1:
        transfer_paths = args.transfer * len(run_dirs)
    elif len(args.transfer) == len(run_dirs):
        transfer_paths = args.transfer
    else:
        raise SystemExit(f"--transfer must have length 1 (broadcast) or match "
                          f"--run-dirs ({len(run_dirs)}), got {len(args.transfer)}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # One bad cosmology (missing/corrupt alms, a transfer.npz that failed to build
    # upstream, ...) must NOT blank out every diagnostic plot for the cosmologies
    # that DID succeed -- losing that work to an unrelated crash on cosmology N is
    # expensive and, worse, silent (the calling shell script doesn't check
    # python's exit code either).
    # Real incident (2026-07-13, job 4200753): cosmo_000054 had no disco_sim/ data
    # at all (a data-availability gap, not a code bug -- see split_val_cosmos's
    # docstring for the actual fix at the SELECTION point) and crashed apply() on
    # its FileNotFoundError, aborting the whole run before reaching ANY plotting
    # even though the other 2/3 cosmologies had already finished successfully.
    corrected_by_run = {}
    for run, tpath in zip(run_dirs, transfer_paths):
        cosmo_label = f"{run.parent.name}/{run.name}"
        out_path = (Path(args.out_counts_dir) / f"{run.parent.name}_counts.npz"
                   if args.out_counts_dir else None)
        try:
            # --reuse-counts: skip apply() entirely and load the corrected shells a
            # PREVIOUS run already produced. apply() is fully deterministic given
            # (transfer, --seed, --ell-min-mpc, --no-clip), so re-running it just to
            # redraw a plot burns minutes recomputing a byte-identical array.
            # Iterating on the DIAGNOSTICS (which is most of what we do) is what
            # this is for. Only valid if those knobs are unchanged -- change any of
            # them and you must regenerate, not reuse.
            reuse = (Path(args.reuse_counts) / f"{run.parent.name}_counts.npz"
                    if args.reuse_counts else None)
            if reuse and reuse.exists():
                print(f"=== [apply_transfer] {cosmo_label}  (REUSING counts {reuse}) ===",
                      flush=True)
                corrected_by_run[run] = np.load(reuse, mmap_mode="r")["shells"]
            else:
                if reuse:
                    print(f"[apply_transfer] --reuse-counts: {reuse} not found -- "
                          f"recomputing {cosmo_label} from scratch", flush=True)
                print(f"=== [apply_transfer] {cosmo_label}  (transfer={tpath}) ===",
                      flush=True)
                corrected_by_run[run] = apply(args, run, tpath, out_path)
        except Exception as e:
            print(f"[apply_transfer] ERROR: {cosmo_label} failed ({e!r}) -- "
                  f"skipping it, continuing with the rest", flush=True)

    ok_run_dirs = list(corrected_by_run.keys())
    if not ok_run_dirs:
        raise SystemExit("[apply_transfer] every cosmology failed -- nothing to plot "
                         "(see ERROR lines above)")
    if len(ok_run_dirs) < len(run_dirs):
        failed = [r for r in run_dirs if r not in corrected_by_run]
        print(f"[apply_transfer] WARNING: {len(failed)}/{len(run_dirs)} cosmologies "
              f"failed and are excluded from all plots below: "
              f"{[f'{r.parent.name}/{r.name}' for r in failed]}", flush=True)

    if args.patch_shells:
        plot_patches(args, ok_run_dirs, corrected_by_run)
    if args.fullsky_shells or args.fullsky_shell_indices:
        plot_full_sky(args, ok_run_dirs, corrected_by_run)
    if args.n_zbins > 0:
        plot_cl_zbin_grid(args, ok_run_dirs, corrected_by_run)
    if args.kappa:
        plot_kappa(args, ok_run_dirs, corrected_by_run)

    print(f"[apply_transfer] done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
