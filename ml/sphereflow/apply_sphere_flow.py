#!/usr/bin/env python3
"""Sample the sphere-flow model (formulation=direct) on held-out cosmologies and
run it through the SAME shared analysis/ diagnostics apply_transfer.py uses --
for direct, apples-to-apples comparison against the transfer-function baseline's
plots.

Why sphereflow, not the other ML candidates (survey done 2026-07-14, see
[[deepsphere-shell-correction]] memory): of almflow / sphereflow / unet_diff /
unet_flow / unet_resflow, sphereflow (formulation=direct, run 3826942) is the
only one with BOTH a complete, non-crashed checkpoint AND no catastrophic Cl
blow-up on held-out data. unet_flow/unet_resflow diverge on the faintest shell
(Cl ratio 23x-900x truth); unet_diff is a deterministic regressor that barely
beats predicting nothing (~1% MSE gain), reproducing the same power-suppression
failure that killed the original deepsphere approach; almflow's only saved run
had just 660 training steps with a non-converged loss. sphereflow/3826942
measured (from its own training log) shell 30/50 Cl ratio 0.96-1.03 across ALL
ell bands checked -- competitive with the transfer function -- though it still
undercorrects the faintest shell (shell 3) at low ell (0.24 vs 1.0). This script
exists to check that more rigorously, on more shells/statistics than a 3-shell
spot check, using the same tooling apply_transfer.py already validated.

Held-out cosmologies: as of 2026-07-14, train_sphere_flow.py holds out MULTIPLE
cosmologies (--test-cosmos, or auto-selected via --val-frac/--val-seed through
the SAME split_val_cosmos the transfer pipeline uses) and saves that exact set
into meta.npz -- if --run-dirs is omitted below, this script reads it back
automatically (capped at --max-cosmologies) so it always evaluates on
whatever this checkpoint actually held out, never guessing.

Older checkpoints (e.g. run 3826942, trained before this change) predate
test_cosmos and were LOO-style on a single `--test-cosmo cosmo_000122` (the
old default) -- for those this script falls back to cosmo_000122/run_0, which
is also the transfer function's own long-standing single-cosmology LOO
validation cosmology (see [[transfer-fn-positivity-tradeoff]]), keeping old
and new results comparable. Do not pass OTHER cosmologies via --run-dirs
unless you know they were excluded from THIS checkpoint's training run.

Cost note: unlike apply_transfer.py's apply() (closed-form, corrects all ~69
shells of a cosmology in one cheap vectorized pass), sphere-flow's correction is
an ODE integration through a 6-layer graph-conv net for EVERY shell -- expensive.
So corrected_by_run here is a LAZY, per-shell cache (LazyCorrected) that only
ever samples the specific shells a plot stage actually needs, instead of
eagerly correcting the whole cosmology up front. --compile/--amp/--patch-batch
are on/tuned by default (bf16 autocast + torch.compile reduce-overhead + a
bigger inference batch than training's memory-bound sweet spot) -- see
sphere_flow.sample_ode's docstring for why this doesn't lose precision.

Only formulation='direct' checkpoints are supported (condition on raw DISCO,
generate the high-res signal directly) -- a 'residual' checkpoint would need the
transfer-corrected map as its conditioning input, not the raw low map the
plotting stages below load.

  python apply_sphere_flow.py \\
      --model-dir /capstor/scratch/cscs/damrein/outputs/sphereflow/3826942 \\
      --data-root /capstor/scratch/cscs/damrein/cosmogridv1 \\
      --out-dir /capstor/scratch/cscs/damrein/outputs/sphereflow/3826942/compare

  # quick 3-shell smoke test only (skip the patch/full-sky/zbin diagnostics):
  python apply_sphere_flow.py --model-dir <out> --data-root <grid> \\
      --patch-shells --fullsky-shells --n-zbins 0 \\
      --fullsky-shell-indices 3 30 50 --out-dir <out>/eval

Run inside the pytorch uenv venv.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import healpy as hp
import torch

import sphere_flow as sf

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


def load_model(model_dir, device, compile: bool = False):
    meta = dict(np.load(Path(model_dir) / "meta.npz", allow_pickle=True))
    L = sf.healpix_laplacian(int(meta["nside"]), order=int(meta["order"]))
    net = sf.SphereFlowNet(L, cond_dim=int(meta["cond_dim"]), hidden=int(meta["hidden"]),
                           n_layers=int(meta["n_layers"]), K=int(meta["K"])).to(device)
    net.load_state_dict(torch.load(Path(model_dir) / "sphere_flow.pth",
                                   map_location=device))
    net.eval()
    if compile:
        # mode="reduce-overhead" (CUDA graphs) is a good fit here: correct_shell
        # always calls net() with the SAME (patch_batch, npix) shape -- n_patches
        # (12*order^2) divides evenly by the patch_batch values this codebase
        # uses (256, 512, ...), so there is no ragged final batch to trigger a
        # shape-mismatch recompile.
        net = torch.compile(net, mode="reduce-overhead")
    return net, meta


def cosmo_vector(params_yml, meta):
    import yaml
    p = yaml.safe_load(Path(params_yml).read_text())
    keys = sorted(k for k, v in p.items() if _is_num(v))
    v = np.array([float(p[k]) for k in keys], dtype=np.float64)
    n = len(meta["cosmo_mean"])
    v = np.pad(v, (0, max(n - len(v), 0)))[:n]
    return ((v - meta["cosmo_mean"]) / meta["cosmo_std"]).astype(np.float32)


def _is_num(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False


@torch.no_grad()
def correct_shell(net, meta, in_map, cosmo_vec, device, steps, patch_batch, amp: bool = False):
    """Flow-corrected physical map for one shell, conditioned on the input map.

    formulation='direct'  : the sample IS the corrected signal.
    formulation='residual': corrected signal = cond + resid_scale * sample.
    """
    order = int(meta["order"])
    scale, soft = float(meta["sig_scale"]), float(meta["softening"])
    rscale = float(meta["resid_scale"])
    formulation = str(meta.get("formulation", "residual"))

    # in_map is RING (the .npy stacks are RING). Reorder RING->NESTED so
    # map_to_patches' contiguous slices are compact superpixels matching the
    # nest=True graph Laplacian -- same reorder make_patch_dataset.py applies at
    # train time. mean is order-invariant. (See [[healpix-ring-nested-ordering]].)
    in_nest = hp.reorder(np.asarray(in_map, dtype=np.float64), r2n=True)
    mean = max(float(in_nest.mean()), 1e-12)
    d_in = in_nest[None] / mean - 1.0
    cond = sf.map_to_patches(sf.signal_forward(d_in, scale, soft), order)  # (P, m)

    cosmo = torch.from_numpy(cosmo_vec[None]).to(device)
    out = np.empty_like(cond)
    for b in range(0, cond.shape[0], patch_batch):
        c = torch.from_numpy(cond[b:b + patch_batch]).to(device)
        r = sf.sample_ode(net, c, cosmo.expand(c.shape[0], -1), steps=steps, amp=amp)
        out[b:b + patch_batch] = r.cpu().numpy()

    if formulation == "direct":
        sig = sf.patches_to_maps(out, order, 1)[0]
    else:
        sig = sf.patches_to_maps(cond + rscale * out, order, 1)[0]
    delta = sf.signal_inverse(sig, scale, soft)
    # patches_to_maps reassembles the NESTED superpixels, so this map is NESTED.
    # Reorder NESTED->RING so the returned map is RING, matching the raw low/high
    # shells and what every analysis/ diagnostic assumes (od_cl/anafast,
    # extract_patch's nest=False gnomonic, weak_lensing kappa).
    corrected_nest = (mean * (1.0 + delta)).astype(np.float32)
    return hp.reorder(corrected_nest, n2r=True)


@torch.no_grad()
def overlap_correct_shell(net, meta, in_map, cosmo_vec, device, steps, patch_batch,
                          nside_centers, amp: bool = False, taper_power: float = 8.0):
    """Flow-corrected physical map for one shell, OVERLAPPING taper-blended
    patches instead of correct_shell's disjoint, non-overlapping blocks -- for
    checkpoints trained by the 2026-07-20 overlap-capable make_patch_dataset.py
    (meta['patch_mode']=='overlap'; see sphere_flow.py's "OVERLAPPING patch
    geometry" section). Same formulation handling / normalization / RING<->NESTED
    convention as correct_shell -- only the patch extraction + reassembly differ.

    taper_power: sphereflow's ODE draws x0 ~ N(0,I) fresh PER PATCH (unlike
    unet's deterministic x0=low), so -- exactly like diffusion's own overlap
    tiling -- overlapping patches predict INDEPENDENT stochastic samples of the
    same sky region, and a plain (taper_power=1) weighted average shrinks the
    injected small-scale power (a weighted mean of N_eff independent draws keeps
    the conditional-mean part but damps the stochastic part by 1/sqrt(N_eff)).
    Default 8.0 is an UNTUNED starting point reasoned from diffusion's own
    measured knee (p=32 at ~16x mean overlap, see analysis.patch_tiling and
    diffusion/apply_diffusion.py's --taper-power docstring) and this scheme's
    similar ~16x mean overlap (see sphere_flow.auto_overlap_nside_centers) --
    re-measure the actual Cl-ratio-vs-taper_power knee on a real trained
    checkpoint before trusting this value, the same way diffusion's was
    calibrated post-hoc, not assumed."""
    order = int(meta["order"])
    nside = int(meta["nside"])
    scale, soft = float(meta["sig_scale"]), float(meta["softening"])
    rscale = float(meta["resid_scale"])
    formulation = str(meta.get("formulation", "residual"))

    idx, _centers = sf.healpix_overlap_index_maps(nside, order, nside_centers)  # (n_centers, npix_patch) NESTED ids
    ring_idx = hp.nest2ring(nside, idx)
    taper = sf.patch_angular_taper(nside, order, taper_power=taper_power)        # (npix_patch,)

    in_map64 = np.asarray(in_map, dtype=np.float64)
    mean = max(float(in_map64.mean()), 1e-12)
    d_in = in_map64 / mean - 1.0
    signal_full = sf.signal_forward(d_in[None], scale, soft)[0]                  # (npix,) RING, signal space

    cosmo = torch.from_numpy(cosmo_vec[None]).to(device)
    n_centers = idx.shape[0]
    accum = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    weight = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    for b in range(0, n_centers, patch_batch):
        batch_ring_idx = ring_idx[b:b + patch_batch]                             # (B, npix_patch)
        cond_np = signal_full[batch_ring_idx]                                    # (B, npix_patch)
        c = torch.from_numpy(cond_np.astype(np.float32)).to(device)
        r = sf.sample_ode(net, c, cosmo.expand(c.shape[0], -1), steps=steps, amp=amp)
        out = r.cpu().numpy().astype(np.float64)                                 # (B, npix_patch)
        patch_sig = out if formulation == "direct" else cond_np + rscale * out
        for k in range(batch_ring_idx.shape[0]):
            ring_ids = batch_ring_idx[k]
            np.add.at(accum, ring_ids, patch_sig[k] * taper)
            np.add.at(weight, ring_ids, taper)
        print(f"  [sphereflow-overlap]   {min(b + patch_batch, n_centers)}/{n_centers} "
              f"patches", flush=True)

    covered = weight > 0
    if not covered.all():
        n_gap = int((~covered).sum())
        print(f"  [sphereflow-overlap]   filling {n_gap} residual gap pixels "
              f"({100 * n_gap / len(covered):.3f}%) with the input signal", flush=True)
    sig_blend = np.where(covered, np.divide(accum, weight, out=np.zeros_like(accum),
                                            where=covered), signal_full)
    delta = sf.signal_inverse(sig_blend, scale, soft)
    corrected = (mean * (1.0 + delta)).astype(np.float32)
    return corrected                    # already RING (gather/scatter used RING ids throughout)


def _resolve_run_dir(data_root, cosmo_name):
    """cosmo name -> its run dir, same convention as train_sphere_flow.build_runs
    (first subdir starting with "run_", else the cosmology dir itself)."""
    c = Path(data_root) / cosmo_name
    run = next((r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")),
              None) if c.is_dir() else None
    return run or c


class LazyCorrected:
    """corrected[s] triggers sphere-flow ODE sampling for shell s ONCE, cached.

    Plug-compatible with the `corrected[s]` indexing apply_transfer.py's
    plot_patches/plot_full_sky/plot_cl_zbin_grid rely on, so this file can mirror
    their exact statistics/layout without eagerly correcting all ~69 shells of a
    cosmology up front (ODE sampling is the expensive part; apply_transfer's
    closed-form transfer correction is not, so it can afford to be eager).
    """

    def __init__(self, net, meta, low_all, cosmo_base, device, steps, patch_batch, amp=True,
                nside_centers=None, taper_power=8.0):
        self.net, self.meta = net, meta
        self.low_all = low_all
        self.cosmo_base = cosmo_base
        self.device, self.steps, self.patch_batch, self.amp = device, steps, patch_batch, amp
        self.n_shells = low_all.shape[0]
        self._cache = {}
        # Dispatch on how this checkpoint's patches were built (see
        # train_sphere_flow.py's meta['patch_mode'] and sphere_flow.py's
        # "OVERLAPPING patch geometry" section) -- a checkpoint predating the
        # 2026-07-20 overlap change has no 'patch_mode' key and MUST use the old
        # disjoint reconstruction (it never saw a rotated patch boundary).
        self.overlap = str(meta.get("patch_mode", "disjoint")) == "overlap"
        self.nside_centers = nside_centers or sf.auto_overlap_nside_centers(int(meta["order"]))
        self.taper_power = taper_power

    def __getitem__(self, s):
        s = int(s)
        if s not in self._cache:
            shell_norm = np.float32(s / max(self.n_shells - 1, 1))
            cosmo_vec = np.concatenate([self.cosmo_base, [shell_norm]]).astype(np.float32)
            in_map = np.asarray(self.low_all[s], dtype=np.float32)
            if self.overlap:
                print(f"  [sphereflow-overlap] sampling shell {s} ({self.steps} ODE "
                      f"steps, nside_centers={self.nside_centers}, "
                      f"taper_power={self.taper_power})...", flush=True)
                self._cache[s] = overlap_correct_shell(
                    self.net, self.meta, in_map, cosmo_vec, self.device, self.steps,
                    self.patch_batch, self.nside_centers, amp=self.amp,
                    taper_power=self.taper_power)
            else:
                print(f"  [sphereflow] sampling shell {s} ({self.steps} ODE steps)...", flush=True)
                self._cache[s] = correct_shell(self.net, self.meta, in_map, cosmo_vec,
                                                   self.device, self.steps, self.patch_batch,
                                                   amp=self.amp)
        return self._cache[s]


def extract_patch(shell_map: np.ndarray, nside: int, center_ipix: int, psi: float,
                  patch_size: int, reso_arcmin: float) -> np.ndarray:
    """Gnomonic-project one patch. Identical to apply_transfer.py's helper (kept
    as a small local copy so this script has no dependency on ml/transfer/)."""
    lon, lat = hp.pix2ang(nside, int(center_ipix), nest=False, lonlat=True)
    proj = hp.projector.GnomonicProj(rot=(lon, lat, psi), xsize=patch_size,
                                     ysize=patch_size, reso=reso_arcmin)
    vec2pix = lambda x, y, z, _ns=nside: hp.vec2pix(_ns, x, y, z, nest=False)
    return proj.projmap(shell_map, vec2pix)


# ---------------------------------------------------------------------------
# plot_patches / plot_full_sky / plot_cl_zbin_grid / plot_kappa: same statistics
# and shared analysis/ plotting calls as apply_transfer.py's stages of the same
# name, adapted to (a) a parametrized method_label instead of a hardcoded
# "transfer(+Poisson)" string, and (b) LazyCorrected instead of an eagerly
# materialized (n_shells, npix) array.
# ---------------------------------------------------------------------------

def plot_patches(args, run_dirs: list[Path], corrected_by_run: dict, method_label: str):
    nside = args.nside
    reso_arcmin = hp.nside2resol(nside, arcmin=True)
    npix = hp.nside2npix(nside)

    arrays = {run: (np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r"),
                    np.load(run / f"high_shells_nside={nside}.npy", mmap_mode="r"))
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

    all_cosmos = [f"{r.parent.name}/{r.name}" for r in run_dirs]
    out_dir = Path(args.out_dir)
    plot_example_patch_grid(
        rows, out_dir / "example_patches.png", corrected_label=f"corrected ({method_label})",
        suptitle=f"{method_label}: validated on {len(run_dirs)} held-out "
                 f"cosmologies: {all_cosmos} (held out of TRAINING) -- example patches "
                 "(log1p overdensity) + per-patch power ratio\n(same layout/statistic as "
                 "transfer/apply_transfer.py's example_patches.png)")

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


def plot_full_sky(args, run_dirs: list[Path], corrected_by_run: dict, method_label: str):
    nside = args.nside
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {run: (np.load(run / f"low_shells_nside={nside}.npy", mmap_mode="r"),
                    np.load(run / f"high_shells_nside={nside}.npy", mmap_mode="r"))
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


def plot_cl_zbin_grid(args, run_dirs: list[Path], corrected_by_run: dict, method_label: str):
    """The primary Cl diagnostic, same as apply_transfer.py's stage of the same
    name: one row per held-out cosmology x one column per redshift bin, with a
    percentile band. Cheap PER SHELL here (just od_cl on arrays already in hand)
    but each shell not already touched by plot_patches/plot_full_sky triggers a
    fresh ODE sample via LazyCorrected -- with --max-cosmologies 1 (this
    checkpoint's only real held-out cosmology) and the default 3x5=15 zbin
    shells, that is at most 15 NEW ODE samples beyond the ones already cached."""
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
        high_all = np.load(run / f"high_shells_nside={nside}.npy", mmap_mode="r")
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

    plot_cl_ratio_pctile_grid(
        grid, out_dir / "cl_ratio_by_zbin_grid.png",
        corrected_label=f"corrected ({method_label}) / true (after)",
        suptitle=f"Full-sky Cl ratio by redshift bin ({method_label})")


def _nz_tag(nz_path) -> str:
    """'bin4' from .../desy3_nz_metacal_bin4.txt (falls back to the file stem)."""
    import re
    m = re.search(r"bin\d+", Path(nz_path).stem)
    return m.group(0) if m else Path(nz_path).stem


def plot_kappa(args, run_dirs: list[Path], corrected_by_run: dict, method_label: str):
    """Weak-lensing kappa map diagnostic -- EXPENSIVE here: unlike
    apply_transfer.py (corrected is already the full array), every usable shell
    in [--kappa-zi, --kappa-zf] not already cached triggers a fresh ODE sample.
    For the default z range that is typically several dozen shells. Off unless
    --kappa is passed."""
    out_dir = Path(args.out_dir)
    nside = args.nside
    # One full diagnostic set PER n(z) BIN (2026-07-16; --kappa-nz takes several,
    # default DES-Y3 metacal bin1 + bin4): bin1 peaks at z~0.23 (low-z, hardest
    # correction), bin4 at z~0.98 (less correction, most cosmological weight).
    # Shells are gathered/corrected ONCE per cosmology up to max(--kappa-zf); each
    # bin's kappa_map integrates only its own [zi, zf] window (UFalcon skips
    # shells outside internally), so extra bins cost kappa_map calls only.
    nz_list = list(args.kappa_nz)
    zf_list = (list(args.kappa_zf) if len(args.kappa_zf) == len(nz_list)
               else [args.kappa_zf[0]] * len(nz_list))
    tags = [_nz_tag(nz) for nz in nz_list]
    zf_max = max(zf_list)
    print(f"[plot_kappa] building kappa maps for ALL {len(run_dirs)} held-out "
          f"cosmologies | n(z) bins: "
          + ", ".join(f"{t} (zf={zf:g})" for t, zf in zip(tags, zf_list))
          + f" | zi={args.kappa_zi}, nside={args.kappa_nside} -- corrects every usable shell, may be slow", flush=True)

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
        high_all = np.load(run / f"high_shells_nside={nside}.npy", mmap_mode="r")
        corrected = corrected_by_run[run]
        low_shells = np.stack([np.asarray(low_all[int(s)], np.float64) for s in usable])
        high_shells = np.stack([np.asarray(high_all[int(s)], np.float64) for s in usable])
        corr_shells = np.stack([np.asarray(corrected[int(s)], np.float64) for s in usable])
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
    p.add_argument("--model-dir", required=True,
                   help="sphereflow run dir with sphere_flow.pth + meta.npz, e.g. "
                        "/capstor/scratch/cscs/damrein/outputs/sphereflow/3826942")
    p.add_argument("--data-root", default="/capstor/scratch/cscs/damrein/cosmogridv1")
    p.add_argument("--run-dirs", nargs="*", default=None,
                   help="Held-out cosmology run dirs. Default: read this checkpoint's "
                        "OWN held-out set from meta.npz's test_cosmos (capped at "
                        "--max-cosmologies), or cosmo_000122/run_0 for older checkpoints "
                        "that predate saving it. Do not pass other cosmologies unless you "
                        "know they were excluded from THIS checkpoint's training too.")
    p.add_argument("--nside", type=int, default=None,
                   help="Resolution of the low/high_shells_nside=*.npy data files to "
                        "load. Default: the MODEL's own nside from meta.npz -- the net "
                        "can only correct maps at the resolution it was trained at, so "
                        "any other value is an error (a 512 model + the old hardcoded "
                        "2048 default silently looked for the wrong files, job 4221138).")
    p.add_argument("--lmax", type=int, default=3000,
                   help="Capped internally at 3*nside-1 (e.g. 1535 for a 512 model).")
    p.add_argument("--steps", type=int, default=50, help="ODE integration steps/shell.")
    p.add_argument("--patch-batch", type=int, default=512,
                   help="Patches sampled per forward-batch. 3072 (order=16's "
                        "12*order^2 patch count) divides evenly by 256/512/768/1024/"
                        "1536/3072 -- keep it one of those so torch.compile's "
                        "reduce-overhead CUDA graphs don't hit a ragged final batch. "
                        "512 (vs. training's memory-bound 256 sweet spot) is safe here: "
                        "no backward pass / optimizer state at inference time.")
    p.add_argument("--compile", action="store_true", default=True,
                   help="torch.compile the net (mode=reduce-overhead) before sampling. "
                        "On by default -- adds a one-time compile cost, then speeds up "
                        "every subsequent step/shell.")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--amp", action="store_true", default=True,
                   help="bf16 autocast during ODE sampling -- on by default here.")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--nside-centers", type=int, default=None,
                   help="OVERLAP checkpoints only (meta['patch_mode']=='overlap'): "
                        "center-grid nside for the overlapping-patch reconstruction "
                        "sweep (see sphere_flow.healpix_overlap_index_maps). Default: "
                        "auto-scaled from --order via "
                        "sphere_flow.auto_overlap_nside_centers (~16x mean overlap, "
                        "matching analysis.patch_tiling's own target density). "
                        "Ignored for pre-2026-07-20 disjoint checkpoints.")
    p.add_argument("--taper-power", type=float, default=8.0,
                   help="OVERLAP checkpoints only: blend-weight sharpening (see "
                        "overlap_correct_shell's docstring) -- UNTUNED default, "
                        "reasoned from diffusion's own measured p=32 knee at a "
                        "similar ~16x overlap; re-measure the Cl-ratio-vs-taper_power "
                        "knee on a real trained checkpoint before trusting it.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method-label", default=None,
                   help="Defaults to 'sphereflow ({formulation}, {model-dir name})'.")

    p.add_argument("--patch-shells", type=int, nargs="*", default=[5, 10, 15, 30, 50],
                   help="Shells for example_patches.png + pctile band. Empty to skip.")
    p.add_argument("--n-per-shell", type=int, default=1)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--n-pctile-patches", type=int, default=200)

    p.add_argument("--fullsky-shell-indices", type=int, nargs="*", default=[],
                   help="Shells for individual cl_shell*.png. Empty (default) skips "
                        "these -- cl_ratio_by_zbin_grid.png already covers the real "
                        "Cl ratio across every held-out cosmology and redshift bin.")
    p.add_argument("--fullsky-shells", type=int, nargs="*",
                   default=[3, 5, 8, 12, 16, 20, 25, 30, 36, 42, 50, 58, 66],
                   help="Shells for the full-sky moments/histogram plots. Empty to "
                        "skip. Densified 2026-07-16 (was 5 10 15 30 50): with only "
                        "5 shells a spike at one shell is indistinguishable from a trend.")

    p.add_argument("--zbin-start", type=int, default=0)
    p.add_argument("--n-zbins", type=int, default=3)
    p.add_argument("--n-shells-per-zbin", type=int, default=5)
    p.add_argument("--max-cosmologies", type=int, default=3,
                   help="Held-out cosmologies to include as grid rows -- ALSO caps how "
                        "many of the checkpoint's own held-out cosmologies (meta.npz's "
                        "test_cosmos) are used at all when --run-dirs is omitted, since "
                        "ODE sampling cost scales with cosmology count.")

    p.add_argument("--kappa", action="store_true",
                   help="build weak-lensing kappa maps for every held-out cosmology. "
                        "EXPENSIVE: corrects every usable shell (dozens), each a 50-step "
                        "ODE sample, unlike apply_transfer.py's cheap closed-form kappa.")
    p.add_argument("--info-npz", default="compressed_shells.npz",
                   help="only used for --kappa's shell_info (lower_z/upper_z).")
    p.add_argument("--kappa-nz", nargs="+",
                   default=["/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin1.txt",
                            "/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin4.txt"],
                   help="one or more n(z) distributions; a FULL kappa diagnostic set "
                        "per bin, files tagged _bin1/_bin4/... (bin1: low-z, hardest "
                        "correction; bin4: high-z, most cosmological weight).")
    p.add_argument("--kappa-nside", type=int, default=1024)
    p.add_argument("--kappa-zi", type=float, default=0.0)
    p.add_argument("--kappa-zf", type=float, nargs="+", default=[1.05, 1.85],
                   help="integration upper redshift PER --kappa-nz entry (single "
                        "value broadcasts): 1.05 holds >=95%% of bin1's n(z), "
                        "1.85 ~99%% of bin4's.")
    p.add_argument("--kappa-lmax", type=int, default=2048)

    p.add_argument("--out-dir", required=True, help="Where all plots are written.")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = load_model(args.model_dir, dev, compile=args.compile)
    model_nside = int(meta["nside"])
    if args.nside is None:
        args.nside = model_nside
        print(f"[apply_sphere_flow] using the model's own nside={model_nside} "
              f"(from meta.npz) for the data files", flush=True)
    elif args.nside != model_nside:
        raise SystemExit(
            f"--nside {args.nside} does not match the model's trained nside "
            f"{model_nside} (meta.npz) -- the net can only correct maps at its "
            f"trained resolution. Drop --nside to use the model's value.")
    formulation = str(meta.get("formulation", "residual"))
    if formulation != "direct":
        raise SystemExit(
            f"this script only supports formulation='direct' checkpoints (condition "
            f"on raw DISCO) -- {args.model_dir} is '{formulation}'. A residual "
            f"checkpoint needs the T-corrected map (tcorr_shells_nside=*.npy) as its "
            f"conditioning input, not the raw low map the plotting stages here load -- "
            f"not currently supported (no known-good residual checkpoint exists; the "
            f"only complete, non-crashed sphereflow checkpoint from the 2026-07-14 "
            f"survey is direct/3826942).")
    method_label = args.method_label or f"sphereflow ({formulation}, {Path(args.model_dir).name})"

    if args.run_dirs:
        run_dirs = [Path(r) for r in args.run_dirs]
    elif "test_cosmos" in meta and np.asarray(meta["test_cosmos"]).size > 0:
        # Checkpoints trained after 2026-07-14 save their OWN held-out set (see
        # train_sphere_flow.py) -- use it directly instead of guessing, capped
        # at --max-cosmologies (ODE sampling is expensive per cosmology).
        held_out = np.asarray(meta["test_cosmos"]).tolist()[:args.max_cosmologies]
        run_dirs = [_resolve_run_dir(args.data_root, c) for c in held_out]
        print(f"[apply_sphere_flow] --run-dirs not given -- using this checkpoint's "
              f"own held-out set from meta.npz (capped at --max-cosmologies="
              f"{args.max_cosmologies} of {np.asarray(meta['test_cosmos']).size}): "
              f"{held_out}", flush=True)
    else:
        # Pre-2026-07-14 checkpoints (e.g. 3826942) predate saving test_cosmos and
        # were trained LOO-style via the old --test-cosmo cosmo_000122 default.
        run_dirs = [Path(args.data_root) / "cosmo_000122" / "run_0"]
        print("[apply_sphere_flow] --run-dirs not given and this checkpoint has no "
              "saved test_cosmos -- falling back to cosmo_000122/run_0 (the old "
              "single-cosmology --test-cosmo default).", flush=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # One bad cosmology must not blank every plot -- see apply_transfer.py's own
    # rationale for the same try/except pattern.
    corrected_by_run = {}
    for run in run_dirs:
        cosmo_label = f"{run.parent.name}/{run.name}"
        try:
            low_all = np.load(run / f"low_shells_nside={args.nside}.npy", mmap_mode="r")
            cosmo_base = (cosmo_vector(run / "params.yml", meta)
                         if (run / "params.yml").exists()
                         else np.zeros(int(meta["cond_dim"]) - 1, np.float32))
            corrected_by_run[run] = LazyCorrected(net, meta, low_all, cosmo_base, dev,
                                                  args.steps, args.patch_batch, amp=args.amp,
                                                  nside_centers=args.nside_centers,
                                                  taper_power=args.taper_power)
            print(f"=== [apply_sphere_flow] {cosmo_label} ready "
                  f"(shells corrected on demand) ===", flush=True)
        except Exception as e:
            print(f"[apply_sphere_flow] ERROR: {cosmo_label} failed to load "
                  f"({e!r}) -- skipping it, continuing with the rest", flush=True)

    ok_run_dirs = list(corrected_by_run.keys())
    if not ok_run_dirs:
        raise SystemExit("[apply_sphere_flow] every cosmology failed -- "
                         "nothing to plot (see ERROR lines above)")
    if len(ok_run_dirs) < len(run_dirs):
        failed = [r for r in run_dirs if r not in corrected_by_run]
        print(f"[apply_sphere_flow] WARNING: {len(failed)}/{len(run_dirs)} "
              f"cosmologies failed and are excluded from all plots below: "
              f"{[f'{r.parent.name}/{r.name}' for r in failed]}", flush=True)

    if args.patch_shells:
        plot_patches(args, ok_run_dirs, corrected_by_run, method_label)
    if args.fullsky_shells or args.fullsky_shell_indices:
        plot_full_sky(args, ok_run_dirs, corrected_by_run, method_label)
    if args.n_zbins > 0:
        plot_cl_zbin_grid(args, ok_run_dirs, corrected_by_run, method_label)
    if args.kappa:
        plot_kappa(args, ok_run_dirs, corrected_by_run, method_label)

    print(f"[apply_sphere_flow] done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
