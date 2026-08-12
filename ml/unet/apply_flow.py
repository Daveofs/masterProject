#!/usr/bin/env python3
"""Apply a trained jbucko flow model to HELD-OUT test patches and evaluate.

Embeds his code directly: FlowUNet + sample_ode from flow_model.py, and the patch
loading / split / transform from dataset.py. Nothing is reimplemented here.

Test data = the validation (held-out COSMOLOGY) split of the patch dataset -- cosmologies
the model never saw in training (dataset.split_by_cosmo). For each patch we integrate the
flow ODE from x0 = low (whichever --space the checkpoint was trained in -- default
'delta', the linear overdensity od_cl measures; 'log1p' only for pre-2026-07-18
checkpoints, see dataset.raw_to_delta_pair) to a predicted high, then compare
corrected vs the true high by:
  * the 2D-FFT radial power spectrum ratio (the flat-patch analogue of the C_ell ratio --
    the statistic we care about: does the correction restore small-scale power?),
  * example low / corrected / high patch triptychs (judge by eye).

Because a flow SAMPLE (not a conditional mean) carries the full high-field variance, the
corrected power should track the truth at small scales where a deterministic regressor
would sag -- that is the whole reason for this approach.

SHARED EVAL SET: the figures below are the SAME statistics, shared analysis/ plotting
code, same shells and same kappa resolution as transfer/apply_transfer.py (the
reference) and sphereflow/apply_sphere_flow.py, so the three pipelines' outputs are
comparable file-by-file:
    example_patches.png, patch_power_ratio_pctile_band.png,
    moments_vs_shell.png / example_histograms.png (full-sky one-point PDF, needs
    --data-root -- the patch-pooled versions are written as patch_*.png and have no
    counterpart in the other two pipelines),
    cl_ratio_by_zbin_grid.png (--data-root),
    kappa_cl_per_cosmology.png, kappa_cl_pctile_band.png, kappa_moments_scatter.png
    (--kappa).
power_spectrum_ratio.png (mean 2D-FFT power, loglog) is extra, unique to this script.

  python apply_flow.py --patch-dir <dir> --model <run>/best.pt --out-dir <run>/eval
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import healpy as hp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# embed his files: this script lives alongside flow_model.py / dataset.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_model import FlowUNet, sample_ode, compose_corrected, cutoff_from_chi  # noqa: E402
from dataset import (PatchDataset, split_by_cosmo, transform_pair,  # noqa: E402
                     low_to_field, field_to_counts, cosmo_z_vector, COSMO_FIELDS)

# every plotting/tiling/full-sky-power routine comes from ../analysis -- nothing
# reimplemented here (this is the ONLY diagnostics script for the flow model now;
# infer_full_sky.py was merged in and deleted, see the full-sky section below).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.plotting import (plot_example_patch_grid, plot_pctile_band_ratio,  # noqa: E402
                               plot_histogram_grid, plot_moments_vs_shell,
                               plot_cl_shell, plot_cl_ratio_pctile_grid,
                               plot_kappa_cl_grid, plot_kappa_moments_scatter)
from analysis.moments import moments                      # noqa: E402
from analysis.patch_tiling import auto_nside_centers, reconstruct_shell  # noqa: E402
from analysis.full_sky import od_cl, zbin_shell_samples, shell_redshifts    # noqa: E402
from analysis.example_patches import patch_plan, extract_patch      # noqa: E402
from analysis.transforms import log1p_delta_pair                    # noqa: E402
from analysis import weak_lensing                          # noqa: E402


def stack_cosmo_z(ds: PatchDataset, dev, dtype) -> torch.Tensor:
    """Stack (cosmo, z) across every sample in ds, on dev, into the (N,8) cosmo_z
    tensor FlowUNet expects (see dataset.cosmo_z_vector) -- the eval-time analogue
    of train_flow.py's per-batch cosmo_z construction."""
    cosmo = torch.stack([ds[i]["cosmo"] for i in range(len(ds))]).to(dev, dtype)
    z = torch.tensor([ds[i]["z"] for i in range(len(ds))], device=dev, dtype=dtype)
    return cosmo_z_vector(cosmo, z)


def radial_power_batch(imgs: torch.Tensor, n_bins: int = None):
    """(B,1,H,W) -> (B,n_bins) PER-SAMPLE 2D-FFT power per radial wavenumber bin.

    Batched-on-GPU torch version, used for the aggregate power_spectrum_ratio.png
    (mean over samples) AND the pctile-band plot (needs the per-sample spread, not
    just the mean) -- a deliberate, performance-only duplication of
    analysis.radial_power's single-image numpy version, which the example_patches.png
    rows below use instead (a handful of images, GPU batching buys nothing there)."""
    B, _, H, W = imgs.shape
    dev = imgs.device
    n_bins = n_bins or H // 2
    fy = torch.fft.fftfreq(H, device=dev) * H
    fx = torch.fft.rfftfreq(W, device=dev) * W
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = torch.clamp((r / r.max() * (n_bins - 1)).long(), 0, n_bins - 1).view(-1)
    counts = torch.bincount(bins, minlength=n_bins).clamp(min=1).float()
    f = torch.fft.rfft2(imgs.squeeze(1))
    power = (f.real ** 2 + f.imag ** 2).view(B, -1)              # (B, H*(W//2+1))
    binned = torch.zeros(B, n_bins, device=dev).scatter_add_(1, bins.expand(B, -1), power)
    return (binned / counts).cpu().numpy()


def _nz_tag(nz_path) -> str:
    """'bin4' from .../desy3_nz_metacal_bin4.txt (falls back to the file stem)."""
    import re
    m = re.search(r"bin\d+", Path(nz_path).stem)
    return m.group(0) if m else Path(nz_path).stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True)
    p.add_argument("--model", required=True, help="checkpoint (best.pt / last.pt)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    # Shells / size for the SHARED example-patch figure (analysis/example_patches.py).
    # Defaults match transfer/apply_transfer.py so all three pipelines draw the same
    # rows; changing them here alone would desynchronise the comparison.
    p.add_argument("--patch-shells", type=int, nargs="*", default=[5, 10, 15, 30, 50],
                   help="Shells for example_patches.png (shared across pipelines).")
    p.add_argument("--patch-size", type=int, default=256,
                   help="Gnomonic patch size in px for example_patches.png.")
    p.add_argument("--n-eval", type=int, default=512, help="held-out patches to evaluate")
    p.add_argument("--eval-batch", type=int, default=256,
                   help="mini-batch size for the ODE sampling pass (memory control). "
                        "256 (was 64) -- UNTESTED bump, reasoned from FlowUNet's modest "
                        "size (base_channels=32, 256x256 patches) leaving a GH200's "
                        "96GB HBM mostly idle at the old default; profile and adjust "
                        "if it OOMs or doesn't help.")
    p.add_argument("--amp", action="store_true", default=True,
                   help="bf16 autocast during ODE sampling -- same pattern as "
                        "sphere_flow.sample_ode's own amp flag (see "
                        "flow_model.sample_ode's docstring). On by default; "
                        "UNTESTED for this sampler specifically, --no-amp to disable "
                        "if results look off.")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--example-shells", type=int, nargs="+", default=[5, 10, 15, 30, 50],
                   help="shell indices to show as rows in example_patches.png (one "
                        "held-out patch per shell, picked via patch metadata) and to "
                        "pool for the moments/histogram plots. Default matches "
                        "transfer/apply_transfer.py's --patch-shells/--fullsky-shells "
                        "and sphereflow's, so the three pipelines' figures line up.")
    p.add_argument("--steps", type=int, default=8, help="Euler ODE steps")
    p.add_argument("--n-stat-patches", type=int, default=64,
                   help="held-out patches PER --example-shells shell to pool for "
                        "patch_moments_vs_shell.png / patch_example_histograms.png -- the "
                        "one-point PDF (esp. skew/kurtosis of a sparse count field) "
                        "is far noisier per-patch than the power ratio, so this "
                        "needs many patches, unlike the single visual example patch")
    # --- optional full-sky reconstruction + REAL angular Cl (skipped unless
    # --data-root is given): the power-ratio panels above are a flat 2D-FFT on one
    # --patch-dir patch, bounded by that patch's own Nyquist wavenumber -- nowhere
    # near the full ell~3000 range a real spherical-harmonic transform covers. This
    # tiles the WHOLE sphere (analysis.patch_tiling) and runs healpy.anafast
    # (analysis.full_sky.od_cl) to see the real high-ell behavior. ---
    p.add_argument("--data-root", default=None,
                   help="prepare_maps.py output (full-sky low/high shell stacks). "
                        "If given, ALSO reconstructs the whole sphere and computes "
                        "the real full-sky Cl -> cl_shell*.png, cl_ratio_by_zbin_grid.png, "
                        "moments_vs_shell.png, example_histograms.png (the full-sky, "
                        "cross-pipeline-comparable one-point PDF -- WITHOUT --data-root only "
                        "the patch_-prefixed patch-pooled versions are written)")
    p.add_argument("--cosmo", default=None,
                   help="cosmology for the full-sky reconstruction; default: first "
                        "held-out (val) cosmology")
    p.add_argument("--run", default="run_0")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[0, 34, 68],
                   help="shells to render as individual cl_shell*.png (2-panel "
                        "Cl+ratio); pass with no values to skip these")
    p.add_argument("--nside-centers", type=int, default=None,
                   help="default: auto-scaled from the data's nside so patch overlap "
                        "is consistent (see analysis.patch_tiling.auto_nside_centers)")
    p.add_argument("--fullsky-patch-size", type=int, default=256,
                   help="gnomonic tile size for full-sky reconstruction -- should "
                        "match the patch size this checkpoint was trained on")
    p.add_argument("--taper-power", type=float, default=1.0,
                   help="blend-weight sharpening for the full-sky tiling (see "
                        "analysis.patch_tiling.tile_and_predict) -- SAME knob "
                        "diffusion/apply_diffusion.py exposes, added here for tooling "
                        "parity/comparability, not because unet needs a different "
                        "value: unet's flow starts from x0=low (deterministic, no "
                        "noise draw), so overlapping tiles predicting the same sky "
                        "pixel already AGREE and 1.0 (plain cosine-taper averaging) "
                        "is optimal -- it is free seam smoothing, not lossy averaging "
                        "of independent samples. Sharpening toward nearest-tile-wins "
                        "(as diffusion's tuned p=32 does) only matters for a "
                        "genuinely STOCHASTIC per-tile model (diffusion draws a fresh "
                        "x_T per tile; sphereflow's ODE also starts from noise, but "
                        "its HEALPix-superpixel tiling has no overlap to average over "
                        "in the first place). Override only to A/B-test this claim on "
                        "your own checkpoint.")
    p.add_argument("--lmax", type=int, default=1500,
                   help="band limit for the full-sky Cl. 1500 is the common "
                        "N_side=512 footing all three pipelines are scored on; "
                        "clamped to 3*nside-1 regardless.")
    # --- example_full_sky.png: Cl-ratio-by-redshift-bin pctile grid (rows = held-out
    # cosmologies, columns = redshift/shell bins) -- no images (see example_patches.png
    # for those), just the aggregate two-point check. Each cell needs
    # --n-shells-per-zbin full-sky reconstructions, so rows*cols*shells-per-zbin is
    # the real cost knob -- these defaults are deliberately small; widen once the
    # per-shell cost on this setup is known (same philosophy as --shell-indices above). ---
    p.add_argument("--zbin-start", type=int, default=5,
                   help="first shell in the Cl-ratio-by-redshift-bin grid. 5 (was 0, "
                        "changed 2026-07-20): shell 0 shows weird behaviour in the "
                        "grid's first panel (user-observed, job 4247908) -- excluded, "
                        "same default as transfer/diffusion/sphereflow so all "
                        "pipelines' grids keep binning the SAME shells. The rest of "
                        "the faint/noisy range (shells 5+) stays in -- that IS the "
                        "regime the generative models are supposed to win in.")
    p.add_argument("--n-zbins", type=int, default=3,
                   help="number of redshift/shell-index bins (columns) spanning "
                        "[--zbin-start, last shell]")
    p.add_argument("--n-shells-per-zbin", type=int, default=5,
                   help="shells sampled (evenly spaced) per bin, each fully "
                        "reconstructed+Cl'd -- more = smoother percentile band, more compute")
    p.add_argument("--max-cosmologies", type=int, default=3,
                   help="held-out cosmologies to include as rows (each row costs "
                        "--n-zbins * --n-shells-per-zbin full-sky reconstructions)")
    # --- weak-lensing kappa map diagnostic (analysis.weak_lensing): reduces the
    # WHOLE usable lightcone [--kappa-zi, --kappa-zf] of low/corrected/high into one
    # kappa map each via UFalcon, for EVERY held-out cosmology (not capped by
    # --max-cosmologies -- one kappa map per cosmology is comparatively cheap once
    # its shells exist, but "corrected" needs every usable shell reconstructed via
    # full-sky tiling, not just a --n-shells-per-zbin sample, so this is still the
    # most expensive optional section here). Off by default. ---
    p.add_argument("--kappa", action="store_true",
                   help="build weak-lensing kappa maps (low/corrected/high) for "
                        "EVERY held-out cosmology and compute their Cl + moments. "
                        "Requires --data-root. Expensive: see module docstring.")
    p.add_argument("--kappa-nz", nargs="+",
                   default=["/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin1.txt",
                            "/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin4.txt"],
                   help="one or more n(z) distributions; a FULL kappa diagnostic set "
                        "per bin, files tagged _bin1/_bin4/... (bin1: low-z, hardest "
                        "correction; bin4: high-z, most cosmological weight).")
    p.add_argument("--kappa-nside", type=int, default=512,
                   help="output kappa map nside. MUST match the nside the shells were "
                        "corrected at (--nside): the thesis compares all pipelines on a "
                        "common N_side=512 footing, and building kappa at 1024 from 512 "
                        "shells is pure upsampling -- it invents no information and "
                        "produces spectra out to ell~3071 that above the shells' own band "
                        "limit (3*512-1 = 1535) are interpolation and pixel-window "
                        "artefacts, not signal. It must also not go LOW: the transfer "
                        "function is ~1 below ell~350 and does essentially all its work "
                        "above it (measured on transfer_cosmo_000003.npz: max|T-1| is "
                        "0.002-0.025 over ell<=350 on shells 10-30, versus 0.15-1.10 over "
                        "ell 351-3000), so at e.g. nside=128 (ell<~383) the kappa "
                        "comparison would show corrected ~ low ~ high and say nothing.")
    p.add_argument("--kappa-zi", type=float, default=0.0)
    p.add_argument("--kappa-zf", type=float, nargs="+", default=[1.05, 1.85],
                   help="integration upper redshift PER --kappa-nz entry (single "
                        "value broadcasts): 1.05 holds >=95%% of bin1's n(z), "
                        "1.85 ~99%% of bin4's.")
    p.add_argument("--kappa-lmax", type=int, default=1500,
                   help="angular power spectrum lmax for the kappa maps. Must match the "
                        "shells' own band limit for a fair comparison (1500 on the "
                        "N_side=512 footing) -- and must reach well past ell~350 or the "
                        "correction is invisible; see --kappa-nside.")
    p.add_argument("--kappa-max-cosmologies", type=int, default=3,
                   help="held-out cosmologies to build kappa maps for. This is THE "
                        "cost knob of the whole script: each one needs every usable "
                        "shell (~47 of 69 for zf=1.05) fully reconstructed by tiling "
                        "the sphere and integrating the flow ODE per patch. Since "
                        "the tiling geometry got cached (analysis.patch_tiling), the "
                        "GPU ODE integration is what dominates -- so bound this "
                        "rather than the (now cheap) tiling. 0 = all held-out.")
    args = p.parse_args()

    # Multi-GPU eval: launch with `torchrun --nproc_per_node=N apply_flow.py ...` to
    # split the two DOMINANT cost sections (the zbin-grid and --kappa full-sky
    # reconstructions -- both explicitly flagged as "the real cost knob" in their own
    # --help text below, since each held-out cosmology needs many independent flow-ODE
    # reconstructions) across N GPUs, one held-out cosmology's reconstruction work per
    # rank at a time. Everything else (patch-level diagnostics) is cheap and NOT
    # worth the added complexity of splitting -- it stays on rank 0 only, so with
    # world_size>1 the other ranks simply skip it instead of wastefully duplicating
    # it. Falls back to single-process/single-GPU exactly as before when launched
    # with plain `python` (world_size defaults to 1). See diffusion/apply_diffusion.py
    # for the sibling implementation of this same pattern.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist
        # Default NCCL collective timeout is 10 minutes -- far too short here: rank 0
        # does substantial EXTRA serial work (the rank-0-only patch/single-cosmology
        # diagnostics below, each a full-sky tiled ODE reconstruction) before it ever
        # reaches the first dist.all_gather_object() in the zbin-grid section, while
        # the other ranks reach it almost immediately and wait -- see
        # diffusion/apply_diffusion.py's identical fix (production timeout, job
        # 4247489). 4h comfortably covers a single rank's full eval workload within
        # this job's 10-12h walltime.
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location=dev)
    cfg = ckpt.get("args", {})
    # older checkpoints predate this flag entirely (no cosmo_mlp in their state_dict),
    # so the fallback must be False, not the FlowUNet default of True.
    use_cosmo_cond = bool(cfg.get("use_cosmo_cond", False))
    # High-pass residual formulation (see flow_model.py): MUST use the exact cutoff
    # the target was built with at train time, so THE CHECKPOINT IS AUTHORITATIVE --
    # deliberately not a CLI flag, which could silently disagree with training and
    # break the composition guarantee without any error.
    #
    # HARD ERROR, not a default, on either of the two older formulations:
    #   no hp_scale_mpc_h -> trained with a fixed ANGULAR cutoff, correct for at most
    #                        one shell; applying a per-shell filter to it would filter
    #                        at a scale it never saw.
    #   no hp_cutoff      -> older still: trained to flow the WHOLE field, so
    #                        compose_corrected would highpass an output that was never
    #                        pinned at large scales, silently corrupting it (job
    #                        4247908: kappa Cl collapsed to ~0.55 at low ell).
    # Three checkpoint generations, distinguished by which keys their saved args have:
    #   hp_scale_mpc_h present  -> current, per-shell comoving cutoff. Accepted.
    #   hp_cutoff only          -> retired FIXED ANGULAR filter. Its residual has one
    #                              onset multipole for every shell, so filtering it
    #                              per-shell here would cut at a scale it never saw.
    #   neither                 -> v1 FULL-FIELD model, trained to output the whole
    #                              high map. compose_corrected would add a whole field
    #                              on top of the low map (~4x power, ~40x variance) --
    #                              plausible-looking garbage, not a crash, which is
    #                              exactly why this refuses instead of falling back.
    # Note hp_cutoff is ABSENT from current checkpoints (the flag was removed), so
    # presence of hp_cutoff cannot be the test for "new enough" -- hp_scale_mpc_h is.
    if not float(cfg.get("hp_scale_mpc_h", 0.0)):
        if "hp_cutoff" in cfg:
            raise SystemExit(
                f"{args.model} was trained with the retired FIXED ANGULAR cutoff "
                f"(hp_cutoff={cfg.get('hp_cutoff')}, hp_transition={cfg.get('hp_transition')}). "
                f"That filter is gone: the cutoff is now per-shell, from a fixed comoving "
                f"scale. Retrain with the current train_flow.py (--hp-scale-mpc-h 17.0).")
        raise SystemExit(
            f"{args.model} predates the high-pass residual formulation entirely (no "
            f"'hp_scale_mpc_h' and no 'hp_cutoff' in its saved args) -- it was trained on "
            f"the WHOLE field, so composing its output as low + highpass(sample) would "
            f"silently corrupt it. Retrain with the current train_flow.py.")
    hp_scale_mpc_h = float(cfg["hp_scale_mpc_h"])
    # Absent => a pre-2026-07-21 checkpoint would have already failed the guards
    # above, so any checkpoint reaching here is new enough that 'delta' (the
    # current default) is the correct fallback, not a guess.
    space = str(cfg.get("space", "delta"))
    net = FlowUNet(in_channels=1, out_channels=1,
                   base_channels=int(cfg.get("base_channels", 32)),
                   time_emb_dim=int(cfg.get("time_emb_dim", 128)),
                   use_cosmo_cond=use_cosmo_cond).to(dev)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"[eval] checkpoint use_cosmo_cond={use_cosmo_cond} "
          f"hp_scale_mpc_h={hp_scale_mpc_h} space={space}", flush=True)

    def _cut(shell_chi):
        """Cutoff from comoving distance -- the same quantity training took from the
        patch metadata (flow_model.cutoff_from_chi). Scalar, or a (B,) tensor when a
        batch mixes shells."""
        chi = shell_chi if torch.is_tensor(shell_chi) else float(shell_chi)
        return cutoff_from_chi(chi, hp.nside2resol(nside_src, arcmin=True), hp_scale_mpc_h)

    def sample(x0, cosmo_z, shell_chi):
        """Integrate the flow ODE from x0 (=low field) and compose the corrected
        field via compose_corrected -- the hard guarantee that large scales end up
        EXACTLY x0's, independent of any drift the finite-step ODE accumulated. Every
        call site below goes through this, so none of them can forget the compose
        step (see flow_model.compose_corrected's docstring)."""
        raw = sample_ode(net, x0, n_steps=args.steps, cosmo_z=cosmo_z, amp=args.amp)
        # Same filter as training. If these two ever disagree the composition
        # guarantee (large scales end up EXACTLY x0's) silently stops holding.
        return compose_corrected(x0, raw - x0, _cut(shell_chi))

    # held-out cosmologies = our test data
    _, val_idx, val_cosmos = split_by_cosmo(args.patch_dir, args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed)

    # one held-out patch per requested shell, via the patch dataset's own metadata
    # (shell_idx field) -- for the example_patches.png rows specifically. Also
    # reused below by the full-sky section (cosmo params + shell redshift lookup).
    meta = np.load(Path(args.patch_dir) / "metadata.npy")

    nside_src = int(np.unique(meta["nside_source"])[0])
    if len(np.unique(meta["nside_source"])) > 1:
        raise RuntimeError(f"--patch-dir mixes multiple source nsides: "
                           f"{np.unique(meta['nside_source'])} -- the high-pass cutoff "
                           f"is derived from the pixel scale, so one value cannot serve both.")

    val_shell_idx = meta["shell_idx"][val_idx]
    example_pick, example_shells = [], []
    for s in args.example_shells:
        cand = val_idx[val_shell_idx == s]
        if len(cand) == 0:
            print(f"[eval] WARNING: no held-out patch for shell {s}, skipping", flush=True)
            continue
        example_pick.append(int(rng.choice(cand)))
        example_shells.append(s)

    # Patch-level diagnostics are cheap (a handful of batched GPU passes over a few
    # hundred patches) -- not worth splitting across ranks. rank 0 only when
    # distributed, so the other ranks don't waste time redundantly regenerating the
    # same files.
    if rank == 0:
        pick = val_idx[rng.permutation(len(val_idx))[:args.n_eval]]
        ds = PatchDataset(args.patch_dir, pick)
        print(f"[eval] {len(val_idx)} held-out patches from {len(val_cosmos)} cosmologies "
              f"{val_cosmos}; evaluating {len(pick)} | ODE steps={args.steps}", flush=True)

        chi_all = torch.tensor([ds[i]["shell_com"] for i in range(len(ds))], device=dev)                            # (N,) Mpc/h, per patch
        low_all = torch.stack([ds[i]["low"] for i in range(len(ds))])       # (N,1,H,W) raw, CPU
        high_all = torch.stack([ds[i]["high"] for i in range(len(ds))])
        cosmo_z_all = stack_cosmo_z(ds, dev, low_all.dtype)                 # (N,8)
        # ODE-sample in mini-batches: the full n_eval stack through an 8-level UNet at once
        # is needlessly memory-heavy (and fragile on a shared/partially-occupied GPU) --
        # only the final comparison needs everything gathered, not the forward pass.
        mb = args.eval_batch
        low_f_parts, high_f_parts, corr_f_parts = [], [], []
        for b in range(0, low_all.shape[0], mb):
            lo = low_all[b:b + mb].to(dev); hi = high_all[b:b + mb].to(dev)
            lo_f, hi_f = transform_pair(lo, hi, space)
            with torch.no_grad():
                co_f = sample(lo_f, cosmo_z_all[b:b + mb], chi_all[b:b + mb].to(lo_f.dtype))
            low_f_parts.append(lo_f); high_f_parts.append(hi_f); corr_f_parts.append(co_f)
        low_f = torch.cat(low_f_parts); high_f = torch.cat(high_f_parts)
        corr_f = torch.cat(corr_f_parts)

        mse_low = torch.mean((low_f - high_f) ** 2).item()
        mse_corr = torch.mean((corr_f - high_f) ** 2).item()

        pr_low_stack = radial_power_batch(low_f)
        pr_corr_stack = radial_power_batch(corr_f)
        pr_high_stack = radial_power_batch(high_f)
        pr_low, pr_corr, pr_high = pr_low_stack.mean(0), pr_corr_stack.mean(0), pr_high_stack.mean(0)
        k = np.arange(len(pr_high))
        with np.errstate(divide="ignore", invalid="ignore"):
            lo_r_stack, co_r_stack = pr_low_stack / pr_high_stack, pr_corr_stack / pr_high_stack
        lo_r, co_r = pr_low / pr_high, pr_corr / pr_high

        def band(r, a, b):
            return float(np.nanmean(r[a:b]))
        nb = len(k)
        print(f"[eval] MSE({space}) low={mse_low:.4e} corrected={mse_corr:.4e} "
              f"({100*(1-mse_corr/max(mse_low,1e-12)):+.0f}%)", flush=True)
        print(f"[eval] power ratio to high | low  (mid,high-k)=({band(lo_r,nb//4,nb//2):.3f},"
              f"{band(lo_r,nb//2,nb):.3f}) | corrected=({band(co_r,nb//4,nb//2):.3f},"
              f"{band(co_r,nb//2,nb):.3f})   [target 1.0]", flush=True)

        # --- power spectrum ratio plot ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].loglog(k[1:], pr_low[1:], ":", color="seagreen", label="low (DISCO)")
        ax[0].loglog(k[1:], pr_corr[1:], "-", color="steelblue", label="flow-corrected")
        ax[0].loglog(k[1:], pr_high[1:], "--", color="tomato", label="high (CosmoGrid)")
        ax[0].set_xlabel("radial wavenumber bin"); ax[0].set_ylabel("power")
        ax[0].legend(fontsize=8); ax[0].set_title("2D-FFT radial power (mean over held-out patches)")
        ax[1].semilogx(k[1:], lo_r[1:], ":", color="seagreen", label="low/high")
        ax[1].semilogx(k[1:], co_r[1:], "-", color="steelblue", label="corrected/high")
        ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0.4, 1.6)
        ax[1].set_xlabel("radial wavenumber bin"); ax[1].set_ylabel("power ratio to high")
        ax[1].legend(fontsize=8); ax[1].set_title("power ratio (the target metric)")
        fig.tight_layout(); fig.savefig(out_dir / "power_spectrum_ratio.png", dpi=150)
        plt.close(fig)

        # --- pctile-band ratio plot: same statistic as above, but showing the per-patch
        # spread (not just the mean) so a systematic bias is distinguishable from noise ---
        plot_pctile_band_ratio(
            k, {"low / high (baseline, no model)": lo_r_stack, "flow pred / high": co_r_stack},
            out_dir / "patch_power_ratio_pctile_band.png", xlabel="radial wavenumber bin",
            ylim=(0.4, 1.6),
            title=f"power ratio: flow vs baseline ({len(pick)} val patches, 16-84th pctile band)")

        # --- example triptychs + per-patch power ratio (4th column), one row per shell ---
        # shared with every other pipeline (analysis.plotting.plot_example_patch_grid) so
        # the figure is visually identical by construction, not by convention.
        ns = len(example_pick)
        ex_ds = PatchDataset(args.patch_dir, np.array(example_pick, dtype=np.int64))
        ex_low = torch.stack([ex_ds[i]["low"] for i in range(ns)]).to(dev)
        ex_high = torch.stack([ex_ds[i]["high"] for i in range(ns)]).to(dev)
        ex_cosmo_z = stack_cosmo_z(ex_ds, dev, ex_low.dtype)
        ex_low_f, ex_high_f = transform_pair(ex_low, ex_high, space)
        with torch.no_grad():
            ex_chi = torch.tensor([ex_ds[i]["shell_com"] for i in range(ns)],
                                  device=dev, dtype=ex_low_f.dtype)
            ex_corr_f = sample(ex_low_f, ex_cosmo_z, ex_chi)

        rows = [(f"shell {example_shells[i]}", ex_low_f[i, 0].cpu().numpy(),
                ex_corr_f[i, 0].cpu().numpy(), ex_high_f[i, 0].cpu().numpy())
               for i in range(ns)]
        # of view the nside=2048 transfer figures use, making them comparable.
        plot_example_patch_grid(rows, out_dir / "example_patches.png",
                                corrected_label="flow-corrected",
                                suptitle=f"held-out test patches ({space} overdensity) + "
                                         "per-patch power ratio")

        # --- per-shell moments + histograms (raw counts) -- pools ALL held-out patches of
        # each --example-shells shell (capped by --n-stat-patches), not just the single
        # visual example patch above, since the one-point PDF is much noisier per-patch. ---
        n_stat = args.n_stat_patches
        moment_shells, mom_low, mom_corr, mom_high, hist_rows = [], [], [], [], []
        for s in example_shells:
            cand = val_idx[val_shell_idx == s]
            cand = cand[rng.permutation(len(cand))[:n_stat]]
            sd = PatchDataset(args.patch_dir, cand)
            s_low = torch.stack([sd[i]["low"] for i in range(len(sd))]).to(dev)
            s_high = torch.stack([sd[i]["high"] for i in range(len(sd))]).to(dev)
            s_cosmo_z = stack_cosmo_z(sd, dev, s_low.dtype)
            s_low_mean = s_low.mean(dim=(2, 3), keepdim=True)
            s_low_f, _ = transform_pair(s_low, s_high, space)
            with torch.no_grad():
                s_chi = torch.tensor([sd[i]["shell_com"] for i in range(len(sd))],
                                     device=dev, dtype=s_low_f.dtype)
                s_corr_f = sample(s_low_f, s_cosmo_z, s_chi)
            s_corr_raw = field_to_counts(s_corr_f, s_low_mean, space)

            low_np, high_np, corr_np = s_low.cpu().numpy(), s_high.cpu().numpy(), s_corr_raw.cpu().numpy()
            moment_shells.append(s)
            # ONE moments() call PER PATCH (not one pooling all n_stat patches together),
            # so plot_moments_vs_shell can draw the patch-to-patch spread as a pctile
            # band instead of collapsing it into a single number per shell.
            mom_low.append([moments(low_np[k]) for k in range(low_np.shape[0])])
            mom_corr.append([moments(corr_np[k]) for k in range(corr_np.shape[0])])
            mom_high.append([moments(high_np[k]) for k in range(high_np.shape[0])])
            hist_rows.append((f"shell {s}", low_np.ravel(), corr_np.ravel(), high_np.ravel()))
            print(f"[eval] shell {s} moments (n={len(sd)} patches): low var median="
                  f"{np.median([m['variance'] for m in mom_low[-1]]):.3g} flow-pred var median="
                  f"{np.median([m['variance'] for m in mom_corr[-1]]):.3g} high var median="
                  f"{np.median([m['variance'] for m in mom_high[-1]]):.3g}", flush=True)

        plot_moments_vs_shell(
            moment_shells, {"low": mom_low, "high (true)": mom_high, "flow pred": mom_corr},
            out_dir / "patch_moments_vs_shell.png",
            suptitle=f"moments vs. shell depth. Median + 16-84th pctile band across "
                     f"up to {n_stat} held-out patches/shell (raw counts)")
        plot_histogram_grid(
            hist_rows, out_dir / "patch_example_histograms.png", corrected_label="flow-corrected",
            suptitle=f"held-out patches, raw pixel-count histogram per shell ({n_stat} patches/shell)")

    # ============= optional: full-sky reconstruction + REAL angular Cl =============
    # Skipped entirely unless --data-root is given. Tiles the WHOLE sphere (not one
    # flat patch) via analysis.patch_tiling and runs the real spherical-harmonic
    # transform (analysis.full_sky.od_cl) -- the genuine "how does it behave at very
    # high ell" answer the flat 2D-FFT power ratio above structurally cannot give
    # (bounded by that patch's own Nyquist wavenumber).
    if args.data_root:
        nside = nside_src
        nside_centers = args.nside_centers or auto_nside_centers(nside, args.fullsky_patch_size)
        lmax = min(args.lmax, 3 * nside - 1)
        ells = np.arange(lmax + 1)

        def lookup_cosmo_z_fs(cosmo_name: str, shell_idx: int) -> np.ndarray:
            """(8,) cosmo_z vector for one full-sky shell/cosmology -- cosmo params
            constant per cosmology, redshift constant per shell_idx, both looked up
            from the patch dataset's own metadata (authoritative, same source
            train_flow.py used)."""
            rows_cosmo = meta[meta["cosmo"] == cosmo_name]
            if len(rows_cosmo) == 0:
                raise ValueError(f"lookup_cosmo_z_fs: no patch-dir metadata row for "
                                 f"cosmology {cosmo_name!r}")
            cosmo_vec = np.array([[rows_cosmo[0][f] for f in COSMO_FIELDS]], dtype=np.float32)
            rows_shell = meta[meta["shell_idx"] == shell_idx]
            if len(rows_shell) == 0:
                raise ValueError(f"lookup_cosmo_z_fs: no patch-dir metadata row for "
                                 f"shell_idx={shell_idx} (any cosmology)")
            z = np.array([0.5 * (rows_shell[0]["lower_z"] + rows_shell[0]["upper_z"])], dtype=np.float32)
            return cosmo_z_vector(cosmo_vec, z)[0]

        def lookup_chi_fs(shell_idx: int) -> float:
            """Comoving distance (Mpc/h) of one full-sky shell, from the patch
            metadata -- the same field the dataset feeds training, so the filter here
            is identical to the one the target was built with."""
            rows_shell = meta[meta["shell_idx"] == shell_idx]
            if len(rows_shell) == 0:
                raise ValueError(f"lookup_chi_fs: no patch-dir metadata row for "
                                 f"shell_idx={shell_idx} (any cosmology)")
            return float(rows_shell[0]["shell_com"])

        def make_predict_batch_fs(cosmo_z_vec: np.ndarray | None, shell_chi: float):
            cosmo_z_t = None if cosmo_z_vec is None else torch.from_numpy(cosmo_z_vec).to(dev)

            def predict_batch(low_batch: np.ndarray) -> np.ndarray:
                low_t = torch.from_numpy(low_batch).unsqueeze(1).to(dev)
                low_f, low_mean = low_to_field(low_t, space)
                cz = (None if cosmo_z_t is None else
                     cosmo_z_t.unsqueeze(0).expand(low_t.shape[0], -1).to(low_f.dtype))
                with torch.no_grad():
                    pred_f = sample(low_f, cz, shell_chi)
                return field_to_counts(pred_f, low_mean, space).squeeze(1).cpu().numpy()
            return predict_batch

        def reconstruct(low_shell, cosmo_name, shell_idx):
            cosmo_z_vec = lookup_cosmo_z_fs(cosmo_name, shell_idx) if use_cosmo_cond else None
            predict_batch = make_predict_batch_fs(cosmo_z_vec, lookup_chi_fs(shell_idx))
            return reconstruct_shell(predict_batch, low_shell, nside_centers,
                                     args.fullsky_patch_size, args.eval_batch,
                                     taper_power=args.taper_power)


        # --- example_patches.png, REDRAWN from the full sky so that all three
        # pipelines show the SAME patches (analysis/example_patches.patch_plan).
        # Overwrites the patch-dataset version written earlier in this script: that
        # one indexed arbitrary tiles out of the training dataset, so its rows showed
        # different cosmologies and different sky positions than transfer's figure and
        # the three could not be compared row by row. Drawing from the reconstructed
        # sphere at the plan's (cosmology, centre pixel, rotation) fixes that, and is
        # also the more honest picture: it is the tiled+blended product a user of this
        # pipeline actually gets, not a single isolated network input.
        if rank == 0:
            try:
                ep_plan = patch_plan(args.seed, args.patch_shells, 1,
                                     val_cosmos, nside)
                reso_arcmin = hp.nside2resol(nside, arcmin=True)
                ep_rows = []
                for s, cname, cipix, psi in ep_plan:
                    c_dir = Path(args.data_root) / cname / args.run
                    lo_all = np.load(c_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
                    hi_all = np.load(c_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
                    lo_sh = np.asarray(lo_all[s], np.float32)
                    print(f"[eval] example patch: reconstructing {cname} shell {s}...",
                          flush=True)
                    co_sh = reconstruct(lo_sh, cname, s)
                    hi_sh = np.asarray(hi_all[s], np.float32)
                    lo_p = extract_patch(lo_sh.astype(np.float64), nside, cipix, psi,
                                         args.patch_size, reso_arcmin)
                    co_p = extract_patch(np.asarray(co_sh, np.float64), nside, cipix, psi,
                                         args.patch_size, reso_arcmin)
                    hi_p = extract_patch(hi_sh.astype(np.float64), nside, cipix, psi,
                                         args.patch_size, reso_arcmin)
                    lo_l, hi_l = log1p_delta_pair(lo_p, hi_p)
                    co_l, _ = log1p_delta_pair(co_p, hi_p)
                    ep_rows.append((f"shell {s} ({cname})", lo_l, co_l, hi_l))
                plot_example_patch_grid(ep_rows, out_dir / "example_patches.png",
                                        corrected_label='flow-corrected')
            except Exception as e:
                print(f"[eval] full-sky example patches FAILED ({type(e).__name__}: {e}); "
                      f"keeping the patch-dataset version", flush=True)

        # --- single-cosmology diagnostics: standalone cl_shell*.png + full-sky
        # moments/histograms (--cosmo, default: first held-out cosmology). Modest cost
        # (a few shells x a few cosmologies) next to the zbin-grid/kappa sections below
        # -- rank 0 only. ---
        if rank == 0:
            fs_cosmo = args.cosmo
            if fs_cosmo is None:
                fs_cosmo = val_cosmos[0]
                print(f"[eval] --cosmo not given for full-sky reconstruction, using "
                      f"held-out cosmology {fs_cosmo} (full held-out set: {val_cosmos})", flush=True)

            run_dir = Path(args.data_root) / fs_cosmo / args.run
            low_full_all = np.load(run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
            high_full_all = np.load(run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
            print(f"[eval] full-sky: {fs_cosmo}/{args.run} nside={nside} "
                  f"nside_centers={nside_centers} ({12*nside_centers**2:,} centers) "
                  f"patch_size={args.fullsky_patch_size}", flush=True)

            for s in args.shell_indices:
                low_shell = np.asarray(low_full_all[s], np.float32)
                high_shell = np.asarray(high_full_all[s], np.float32)
                print(f"[eval] full-sky shell {s}: tiling + predicting...", flush=True)
                pred_filled = reconstruct(low_shell, fs_cosmo, s)
                cl_lo, cl_c, cl_hi = od_cl(low_shell, lmax), od_cl(pred_filled, lmax), od_cl(high_shell, lmax)
                plot_cl_shell(s, ells, cl_lo, cl_c, cl_hi, out_dir / f"cl_shell{s:03d}.png")

            if example_shells:
                # POOLED across held-out cosmologies (capped by --max-cosmologies, same
                # knob the Cl-ratio-by-zbin-grid section below uses) -- one reconstruct()
                # per (cosmology, shell), so plot_moments_vs_shell can draw the
                # cosmology-to-cosmology spread as a pctile band instead of the old
                # single-cosmology curve. Brings this in line with transfer/sphereflow's
                # own full-sky moments (both already pool across their held-out set).
                fs_cosmos = list(val_cosmos[:args.max_cosmologies])
                print(f"[eval] full-sky moments: pooling {len(fs_cosmos)} of "
                      f"{len(val_cosmos)} held-out cosmologies (max-cosmologies="
                      f"{args.max_cosmologies}): {fs_cosmos}", flush=True)
                mom_low_per_s = {s: [] for s in example_shells}
                mom_pred_per_s = {s: [] for s in example_shells}
                mom_high_per_s = {s: [] for s in example_shells}
                hist_low_per_s = {s: [] for s in example_shells}
                hist_pred_per_s = {s: [] for s in example_shells}
                hist_high_per_s = {s: [] for s in example_shells}
                for c in fs_cosmos:
                    c_run_dir = Path(args.data_root) / c / args.run
                    c_low_all = np.load(c_run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r")
                    c_high_all = np.load(c_run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r")
                    for s in example_shells:
                        low_shell = np.asarray(c_low_all[s], np.float32)
                        high_shell = np.asarray(c_high_all[s], np.float32)
                        print(f"[eval] full-sky {c} shell {s}: tiling + predicting "
                              f"(moments/hist)...", flush=True)
                        pred_filled = reconstruct(low_shell, c, s)
                        mom_low_per_s[s].append(moments(low_shell))
                        mom_high_per_s[s].append(moments(high_shell))
                        mom_pred_per_s[s].append(moments(pred_filled))
                        hist_low_per_s[s].append(low_shell.ravel())
                        hist_pred_per_s[s].append(pred_filled.ravel())
                        hist_high_per_s[s].append(high_shell.ravel())

                fs_mom_low = [mom_low_per_s[s] for s in example_shells]
                fs_mom_high = [mom_high_per_s[s] for s in example_shells]
                fs_mom_pred = [mom_pred_per_s[s] for s in example_shells]
                fs_hist_rows = [(f"shell {s}", np.concatenate(hist_low_per_s[s]),
                                np.concatenate(hist_pred_per_s[s]),
                                np.concatenate(hist_high_per_s[s])) for s in example_shells]
                plot_moments_vs_shell(
                    example_shells, {"low": fs_mom_low, "high (true)": fs_mom_high, "flow pred": fs_mom_pred},
                    out_dir / "moments_vs_shell.png",
                    suptitle=f"moments vs. shell depth -- full-sky reconstruction (raw counts). "
                            f"Median + 16-84th pctile band ACROSS {len(fs_cosmos)} held-out "
                            f"cosmologies (one sample per cosmology; see heldout_cosmo_params.png "
                            f"for their parameters).")
                plot_histogram_grid(
                    fs_hist_rows, out_dir / "example_histograms.png", corrected_label="flow-corrected",
                    suptitle=f"full-sky raw pixel-count histogram per shell, pooled over "
                            f"{len(fs_cosmos)} held-out cosmologies: {fs_cosmos}")

        # --- example_full_sky.png: Cl-ratio-by-redshift-bin pctile grid. One row per
        # held-out cosmology (up to --max-cosmologies), one column per redshift/shell
        # bin (analysis.full_sky.zbin_shell_samples) -- percentile band across
        # --n-shells-per-zbin sampled shells per bin. No images (already in
        # example_patches.png); this is purely the aggregate two-point check. ---
        # n_shells_total computed independently of the rank-0-only block above (every
        # rank needs it for the parallel zbin-grid/kappa sections below).
        n_shells_total = np.load(
            Path(args.data_root) / val_cosmos[0] / args.run / f"low_shells_nside={nside}.npy",
            mmap_mode="r").shape[0]
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
        # label bins by redshift rather than shell index (see zbin_shell_samples):
        # the shell grid is shared across CosmoGridV1, so any run's table serves
        zbins = zbin_shell_samples(n_shells_total, args.zbin_start, args.n_zbins,
                                   args.n_shells_per_zbin,
                                   shell_z=shell_redshifts(
                                       Path(args.data_root) / val_cosmos[0] / args.run))
        grid_cosmos = list(val_cosmos[:args.max_cosmologies])
        if rank == 0:
            print(f"[eval] full-sky: Cl-ratio-by-redshift-bin grid for {len(grid_cosmos)} "
                  f"cosmologies {grid_cosmos} x {len(zbins)} bins {[b[0] for b in zbins]} "
                  f"(split across {world_size} rank(s))", flush=True)

        # DOMINANT cost of this script (see --max-cosmologies help): each cosmology
        # needs len(zbins)*n_shells_per_zbin independent full-sky flow-ODE
        # reconstructions. Splitting whole COSMOLOGIES round-robin across ranks (the
        # first cut of this) starves ranks whenever world_size > len(grid_cosmos) --
        # e.g. 4 GPUs but --max-cosmologies 3 leaves one rank with literally nothing
        # to do for this entire section (confirmed in production on diffusion's
        # sibling script, job 4247847). Split at SHELL granularity instead -- flatten
        # every (cosmology, bin, shell) reconstruction into one task list and
        # round-robin THAT across ranks, so work divides evenly no matter how
        # grid_cosmos compares to world_size.
        tasks = [(ci, bi, si, int(s))
                for ci, c in enumerate(grid_cosmos)
                for bi, (bin_label, shells) in enumerate(zbins)
                for si, s in enumerate(shells)]
        shell_cache: dict[int, tuple] = {}   # cosmo_idx -> (c_low_all, c_high_all), per rank

        def _shell_arrays(ci):
            if ci not in shell_cache:
                c_run_dir = Path(args.data_root) / grid_cosmos[ci] / args.run
                shell_cache[ci] = (
                    np.load(c_run_dir / f"low_shells_nside={nside}.npy", mmap_mode="r"),
                    np.load(c_run_dir / f"high_shells_nside={nside}.npy", mmap_mode="r"))
            return shell_cache[ci]

        local_grid = []
        for ci, bi, si, s in tasks[rank::world_size]:
            c = grid_cosmos[ci]
            c_low_all, c_high_all = _shell_arrays(ci)
            low_shell = np.asarray(c_low_all[s], np.float32)
            high_shell = np.asarray(c_high_all[s], np.float32)
            bin_label, shells = zbins[bi]
            print(f"[eval][rank{rank}] full-sky {c} shell {s} ({bin_label}): "
                  f"tiling + predicting...", flush=True)
            pred_filled = reconstruct(low_shell, c, s)
            cl_lo = od_cl(low_shell, lmax); cl_c = od_cl(pred_filled, lmax)
            cl_hi = od_cl(high_shell, lmax)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo_ratio = cl_lo / cl_hi; co_ratio = cl_c / cl_hi
            local_grid.append((ci, bi, si, lo_ratio, co_ratio))

        if distributed:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_grid)
            flat_grid = [item for part in gathered for item in part]
        else:
            flat_grid = local_grid

        if rank == 0:
            # Reassemble the (cosmology x bin x shell) grid from the flat, task-order
            # -independent results -- pre-size every panel's ratio stacks so the merge
            # doesn't depend on which rank finished which task first.
            panels_per_cosmo = [[None] * len(zbins) for _ in grid_cosmos]
            for ci, bi, si, lo_ratio, co_ratio in flat_grid:
                bin_label, shells = zbins[bi]
                if panels_per_cosmo[ci][bi] is None:
                    panels_per_cosmo[ci][bi] = (bin_label, shells, ells,
                                                [None] * len(shells), [None] * len(shells))
                panels_per_cosmo[ci][bi][3][si] = lo_ratio
                panels_per_cosmo[ci][bi][4][si] = co_ratio
            grid = [(f"{c}/{args.run}",
                    [(bl, sh, el, np.array(lo), np.array(co))
                     for bl, sh, el, lo, co in panels_per_cosmo[ci]])
                   for ci, c in enumerate(grid_cosmos)]
            # Per-(cosmology, shell) ratio dump: the pooled percentile band above is
            # anonymous -- when it grows a lobe (e.g. the 2026-07-22 hp0.05/0.12
            # runs' 16th-pctile power-loss on faint shells) there is no way to tell
            # WHICH cosmology/shell drives it from the png alone. Keys:
            # low_{cosmo}_s{shell} / corrected_{cosmo}_s{shell}, each Cl(x)/Cl(true).
            np.savez_compressed(
                out_dir / "cl_ratio_by_zbin_data.npz",
                cosmos=np.array(grid_cosmos), lmax=lmax,
                bin_labels=np.array([bl for bl, _ in zbins]),
                bin_shells=np.array([sh for _, sh in zbins], dtype=object),
                **{f"{kind}_{grid_cosmos[ci]}_s{int(zbins[bi][1][si])}": r
                   for ci, bi, si, lo_r, co_r in flat_grid
                   for kind, r in (("low", lo_r), ("corrected", co_r))})
            plot_cl_ratio_pctile_grid(
                grid, out_dir / "cl_ratio_by_zbin_grid.png",
                corrected_label="corrected (flow) / true (after)",
                suptitle="Full-sky Cl ratio by redshift bin (flow)")

        # --- weak-lensing kappa map diagnostic (analysis.weak_lensing): reduces
        # the WHOLE usable lightcone into ONE kappa map each for low/corrected/high,
        # for EVERY held-out cosmology -- the comparison this whole pipeline is
        # ultimately in service of (kappa maps, not individual shells). Each
        # cosmology's OWN params.yml is loaded fresh (H0, Om, Ob, O_nu, ... --
        # "further values" beyond the 6 COSMO_FIELDS this script otherwise uses)
        # since weak_lensing_ufalcon.py's hardcoded example cosmology was checked
        # and does NOT match the lightcone it actually loads. ---
        if args.kappa:
            kappa_cosmos = (list(val_cosmos) if args.kappa_max_cosmologies <= 0
                           else list(val_cosmos[:args.kappa_max_cosmologies]))
            # One full diagnostic set PER n(z) BIN (2026-07-16; default DES-Y3
            # metacal bin1 + bin4): bin1 peaks at z~0.23 (low-z, hardest
            # correction), bin4 at z~0.98 (less correction, most cosmological
            # weight). Shells are RECONSTRUCTED once per cosmology up to
            # max(--kappa-zf) -- the expensive part -- then each bin's kappa_map
            # integrates only its own [zi, zf] window (UFalcon skips shells
            # outside internally), so extra bins cost kappa_map calls only.
            nz_list = list(args.kappa_nz)
            zf_list = (list(args.kappa_zf) if len(args.kappa_zf) == len(nz_list)
                       else [args.kappa_zf[0]] * len(nz_list))
            kappa_tags = [_nz_tag(nz) for nz in nz_list]
            zf_max = max(zf_list)
            if rank == 0:
                print(f"[eval] kappa: building kappa maps for {len(kappa_cosmos)} of "
                      f"{len(val_cosmos)} held-out cosmologies {kappa_cosmos} | n(z) bins: "
                      + ", ".join(f"{t} (zf={zf:g})" for t, zf in zip(kappa_tags, zf_list))
                      + f" | zi={args.kappa_zi}, nside={args.kappa_nside} "
                      + f"(split across {world_size} rank(s))", flush=True)

            # THE most expensive section (see --kappa-max-cosmologies help): every
            # usable shell (often dozens per cosmology) needs a fresh reconstruction.
            # Whole-COSMOLOGY round-robin splitting starves ranks the same way the
            # zbin grid did above whenever world_size > len(kappa_cosmos) -- split at
            # SHELL granularity instead. Cheap, GPU-free per-cosmology metadata (usable
            # shells, z bounds, cosmo params) is computed identically on every rank
            # first; only the expensive reconstruct() calls are distributed. The
            # reconstructed shells themselves (not just derived Cl/moments) are
            # gathered -- at these nside's (<=2048) that is at most a few GB total,
            # trivial next to the GPU time saved -- because kappa_map integrates ALL
            # of a cosmology's usable shells together, so whichever rank assembles the
            # final kappa map needs the complete per-cosmology stack.
            cosmo_meta = []
            for c in kappa_cosmos:
                c_run_dir = Path(args.data_root) / c / args.run
                cosmo_params = weak_lensing.load_cosmo_yaml(c_run_dir)
                shell_info = np.load(c_run_dir / "compressed_shells.npz",
                                     allow_pickle=True, mmap_mode="r")["shell_info"]
                lower_z_all = shell_info["lower_z"]; upper_z_all = shell_info["upper_z"]
                usable = np.where(weak_lensing.usable_shell_mask(
                    lower_z_all, upper_z_all, args.kappa_zi, zf_max))[0]
                cosmo_meta.append(dict(c=c, run_dir=c_run_dir, cosmo_params=cosmo_params,
                                       usable=usable, lower_z=lower_z_all[usable],
                                       upper_z=upper_z_all[usable]))
            tasks = [(ci, sp, int(s)) for ci, m in enumerate(cosmo_meta)
                    for sp, s in enumerate(m["usable"])]
            if rank == 0:
                print(f"[eval] kappa: {len(tasks)} shell-reconstruction tasks across "
                      f"{len(kappa_cosmos)} cosmologies, split across {world_size} "
                      f"rank(s)", flush=True)
            kappa_shell_cache: dict[int, tuple] = {}

            def _kappa_shell_arrays(ci):
                if ci not in kappa_shell_cache:
                    rd = cosmo_meta[ci]["run_dir"]
                    kappa_shell_cache[ci] = (
                        np.load(rd / f"low_shells_nside={nside}.npy", mmap_mode="r"),
                        np.load(rd / f"high_shells_nside={nside}.npy", mmap_mode="r"))
                return kappa_shell_cache[ci]

            local_recon = []
            for ci, sp, s in tasks[rank::world_size]:
                c_low_all, _ = _kappa_shell_arrays(ci)
                print(f"[eval][rank{rank}] kappa {cosmo_meta[ci]['c']}: "
                      f"reconstructing shell {s}...", flush=True)
                corr_shell = reconstruct(np.asarray(c_low_all[s], np.float32),
                                        cosmo_meta[ci]["c"], s).astype(np.float64)
                local_recon.append((ci, sp, corr_shell))

            if distributed:
                gathered = [None] * world_size
                dist.all_gather_object(gathered, local_recon)
                flat_recon = [item for part in gathered for item in part]
            else:
                flat_recon = local_recon

            if rank == 0:
                kappa_cosmo_labels = [f"{m['c']}/{args.run}" for m in cosmo_meta]
                kacc = {t: {k: [] for k in ("cl_low", "cl_corr", "cl_high",
                                            "mom_low", "mom_corr", "mom_high")}
                       for t in kappa_tags}
                for ci, m in enumerate(cosmo_meta):
                    n_usable = len(m["usable"])
                    corr_shells = np.empty((n_usable,) + flat_recon[0][2].shape, dtype=np.float64)
                    for ci2, sp, corr_shell in flat_recon:
                        if ci2 == ci:
                            corr_shells[sp] = corr_shell
                    c_low_all, c_high_all = _kappa_shell_arrays(ci)
                    low_shells = np.stack([np.asarray(c_low_all[int(s)], np.float64) for s in m["usable"]])
                    high_shells = np.stack([np.asarray(c_high_all[int(s)], np.float64) for s in m["usable"]])
                    for nz, zf, tag in zip(nz_list, zf_list, kappa_tags):
                        kw = dict(nside=args.kappa_nside, zi=args.kappa_zi, zf=zf)
                        kappa_low = weak_lensing.kappa_map(low_shells, m["lower_z"], m["upper_z"], m["cosmo_params"], nz, **kw)
                        kappa_corr = weak_lensing.kappa_map(corr_shells, m["lower_z"], m["upper_z"], m["cosmo_params"], nz, **kw)
                        kappa_high = weak_lensing.kappa_map(high_shells, m["lower_z"], m["upper_z"], m["cosmo_params"], nz, **kw)
                        a = kacc[tag]
                        a["cl_low"].append(weak_lensing.kappa_cl(kappa_low, args.kappa_lmax))
                        a["cl_corr"].append(weak_lensing.kappa_cl(kappa_corr, args.kappa_lmax))
                        a["cl_high"].append(weak_lensing.kappa_cl(kappa_high, args.kappa_lmax))
                        a["mom_low"].append(moments(kappa_low)); a["mom_corr"].append(moments(kappa_corr))
                        a["mom_high"].append(moments(kappa_high))
                    print(f"[eval] kappa {m['c']}: done ({n_usable} shells)", flush=True)

                kappa_ells = np.arange(args.kappa_lmax + 1)
                # Same two views of the kappa Cl as transfer/apply_transfer.py and
                # sphereflow/apply_sphere_flow.py, per n(z) bin (_bin1/_bin4/...).
                for nz, zf, tag in zip(nz_list, zf_list, kappa_tags):
                    a = kacc[tag]
                    kappa_suptitle_common = (f"{len(kappa_cosmo_labels)} held-out cosmologies (flow) | "
                                             f"n(z)={Path(nz).name} | "
                                             f"z in [{args.kappa_zi:g},{zf:g}]"
                                             f" | kappa nside={args.kappa_nside}, lmax={args.kappa_lmax}")
                    plot_kappa_cl_grid(
                        kappa_cosmo_labels, kappa_ells, a["cl_low"], a["cl_corr"], a["cl_high"],
                        out_dir / f"kappa_cl_per_cosmology_{tag}.png", corrected_label="flow-corrected",
                        suptitle=f"weak-lensing kappa Cl per cosmology, {kappa_suptitle_common}")

                    with np.errstate(divide="ignore", invalid="ignore"):
                        k_lo_stack = np.array([lo / hi for lo, hi in zip(a["cl_low"], a["cl_high"])])
                        k_co_stack = np.array([co / hi for co, hi in zip(a["cl_corr"], a["cl_high"])])
                    # Persist the kappa Cl stacks behind this figure. Without them a plot-only
                    # change (y-range, fonts, labels) costs a full inference rerun -- exactly what
                    # the missing cl_ratio_by_zbin_data.npz cost once already.
                    np.savez(out_dir / f"kappa_cl_data_{tag}.npz", ells=kappa_ells,
                             cl_low=np.asarray(a["cl_low"]), cl_corr=np.asarray(a["cl_corr"]),
                             cl_high=np.asarray(a["cl_high"]),
                             cosmo_labels=np.array([str(c) for c in kappa_cosmo_labels]))
                    plot_pctile_band_ratio(
                        kappa_ells[1:], {"low / high (baseline, no model)": k_lo_stack[:, 1:],
                                         "flow-corrected / high": k_co_stack[:, 1:]},
                        out_dir / f"kappa_cl_pctile_band_{tag}.png", xlabel=r"$\ell$", ylim=None,
                        title=f"weak-lensing kappa Cl ratio to truth ({tag}) -- median + 16-84th "
                              f"pctile band ACROSS {len(kappa_cosmo_labels)} held-out cosmologies")

                    plot_kappa_moments_scatter(
                        kappa_cosmo_labels, a["mom_low"], a["mom_corr"], a["mom_high"],
                        out_dir / f"kappa_moments_scatter_{tag}.png", corrected_label="flow-corrected",
                        suptitle=f"weak-lensing kappa map moments, {kappa_suptitle_common}")
                    # persist the same numbers so the three pipelines can be put on
                    # one axes later without re-running any reconstruction
                    weak_lensing.save_kappa_moment_summary(
                        out_dir / f"kappa_moments_{tag}.npz", kappa_cosmo_labels,
                        a["mom_low"], a["mom_corr"], a["mom_high"],
                        method_label="unet-flow", tag=tag)
    # ============= end optional full-sky section =============

    if rank == 0:
        print(f"[eval] figures -> {out_dir}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
