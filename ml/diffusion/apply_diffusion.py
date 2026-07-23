#!/usr/bin/env python3
"""Apply a trained EDM diffusion model to HELD-OUT test patches and evaluate.

DELIBERATE structural duplicate of unet/apply_flow.py (see
feedback-decoupled-pipeline-modules memory): same plotting/tiling/full-sky-power
calls, same --example-shells convention, same output filenames, so its figures are
directly comparable file-by-file to unet's, transfer's, and sphereflow's. Only the
model + sampling differ: DenoiserUNet+EDMPrecond and model.sample_heun instead of
FlowUNet and flow_model.sample_ode.

Test data = the validation (held-out COSMOLOGY) split of the patch dataset. For each
patch we run the EDM Heun ODE sampler from x_T ~ N(0, sigma_max^2 I) conditioned on
low (log1p-delta) to draw a high-pass RESIDUAL, then compose the corrected map as
low + highpass(sample) (model.compose_corrected -- large scales pinned to the low
map, see model.py's docstring for why), and compare corrected vs the true high by:
  * the 2D-FFT radial power spectrum ratio (the flat-patch analogue of the C_ell
    ratio -- does the correction restore small-scale power?),
  * example low / corrected / high patch triptychs (judge by eye).

Because a diffusion SAMPLE (not a conditional mean) carries the full high-field
variance, the corrected SMALL-SCALE power should track the truth where a
deterministic regressor would sag -- same rationale as the flow models. Unlike the
first (full-field) version of this pipeline, the large scales are NOT generated: they
come from the low map, so the full-sky / kappa Cl at low ell is preserved by
construction (the earlier full-field run destroyed it -- see diffusion-pipeline-build
memory).

SHARED EVAL SET: same statistics/shared analysis/ plotting code/shells/kappa
resolution as transfer/apply_transfer.py, sphereflow/apply_sphere_flow.py, and
unet/apply_flow.py:
    example_patches.png, patch_power_ratio_pctile_band.png,
    moments_vs_shell.png / example_histograms.png, cl_ratio_by_zbin_grid.png
    (--data-root), kappa_cl_per_cosmology.png, kappa_cl_pctile_band.png,
    kappa_moments_scatter.png (--kappa).
power_spectrum_ratio.png (mean 2D-FFT power, loglog) is extra, unique to this script
(same as unet's).

  python apply_diffusion.py --patch-dir <dir> --model <run>/best.pt --out-dir <run>/eval
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DenoiserUNet, EDMPrecond, sample_heun, compose_corrected  # noqa: E402
from dataset import (PatchDataset, split_by_cosmo, transform_pair,  # noqa: E402
                     low_to_field, field_to_counts, cosmo_z_vector, COSMO_FIELDS)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.plotting import (plot_example_patch_grid, plot_pctile_band_ratio,  # noqa: E402
                               plot_histogram_grid, plot_moments_vs_shell,
                               plot_cl_shell, plot_cl_ratio_pctile_grid,
                               plot_kappa_cl_grid, plot_kappa_moments_scatter)
from analysis.moments import moments                      # noqa: E402
from analysis.patch_tiling import auto_nside_centers, reconstruct_shell  # noqa: E402
from analysis.full_sky import od_cl, zbin_shell_samples    # noqa: E402
from analysis import weak_lensing                          # noqa: E402


def stack_cosmo_z(ds: PatchDataset, dev, dtype) -> torch.Tensor:
    """Stack (cosmo, z) across every sample in ds, on dev, into the (N,8) cosmo_z
    tensor DenoiserUNet expects (see dataset.cosmo_z_vector)."""
    cosmo = torch.stack([ds[i]["cosmo"] for i in range(len(ds))]).to(dev, dtype)
    z = torch.tensor([ds[i]["z"] for i in range(len(ds))], device=dev, dtype=dtype)
    return cosmo_z_vector(cosmo, z)


def radial_power_batch(imgs: torch.Tensor, n_bins: int = None):
    """(B,1,H,W) -> (B,n_bins) PER-SAMPLE 2D-FFT power per radial wavenumber bin.
    Same batched-on-GPU implementation as unet/apply_flow.py's."""
    B, _, H, W = imgs.shape
    dev = imgs.device
    n_bins = n_bins or H // 2
    fy = torch.fft.fftfreq(H, device=dev) * H
    fx = torch.fft.rfftfreq(W, device=dev) * W
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = torch.clamp((r / r.max() * (n_bins - 1)).long(), 0, n_bins - 1).view(-1)
    counts = torch.bincount(bins, minlength=n_bins).clamp(min=1).float()
    f = torch.fft.rfft2(imgs.squeeze(1))
    power = (f.real ** 2 + f.imag ** 2).view(B, -1)
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
    p.add_argument("--n-eval", type=int, default=512, help="held-out patches to evaluate")
    p.add_argument("--eval-batch", type=int, default=256,
                   help="mini-batch size for the diffusion sampling pass (memory control). "
                        "UNTESTED bump from the old default of 64 -- matches unet's own "
                        "bump, watch for OOM on the first real run.")
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                   help="run each Heun precond() call under bf16 autocast (default: on). "
                        "See model.sample_heun's docstring -- Heun does up to 2 precond() "
                        "calls/step, so this is the most expensive sampler of the three "
                        "pipelines and benefits the most from amp.")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="disable bf16 autocast (fp32 sampling, slower).")
    p.add_argument("--example-shells", type=int, nargs="+", default=[5, 10, 15, 30, 50],
                   help="shell indices to show as rows in example_patches.png. Default "
                        "matches transfer/unet/sphereflow's, so all four pipelines' "
                        "figures line up.")
    p.add_argument("--steps", type=int, default=32,
                   help="Heun ODE steps (EDM sampler). Much higher than the flow "
                        "models' default of ~8: a real diffusion trajectory is not a "
                        "straight line, so it needs materially more function "
                        "evaluations -- see model.sample_heun.")
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0, help="EDM Karras schedule exponent")
    p.add_argument("--taper-power", type=float, default=32.0,
                   help="blend-weight sharpening for the full-sky tiling (see "
                        "analysis.patch_tiling.tile_and_predict). Each sky pixel is "
                        "covered by ~16 overlapping tiles whose INDEPENDENT diffusion "
                        "samples get weight-averaged; at the default cosine taper "
                        "(power 1) that shrinks the stochastic part of the correction "
                        "to ~0.41 of its amplitude (the conditional-mean part "
                        "survives), which showed up as a huge downward Cl percentile "
                        "band on faint shells. Sharpening toward nearest-tile-wins "
                        "restores it. Measured on the real delta-512 checkpoint, "
                        "faint shell 5 corrected/high at ell 800-1535: p=1 0.597, "
                        "p=16 0.781, p=32 0.800, p=128 0.816 (saturates; low/high "
                        "baseline 0.535) -- 32 is the knee. Only the full-sky "
                        "sections use this; 1.0 reproduces the old blend.")
    p.add_argument("--n-stat-patches", type=int, default=64,
                   help="held-out patches PER --example-shells shell to pool for "
                        "patch_moments_vs_shell.png / patch_example_histograms.png")
    p.add_argument("--data-root", default=None,
                   help="prepare_maps.py output (full-sky low/high shell stacks). "
                        "If given, ALSO reconstructs the whole sphere and computes "
                        "the real full-sky Cl -> cl_shell*.png, cl_ratio_by_zbin_grid.png, "
                        "moments_vs_shell.png, example_histograms.png")
    p.add_argument("--cosmo", default=None,
                   help="cosmology for the full-sky reconstruction; default: first "
                        "held-out (val) cosmology")
    p.add_argument("--run", default="run_0")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[0, 34, 68],
                   help="shells to render as individual cl_shell*.png; pass with no "
                        "values to skip these")
    p.add_argument("--nside-centers", type=int, default=None,
                   help="default: auto-scaled from the data's nside (see "
                        "analysis.patch_tiling.auto_nside_centers)")
    p.add_argument("--fullsky-patch-size", type=int, default=256,
                   help="gnomonic tile size for full-sky reconstruction -- should "
                        "match the patch size this checkpoint was trained on")
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--zbin-start", type=int, default=5,
                   help="first shell in the Cl-ratio-by-redshift-bin grid. 5 (was 0, "
                        "changed 2026-07-20): shell 0 shows weird behaviour in the "
                        "grid's first panel (user-observed, job 4247908) -- excluded "
                        "while shells 1-4 stay out too since the panel is defined by "
                        "its start, not a per-shell mask. Same default as "
                        "transfer/unet/sphereflow so all pipelines' grids keep "
                        "binning the SAME shells.")
    p.add_argument("--n-zbins", type=int, default=3)
    p.add_argument("--n-shells-per-zbin", type=int, default=5)
    p.add_argument("--max-cosmologies", type=int, default=3)
    p.add_argument("--kappa", action="store_true",
                   help="build weak-lensing kappa maps (low/corrected/high) for "
                        "EVERY held-out cosmology and compute their Cl + moments. "
                        "Requires --data-root. Expensive: see module docstring.")
    p.add_argument("--kappa-nz", nargs="+",
                   default=["/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin1.txt",
                            "/capstor/scratch/cscs/damrein/redshift_distribution/desy3_nz_metacal_bin4.txt"])
    p.add_argument("--kappa-nside", type=int, default=1024)
    p.add_argument("--kappa-zi", type=float, default=0.0)
    p.add_argument("--kappa-zf", type=float, nargs="+", default=[1.05, 1.85])
    p.add_argument("--kappa-lmax", type=int, default=2048)
    p.add_argument("--kappa-max-cosmologies", type=int, default=3)
    args = p.parse_args()

    # Multi-GPU eval: launch with `torchrun --nproc_per_node=N apply_diffusion.py ...`
    # to split the two DOMINANT cost sections (zbin-grid, kappa -- both explicitly
    # flagged as "the real cost knob" in their own comments below, since each held-out
    # cosmology needs dozens of independent Heun ODE reconstructions) across N GPUs,
    # one held-out cosmology's full reconstruction work per rank at a time. Everything
    # else (patch-level diagnostics, the cheap single-cosmology full-sky sections) is
    # NOT worth the added complexity of splitting -- it stays on rank 0 only, so with
    # world_size>1 those sections simply run once instead of being wastefully
    # duplicated on every rank. Falls back to single-process/single-GPU exactly as
    # before when launched with plain `python` (world_size defaults to 1).
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist
        # Default NCCL collective timeout is 10 minutes -- far too short here: rank 0
        # does substantial EXTRA serial work (the rank-0-only patch/single-cosmology
        # diagnostics below, each a full-sky tiled Heun reconstruction) before it ever
        # reaches the first dist.all_gather_object() in the zbin-grid section, while
        # the other ranks reach it almost immediately and wait -- exactly the
        # DistBackendError timeout seen in production (job 4247489, rank 3 timed out
        # after 600s waiting on rank 0). 4h comfortably covers a single rank's full
        # eval workload within this job's 10-12h walltime.
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location=dev)
    cfg = ckpt.get("args", {})
    use_cosmo_cond = bool(cfg.get("use_cosmo_cond", False))
    sigma_data = float(cfg.get("sigma_data", 0.5))
    # High-pass residual formulation (see model.py): MUST use the exact cutoff the
    # target was built with at train time, so the checkpoint is authoritative.
    #
    # HARD ERROR, not a default: a checkpoint without hp_cutoff is a v1 FULL-FIELD
    # model (trained to output the whole high map, before the residual redesign).
    # Composing its output as low + highpass(sample) is semantically wrong -- it adds
    # a whole field on top of the low map instead of a small-scale residual, giving
    # ~4x the correct power and ~40x the correct variance. That is NOT a crash, it is
    # plausible-looking garbage, which is exactly why this must refuse rather than
    # fall back (it silently produced a full bogus eval for the nside=512 v1
    # checkpoint before this guard existed -- see diffusion-pipeline-build memory).
    if "hp_cutoff" not in cfg:
        raise SystemExit(
            f"{args.model} is a v1 FULL-FIELD checkpoint (no 'hp_cutoff' in its saved "
            f"args; sigma_data={sigma_data:.4f} was measured on the full field, not the "
            f"residual). This script implements the HIGH-PASS RESIDUAL formulation and "
            f"would silently produce meaningless results for it (~4x power, ~40x "
            f"variance). Use a checkpoint trained by the current train_diffusion.py, "
            f"e.g. outputs/diffusionruns/diffusion_cosmogridv1_nside2048_patch256_"
            f"n100000_ch32_b32_e40/best.pt, or retrain this configuration.")
    hp_cutoff = float(cfg["hp_cutoff"])
    hp_transition = float(cfg["hp_transition"])
    # Which field the residual was modelled in. Absent => a pre-2026-07-18 checkpoint,
    # which was necessarily log1p -- so this default is CORRECT for old checkpoints,
    # not a silent guess (unlike the hp_cutoff case guarded above).
    space = str(cfg.get("space", "log1p"))
    net = DenoiserUNet(in_channels=2, out_channels=1,
                       base_channels=int(cfg.get("base_channels", 32)),
                       noise_emb_dim=int(cfg.get("noise_emb_dim", 128)),
                       use_cosmo_cond=use_cosmo_cond).to(dev)
    precond = EDMPrecond(net, sigma_data=sigma_data).to(dev)
    precond.load_state_dict(ckpt["model"])
    precond.eval()
    print(f"[eval] checkpoint use_cosmo_cond={use_cosmo_cond} sigma_data={sigma_data:.4f} "
          f"hp_cutoff={hp_cutoff} hp_transition={hp_transition} space={space}", flush=True)

    def sample(cond, cosmo_z, noise=None):
        """Draw a high-pass-residual diffusion sample and RETURN THE COMPOSED corrected
        log-map (low + highpass(sample)) -- so every call site below gets the corrected
        map directly, exactly as the old full-field sample() returned it. The large
        scales of the returned map are pinned to `cond` (the low map) by construction
        (see model.compose_corrected).

        noise: optional unit-variance initial state (see model.sample_heun). Left None
        for the standalone PATCH diagnostics below (independent draws are correct
        there -- those patches don't overlap and are never blended); supplied by the
        FULL-SKY path, where overlapping tiles must share one global sphere noise
        field or the blend averages the generated structure away."""
        r = sample_heun(precond, cond, n_steps=args.steps, cosmo_z=cosmo_z,
                        sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho,
                        noise=noise, amp=args.amp)
        return compose_corrected(cond, r, hp_cutoff, hp_transition)

    # held-out cosmologies = our test data
    _, val_idx, val_cosmos = split_by_cosmo(args.patch_dir, args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed)

    meta = np.load(Path(args.patch_dir) / "metadata.npy")
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
    # hundred patches, not the dozens-of-full-shell-reconstructions cost the full-sky
    # sections below pay) -- not worth splitting across ranks. rank 0 only when
    # distributed, so the other ranks don't waste time redundantly regenerating the
    # same files.
    if rank == 0:
        pick = val_idx[rng.permutation(len(val_idx))[:args.n_eval]]
        ds = PatchDataset(args.patch_dir, pick)
        print(f"[eval] {len(val_idx)} held-out patches from {len(val_cosmos)} cosmologies "
              f"{val_cosmos}; evaluating {len(pick)} | Heun steps={args.steps}", flush=True)

        low_all = torch.stack([ds[i]["low"] for i in range(len(ds))])
        high_all = torch.stack([ds[i]["high"] for i in range(len(ds))])
        cosmo_z_all = stack_cosmo_z(ds, dev, low_all.dtype)
        mb = args.eval_batch
        low_log_parts, high_log_parts, corr_log_parts = [], [], []
        for b in range(0, low_all.shape[0], mb):
            lo = low_all[b:b + mb].to(dev); hi = high_all[b:b + mb].to(dev)
            lo_log, hi_log = transform_pair(lo, hi, space)
            with torch.no_grad():
                co_log = sample(lo_log, cosmo_z_all[b:b + mb])
            low_log_parts.append(lo_log); high_log_parts.append(hi_log); corr_log_parts.append(co_log)
        low_log = torch.cat(low_log_parts); high_log = torch.cat(high_log_parts)
        corr_log = torch.cat(corr_log_parts)

        mse_low = torch.mean((low_log - high_log) ** 2).item()
        mse_corr = torch.mean((corr_log - high_log) ** 2).item()

        pr_low_stack = radial_power_batch(low_log)
        pr_corr_stack = radial_power_batch(corr_log)
        pr_high_stack = radial_power_batch(high_log)
        pr_low, pr_corr, pr_high = pr_low_stack.mean(0), pr_corr_stack.mean(0), pr_high_stack.mean(0)
        k = np.arange(len(pr_high))
        with np.errstate(divide="ignore", invalid="ignore"):
            lo_r_stack, co_r_stack = pr_low_stack / pr_high_stack, pr_corr_stack / pr_high_stack
        lo_r, co_r = pr_low / pr_high, pr_corr / pr_high

        def band(r, a, b):
            return float(np.nanmean(r[a:b]))
        nb = len(k)
        print(f"[eval] MSE(log-delta) low={mse_low:.4e} corrected={mse_corr:.4e} "
              f"({100*(1-mse_corr/max(mse_low,1e-12)):+.0f}%)", flush=True)
        print(f"[eval] power ratio to high | low  (mid,high-k)=({band(lo_r,nb//4,nb//2):.3f},"
              f"{band(lo_r,nb//2,nb):.3f}) | corrected=({band(co_r,nb//4,nb//2):.3f},"
              f"{band(co_r,nb//2,nb):.3f})   [target 1.0]", flush=True)

        # --- power spectrum ratio plot ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].loglog(k[1:], pr_low[1:], ":", color="seagreen", label="low (DISCO)")
        ax[0].loglog(k[1:], pr_corr[1:], "-", color="steelblue", label="diffusion-corrected")
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

        plot_pctile_band_ratio(
            k, {"low / high (baseline, no model)": lo_r_stack, "diffusion pred / high": co_r_stack},
            out_dir / "patch_power_ratio_pctile_band.png", xlabel="radial wavenumber bin",
            ylim=(0.4, 1.6),
            title=f"power ratio: diffusion vs baseline ({len(pick)} val patches, 16-84th pctile band)")

        ns = len(example_pick)
        ex_ds = PatchDataset(args.patch_dir, np.array(example_pick, dtype=np.int64))
        ex_low = torch.stack([ex_ds[i]["low"] for i in range(ns)]).to(dev)
        ex_high = torch.stack([ex_ds[i]["high"] for i in range(ns)]).to(dev)
        ex_cosmo_z = stack_cosmo_z(ex_ds, dev, ex_low.dtype)
        ex_low_log, ex_high_log = transform_pair(ex_low, ex_high, space)
        with torch.no_grad():
            ex_corr_log = sample(ex_low_log, ex_cosmo_z)

        rows = [(f"shell {example_shells[i]}", ex_low_log[i, 0].cpu().numpy(),
                ex_corr_log[i, 0].cpu().numpy(), ex_high_log[i, 0].cpu().numpy())
               for i in range(ns)]
        plot_example_patch_grid(rows, out_dir / "example_patches.png",
                                corrected_label="diffusion-corrected",
                                suptitle="held-out test patches (log1p overdensity) + "
                                         "per-patch power ratio")

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
            s_low_log, _ = transform_pair(s_low, s_high, space)
            with torch.no_grad():
                s_corr_log = sample(s_low_log, s_cosmo_z)
            s_corr_raw = field_to_counts(s_corr_log, s_low_mean, space)

            low_np, high_np, corr_np = s_low.cpu().numpy(), s_high.cpu().numpy(), s_corr_raw.cpu().numpy()
            moment_shells.append(s)
            # ONE moments() call PER PATCH (see unet/apply_flow.py's identical fix) so
            # plot_moments_vs_shell's pctile-band contract (list-of-samples per shell,
            # not one pooled dict) is satisfied here too.
            mom_low.append([moments(low_np[k]) for k in range(low_np.shape[0])])
            mom_corr.append([moments(corr_np[k]) for k in range(corr_np.shape[0])])
            mom_high.append([moments(high_np[k]) for k in range(high_np.shape[0])])
            hist_rows.append((f"shell {s}", low_np.ravel(), corr_np.ravel(), high_np.ravel()))
            print(f"[eval] shell {s} moments (n={len(sd)} patches): low var median="
                  f"{np.median([m['variance'] for m in mom_low[-1]]):.3g} diffusion-pred var median="
                  f"{np.median([m['variance'] for m in mom_corr[-1]]):.3g} high var median="
                  f"{np.median([m['variance'] for m in mom_high[-1]]):.3g}", flush=True)

        plot_moments_vs_shell(
            moment_shells, {"low": mom_low, "high (true)": mom_high, "diffusion pred": mom_corr},
            out_dir / "patch_moments_vs_shell.png",
            suptitle=f"moments vs. shell depth. Median + 16-84th pctile band across "
                     f"up to {n_stat} held-out patches/shell (raw counts)")
        plot_histogram_grid(
            hist_rows, out_dir / "patch_example_histograms.png", corrected_label="diffusion-corrected",
            suptitle=f"held-out patches, raw pixel-count histogram per shell ({n_stat} patches/shell)")

    # ============= optional: full-sky reconstruction + REAL angular Cl =============
    if args.data_root:
        nside = int(np.unique(meta["nside_source"])[0])
        if len(np.unique(meta["nside_source"])) > 1:
            raise RuntimeError(f"--patch-dir mixes multiple source nsides: {np.unique(meta['nside_source'])}")
        nside_centers = args.nside_centers or auto_nside_centers(nside, args.fullsky_patch_size)
        lmax = min(args.lmax, 3 * nside - 1)
        ells = np.arange(lmax + 1)

        def lookup_cosmo_z_fs(cosmo_name: str, shell_idx: int) -> np.ndarray:
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

        def make_predict_batch_fs(cosmo_z_vec: np.ndarray | None):
            """Per-tile prediction for the full-sky reconstruction.

            NOTE ON TILE NOISE (2026-07-18): each tile deliberately draws its OWN white
            x_T. A "shared global sphere noise" variant was tried -- crop every tile's
            initial state from one white (npix,) field via its HEALPix indices, so
            overlapping tiles agree and survive patch_tiling's averaging blend -- and it
            was MEASURABLY HARMFUL, so it was reverted. The gnomonic index map duplicates
            ~17% of pixels within a tile, making the crop ~8% lag-1 correlated instead of
            white; the denoiser is trained on white x_T, so that is out-of-distribution
            input and it degraded high-k power from 1.013 (white) to 0.799 (cropped) --
            worse than applying no model at all (0.956). Do NOT reintroduce shared noise
            without first making the crop genuinely white in the PATCH grid.

            Returns raw counts. In the default 'delta' space the correction is
            ADDITIVE (corrected = delta_low + highpass(sample)), so averaging
            overlapping tiles in counts space is linear and unbiased -- no special
            blending space is needed. (A log-space/geometric-mean blend was tried while
            the model was still multiplicative and EXPLODED on sparse shells:
            zero-count pixels clamp to log(1e-6) = -13.8 and dominate the average,
            giving Cl ratios ~3e4. Do not reintroduce it.)
            """
            cosmo_z_t = None if cosmo_z_vec is None else torch.from_numpy(cosmo_z_vec).to(dev)

            def predict_batch(low_batch: np.ndarray) -> np.ndarray:
                low_t = torch.from_numpy(low_batch).unsqueeze(1).to(dev)
                low_f, low_mean = low_to_field(low_t, space)
                cz = (None if cosmo_z_t is None else
                     cosmo_z_t.unsqueeze(0).expand(low_t.shape[0], -1).to(low_f.dtype))
                with torch.no_grad():
                    pred_f = sample(low_f, cz)
                return field_to_counts(pred_f, low_mean, space).squeeze(1).cpu().numpy()
            return predict_batch

        def reconstruct(low_shell, cosmo_name, shell_idx):
            """predict_batch returns raw COUNTS, so patch_tiling's blend and its
            default gap fill (raw DISCO counts for the ~0.006% pixels no tile covers)
            are both already in the right space -- no fill_map override needed."""
            cosmo_z_vec = lookup_cosmo_z_fs(cosmo_name, shell_idx) if use_cosmo_cond else None
            predict_batch = make_predict_batch_fs(cosmo_z_vec)
            return reconstruct_shell(predict_batch, low_shell, nside_centers,
                                     args.fullsky_patch_size, args.eval_batch,
                                     taper_power=args.taper_power)

        # Single-cosmology diagnostics (cl_shell*.png, example_shells moments/hist):
        # cheap relative to the zbin-grid/kappa sections below (one cosmology, a
        # handful of shells) -- rank 0 only.
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
                fs_mom_low, fs_mom_pred, fs_mom_high, fs_hist_rows = [], [], [], []
                for s in example_shells:
                    low_shell = np.asarray(low_full_all[s], np.float32)
                    high_shell = np.asarray(high_full_all[s], np.float32)
                    print(f"[eval] full-sky shell {s}: tiling + predicting (moments/hist)...", flush=True)
                    pred_filled = reconstruct(low_shell, fs_cosmo, s)
                    # single-element lists: plot_moments_vs_shell now expects a list of
                    # per-sample dicts per shell (see analysis/plotting.py); this section
                    # stays single-cosmology (unlike unet/transfer/sphereflow's pooled
                    # full-sky moments), so there is exactly one sample per shell here.
                    fs_mom_low.append([moments(low_shell)]); fs_mom_high.append([moments(high_shell)])
                    fs_mom_pred.append([moments(pred_filled)])
                    fs_hist_rows.append((f"shell {s}", low_shell.ravel(), pred_filled.ravel(), high_shell.ravel()))
                import yaml as _yaml
                _cp = _yaml.safe_load((run_dir / "params.yml").read_text()) if (run_dir / "params.yml").exists() else {}
                _note = (f"{fs_cosmo}:  s8={_cp.get('s8', float('nan')):<8.4g} Om={_cp.get('Om', float('nan')):<8.4g} "
                         f"Ob={_cp.get('Ob', float('nan')):<8.4g} H0={_cp.get('H0', float('nan')):<7.4g} "
                         f"ns={_cp.get('ns', float('nan')):<8.4g} w0={_cp.get('w0', float('nan')):<8.4g}") if _cp else None
                plot_moments_vs_shell(
                    example_shells, {"low": fs_mom_low, "high (true)": fs_mom_high, "diffusion pred": fs_mom_pred},
                    out_dir / "moments_vs_shell.png", note=_note,
                    suptitle=f"moments vs. shell depth -- full-sky reconstruction (raw counts), "
                            f"ONE cosmology: {fs_cosmo}/{args.run} (parameters below)")
                plot_histogram_grid(
                    fs_hist_rows, out_dir / "example_histograms.png", corrected_label="diffusion-corrected",
                    suptitle=f"full-sky raw pixel-count histogram per shell\n{fs_cosmo}/{args.run}")

        # n_shells_total computed independently of the rank-0-only block above (every
        # rank needs it for the parallel zbin-grid/kappa sections below) -- read fresh
        # from the first held-out cosmology's own shell stack rather than reusing
        # rank 0's low_full_all, which does not exist on other ranks.
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
        zbins = zbin_shell_samples(n_shells_total, args.zbin_start, args.n_zbins,
                                   args.n_shells_per_zbin)
        grid_cosmos = list(val_cosmos[:args.max_cosmologies])
        if rank == 0:
            print(f"[eval] full-sky: Cl-ratio-by-redshift-bin grid for {len(grid_cosmos)} "
                  f"cosmologies {grid_cosmos} x {len(zbins)} bins {[b[0] for b in zbins]} "
                  f"(split across {world_size} rank(s))", flush=True)

        # DOMINANT cost of this script (see module docstring / --max-cosmologies help):
        # each cosmology needs len(zbins)*n_shells_per_zbin independent full-sky Heun
        # reconstructions. Splitting whole COSMOLOGIES round-robin across ranks (the
        # first cut of this) starves ranks whenever world_size > len(grid_cosmos) --
        # e.g. 4 GPUs but --max-cosmologies 3 leaves one rank with literally nothing
        # to do for this entire section (confirmed in production, job 4247847: rank 3
        # sat idle the whole run). Split at SHELL granularity instead -- flatten every
        # (cosmology, bin, shell) reconstruction into one task list and round-robin
        # THAT across ranks, so work divides evenly no matter how grid_cosmos compares
        # to world_size (typically dozens of shell-tasks vs. a handful of GPUs).
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
                corrected_label="corrected (diffusion) / true (after)",
                suptitle="Full-sky Cl ratio by redshift bin (diffusion)")

        if args.kappa:
            kappa_cosmos = (list(val_cosmos) if args.kappa_max_cosmologies <= 0
                           else list(val_cosmos[:args.kappa_max_cosmologies]))
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
                for nz, zf, tag in zip(nz_list, zf_list, kappa_tags):
                    a = kacc[tag]
                    kappa_suptitle_common = (f"{len(kappa_cosmo_labels)} held-out cosmologies (diffusion) | "
                                             f"n(z)={Path(nz).name} | "
                                             f"z in [{args.kappa_zi:g},{zf:g}]"
                                             f" | kappa nside={args.kappa_nside}, lmax={args.kappa_lmax}")
                    plot_kappa_cl_grid(
                        kappa_cosmo_labels, kappa_ells, a["cl_low"], a["cl_corr"], a["cl_high"],
                        out_dir / f"kappa_cl_per_cosmology_{tag}.png", corrected_label="diffusion-corrected",
                        suptitle=f"weak-lensing kappa Cl per cosmology, {kappa_suptitle_common}")

                    with np.errstate(divide="ignore", invalid="ignore"):
                        k_lo_stack = np.array([lo / hi for lo, hi in zip(a["cl_low"], a["cl_high"])])
                        k_co_stack = np.array([co / hi for co, hi in zip(a["cl_corr"], a["cl_high"])])
                    plot_pctile_band_ratio(
                        kappa_ells[1:], {"low / high (baseline, no model)": k_lo_stack[:, 1:],
                                         "diffusion-corrected / high": k_co_stack[:, 1:]},
                        out_dir / f"kappa_cl_pctile_band_{tag}.png", xlabel=r"$\ell$", ylim=(0.4, 1.6),
                        title=f"weak-lensing kappa Cl ratio to truth ({tag}) -- median + 16-84th "
                              f"pctile band ACROSS {len(kappa_cosmo_labels)} held-out cosmologies")

                    plot_kappa_moments_scatter(
                        kappa_cosmo_labels, a["mom_low"], a["mom_corr"], a["mom_high"],
                        out_dir / f"kappa_moments_scatter_{tag}.png", corrected_label="diffusion-corrected",
                        suptitle=f"weak-lensing kappa map moments, {kappa_suptitle_common}")
    # ============= end optional full-sky section =============

    if rank == 0:
        print(f"[eval] figures -> {out_dir}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
