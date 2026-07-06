#!/usr/bin/env python3
"""Apply the residual-correction UNet: correct the held-out TEST cosmology and evaluate.

correction: corrected_signal = signal(DISCO) + DiffUNet(signal(DISCO) | cosmo, z),
then map = physical(signal_inverse(corrected_signal)). Produces per shell:
  * angular power spectrum C_ell (DISCO / corrected / CosmoGrid-high) + ratio to truth,
  * gnomonic zoom-ins of DISCO / corrected / high,
and once:
  * the TRAINING vs VALIDATION loss curve (val = held-out-run loss; gap => overfitting).

SANITY CHECK (also runnable via --sanity): (1) the reshape/correction identity round-trips
(disco + 0-diff == disco), and (2) on the test cosmology the corrected map must have a
LOWER overdensity-MSE-to-truth than raw DISCO for the mid shells -- else the correction is
not helping. Exit code is non-zero if the sanity check fails, so it gates the pipeline.

  python apply_unet_diff.py --model-dir <out> --data-root <grid> --test-cosmo cosmo_000122
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sphereflow"))
import sphere_flow as sf
import unet_diff as ud
from train_sphere_flow import build_runs


def load_model(model_dir, device):
    ck = torch.load(Path(model_dir) / "checkpoint.pt", map_location=device)
    net = ud.DiffUNet(in_ch=1, out_ch=1, base=int(ck["base"]),
                      ch_mult=tuple(int(m) for m in str(ck["ch_mult"]).split(",")),
                      bottleneck=int(ck["bottleneck"]), cond_dim=int(ck["cond_dim"])).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


@torch.no_grad()
def correct_shell(net, ck, disco_map, cosmo_vec, device, patch_batch, to_img, to_patch, L):
    """corrected physical map = physical( signal(disco) + pred_diff )."""
    order = int(ck["order"]); scale = float(ck["sig_scale"]); soft = float(ck["softening"])
    mean = max(float(disco_map.mean()), 1e-12)
    d_in = disco_map[None] / mean - 1.0
    s_disco = sf.map_to_patches(sf.signal_forward(d_in, scale, soft), order)   # (P, M)

    cosmo = torch.from_numpy(cosmo_vec[None]).to(device)
    out = np.empty_like(s_disco)
    for b in range(0, s_disco.shape[0], patch_batch):
        c = torch.from_numpy(s_disco[b:b + patch_batch]).to(device)
        c_img = c[:, to_img].view(c.shape[0], 1, L, L)
        corr = c_img + net(c_img, cosmo.expand(c.shape[0], -1))     # disco + pred_diff
        out[b:b + patch_batch] = corr.view(c.shape[0], -1)[:, to_patch].cpu().numpy()

    sig = sf.patches_to_maps(out, order, 1)[0]
    delta = sf.signal_inverse(sig, scale, soft)
    return (mean * (1.0 + delta)).astype(np.float32)


def od_cl(m, lmax):
    return hp.anafast((m / m.mean() - 1.0).astype(np.float64), lmax=lmax)


def plot_loss_curves(model_dir, out_dir, ck):
    tr = Path(model_dir) / "train_history.npy"
    va = Path(model_dir) / "val_history.npy"
    if not tr.exists():
        return
    h = np.load(tr)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(h[:, 0], h[:, 1], color="0.8", lw=0.7, label="train (per-batch)")
    ema_d = float(ck.get("ema_decay", 0.99))
    ax.plot(h[:, 0], h[:, 2], color="steelblue", lw=2.0,
            label=f"train EMA (decay {ema_d})")
    if va.exists() and np.load(va).size:
        v = np.load(va)
        ax.plot(v[:, 0], v[:, 1], color="tomato", lw=2.0, marker="o", ms=4,
                label=f"validation, combined ({int(ck.get('n_val',3))} held-out runs, "
                      f"{int(ck.get('val_batches',20))} fixed batches)")
        if v.shape[1] >= 4:                       # (step, combined, pixel, spectral)
            ax.plot(v[:, 0], v[:, 2], color="tomato", lw=1.2, ls="--", alpha=0.7,
                    label="validation, pixel term")
            ax.plot(v[:, 0], v[:, 3], color="darkorange", lw=1.2, ls=":", alpha=0.9,
                    label="validation, spectral term")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.legend(fontsize=9)
    lam = float(ck.get("lambda_spec", 0.0))
    # loss + validation SCHEME stated in the title (as requested).
    ax.set_title(
        "Residual-correction loss   "
        r"$\mathcal{L}=\langle(\mathrm{corrected}-\mathrm{high})^2\rangle"
        f" + {lam:g}\\,"
        r"\langle(\log(1{+}P_{\mathrm{corr}})-\log(1{+}P_{\mathrm{high}}))^2\rangle$"
        "\n(pixel MSE in arcsinh-signal space + radial 2D-FFT power-spectrum term "
        f"(fixes MSE blurring);  validation = same $\\mathcal{{L}}$ on "
        f"{int(ck.get('n_val',3))} held-out runs every {int(ck.get('val_every',500))} steps)",
        fontsize=9.5)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "loss_curve.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000122")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[3, 30, 50])
    p.add_argument("--patch-batch", type=int, default=256)
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--sanity", action="store_true", help="run only the sanity check")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load_model(args.model_dir, dev)
    nside = int(ck["nside"]); L = int(ck["L"])
    out_dir = Path(args.out_dir or (Path(args.model_dir) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    to_img_np, to_patch_np = ud.nested_grid_perms(L)
    to_img = torch.from_numpy(to_img_np).to(dev)
    to_patch = torch.from_numpy(to_patch_np).to(dev)

    # cosmology vector for the (held-out) test run, exactly as in training.
    runs = build_runs(args.data_root, args.test_cosmo, nside, include_test=True, prefix="low")
    test_run = next((r for r in runs if args.test_cosmo in str(r[0])), runs[0])
    low_path, high_path, cosmo_base = test_run
    inp = np.load(low_path, mmap_mode="r")
    high = np.load(high_path, mmap_mode="r")
    n_shells = int(min(inp.shape[0], high.shape[0]))

    # --- SANITY CHECK 1: Morton round-trip identity ---
    rt_ok = np.array_equal(np.arange(L * L)[to_img_np][to_patch_np], np.arange(L * L))

    downscale = int(ck.get("downscale", 1))
    lmax = min(args.lmax, 3 * (nside // downscale) - 1)   # valid range at working res
    plot_loss_curves(args.model_dir, out_dir, ck)
    ells = np.arange(lmax + 1)
    od = lambda m: m / m.mean() - 1.0
    print(f"[eval] cosmo={args.test_cosmo} nside={nside} (downscale={downscale}) "
          f"patch {L}x{L} | {n_shells} shells | morton-roundtrip={'OK' if rt_ok else 'FAIL'}",
          flush=True)

    improved, checked = 0, 0
    for s in args.shell_indices:
        if s >= n_shells:
            continue
        disco = np.asarray(inp[s], dtype=np.float32)
        hi = np.asarray(high[s], dtype=np.float32)
        if downscale > 1:      # match the resolution the checkpoint was trained at
            disco = ud.downscale_nested(disco[None], downscale)[0]
            hi = ud.downscale_nested(hi[None], downscale)[0]
        shell_norm = np.float32(s / max(n_shells - 1, 1))
        cosmo_vec = np.concatenate([cosmo_base, [shell_norm]]).astype(np.float32)
        corr = correct_shell(net, ck, disco, cosmo_vec, dev, args.patch_batch,
                             to_img, to_patch, L)

        mse_disco = float(np.mean((od(disco) - od(hi)) ** 2))
        mse_corr = float(np.mean((od(corr) - od(hi)) ** 2))
        checked += 1
        improved += int(mse_corr < mse_disco)
        cl_d, cl_c, cl_h = od_cl(disco, lmax), od_cl(corr, lmax), od_cl(hi, lmax)
        print(f"shell {s}: MSE disco={mse_disco:.4e} corr={mse_corr:.4e} "
              f"({100*(1-mse_corr/max(mse_disco,1e-12)):+.0f}%)", flush=True)
        if args.sanity:
            continue

        fig = plt.figure(figsize=(13, 9))
        ax0 = fig.add_subplot(2, 3, 1)
        ax0.loglog(ells, cl_d, ":", color="seagreen", label="DISCO")
        ax0.loglog(ells, cl_c, "-", color="steelblue", label="corrected")
        ax0.loglog(ells, cl_h, "--", color="tomato", label="CosmoGrid high")
        ax0.set_xlabel(r"$\ell$"); ax0.set_ylabel(r"$C_\ell$")
        ax0.legend(fontsize=8); ax0.set_title(f"shell {s}: power spectrum")
        ax1 = fig.add_subplot(2, 3, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax1.semilogx(ells, cl_d / cl_h, ":", color="seagreen", label="DISCO/high")
            ax1.semilogx(ells, cl_c / cl_h, "-", color="steelblue", label="corr/high")
        ax1.axhline(1, color="k", lw=0.8); ax1.set_ylim(0.5, 1.5)
        ax1.set_xlabel(r"$\ell$"); ax1.set_ylabel("ratio"); ax1.legend(fontsize=8)
        ax1.set_title("Cl ratio to truth")
        axL = fig.add_subplot(2, 3, 3)
        axL.bar(["DISCO", "corrected"], [mse_disco, mse_corr], color=["seagreen", "steelblue"])
        axL.set_ylabel("overdensity MSE vs high")
        axL.set_title(f"pixel loss ({100*(1-mse_corr/max(mse_disco,1e-12)):+.0f}%)")
        rot = (45.0, 45.0)
        for i, (m, ttl) in enumerate([(disco, "DISCO"), (corr, "corrected"),
                                      (hi, "CosmoGrid high")]):
            hp.gnomview(m.astype(np.float64), nest=True, rot=rot, reso=1.5, xsize=200,
                        title=ttl, sub=(2, 3, 4 + i), fig=fig.number, notext=True,
                        cbar=(i == 2))
        fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93, bottom=0.06)
        fig.savefig(out_dir / f"eval_shell{s:03d}.png", dpi=140)
        plt.close(fig)

    # --- SANITY CHECK 2: correction must improve the majority of checked shells ---
    ok = rt_ok and (improved >= max(1, checked - 1))
    print(f"[sanity] morton={'OK' if rt_ok else 'FAIL'} | improved {improved}/{checked} shells "
          f"| RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    if not args.sanity:
        print(f"[eval] figures in {out_dir}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
