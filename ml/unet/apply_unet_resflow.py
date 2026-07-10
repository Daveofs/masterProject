#!/usr/bin/env python3
"""Sample the conditional residual flow and evaluate against CosmoGrid.

corrected_signal = signal(DISCO) + resid_scale * r,  r ~ p(residual | DISCO, cosmo, z)
then map = physical(signal_inverse(corrected_signal)).

Unlike the deterministic diff model (which must shrink its output to corr*target and so
systematically LOSES small-scale power), a flow SAMPLE carries the full residual
variance -- the corrected map should have the right small-scale Cl and realistic
non-Gaussian structure. It is not pixel-exact by construction, so we report the Cl
ratio (the statistic that matters) alongside the pixel MSE.
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
import unet_flow as uf
import unet_diff as ud
from train_sphere_flow import build_runs


def load_model(model_dir, device):
    ck = torch.load(Path(model_dir) / "checkpoint.pt", map_location=device)
    net = uf.SimpleUNet(in_ch=2, out_ch=1, base=int(ck["base"]),
                        ch_mult=tuple(int(m) for m in str(ck["ch_mult"]).split(",")),
                        cond_dim=int(ck["cond_dim"]), time_dim=int(ck["time_dim"])).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


@torch.no_grad()
def correct_shell(net, ck, disco_map, cosmo_vec, device, patch_batch, to_img, to_patch,
                  L, steps):
    order = int(ck["order"]); scale = float(ck["sig_scale"]); soft = float(ck["softening"])
    rscale = float(ck["resid_scale"])
    mean = max(float(disco_map.mean()), 1e-12)
    d_in = disco_map[None] / mean - 1.0
    s_disco = sf.map_to_patches(sf.signal_forward(d_in, scale, soft), order)

    cosmo = torch.from_numpy(cosmo_vec[None]).to(device)
    out = np.empty_like(s_disco)
    for b in range(0, s_disco.shape[0], patch_batch):
        c = torch.from_numpy(s_disco[b:b + patch_batch]).to(device)
        c_img = c[:, to_img].view(c.shape[0], 1, L, L)
        r = uf.sample_ode(net, c_img, cosmo.expand(c.shape[0], -1), steps=steps)
        corr = c_img + rscale * r                       # cond + resid_scale * sample
        out[b:b + patch_batch] = corr.view(c.shape[0], -1)[:, to_patch].cpu().numpy()

    sig = sf.patches_to_maps(out, order, 1)[0]
    delta = sf.signal_inverse(sig, scale, soft)
    return (mean * (1.0 + delta)).astype(np.float32)


def od_cl(m, lmax):
    return hp.anafast((m / m.mean() - 1.0).astype(np.float64), lmax=lmax)


def plot_loss_curves(model_dir, out_dir, ck):
    import textwrap
    tr, va = Path(model_dir) / "train_history.npy", Path(model_dir) / "val_history.npy"
    if not tr.exists():
        return
    h = np.load(tr)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(h[:, 0], h[:, 1], color="0.8", lw=0.7, label="train (per-batch)")
    ax.plot(h[:, 0], h[:, 2], color="steelblue", lw=2.0, label="train EMA")
    if va.exists() and np.load(va).size:
        v = np.load(va)
        ax.plot(v[:, 0], v[:, 1], color="tomato", lw=2.0, marker="o", ms=4,
                label="VALIDATION (held-out runs, never trained on)")
    ax.set_xlabel("step"); ax.set_ylabel("flow-matching loss"); ax.set_yscale("log")
    ax.legend(fontsize=9)
    desc = ("loss = < || v_theta(x_t, t | DISCO, cosmo, z) - (x1 - x0) ||^2 >, the rectified-flow "
            "velocity error. x1 = (signal(high) - signal(DISCO)) / resid_scale is the normalized "
            f"residual, x0 ~ N(0,I). VALIDATION = same loss on {int(ck.get('n_val',3))} held-out "
            f"cosmology runs ({int(ck.get('val_batches',20))} fixed batches spanning many shells, "
            f"NEVER used for gradients), every {int(ck.get('val_every',500))} steps.")
    ax.set_title("Conditional residual flow matching\n" + textwrap.fill(desc, width=100),
                 fontsize=8.8, linespacing=1.4)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "loss_curve.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000122")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[3, 30, 50])
    p.add_argument("--patch-batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=50, help="Euler steps for the ODE sampler")
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load_model(args.model_dir, dev)
    nside = int(ck["nside"]); L = int(ck["L"]); downscale = int(ck.get("downscale", 1))
    out_dir = Path(args.out_dir or (Path(args.model_dir) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    to_img_np, to_patch_np = ud.nested_grid_perms(L)
    to_img = torch.from_numpy(to_img_np).to(dev)
    to_patch = torch.from_numpy(to_patch_np).to(dev)

    runs = build_runs(args.data_root, args.test_cosmo, nside, include_test=True, prefix="low")
    test_run = next((r for r in runs if args.test_cosmo in str(r[0])), runs[0])
    low_path, high_path, cosmo_base = test_run
    inp = np.load(low_path, mmap_mode="r")
    high = np.load(high_path, mmap_mode="r")
    n_shells = int(min(inp.shape[0], high.shape[0]))

    lmax = min(args.lmax, 3 * (nside // downscale) - 1)
    plot_loss_curves(args.model_dir, out_dir, ck)
    ells = np.arange(lmax + 1)
    od = lambda m: m / m.mean() - 1.0
    print(f"[eval] cosmo={args.test_cosmo} nside={nside} (downscale={downscale}) "
          f"patch {L}x{L} | ODE steps={args.steps}", flush=True)

    for s in args.shell_indices:
        if s >= n_shells:
            continue
        disco = np.asarray(inp[s], dtype=np.float32)
        hi = np.asarray(high[s], dtype=np.float32)
        if downscale > 1:
            disco = ud.downscale_nested(disco[None], downscale)[0]
            hi = ud.downscale_nested(hi[None], downscale)[0]
        shell_norm = np.float32(s / max(n_shells - 1, 1))
        cosmo_vec = np.concatenate([cosmo_base, [shell_norm]]).astype(np.float32)
        corr = correct_shell(net, ck, disco, cosmo_vec, dev, args.patch_batch,
                             to_img, to_patch, L, args.steps)

        mse_disco = float(np.mean((od(disco) - od(hi)) ** 2))
        mse_corr = float(np.mean((od(corr) - od(hi)) ** 2))
        cl_d, cl_c, cl_h = od_cl(disco, lmax), od_cl(corr, lmax), od_cl(hi, lmax)

        def band(cl, a, b):
            a, b = min(a, lmax), min(b, lmax)
            return float(np.nanmean(cl[a:b] / cl_h[a:b])) if b > a else float("nan")
        print(f"shell {s}: MSE disco={mse_disco:.3e} corr={mse_corr:.3e} | "
              f"Cl ratio (disco,corr): l500-1000 ({band(cl_d,500,1000):.3f},{band(cl_c,500,1000):.3f}) "
              f"l1000-2000 ({band(cl_d,1000,2000):.3f},{band(cl_c,1000,2000):.3f}) "
              f"l2000-2900 ({band(cl_d,2000,2900):.3f},{band(cl_c,2000,2900):.3f})", flush=True)

        fig = plt.figure(figsize=(13, 9))
        ax0 = fig.add_subplot(2, 3, 1)
        ax0.loglog(ells, cl_d, ":", color="seagreen", label="DISCO")
        ax0.loglog(ells, cl_c, "-", color="steelblue", label="flow-corrected")
        ax0.loglog(ells, cl_h, "--", color="tomato", label="CosmoGrid high")
        ax0.set_xlabel(r"$\ell$"); ax0.set_ylabel(r"$C_\ell$")
        ax0.legend(fontsize=8); ax0.set_title(f"shell {s}: power spectrum")
        ax1 = fig.add_subplot(2, 3, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax1.semilogx(ells, cl_d / cl_h, ":", color="seagreen", label="DISCO/high")
            ax1.semilogx(ells, cl_c / cl_h, "-", color="steelblue", label="flow/high")
        ax1.axhline(1, color="k", lw=0.8); ax1.set_ylim(0.5, 1.5)
        ax1.set_xlabel(r"$\ell$"); ax1.set_ylabel("ratio"); ax1.legend(fontsize=8)
        ax1.set_title("Cl ratio to truth (the target metric)")
        axL = fig.add_subplot(2, 3, 3)
        axL.bar(["DISCO", "flow"], [mse_disco, mse_corr], color=["seagreen", "steelblue"])
        axL.set_ylabel("overdensity MSE vs high")
        axL.set_title("pixel MSE (a SAMPLE is not pixel-exact\nby construction -- judge Cl)",
                      fontsize=8)
        rot = (45.0, 45.0)
        for i, (m, ttl) in enumerate([(disco, "DISCO"), (corr, "flow-corrected"),
                                      (hi, "CosmoGrid high")]):
            hp.gnomview(m.astype(np.float64), nest=True, rot=rot, reso=1.5, xsize=200,
                        title=ttl, sub=(2, 3, 4 + i), fig=fig.number, notext=True,
                        cbar=(i == 2))
        fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93, bottom=0.06)
        fig.savefig(out_dir / f"eval_shell{s:03d}.png", dpi=140)
        plt.close(fig)

    print(f"[eval] figures in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
