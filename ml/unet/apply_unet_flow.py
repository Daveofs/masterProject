#!/usr/bin/env python3
"""Apply a trained 2D-UNet flow model: correct DISCO shells and evaluate.

Loads a checkpoint from train_unet_flow.py, conditions on the raw DISCO (low-res)
map + cosmology/redshift, flow-samples a corrected high-res realization patch-wise,
stitches the patches back into full HEALPix maps, and produces per-shell diagnostic
figures:

  * angular power spectrum C_ell (DISCO / flow-corrected / CosmoGrid-high) + ratio,
  * gnomonic zoom-ins of the three maps,
  * a LOSS panel: pixel-space MSE of DISCO vs high and flow-corrected vs high (the
    correction is "good" when it lowers this loss toward 0), plus the training
    flow-matching loss curve if loss_history.npy is present.

Run inside the pytorch uenv venv (needs a GPU):
  python apply_unet_flow.py --model-dir <out> --data-root <grid> --test-cosmo cosmo_000122
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sphere_flow as sf
import unet_flow as uf
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
def correct_shell(net, ck, in_map, cosmo_vec, device, steps, patch_batch,
                  to_img, to_patch, L):
    """Flow-corrected physical map for one shell, conditioned on the DISCO map."""
    order = int(ck["order"]); scale = float(ck["sig_scale"]); soft = float(ck["softening"])
    mean = max(float(in_map.mean()), 1e-12)
    d_in = in_map[None] / mean - 1.0
    cond = sf.map_to_patches(sf.signal_forward(d_in, scale, soft), order)  # (P, M)

    cosmo = torch.from_numpy(cosmo_vec[None]).to(device)
    out = np.empty_like(cond)
    for b in range(0, cond.shape[0], patch_batch):
        c = torch.from_numpy(cond[b:b + patch_batch]).to(device)
        c_img = c[:, to_img].view(c.shape[0], 1, L, L)          # nested patch -> image
        r = uf.sample_ode(net, c_img, cosmo.expand(c.shape[0], -1), steps=steps)
        out[b:b + patch_batch] = r.view(r.shape[0], -1)[:, to_patch].cpu().numpy()

    sig = sf.patches_to_maps(out, order, 1)[0]
    delta = sf.signal_inverse(sig, scale, soft)
    return (mean * (1.0 + delta)).astype(np.float32)


def od_cl(m, lmax):
    return hp.anafast((m / m.mean() - 1.0).astype(np.float64), lmax=lmax)


def plot_loss_curve(model_dir, out_dir):
    """Training flow-matching loss curve, if loss_history.npy was saved."""
    hp_path = Path(model_dir) / "loss_history.npy"
    if not hp_path.exists():
        return
    h = np.load(hp_path)
    if h.ndim != 2 or h.shape[0] < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(h[:, 0], h[:, 1], color="0.7", lw=0.8, label="loss")
    ax.plot(h[:, 0], h[:, 2], color="steelblue", lw=1.8, label="EMA")
    ax.set_xlabel("step"); ax.set_ylabel("flow-matching loss")
    ax.set_yscale("log"); ax.legend(); ax.set_title("training loss")
    fig.tight_layout(); fig.savefig(Path(out_dir) / "loss_curve.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000122")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[3, 30, 50])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--patch-batch", type=int, default=256)
    p.add_argument("--lmax", type=int, default=3000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load_model(args.model_dir, dev)
    nside = int(ck["nside"]); L = int(ck["L"])
    out_dir = Path(args.out_dir or (Path(args.model_dir) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    to_img, to_patch = uf.nested_grid_perms(L)
    to_img = torch.from_numpy(to_img).to(dev)
    to_patch = torch.from_numpy(to_patch).to(dev)

    # Reuse build_runs so the cosmology vector EXACTLY matches training.
    runs = build_runs(args.data_root, args.test_cosmo, nside, include_test=True, prefix="low")
    test_run = next((r for r in runs if args.test_cosmo in str(r[0])), runs[0])
    low_path, high_path, cosmo_base = test_run
    inp = np.load(low_path, mmap_mode="r")
    high = np.load(high_path, mmap_mode="r")
    n_shells = int(min(inp.shape[0], high.shape[0]))

    plot_loss_curve(args.model_dir, out_dir)
    ells = np.arange(args.lmax + 1)
    print(f"[eval] cosmo={args.test_cosmo} nside={nside} patch {L}x{L} "
          f"| {n_shells} shells | steps={args.steps}", flush=True)

    for s in args.shell_indices:
        if s >= n_shells:
            continue
        disco = np.asarray(inp[s], dtype=np.float32)
        hi = np.asarray(high[s], dtype=np.float32)
        shell_norm = np.float32(s / max(n_shells - 1, 1))
        cosmo_vec = np.concatenate([cosmo_base, [shell_norm]]).astype(np.float32)
        corr = correct_shell(net, ck, disco, cosmo_vec, dev, args.steps,
                             args.patch_batch, to_img, to_patch, L)

        cl_d, cl_c, cl_h = od_cl(disco, args.lmax), od_cl(corr, args.lmax), od_cl(hi, args.lmax)
        # pixel-space overdensity MSE = the "loss" the correction should reduce.
        od = lambda m: m / m.mean() - 1.0
        mse_disco = float(np.mean((od(disco) - od(hi)) ** 2))
        mse_corr = float(np.mean((od(corr) - od(hi)) ** 2))

        fig = plt.figure(figsize=(13, 8))
        # --- row 1: Cl + ratio ---
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
        ax1.set_title("Cl ratio to truth")
        # --- loss panel ---
        axL = fig.add_subplot(2, 3, 3)
        axL.bar(["DISCO", "flow"], [mse_disco, mse_corr],
                color=["seagreen", "steelblue"])
        axL.set_ylabel("overdensity MSE vs high")
        axL.set_title(f"pixel loss (−{100*(1-mse_corr/max(mse_disco,1e-12)):.0f}%)")
        # --- row 2: gnomonic zoom of the three maps (same patch of sky) ---
        rot = (45.0, 45.0)     # (lon, lat) center of the zoom, degrees
        for i, (m, ttl) in enumerate([(disco, "DISCO"), (corr, "flow-corrected"),
                                      (hi, "CosmoGrid high")]):
            hp.gnomview(m.astype(np.float64), nest=True, rot=rot, reso=1.5, xsize=200,
                        title=ttl, sub=(2, 3, 4 + i), fig=fig.number, notext=True,
                        cbar=(i == 2))
        fig.tight_layout()
        fig.savefig(out_dir / f"eval_shell{s:03d}.png", dpi=140)
        plt.close(fig)

        def band(cl, a, b):
            return float(np.nanmean(cl[a:b] / cl_h[a:b]))
        print(f"shell {s}: MSE disco={mse_disco:.4e} flow={mse_corr:.4e} "
              f"({100*(1-mse_corr/max(mse_disco,1e-12)):+.0f}%) | "
              f"Cl flow/high ell800-1500={band(cl_c,800,1500):.3f} "
              f"(disco {band(cl_d,800,1500):.3f})", flush=True)
    print(f"[eval] figures in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
