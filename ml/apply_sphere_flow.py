#!/usr/bin/env python3
"""Sample the trained DeepSphere flow-matching generator and evaluate Cl restoration.

Loads a model trained by train_sphere_flow.py, samples delta_high from noise
conditioned on a test cosmology's low-res shells, inverts to physical density, and
plots the Cl ratio vs the CosmoGrid high-res reference (the real test of whether
small-scale power is restored).

Run inside the pytorch uenv venv, e.g.:
  uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
    'source /capstor/scratch/cscs/damrein/venvs/sphereflow/bin/activate; \
     python apply_sphere_flow.py --model-dir <out_dir> --data-root <...> --test-cosmo cosmo_000001'
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


def load_model(model_dir, device):
    meta = dict(np.load(Path(model_dir) / "meta.npz", allow_pickle=True))
    nside, order, K = int(meta["nside"]), int(meta["order"]), int(meta["K"])
    L = sf.healpix_laplacian(nside, order=order)
    net = sf.SphereFlowNet(L, cond_dim=int(meta["cond_dim"]), hidden=int(meta["hidden"]),
                           n_layers=int(meta["n_layers"]), K=K).to(device)
    net.load_state_dict(torch.load(Path(model_dir) / "sphere_flow.pth", map_location=device))
    net.eval()
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
def correct_shell(net, meta, low_map, cosmo_vec, device, steps, patch_batch):
    """Generate a corrected physical map for one low-res shell."""
    order = int(meta["order"]); scale = float(meta["sig_scale"])
    soft = float(meta["softening"])
    nside = int(meta["nside"]); npix = hp.nside2npix(nside)

    if low_map.shape[0] != npix:
        low_map = hp.ud_grade(low_map, nside, order_in="NESTED", order_out="NESTED")
    low_map = low_map.astype(np.float32)
    mean_low = max(float(low_map.mean()), 1e-12)
    dlow = low_map / mean_low - 1.0
    cond = sf.map_to_patches(sf.signal_forward(dlow, scale, soft)[None], order)  # (P, m)

    cosmo = torch.from_numpy(cosmo_vec[None]).to(device)
    out = np.empty_like(cond)
    for b in range(0, cond.shape[0], patch_batch):
        c = torch.from_numpy(cond[b:b + patch_batch]).to(device)
        cc = cosmo.expand(c.shape[0], -1)
        y = sf.sample_ode(net, c, cc, steps=steps)          # (b, m) signal_high
        out[b:b + patch_batch] = y.cpu().numpy()

    n_maps = 1
    sig_high = sf.patches_to_maps(out, order, n_maps)[0]     # (npix,)
    dhigh = sf.signal_inverse(sig_high, scale, soft)         # overdensity
    return (mean_low * (1.0 + dhigh)).astype(np.float32)


def compute_cl(m, lmax):
    d = m / m.mean() - 1.0 if m.mean() != 0 else m
    return hp.anafast(d.astype(np.float64), lmax=lmax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000001")
    p.add_argument("--low-name", default="shells_nside=2048.npz")
    p.add_argument("--high-name", default="compressed_shells.npz")
    p.add_argument("--shell-indices", type=int, nargs="*", default=[3, 30, 65])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--patch-batch", type=int, default=256)
    p.add_argument("--lmax", type=int, default=4000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = load_model(args.model_dir, dev)
    out_dir = Path(args.out_dir or (Path(args.model_dir) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run = next((r for r in sorted((Path(args.data_root) / args.test_cosmo).iterdir())
                if r.is_dir() and r.name.startswith("run_")),
               Path(args.data_root) / args.test_cosmo)
    low = np.load(run / args.low_name)["shells"]
    high = np.load(run / args.high_name)["shells"]
    cosmo_vec = cosmo_vector(run / "params.yml", meta) if (run / "params.yml").exists() \
        else np.zeros(int(meta["cond_dim"]), np.float32)

    ells = np.arange(args.lmax + 1)
    for idx in args.shell_indices:
        if idx >= low.shape[0]:
            continue
        corr = correct_shell(net, meta, low[idx].astype(np.float32), cosmo_vec,
                             dev, args.steps, args.patch_batch)
        hi = high[idx].astype(np.float64)
        if hi.shape[0] != corr.shape[0]:
            hi = hp.ud_grade(hi, hp.npix2nside(corr.shape[0]),
                             order_in="NESTED", order_out="NESTED")
        cl_l = compute_cl(low[idx].astype(np.float64), args.lmax)
        cl_c = compute_cl(corr.astype(np.float64), args.lmax)
        cl_h = compute_cl(hi, args.lmax)

        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax[0].loglog(ells, cl_l, label="low", color="seagreen")
        ax[0].loglog(ells, cl_c, label="flow-corrected", color="steelblue")
        ax[0].loglog(ells, cl_h, "--", label="high (CosmoGrid)", color="tomato")
        ax[0].set_ylabel(r"$C_\ell$"); ax[0].legend(); ax[0].set_title(f"shell {idx}")
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[1].semilogx(ells, cl_l / cl_h, ":", color="seagreen", label="low/high")
            ax[1].semilogx(ells, cl_c / cl_h, color="steelblue", label="corrected/high")
        ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0, 2)
        ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel("ratio"); ax[1].legend()
        fig.tight_layout(); fig.savefig(out_dir / f"cl_shell{idx:03d}.png", dpi=150)
        plt.close(fig)

        # quick numeric summary
        def band(a, b):
            return float(np.nanmean(cl_c[a:b]) / np.nanmean(cl_h[a:b])), \
                   float(np.nanmean(cl_l[a:b]) / np.nanmean(cl_h[a:b]))
        print(f"shell {idx}: corrected/high vs low/high  "
              f"| ell100-300 {band(100,300)} | ell800-1200 {band(800,1200)} "
              f"| ell1500-2500 {band(1500,2500)}", flush=True)
    print(f"[eval] plots saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
