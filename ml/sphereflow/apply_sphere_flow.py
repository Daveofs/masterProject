#!/usr/bin/env python3
"""Sample the residual sphere-flow model and evaluate against the T baseline.

Loads a model trained by train_sphere_flow.py (formulation: tcorr-residual),
conditions on the T-corrected map, samples the residual, and produces corrected
maps. Evaluation plots compare FOUR Cl curves: raw DISCO low, T-corrected
baseline, flow-corrected, CosmoGrid high — the question being whether the flow
improves on the transfer-function baseline.

Run inside the pytorch uenv venv:
  python apply_sphere_flow.py --model-dir <out> --data-root <grid> --test-cosmo cosmo_000122
"""

from __future__ import annotations
import argparse
import glob
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
    L = sf.healpix_laplacian(int(meta["nside"]), order=int(meta["order"]))
    net = sf.SphereFlowNet(L, cond_dim=int(meta["cond_dim"]), hidden=int(meta["hidden"]),
                           n_layers=int(meta["n_layers"]), K=int(meta["K"])).to(device)
    net.load_state_dict(torch.load(Path(model_dir) / "sphere_flow.pth",
                                   map_location=device))
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
def correct_shell(net, meta, in_map, cosmo_vec, device, steps, patch_batch):
    """Flow-corrected physical map for one shell, conditioned on the input map.

    formulation='direct'  : the sample IS the corrected signal.
    formulation='residual': corrected signal = cond + resid_scale * sample.
    """
    order = int(meta["order"])
    scale, soft = float(meta["sig_scale"]), float(meta["softening"])
    rscale = float(meta["resid_scale"])
    formulation = str(meta.get("formulation", "residual"))

    mean = max(float(in_map.mean()), 1e-12)
    d_in = in_map[None] / mean - 1.0
    cond = sf.map_to_patches(sf.signal_forward(d_in, scale, soft), order)  # (P, m)

    cosmo = torch.from_numpy(cosmo_vec[None]).to(device)
    out = np.empty_like(cond)
    for b in range(0, cond.shape[0], patch_batch):
        c = torch.from_numpy(cond[b:b + patch_batch]).to(device)
        r = sf.sample_ode(net, c, cosmo.expand(c.shape[0], -1), steps=steps)
        out[b:b + patch_batch] = r.cpu().numpy()

    if formulation == "direct":
        sig = sf.patches_to_maps(out, order, 1)[0]
    else:
        sig = sf.patches_to_maps(cond + rscale * out, order, 1)[0]
    delta = sf.signal_inverse(sig, scale, soft)
    return (mean * (1.0 + delta)).astype(np.float32)


def od_cl(m, lmax):
    return hp.anafast((m / m.mean() - 1.0).astype(np.float64), lmax=lmax)


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
    net, meta = load_model(args.model_dir, dev)
    nside = int(meta["nside"])
    out_dir = Path(args.out_dir or (Path(args.model_dir) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    formulation = str(meta.get("formulation", "residual"))
    prefix = "low" if formulation == "direct" else "tcorr"
    base_label = "DISCO (baseline)" if formulation == "direct" else "T-corrected (baseline)"
    run = next((r for r in sorted((Path(args.data_root) / args.test_cosmo).iterdir())
                if r.is_dir() and r.name.startswith("run_")),
               Path(args.data_root) / args.test_cosmo)
    inp = np.load(run / f"{prefix}_shells_nside={nside}.npy", mmap_mode="r")
    high = np.load(run / f"high_shells_nside={nside}.npy", mmap_mode="r")
    n_shells = inp.shape[0]
    cosmo_base = cosmo_vector(run / "params.yml", meta) if (run / "params.yml").exists() \
        else np.zeros(int(meta["cond_dim"]) - 1, np.float32)

    ells = np.arange(args.lmax + 1)
    for s in args.shell_indices:
        if s >= n_shells:
            continue
        tc = np.asarray(inp[s], dtype=np.float32)
        hi = np.asarray(high[s], dtype=np.float32)
        # conditioning = [cosmo params, normalized shell index] (as in training)
        shell_norm = np.float32(s / max(n_shells - 1, 1))
        cosmo_vec = np.concatenate([cosmo_base, [shell_norm]]).astype(np.float32)
        corr = correct_shell(net, meta, tc, cosmo_vec, dev, args.steps, args.patch_batch)

        cl_t, cl_c, cl_h = od_cl(tc, args.lmax), od_cl(corr, args.lmax), od_cl(hi, args.lmax)
        curves = [(base_label, cl_t, "seagreen", ":"),
                  ("flow-corrected", cl_c, "steelblue", "-")]
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for lbl, cl, col, ls in curves:
            ax[0].loglog(ells, cl, ls, color=col, label=lbl)
        ax[0].loglog(ells, cl_h, "--", color="tomato", label="CosmoGrid (high)")
        ax[0].set_ylabel(r"$C_\ell$"); ax[0].legend(); ax[0].set_title(f"shell {s}")
        with np.errstate(divide="ignore", invalid="ignore"):
            for lbl, cl, col, ls in curves:
                ax[1].semilogx(ells, cl / cl_h, ls, color=col, label=f"{lbl}/high")
        ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0.5, 1.5)
        ax[1].set_xlabel(r"$\ell$"); ax[1].set_ylabel("ratio"); ax[1].legend()
        fig.tight_layout(); fig.savefig(out_dir / f"cl_shell{s:03d}.png", dpi=150)
        plt.close(fig)

        def band(cl, a, b):
            return float(np.nanmean(cl[a:b] / cl_h[a:b]))
        print(f"shell {s}: (baseline, flow)/high | "
              f"ell200-500 ({band(cl_t,200,500):.3f}, {band(cl_c,200,500):.3f}) | "
              f"ell800-1500 ({band(cl_t,800,1500):.3f}, {band(cl_c,800,1500):.3f}) | "
              f"ell2000-2900 ({band(cl_t,2000,2900):.3f}, {band(cl_c,2000,2900):.3f})",
              flush=True)
    print(f"[eval] plots in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
