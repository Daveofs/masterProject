#!/usr/bin/env python3
"""Apply the trained alm flow-matching model to correct a test cosmology's shells.

Loads low alms (preprocessed low_alms_lmax{lmax}.npy), whitens with the training
per-ell scale, integrates the flow ODE low->high in whitened space, unwhitens, and
alm2map's back to corrected density shells. Saves a corrected .npz mirroring the
input shell layout.

  python apply_flow_correction.py --model <flow_model_dir> --run-dir <test run dir> \
      --nside 2048 --steps 25 --out corrected.npz
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import healpy as hp
import torch
import yaml

import flow_matching_alm as fm


def load_model(model_dir, device):
    meta = dict(np.load(Path(model_dir) / "flow_meta.npz", allow_pickle=True))
    lmax = int(meta["lmax"])
    model = fm.MLP(int(meta["dim_in"]), cond_dim=int(meta["cond_dim"]),
                   hidden=int(meta["hidden"]), lmax=lmax).to(device)
    model.load_state_dict(torch.load(Path(model_dir) / "flow_mlp.pth", map_location=device))
    model.eval()
    return model, meta


def cond_vector(params_yml, shell_idx, meta):
    p = yaml.safe_load(Path(params_yml).read_text())
    keys = sorted(k for k, v in p.items() if _is_num(v))
    v = np.array([float(p[k]) for k in keys], dtype=np.float32)
    cmean, cstd = meta["cond_mean"], meta["cond_std"]
    n = len(cmean)
    v = np.pad(v, (0, max(n - len(v), 0)))[:n]
    v = (v - cmean) / cstd
    shell_norm = np.float32(shell_idx / float(meta["max_shell_idx"]))
    return np.concatenate([v, [shell_norm]]).astype(np.float32)


def _is_num(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="dir with flow_mlp.pth + flow_meta.npz")
    p.add_argument("--run-dir", required=True, help="test run dir (low_alms + params.yml)")
    p.add_argument("--low-npz", default="shells_nside=2048.npz",
                   help="original low shells (for output layout + nside)")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.model, dev)
    lmax = int(meta["lmax"])
    scale = torch.from_numpy(meta["whiten_scale"]).to(dev)

    run = Path(args.run_dir)
    low_alms = np.load(run / f"low_alms_lmax{lmax}.npy")          # (n_shells, 2*N_alm)
    params_yml = run / "params.yml" if (run / "params.yml").exists() else run.parent / "params.yml"
    n_shells = low_alms.shape[0]
    npix = hp.nside2npix(args.nside)

    corrected = np.zeros((n_shells, npix), dtype=np.float32)
    for start in range(0, n_shells, args.batch_size):
        end = min(start + args.batch_size, n_shells)
        x0 = torch.from_numpy(low_alms[start:end].astype(np.float32)).to(dev) / scale
        cond = torch.from_numpy(np.stack([
            cond_vector(params_yml, i, meta) for i in range(start, end)])).to(dev)
        xc = fm.integrate(model, x0, cond, steps=args.steps) * scale   # unwhiten
        xc = xc.cpu().numpy()
        N_alm = (lmax + 1) * (lmax + 2) // 2
        for j in range(end - start):
            alm = (xc[j, :N_alm] + 1j * xc[j, N_alm:]).astype(np.complex128)
            corrected[start + j] = hp.alm2map(alm, nside=args.nside, lmax=lmax).astype(np.float32)
        print(f"  corrected shells {start}-{end-1}/{n_shells}", flush=True)

    # Preserve any metadata (e.g. 'info') from the original low .npz.
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    low_npz = run / args.low_npz
    extra = {}
    if low_npz.exists():
        d = np.load(low_npz, allow_pickle=False)
        extra = {k: d[k] for k in d.files if k != "shells"}
    np.savez(out, shells=corrected, **extra)
    print(f"[apply] saved corrected shells to {out}", flush=True)


if __name__ == "__main__":
    main()
