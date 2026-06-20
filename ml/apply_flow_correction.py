#!/usr/bin/env python3
"""Apply trained Spherical Harmonic Flow correction to a HEALPix shell.

- Converts input map to complex Alm representation.
- Applies whitened vector normalization.
- Solves the flow trajectory over N Euler ODE steps.
- Reconstructs corrected HEALPix map preserving native data types.
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import healpy as hp
import torch
from mlp import SmallMLP


def load_clean_params(params_path: Path):
    params = yaml.safe_load(params_path.read_text())
    bad_phrases = ["seed", "job", "part", "box", "step", "nside", "path", "dir", "file", "rank", "node", "gpu", "time"]
    valid = [k for k, v in sorted(params.items()) if not any(b in k.lower() for b in bad_phrases) and isinstance(v, (int, float))]
    return np.array([float(params[k]) for k in valid], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--params", type=str, required=True)
    p.add_argument("--shell-index", type=int, default=-1)
    p.add_argument("--steps", type=int, default=10, help="Number of Euler ODE integration steps.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--diagnostic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else input_path.with_name(input_path.stem + "_corrected.npz")

    # 1. Load Whitening Metadata
    metadata = torch.load(model_path.with_name("metadata.pth"), map_location="cpu")
    lmax = metadata["lmax"]
    N_alm = hp.Alm.getsize(lmax)
    
    device = torch.device(args.device)
    data_mean = metadata["data_mean"].to(device)
    data_std = metadata["data_std"].to(device)
    cosmo_mean = metadata["cosmo_mean"].to(device)
    cosmo_std = metadata["cosmo_std"].to(device)

    print(f"Loaded Metadata: lmax={lmax} | Vector dim={metadata['sample_dim']} | ODE Steps={args.steps}")

    # 2. Init Model
    model = SmallMLP(dim_in=metadata["sample_dim"], cond_dim=metadata["cond_dim"], hidden=metadata["hidden"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    # 3. Prepare Conditioning Vector
    raw_cond = load_clean_params(Path(args.params))
    cond_norm = (torch.from_numpy(raw_cond).to(device).unsqueeze(0) - cosmo_mean) / cosmo_std

    data = dict(np.load(input_path, allow_pickle=False))
    shells = data["shells"]
    Nshells, Npix = shells.shape
    nside_full = hp.npix2nside(Npix)

    indices = list(range(Nshells)) if args.shell_index < 0 else [int(args.shell_index)]
    shells_corrected = shells.copy()
    processed_indices, diff_l2_list, diff_max_list = [], [], []

    for idx in indices:
        m = shells[idx]
        
        # Transform map -> Alm vector
        alm_orig = hp.map2alm(m, lmax=lmax, iter=1)
        x0 = np.concatenate([alm_orig.real, alm_orig.imag]).astype(np.float32)
        
        x_curr = (torch.from_numpy(x0).to(device).unsqueeze(0) - data_mean) / data_std

        # ODE Euler Integration
        dt = 1.0 / args.steps
        for step in range(args.steps):
            t_tensor = torch.full((1,), step * dt, dtype=torch.float32, device=device)
            with torch.no_grad():
                v = model(x_curr, t_tensor, cond=cond_norm)
            x_curr = x_curr + v * dt

        # Un-whiten back to physical Alm scale
        pred_phys = (x_curr * data_std + data_mean).squeeze(0).cpu().numpy()
        
        alm_recon = pred_phys[:N_alm] + 1j * pred_phys[N_alm:]
        corrected_map = hp.alm2map(alm_recon, nside=nside_full)

        diff = corrected_map - m
        diff_l2 = float(np.linalg.norm(diff))
        diff_max = float(np.max(np.abs(diff)))

        print(f"  Shell {idx}: Original std: {m.std():.6g} | Corrected std: {corrected_map.std():.6g} | Diff max: {diff_max:.6g}")

        shells_corrected[idx] = corrected_map.astype(shells.dtype)
        processed_indices.append(idx)
        diff_l2_list.append(diff_l2)
        diff_max_list.append(diff_max)

    out_dict = {k: data[k] for k in data.keys()}
    out_dict["shells"] = shells_corrected
    out_dict["corrected_index"] = np.asarray(processed_indices, dtype=np.int64)

    if args.diagnostic:
        out_dict["diff_l2"] = np.asarray(diff_l2_list, dtype=np.float32)
        out_dict["diff_max"] = np.asarray(diff_max_list, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict) # Uncompressed save explicitly preserved per user ledger
    print(f"Saved corrected harmonic output to: {out_path}")


if __name__ == "__main__":
    main()
