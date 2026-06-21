#!/usr/bin/env python3
"""Apply trained Spherical Harmonic Flow correction to HEALPix shells.

Uses per-sample norm normalization matching the training procedure.
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
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--diagnostic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else input_path.with_name(input_path.stem + "_corrected.npz")

    # 1. Load metadata
    metadata = torch.load(model_path.with_name("metadata.pth"), map_location="cpu")
    lmax = metadata["lmax"]
    N_alm = hp.Alm.getsize(lmax)
    n_total_shells = metadata["n_total_shells"]

    device = torch.device(args.device)
    data_mean = metadata["data_mean"].to(device)
    data_std = metadata["data_std"].to(device)
    cosmo_mean = metadata["cosmo_mean"].to(device)
    cosmo_std = metadata["cosmo_std"].to(device)

    print(f"Metadata: lmax={lmax} | dim={metadata['sample_dim']} | cond_dim={metadata['cond_dim']} | steps={args.steps}")

    # 2. Load model
    model = SmallMLP(dim_in=metadata["sample_dim"], cond_dim=metadata["cond_dim"], hidden=metadata["hidden"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    # 3. Prepare cosmology conditioning
    raw_cond = load_clean_params(Path(args.params))
    cond_base = (torch.from_numpy(raw_cond).to(device) - cosmo_mean) / cosmo_std

    # 4. Load input shells
    data = dict(np.load(input_path, allow_pickle=False))
    shells = data["shells"]
    Nshells, Npix = shells.shape
    nside_full = hp.npix2nside(Npix)

    indices = list(range(Nshells)) if args.shell_index < 0 else [int(args.shell_index)]
    shells_corrected = shells.copy()
    processed_indices, diff_l2_list, diff_max_list = [], [], []

    for idx in indices:
        m = shells[idx]

        # Shell index conditioning: normalized to [0, 1]
        shell_norm = torch.tensor([idx / max(n_total_shells - 1, 1)], dtype=torch.float32, device=device)
        cond = torch.cat([cond_base, shell_norm]).unsqueeze(0)  # [1, cond_dim]

        # Map -> Alm -> vector
        alm_orig = hp.map2alm(m, lmax=lmax, iter=1)
        x_raw = np.concatenate([alm_orig.real, alm_orig.imag]).astype(np.float32)
        x_raw_t = torch.from_numpy(x_raw).to(device)

        # Per-sample norm normalization (matching training)
        input_norm = x_raw_t.norm().clamp(min=1e-8)
        x_normed = x_raw_t / input_norm

        # Whiten
        x_curr = ((x_normed - data_mean) / data_std).unsqueeze(0)

        # ODE integration
        dt = 1.0 / args.steps
        for step in range(args.steps):
            t_tensor = torch.full((1,), step * dt, dtype=torch.float32, device=device)
            with torch.no_grad():
                v = model(x_curr, t_tensor, cond=cond)
            x_curr = x_curr + v * dt

        # Un-whiten
        x_out_normed = (x_curr.squeeze(0) * data_std + data_mean)

        # Re-scale: the output is in "normalized" space, scale back by input norm
        # The flow should adjust the norm implicitly, but we use input norm as anchor
        x_out = x_out_normed * input_norm

        pred_phys = x_out.cpu().numpy()
        alm_recon = pred_phys[:N_alm] + 1j * pred_phys[N_alm:]
        corrected_map = hp.alm2map(alm_recon, nside=nside_full)

        diff = corrected_map - m
        diff_l2 = float(np.linalg.norm(diff))
        diff_max = float(np.max(np.abs(diff)))

        # Sanity check: if correction explodes, fall back to original
        ratio = corrected_map.std() / max(m.std(), 1e-10)
        if ratio > 5.0 or ratio < 0.2 or np.any(~np.isfinite(corrected_map)):
            print(f"  Shell {idx}: ⚠️  UNSTABLE (ratio={ratio:.2f}), keeping original")
            corrected_map = m
            diff_l2, diff_max = 0.0, 0.0
        else:
            print(f"  Shell {idx}: std {m.std():.4f} → {corrected_map.std():.4f} | ratio={ratio:.3f} | diff_max={diff_max:.4f}")

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
    np.savez(out_path, **out_dict)
    print(f"Saved corrected output to: {out_path}")


if __name__ == "__main__":
    main()
