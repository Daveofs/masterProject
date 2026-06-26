#!/usr/bin/env python3
"""Apply trained Spherical Harmonic Flow correction — optimized."""

import argparse
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import numpy as np
import healpy as hp
import torch
from MLP import MLP

def load_clean_params(params_path: Path):
    params = yaml.safe_load(params_path.read_text())
    valid_keys = []
    for k, v in sorted(params.items()):
        try:
            float(v)
            valid_keys.append(k)
        except (ValueError, TypeError):
            continue
    vec = np.array([float(params[k]) for k in valid_keys], dtype=np.float32)
    return vec, valid_keys, params

def get_alm_scale_vector(lmax, device):
    """Generates a static vector to whiten the dynamic range of the a_lm spectrum."""
    l_arr, _ = hp.Alm.getlm(lmax)
    scale_complex = (l_arr / 100.0) ** 1.5 + 1.0 
    scale_flat = np.concatenate([scale_complex, scale_complex]).astype(np.float32)
    return torch.tensor(scale_flat, device=device)

def shell_to_alm_vector(args_tuple):
    idx, shell, lmax = args_tuple
    alm = hp.map2alm(shell, lmax=lmax, iter=1)
    x = np.concatenate([alm.real, alm.imag]).astype(np.float32)
    return idx, x

def alm_vector_to_map(args_tuple):
    idx, pred, N_alm, nside = args_tuple
    alm_recon = pred[:N_alm] + 1j * pred[N_alm:]
    return idx, hp.alm2map(alm_recon, nside=nside)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--params", type=str, required=True)
    p.add_argument("--shell-index", type=int, default=-1)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else input_path.with_name(input_path.stem + "_corrected.npz")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load metadata & model
    metadata = torch.load(model_path.with_name("metadata.pth"), map_location="cpu")
    lmax = metadata["lmax"]
    N_alm = hp.Alm.getsize(lmax)

    print(f"Metadata: lmax={lmax} | dim={metadata['sample_dim']} | cond_dim={metadata['cond_dim']}")

    model = MLP(dim_in=metadata["sample_dim"], cond_dim=metadata["cond_dim"],
                hidden=metadata["hidden"], lmax=lmax)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    has_nan_weights = any(torch.isnan(p).any() for p in model.parameters())
    if has_nan_weights:
        raise ValueError("[!] FATAL: The loaded model contains NaN weights. You must retrain.")

    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

    # 2. Load input shells
    data = dict(np.load(input_path, allow_pickle=False))
    shells = data["shells"]
    Nshells, Npix = shells.shape
    nside_full = hp.npix2nside(Npix)

    indices = list(range(Nshells)) if args.shell_index < 0 else [int(args.shell_index)]
    
    raw_cond = load_clean_params(Path(args.params))[0]
    cond_base_raw = torch.from_numpy(raw_cond).float().to(device)

    # Apply the same per-feature normalisation that was used during training.
    # Without this the conditioning is out-of-distribution (e.g. bary_Mc ~5e12).
    cond_mean = torch.tensor(metadata["cond_mean"], dtype=torch.float32, device=device)
    cond_std  = torch.tensor(metadata["cond_std"],  dtype=torch.float32, device=device)
    cond_base = (cond_base_raw - cond_mean) / cond_std

    # Load shell-index normalization constant from training metadata
    max_shell_idx = float(metadata.get("max_shell_idx", max(len(indices) - 1, 1)))

    # Pre-calculate whitening vector
    scale_vec = get_alm_scale_vector(lmax, device)

    t0 = time.time()
    print(f"Computing SHTs (map2alm) with {args.workers} workers...")
    alm_vectors = {}
    sht_args = [(idx, shells[idx].astype(np.float64), lmax) for idx in indices]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(shell_to_alm_vector, a) for a in sht_args]
        for f in as_completed(futures):
            idx, vec = f.result()
            alm_vectors[idx] = vec

    print(f"  SHTs done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"Running ODE integration ({args.steps} steps, batch_size={args.batch_size})...")
    results = {} 

    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        B = len(batch_indices)

        # Step 1: spectral whitening — same as training
        x_batch_raw = torch.stack([
            torch.from_numpy(alm_vectors[idx]) for idx in batch_indices
        ]).to(device) * scale_vec

        # Step 2: per-sample amplitude normalisation — must mirror the training loop.
        # The model was trained on unit-norm inputs; providing unnormalised inputs
        # would move the ODE starting point out of the learned distribution.
        sample_scale = x_batch_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_curr = x_batch_raw / sample_scale

        cond_batch = torch.stack([
            torch.cat([cond_base, torch.tensor([idx / max_shell_idx], dtype=torch.float32, device=device)])
            for idx in batch_indices
        ])

        dt = 1.0 / args.steps

        with torch.no_grad():
            for step in range(args.steps):
                t_tensor = torch.full((B,), step * dt, dtype=torch.float32, device=device)
                
                v = model(x_curr, t_tensor, cond=cond_batch)
                
                if args.diagnostic and batch_start == 0:
                    v_norm = torch.norm(v, dim=-1).mean().item()
                    x_norm = torch.norm(x_curr, dim=-1).mean().item()
                    print(f"    Step {step:02d} | x_norm: {x_norm:.2e} | v_norm: {v_norm:.2e}")
                        
                x_curr = x_curr + v * dt

        # Reverse per-sample normalisation, then reverse spectral whitening
        x_curr = (x_curr * sample_scale) / scale_vec
        x_out = x_curr.cpu().numpy()
        
        for i, idx in enumerate(batch_indices):
            results[idx] = x_out[i]

    print(f"  ODE integration done in {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"Computing inverse SHTs (alm2map) with {args.workers} workers...")

    # Store corrected shells as float32 — alm2map returns a smooth continuous
    # field (float64). Casting back to the original integer dtype (e.g. int32)
    # truncates values in [0, 1) to 0, reducing the mean from ~0.14 to ~0.03
    # and blowing up the overdensity Cl by ~(0.14/0.03)^2 ~ 20x.
    shells_corrected = shells.astype(np.float32)
    inv_args = [(idx, results[idx], N_alm, nside_full) for idx in indices]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(alm_vector_to_map, a) for a in inv_args]
        for f in as_completed(futures):
            idx, corrected_map = f.result()
            shells_corrected[idx] = corrected_map.astype(np.float32)

    print(f"  Inverse SHTs done in {time.time() - t2:.1f}s")

    out_dict = {k: data[k] for k in data.keys()}
    out_dict["shells"] = shells_corrected
    out_dict["corrected_index"] = np.asarray(indices, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensured this uses the requested uncompressed format
    np.savez(out_path, **out_dict)
    
    print(f"\nSaved corrected output to: {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()