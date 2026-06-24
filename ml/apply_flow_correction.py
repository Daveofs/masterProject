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


def shell_to_alm_vector(args_tuple):
    """Convert a single shell to real alm vector (for multiprocessing)."""
    idx, shell, lmax = args_tuple
    alm = hp.map2alm(shell, lmax=lmax, iter=1)
    x = np.concatenate([alm.real, alm.imag]).astype(np.float32)
    return idx, x


def alm_vector_to_map(args_tuple):
    """Convert alm vector back to HEALPix map (for multiprocessing)."""
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
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of shells to process in one batched forward pass")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel workers for SHT transforms")
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

    model = MLP(dim_in=metadata["sample_dim"], cond_dim=metadata["cond_dim"], hidden=metadata["hidden"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    # Compile model for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    except Exception:
        print("torch.compile not available, using eager mode")

    # 2. Load input shells
    data = dict(np.load(input_path, allow_pickle=False))
    shells = data["shells"]
    Nshells, Npix = shells.shape
    nside_full = hp.npix2nside(Npix)

    indices = list(range(Nshells)) if args.shell_index < 0 else [int(args.shell_index)]
    print(f"Processing {len(indices)} shells | nside={nside_full} | N_alm={N_alm}")

    # 3. Prepare cosmology conditioning
    raw_cond = load_clean_params(Path(args.params))[0]
    cond_base = torch.from_numpy(raw_cond).float().to(device)

    # 4. Parallel SHT: map -> alm vectors
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

    # 5. Batched ODE integration on GPU
    t1 = time.time()
    print(f"Running ODE integration ({args.steps} steps, batch_size={args.batch_size})...")

    results = {}  # idx -> corrected alm vector

    # Process in batches
    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        B = len(batch_indices)

        # Stack alm vectors
        x_batch = torch.stack([
            torch.from_numpy(alm_vectors[idx]) for idx in batch_indices
        ]).to(device)  # [B, D]

        # Build conditioning: [cond_base | shell_idx] for each shell
        cond_batch = torch.stack([
            torch.cat([cond_base, torch.tensor([idx], dtype=torch.float32, device=device)])
            for idx in batch_indices
        ])  # [B, cond_dim]

        # Euler ODE integration
        x_curr = x_batch.clone()
        dt = 1.0 / args.steps

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            for step in range(args.steps):
                t_tensor = torch.full((B,), step * dt, dtype=torch.float32, device=device)
                v = model(x_curr, t_tensor, cond=cond_batch)
                x_curr = x_curr + v * dt

        # Move results back to CPU
        x_out = x_curr.cpu().numpy()
        for i, idx in enumerate(batch_indices):
            results[idx] = x_out[i]

        if (batch_start // args.batch_size) % 5 == 0:
            print(f"  Batch {batch_start // args.batch_size + 1}/"
                  f"{(len(indices) + args.batch_size - 1) // args.batch_size}")

    print(f"  ODE integration done in {time.time() - t1:.1f}s")

    # 6. Parallel inverse SHT: alm vectors -> maps
    t2 = time.time()
    print(f"Computing inverse SHTs (alm2map) with {args.workers} workers...")

    shells_corrected = shells.copy()
    inv_args = [(idx, results[idx], N_alm, nside_full) for idx in indices]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(alm_vector_to_map, a) for a in inv_args]
        for f in as_completed(futures):
            idx, corrected_map = f.result()
            shells_corrected[idx] = corrected_map.astype(shells.dtype)

    print(f"  Inverse SHTs done in {time.time() - t2:.1f}s")

    # 7. Save
    out_dict = {k: data[k] for k in data.keys()}
    out_dict["shells"] = shells_corrected
    out_dict["corrected_index"] = np.asarray(indices, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)
    print(f"\nSaved corrected output to: {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
