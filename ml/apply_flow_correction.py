#!/usr/bin/env python3
"""
Apply trained patch-based flow correction to HEALPix shells.

Pixel-space analogue of apply_flow_correction.py.
Operates entirely in pixel space — no SHT required.

Usage
-----
  python apply_patch_correction.py \
      --model ./patch_model/patch_flow_mlp.pth \
      --input shells_nside=2048.npz \
      --params params.yml \
      --steps 25 \
      --device cuda:0 \
      --out corrected.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
import torch

from MLP import PatchMLP, apply_patch_flow, get_or_build_patch_idx


def load_clean_params(params_path: Path):
    params = yaml.safe_load(params_path.read_text())
    valid_keys = sorted(
        k for k, v in params.items()
        if _try_float(v) is not None
    )
    return np.array([float(params[k]) for k in valid_keys], dtype=np.float32)


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--params", type=str, required=True)
    p.add_argument("--shell-index", type=int, default=-1,
                   help="Single shell to correct (-1 = all)")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--chunk-size", type=int, default=1_000)
    p.add_argument("--cache-dir", type=str, default="/capstor/scratch/cscs/damrein/healpy_patch_cache")
    p.add_argument("--diagnostic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else \
        input_path.with_name(input_path.stem + "_patch_corrected.npz")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load metadata + model
    # ------------------------------------------------------------------
    meta_path = model_path.with_name("patch_metadata.pth")
    metadata = torch.load(meta_path, map_location="cpu")

    nside       = metadata["nside"]
    patch_depth = metadata["patch_depth"]
    patch_size  = metadata["patch_size"]
    cond_dim    = metadata["cond_dim"]
    hidden      = metadata["hidden"]
    n_layers    = metadata["n_layers"]

    print(f"Metadata: nside={nside} | patch_size={patch_size} | "
          f"cond_dim={cond_dim} | hidden={hidden} | layers={n_layers}")

    model = PatchMLP(
        patch_size=patch_size,
        cond_dim=cond_dim,
        hidden=hidden,
        n_layers=n_layers,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    if any(torch.isnan(p).any() for p in model.parameters()):
        raise ValueError("FATAL: Model contains NaN weights — retrain required.")

    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Patch indices
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"Loading/building patch indices (nside={nside}, depth={patch_depth})...")
    patch_idx = get_or_build_patch_idx(nside, depth=patch_depth, cache_dir=args.cache_dir)
    print(f"  Patch indices ready in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    raw_cond = load_clean_params(Path(args.params))
    cond_mean = torch.tensor(metadata["cond_mean"], dtype=torch.float32)
    cond_std  = torch.tensor(metadata["cond_std"],  dtype=torch.float32)
    cond_norm = (torch.from_numpy(raw_cond) - cond_mean) / cond_std

    max_shell_idx = float(metadata.get("max_shell_idx", 1))

    # ------------------------------------------------------------------
    # Load shells
    # ------------------------------------------------------------------
    data = dict(np.load(input_path, allow_pickle=False))
    shells = data["shells"]
    Nshells = shells.shape[0]

    indices = list(range(Nshells)) if args.shell_index < 0 else [int(args.shell_index)]
    shells_corrected = shells.astype(np.float32)

    # ------------------------------------------------------------------
    # Apply correction shell by shell
    # ------------------------------------------------------------------
    t1 = time.time()
    for i, idx in enumerate(indices):
        shell_idx_norm = torch.tensor([idx / max_shell_idx], dtype=torch.float32)
        cond = torch.cat([cond_norm, shell_idx_norm]).to(device)

        x_low = shells[idx].astype(np.float32)

        print(f"[{i+1}/{len(indices)}] Shell {idx}  "
              f"(mean={x_low.mean():.4f}, std={x_low.std():.4f})")

        x_corr = apply_patch_flow(
            model=model,
            x_low=x_low,
            cond=cond,
            patch_idx=patch_idx,
            steps=args.steps,
            chunk_size=args.chunk_size,
            device=device,
            diagnostic=args.diagnostic,
        )
        shells_corrected[idx] = x_corr

    print(f"\nAll shells corrected in {time.time()-t1:.1f}s")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dict = {k: data[k] for k in data}
    out_dict["shells"] = shells_corrected
    out_dict["corrected_index"] = np.asarray(indices, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)
    print(f"Saved corrected output to: {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()