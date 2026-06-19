#!/usr/bin/env python3
"""Apply trained `flow_mlp.pth` correction to a shell.

The script:
- loads a model state dict and infers `dim_in` and `cond_dim` from `net.0.weight`.
- loads an NPZ with `shells` (shape [Nshells, Npix]).
- splits each shell into HEALPix patches (NESTED ordering) matching training.
- runs the model on each patch to predict `u = x1 - x0`.
- reconstructs a corrected shell `x0 + u` and saves output NPZ.

Example:
  python apply_flow_correction.py --model models/flow_mlp.pth \
      --input data/cosmo_000001/run_0/shells_nside=2048.npz \
      --nside-patch 16
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import healpy as hp
import torch
from mlp import SmallMLP


def split_into_patches(shell_map, nside_patch=16):
    """Split a full-resolution HEALPix map into patches (NESTED ordering).

    Each patch corresponds to one pixel at nside_patch, containing all the
    sub-pixels from the full-resolution map that fall within it.

    Returns:
        patches: array of shape [n_patches, pixels_per_patch]
    """
    nside_full = hp.npix2nside(len(shell_map))
    assert nside_full >= nside_patch, (
        f"Full nside ({nside_full}) must be >= patch nside ({nside_patch})"
    )
    npix_patch = hp.nside2npix(nside_patch)
    pixels_per_patch = (nside_full // nside_patch) ** 2

    shell_nested = hp.reorder(shell_map, r2n=True)
    patches = shell_nested.reshape(npix_patch, pixels_per_patch)
    return patches


def patches_to_map(patches, nside_patch=16, nside_full=2048):
    """Reassemble patches back into a full HEALPix map (RING ordering)."""
    full_nested = patches.reshape(-1)
    full_ring = hp.reorder(full_nested, n2r=True)
    return full_ring


def load_params_vector(params_path: Path):
    params = yaml.safe_load(params_path.read_text())
    keys = sorted(params.keys())
    return np.array([float(params[k]) for k in keys], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="/capstor/scratch/cscs/damrein/outputs/flow_matching/3703591/flow_mlp.pth")
    p.add_argument("--input", type=str, default="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/shells_nside=2048.npz")
    p.add_argument("--params", type=str, default="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/params.yml",
                   help="Optional params.yml path for conditioning vector")
    p.add_argument("--nside-patch", type=int, default=16,
                   help="Nside for patch grid (must match training)")
    p.add_argument("--shell-index", type=int, default=-1,
                   help="Index of shell to process; -1=process all shells")
    p.add_argument("--t", type=float, default=0.0,
                   help="t value to pass to model (0..1)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Number of patches to process at once (controls GPU memory)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--diagnostic", action="store_true",
                   help="Print diagnostic stats and save prediction arrays to output NPZ")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    out_path = Path(args.out) if args.out else input_path.with_name(input_path.stem + "_corrected.npz")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Load input data
    data = dict(np.load(input_path, allow_pickle=False))
    if "shells" not in data:
        raise KeyError("Input NPZ must contain 'shells' array")

    # 1. Load Normalization Metadata
    metadata_path = model_path.with_name("metadata.pth")
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location="cpu")
        data_mean = float(metadata.get("data_mean", 0.0))
        data_std = float(metadata.get("data_std", 1.0))
        print(f"Loaded normalization stats: mean={data_mean:.6g}, std={data_std:.6g}")
    else:
        print("WARNING: metadata.pth not found. Assuming no normalization (mean=0, std=1).")
        data_mean = 0.0
        data_std = 1.0

    shells = data["shells"]
    Nshells, Npix = shells.shape
    nside_full = hp.npix2nside(Npix)
    nside_patch = args.nside_patch

    n_patches = hp.nside2npix(nside_patch)
    pixels_per_patch = (nside_full // nside_patch) ** 2

    print(f"Input: nside_full={nside_full}, {Nshells} shells")
    print(f"Patches: nside_patch={nside_patch} → {n_patches} patches × {pixels_per_patch} pixels each")

    if args.shell_index < 0:
        indices = list(range(Nshells))
    else:
        idx = int(args.shell_index)
        if not (0 <= idx < Nshells):
            raise IndexError("shell-index out of range")
        indices = [idx]

    # Load model and infer dimensions
    state = torch.load(model_path, map_location="cpu")
    weight_key = None
    for k in state.keys():
        if k.endswith("net.0.weight"):
            weight_key = k
            break
    if weight_key is None:
        raise KeyError("Could not find 'net.0.weight' in state_dict to infer input dims")

    in_features = state[weight_key].shape[1]
    hidden = state[weight_key].shape[0]

    dim_in = pixels_per_patch
    cond_dim = in_features - (dim_in + 1)  # +1 for the time embedding
    if cond_dim < 0:
        raise ValueError(
            f"Inferred cond_dim < 0 ({cond_dim}): model expects in_features={in_features} "
            f"but dim_in={dim_in}+1(time)={dim_in+1}. "
            f"Adjust --nside-patch to match training."
        )

    print(f"Model: dim_in={dim_in}, cond_dim={cond_dim}, hidden={hidden}")

    model = SmallMLP(dim_in=dim_in, cond_dim=cond_dim, hidden=hidden)
    model.load_state_dict(state)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    # Prepare conditioning vector
    cond_vec = None
    if cond_dim > 0:
        if args.params:
            cond_vec = load_params_vector(Path(args.params))
            if cond_dim != cond_vec.size:
                raise ValueError(f"cond_dim mismatch: model expects {cond_dim}, params.yml has {cond_vec.size}")
        else:
            cond_vec = np.zeros((cond_dim,), dtype=np.float32)

    # Init safe shell copy preserving original dtype
    shells_corrected = shells.copy()
    processed_indices = []
    diff_l2_list = []
    diff_max_list = []
    pred_patches_list = []

    for idx in indices:
        m = shells[idx]

        # Extract patches
        x0_patches = split_into_patches(m, nside_patch=nside_patch).astype(np.float32)
        
        # 2. Normalize inputs before feeding to the model
        x0_patches_norm = (x0_patches - data_mean) / data_std

        # Run model in batches
        pred_all_norm = np.zeros_like(x0_patches)
        batch_size = args.batch_size

        for start in range(0, n_patches, batch_size):
            end = min(start + batch_size, n_patches)
            B = end - start

            x0_t = torch.from_numpy(x0_patches_norm[start:end]).to(device)
            t_t = torch.full((B,), float(args.t), dtype=torch.float32, device=device)
            cond_t = torch.from_numpy(cond_vec).to(device).unsqueeze(0).expand(B, -1) if cond_dim > 0 else None

            with torch.no_grad():
                pred = model(x0_t, t_t, cond=cond_t)

            pred_all_norm[start:end] = pred.cpu().numpy()

       # 3. Un-normalize the predicted correction
        pred_phys = pred_all_norm * data_std

        # Reconstruct corrected map: original physical x0 + physical predicted u
        corrected_patches = x0_patches + pred_phys
        corrected_map = patches_to_map(corrected_patches, nside_patch=nside_patch, nside_full=nside_full)
        
        # Diagnostics
        diff = corrected_map - m.astype(np.float32)
        diff_l2 = float(np.linalg.norm(diff))
        diff_max = float(np.max(np.abs(diff)))

        print(f"  Shell {idx}: Original mean: {m.mean():.6g} | Corrected mean: {corrected_map.mean():.6g}")
        print(f"           pred L2={np.linalg.norm(pred_phys):.6g}, diff max={diff_max:.6g}")

        # Cast back to original dtype to save space
        shells_corrected[idx] = corrected_map.astype(shells.dtype)
        processed_indices.append(idx)
        diff_l2_list.append(diff_l2)
        diff_max_list.append(diff_max)
        pred_patches_list.append(pred_phys)

    # Save output
    out_dict = {k: data[k] for k in data.keys()}
    out_dict["shells"] = shells_corrected
    out_dict["corrected_index"] = np.asarray(processed_indices, dtype=np.int64)

    if args.diagnostic:
        out_dict["pred_patches"] = np.asarray(pred_patches_list, dtype=np.float32)
        out_dict["diff_l2"] = np.asarray(diff_l2_list, dtype=np.float32)
        out_dict["diff_max"] = np.asarray(diff_max_list, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)

    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Processed {len(processed_indices)} shell(s)")
    print(f"Patch config: nside_patch={nside_patch}, {n_patches} patches × {pixels_per_patch} px")
    print(f"Correction L2 range: [{np.min(diff_l2_list):.6g}, {np.max(diff_l2_list):.6g}]")
    print(f"Correction max-abs range: [{np.min(diff_max_list):.6g}, {np.max(diff_max_list):.6g}]")
    print(f"Saved output: {out_path}")


if __name__ == "__main__":
    main()
