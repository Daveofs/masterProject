#!/usr/bin/env python3
"""Apply trained `flow_mlp.pth` correction to a shell.

The script:
- loads a model state dict and infers `dim_in` and `cond_dim` from `net.0.weight`.
- loads an NPZ with `shells` (shape [Nshells, Npix]) and downsamples the chosen shell
  to `nside_small` using `healpy.ud_grade` (power selectable).
- splits into `n_patches` if requested and runs the model to predict `u = x1 - x0`.
- reconstructs a corrected shell `x0 + u` and saves a small NPZ with the original
  downsampled shell and corrected version.

Example:
  python apply_flow_correction.py --model models/flow_mlp.pth \
      --input /Users/david/testData/cosmo_000001/shells_nside=512.npz --nside-small 128
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import healpy as hp
import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self, dim_in: int, cond_dim: int = 0, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in + 1 + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_in),
        )

    def forward(self, x, t, cond=None):
        T = t.view(-1, 1)
        if cond is None:
            inp = torch.cat([x, T], dim=-1)
        else:
            inp = torch.cat([x, T, cond], dim=-1)
        return self.net(inp)


def load_params_vector(params_path: Path):
    params = yaml.safe_load(params_path.read_text())
    keys = sorted(params.keys())
    return np.array([float(params[k]) for k in keys], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="/Users/david/Library/CloudStorage/OneDrive-ETHZurich/ETH-Material/Master Project/github/models/flow_mlp.pth")
    p.add_argument("--input", type=str, default="/Users/david/testData/cosmo_000010/shells_nside=512_noisy_shuffle.npz")
    p.add_argument("--params", type=str, default="/Users/david/testData/cosmo_000010/params.yml", help="Optional params.yml path for conditioning vector")
    p.add_argument("--nside-small", type=int, default=128)
    p.add_argument("--n-patches", type=int, default=4)
    p.add_argument("--power", type=float, default=0.0, help="healpy.ud_grade power used for downsampling (like train)")
    p.add_argument("--shell-index", type=int, default=-1, help="Index of shell to process; -1=process all shells")
    p.add_argument("--t", type=float, default=0.0, help="t value to pass to model (0..1).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--diagnostic", action="store_true", help="Print diagnostic stats and save prediction arrays to output NPZ")
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

    data = dict(np.load(input_path, allow_pickle=False))
    if "shells" not in data:
        raise KeyError("Input NPZ must contain 'shells' array")

    shells = data["shells"]
    Nshells, Npix = shells.shape

    if args.shell_index < 0:
        indices = list(range(Nshells))
    else:
        idx = int(args.shell_index)
        if not (0 <= idx < Nshells):
            raise IndexError("shell-index out of range")
        indices = [idx]

    # Downsample like training low map used power=0
    nside_small = int(args.nside_small)
    npix_small = hp.nside2npix(nside_small)
    # split into patches if requested
    n_patches = int(args.n_patches)
    if n_patches < 1:
        raise ValueError("n_patches must be >= 1")
    if npix_small % n_patches != 0:
        raise ValueError("n_patches must evenly divide npix_small")
    patch_npix = npix_small // n_patches

    # determine dim_in expected by model
    state = torch.load(model_path, map_location="cpu")
    # find net.0.weight key
    weight_key = None
    for k in state.keys():
        if k.endswith("net.0.weight"):
            weight_key = k
            break
    if weight_key is None:
        raise KeyError("Could not find 'net.0.weight' in state_dict to infer input dims")
    in_features = state[weight_key].shape[1]
    hidden = state[weight_key].shape[0]

    dim_in = patch_npix
    cond_dim = in_features - (dim_in + 1)
    if cond_dim < 0:
        raise ValueError("Inferred cond_dim < 0: model/input mismatch. Adjust --nside-small/--n-patches to match training.")

    # build model and load weights
    model = SmallMLP(dim_in=dim_in, cond_dim=cond_dim, hidden=hidden)
    model.load_state_dict(state)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    # prepare shell-independent conditioning (broadcast per shell/patch)
    cond_vec = None
    if cond_dim > 0:
        if args.params:
            cond_vec = load_params_vector(Path(args.params))
            if cond_dim != cond_vec.size:
                raise ValueError(f"cond_dim mismatch: model expects {cond_dim}, params.yml has {cond_vec.size}")
        else:
            cond_vec = np.zeros((cond_dim,), dtype=np.float32)

    nside_orig = hp.npix2nside(Npix)
    shells_corrected = np.asarray(shells, dtype=np.float32).copy()
    orig_down_list = []
    corrected_down_list = []
    corrected_up_list = []
    processed_indices = []
    diff_l2_list = []
    diff_max_list = []
    pred_l2_list = []
    pred_max_list = []
    pred_mean_list = []
    pred_patches_list = []

    for idx in indices:
        m = shells[idx]
        m_down = hp.ud_grade(m, nside_out=nside_small, power=args.power).astype(np.float32)

        if n_patches == 1:
            x0 = m_down.reshape(1, -1)
        else:
            x0 = m_down.reshape(n_patches, patch_npix)

        B = x0.shape[0]
        x0_t = torch.from_numpy(x0.astype(np.float32)).to(device)
        t_t = torch.full((B,), float(args.t), dtype=torch.float32, device=device)

        if cond_dim > 0:
            cond_t = torch.from_numpy(cond_vec.astype(np.float32)).to(device).unsqueeze(0).repeat(B, 1)
        else:
            cond_t = None

        with torch.no_grad():
            pred = model(x0_t, t_t, cond=cond_t)
        pred_np = pred.cpu().numpy()

        corrected_patches = (x0 + pred_np).astype(np.float32)
        corrected_down = corrected_patches.reshape(npix_small)
        corrected_up = hp.ud_grade(
            corrected_down,
            nside_out=nside_orig,
            order_in="RING",
            order_out="RING",
            power=args.power,
        ).astype(np.float32)

        shells_corrected[idx] = corrected_up

        try:
            pred_flat = pred_np.reshape(pred_np.shape[0], -1)
            pred_l2 = np.linalg.norm(pred_flat, axis=1)
            pred_max = np.max(np.abs(pred_flat), axis=1)
            pred_mean = np.mean(pred_flat, axis=1)
        except Exception:
            pred_l2 = np.array([np.linalg.norm(pred_np)], dtype=np.float32)
            pred_max = np.array([np.max(np.abs(pred_np))], dtype=np.float32)
            pred_mean = np.array([np.mean(pred_np)], dtype=np.float32)

        diff_down = corrected_down - m_down
        diff_l2 = float(np.linalg.norm(diff_down))
        diff_max = float(np.max(np.abs(diff_down)))

        orig_down_list.append(m_down)
        corrected_down_list.append(corrected_down)
        corrected_up_list.append(corrected_up)
        processed_indices.append(idx)
        diff_l2_list.append(diff_l2)
        diff_max_list.append(diff_max)
        pred_l2_list.append(pred_l2)
        pred_max_list.append(pred_max)
        pred_mean_list.append(pred_mean)
        pred_patches_list.append(pred_np.astype(np.float32))

    out_dict = {k: data[k] for k in data.keys()}
    out_dict["shells"] = shells_corrected
    out_dict["corrected_up"] = np.asarray(corrected_up_list, dtype=np.float32)
    out_dict["orig_down"] = np.asarray(orig_down_list, dtype=np.float32)
    out_dict["corrected_down"] = np.asarray(corrected_down_list, dtype=np.float32)
    out_dict["corrected_index"] = np.asarray(processed_indices, dtype=np.int64)

    print(f"cond_dim inferred: {cond_dim}")
    print(f"Processed {len(processed_indices)} shell(s): first={processed_indices[0]}, last={processed_indices[-1]}")
    print(f"corrected - orig L2 range: min={np.min(diff_l2_list):.6g}, max={np.max(diff_l2_list):.6g}")
    print(f"corrected - orig max abs range: min={np.min(diff_max_list):.6g}, max={np.max(diff_max_list):.6g}")

    if args.diagnostic:
        out_dict["pred_patches"] = np.asarray(pred_patches_list, dtype=np.float32)
        out_dict["diff_l2"] = np.asarray(diff_l2_list, dtype=np.float32)
        out_dict["diff_max"] = np.asarray(diff_max_list, dtype=np.float32)
        out_dict["pred_l2"] = np.asarray(pred_l2_list, dtype=np.float32)
        out_dict["pred_max"] = np.asarray(pred_max_list, dtype=np.float32)
        out_dict["pred_mean"] = np.asarray(pred_mean_list, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_dict)

    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    if len(processed_indices) == 1:
        print(f"Processed shell index: {processed_indices[0]}")
    else:
        print(f"Processed all shell indices: 0..{Nshells-1}")
    print(f"Downsampled nside: {nside_small} npix={npix_small} patches={n_patches}")
    print(f"Upsampled back to nside: {nside_orig}")
    print(f"Saved output: {out_path}")


if __name__ == "__main__":
    main()
