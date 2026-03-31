#!/usr/bin/env python3
"""Perturb or destroy structure in the `shells` array of an NPZ file.

Example:
    python destroy_small_scales.py \
        --input /Users/david/testData/cosmo_000001/shells_nside=512.npz \
        --shuffle-alpha 0.4
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Destroy small-scale structure in shells")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/david/testData/cosmo_000009/shells_nside=512.npz",
        help="Path to input NPZ containing a `shells` array",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to output NPZ (default: input filename with '_destroyed_<mode>' suffix)",
    )
    parser.add_argument(
        "--shuffle-alpha",
        type=float,
        default=0.4,
        help="Fraction of pixels to shuffle per shell (0=no change, 1=full shuffle)",
    )
    parser.add_argument("--sigma", type=float, default=0.01, help="Std dev of Gaussian noise")
    parser.add_argument("--mean", type=float, default=0.0, help="Mean of Gaussian noise")
    # keep only shuffle-related controls: --shuffle-alpha
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _full_shuffle_per_shell(shells: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(shells)
    for i in range(shells.shape[0]):
        flat = shells[i].reshape(-1)
        perm = rng.permutation(flat.size)
        out[i] = flat[perm].reshape(shells[i].shape)
    return out


def _partial_shuffle_per_shell(shells: np.ndarray, rng: np.random.Generator, alpha: float) -> np.ndarray:
    """Shuffle only a fraction `alpha` of pixels per shell.

    alpha=0: return copy of shells
    alpha=1: full shuffle
    0<alpha<1: randomly select ceil(alpha*N) indices and permute them among themselves
    """
    if alpha <= 0.0:
        return shells.copy()
    if alpha >= 1.0:
        return _full_shuffle_per_shell(shells, rng)

    out = shells.copy()
    for i in range(shells.shape[0]):
        flat = shells[i].reshape(-1)
        N = flat.size
        k = int(round(alpha * N))
        if k <= 1:
            # nothing meaningful to permute
            continue
        idx = rng.choice(N, size=k, replace=False)
        permuted = idx.copy()
        rng.shuffle(permuted)
        flat_copy = flat.copy()
        flat_copy[idx] = flat[permuted]
        out[i] = flat_copy.reshape(shells[i].shape)
    return out


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_noisy_shuffle{input_path.suffix}")

    with np.load(input_path, allow_pickle=False) as data:
        if "shells" not in data:
            raise KeyError(f"Expected key 'shells' in {input_path}")

        shells = data["shells"]
        rng = np.random.default_rng(args.seed)
        alpha = float(args.shuffle_alpha)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("--shuffle-alpha must be in [0, 1]")
        noisy_shells = _partial_shuffle_per_shell(shells, rng, alpha)

        out_dict = {k: data[k] for k in data.files}
        out_dict["shells"] = noisy_shells

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out_dict)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("Mode:   shuffle (partial)")
    print(f"shuffle-alpha: {args.shuffle_alpha}")
    print(f"seed: {args.seed}")
    print(f"Shells shape: {noisy_shells.shape}, dtype: {noisy_shells.dtype}")


if __name__ == "__main__":
    main()
