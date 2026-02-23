#!/usr/bin/env python3
"""Downsample DiscoDJ-ready NPZ initial conditions to a lower cubic resolution."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def infer_resolution(n_particles: int) -> int:
    """Infer the cubic resolution from the flattened particle count."""
    res = round(n_particles ** (1.0 / 3.0))
    if res ** 3 != n_particles:
        raise ValueError(
            f"Particle count {n_particles:,} is not a perfect cube."
            " Provide data that can be reshaped to (res, res, res)."
        )
    if res % 4 != 0:
        raise ValueError(f"DiscoDJ currently requires res to be a multiple of 4 (got {res}).")
    return res


def _load_npz(npz_path: Path, dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, Dict[str, np.ndarray]]:
    npz_path = npz_path.expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"IC archive not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    pos = np.asarray(data["pos"], dtype=dtype)
    vel = np.asarray(data["vel"], dtype=dtype)
    mass = np.asarray(data["mass"], dtype=dtype) if "mass" in data.files else None
    metadata = {k: data[k] for k in data.files if k not in {"pos", "vel", "mass"}}
    metadata.setdefault("n_particles", np.array(pos.shape[0], dtype=np.int64))
    metadata.setdefault("source", np.array(str(npz_path)))
    return pos, vel, mass, metadata


def downsample_ic(
    npz_path: Path,
    output_path: Path | None,
    target_res: int,
    dtype: str = "float32",
    seed: int | None = None,
    renormalize_mass: bool = True,
) -> Path:
    """Create a lower-resolution NPZ by randomly sampling particles to a cube of ``target_res``."""
    dtype_np = np.float32 if dtype == "float32" else np.float64
    pos, vel, mass, metadata = _load_npz(npz_path, dtype=dtype_np)
    n_particles = pos.shape[0]
    res_in = infer_resolution(n_particles)

    if target_res >= res_in:
        raise ValueError(f"target_res must be smaller than the source resolution (got {target_res} vs {res_in}).")
    if target_res % 4 != 0:
        raise ValueError("target_res must be a multiple of 4 for DiscoDJ.")

    target_n = target_res ** 3
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(n_particles, size=target_n, replace=False)
    keep_idx.sort()

    pos_ds = pos[keep_idx]
    vel_ds = vel[keep_idx]
    mass_ds = None
    if mass is not None:
        mass_ds = mass[keep_idx]
        if renormalize_mass:
            scale = n_particles / target_n
            mass_ds = mass_ds * scale

    metadata["n_particles"] = np.array(target_n, dtype=np.int64)
    metadata["downsampled_from_res"] = np.array(res_in, dtype=np.int32)
    metadata["downsampled_seed"] = np.array(seed if seed is not None else -1, dtype=np.int64)
    metadata["downsampled_method"] = np.array("random_choice")

    if output_path is None:
        output_path = npz_path.with_name(f"{npz_path.stem}_res{target_res}.npz")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"pos": pos_ds, "vel": vel_ds}
    if mass_ds is not None:
        save_kwargs["mass"] = mass_ds
    save_kwargs.update(metadata)

    np.savez_compressed(output_path, **save_kwargs)
    print(
        f"Downsampled {npz_path} -> {output_path} | res {res_in}^3 -> {target_res}^3 (factor {(res_in / target_res):.1f})"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample a DiscoDJ NPZ IC archive.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the source CosmoML_IC npz.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (defaults to <input>_res<target_res>.npz).",
    )
    parser.add_argument(
        "--target-res",
        type=int,
        required=True,
        help="Target cubic resolution (must be < input res and divisible by 4).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Datatype to use when loading/storing arrays (default: float32).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the sampler (default: 0).")
    parser.add_argument(
        "--no-renorm-mass",
        action="store_true",
        help="Disable mass renormalization after downsampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downsample_ic(
        npz_path=args.input,
        output_path=args.output,
        target_res=args.target_res,
        dtype=args.dtype,
        seed=args.seed,
        renormalize_mass=not args.no_renorm_mass,
    )


if __name__ == "__main__":
    main()
