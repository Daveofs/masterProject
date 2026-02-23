#!/usr/bin/env python3
"""Convert PKDGRAV Tipsy snapshots into DiscoDJ-friendly NumPy bundles.

This script extracts particle positions/velocities/masses from a CosmoML.00000-style
Tipsy file and stores them in a compressed ``.npz`` archive together with metadata
needed by ``run_sim_discodj.py``.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np

try:
    import pynbody  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency hint for the user
    raise SystemExit(
        "pynbody is required to convert PKDGRAV initial conditions.\n"
        "Install it inside your DiscoDJ environment, e.g.\n"
        "  conda install -y -c conda-forge pynbody\n"
        "or\n"
        "  pip install pynbody"
    ) from exc

try:  # Recent pynbody versions renamed the conversion error class
    from pynbody.units import UnitConversionError  # type: ignore
except ImportError:  # pragma: no cover - fallback for pynbody>=2.0
    from pynbody.units import UnitsException as UnitConversionError  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PKDGRAV Tipsy ICs to DiscoDJ npz format.")
    parser.add_argument(
        "tipsy_file",
        type=pathlib.Path,
        help="Path to CosmoML.00000 (Tipsy) produced by pkdgrav3.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output .npz path (defaults to <tipsy_file>.npz in the same directory).",
    )
    parser.add_argument(
        "--boxsize",
        type=float,
        default=900.0,
        help="Comoving box size in Mpc/h to store alongside the ICs (default: 900).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Floating point precision to store in the .npz (default: float32).",
    )
    parser.add_argument(
        "--pos-unit",
        default="Mpc h^-1",
        help=(
            "Target unit for particle positions (pynbody unit string)."
            " Use empty string to keep the native units."
        ),
    )
    parser.add_argument(
        "--vel-unit",
        default="km s^-1",
        help=(
            "Target unit for particle velocities (pynbody unit string)."
            " Use empty string to keep the native units."
        ),
    )
    parser.add_argument(
        "--a-ini",
        type=float,
        default=None,
        help="Override the scale factor stored in the Tipsy header (default: value from file).",
    )
    return parser.parse_args()


def convert_units(sim, field: str, unit: Optional[str]) -> str:
    """Convert ``sim[field]`` to ``unit`` if possible, otherwise keep native units."""
    arr = sim[field]
    original_unit = str(arr.units)
    if unit:
        try:
            arr.convert_units(unit)
        except UnitConversionError:
            print(
                f"Warning: could not convert '{field}' from {original_unit} to {unit}; "
                "keeping native units."
            )
            return original_unit
    return str(arr.units)


def convert_tipsy_to_npz(args: argparse.Namespace) -> pathlib.Path:
    tipsy_path = args.tipsy_file.expanduser().resolve()
    if not tipsy_path.exists():
        raise FileNotFoundError(tipsy_path)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tipsy_path.with_suffix(".npz")
    )

    sim = pynbody.load(str(tipsy_path))
    sim.physical_units()

    pos_unit = convert_units(sim, "pos", args.pos_unit)
    vel_unit = convert_units(sim, "vel", args.vel_unit)

    dtype = np.float32 if args.dtype == "float32" else np.float64

    pos = np.asarray(sim["pos"], dtype=dtype)
    vel = np.asarray(sim["vel"], dtype=dtype)
    mass = np.asarray(sim["mass"], dtype=dtype)

    a_ini = float(args.a_ini if args.a_ini is not None else sim.properties.get("time", 1.0))

    metadata = {
        "a_ini": np.array(a_ini, dtype=np.float32),
        "boxsize": np.array(args.boxsize, dtype=np.float32),
        "pos_unit": np.array(pos_unit),
        "vel_unit": np.array(vel_unit),
        "n_particles": np.array(pos.shape[0], dtype=np.int64),
        "source": np.array(str(tipsy_path)),
    }

    np.savez_compressed(output_path, pos=pos, vel=vel, mass=mass, **metadata)

    print(f"Saved {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
    return output_path


def main() -> None:
    args = parse_args()
    try:
        output_path = convert_tipsy_to_npz(args)
    except Exception as exc:  # pragma: no cover - bubble up with context
        raise SystemExit(f"Conversion failed: {exc}") from exc
    print(f"✅ Conversion complete -> {output_path}")


if __name__ == "__main__":
    main()
