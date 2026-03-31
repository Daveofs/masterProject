from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from discodj import DiscoDJ
from read_tipsy_file import read_tipsy
from visualize import plot_density_slice


@dataclass
class GridSettings:
    res: int


Lbox: float = 900
a_ini: float = 0.02
gs = GridSettings(res=512)

def load_ic(filename_init: Path, dtype: np.dtype = np.float32):
    """Load the converted NPZ bundle and return flattened arrays."""

    print(f'Reading PKDGRAV ICs from {filename_init}')
    p_init, p_header_init = read_tipsy(filename_init, Lbox)

        # Combine position and velocity arrays
    external_ics = np.c_[
        p_init['x'],  p_init['y'],  p_init['z'],
        p_init['vx'], p_init['vy'], p_init['vz']
    ]

    # Convert PKD velocities to DISCO-DJ units
    v_factor = a_ini**2 / np.sqrt(8*np.pi/3) * Lbox
    external_ics[:, 3:] *= v_factor
    
    pos_sorted, vel_sorted = sort_for_lagrangian_x(external_ics[:, :3], external_ics[:, 3:], nx=gs.res)
    return pos_sorted, vel_sorted

def sort_for_lagrangian_x(arr_pos, arr_vel, nx=512):
    """
    Sort a (N,3) particle array by x-coordinate
    so that when reshaped into a (nx, nx, nx, 3) Lagrangian grid,
    slices along axis-0 increase monotonically in x.

    Parameters
    ----------
    arr_pos : ndarray (N, 3)
        Particle positions.
    arr_vel : ndarray (N, 3)
        Particle velocities.
    nx : int
        Grid size per dimension.

    Returns
    -------
    arr_pos_sorted, arr_vel_sorted : ndarrays
        Sorted position and velocity arrays.
    """
    idx = np.argsort(arr_pos[:, 0]) # sort by x-coordinate
    return arr_pos[idx], arr_vel[idx]


def latest_state(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim in (2, 4):
        return arr
    if arr.ndim in (3, 5):
        return arr[-1]
    raise ValueError(f"Unsupported state shape {arr.shape}")


def run_simulation(
    ic_path: Path,
    boxsize: float,
    a_ini_value: float,
    a_end: float,
    n_steps: int,
    res_pm: int,
    res: int | None,
    stepper: str,
    method: str,
    collect_all: bool,
    dtype: str,
) -> Tuple[DiscoDJ, np.ndarray, np.ndarray, np.ndarray]:
    global Lbox, a_ini, gs

    Lbox = float(boxsize)
    a_ini = float(a_ini_value)
    gs = GridSettings(res=int(res) if res is not None else int(res_pm))

    dtype_np = np.float32 if dtype == "float32" else np.float64
    pos, vel = load_ic(ic_path, dtype=dtype_np)

    sim_res = int(res) if res is not None else int(round(pos.shape[0] ** (1.0 / 3.0)))
    dj = DiscoDJ(dim=3, res=sim_res, boxsize=boxsize, precision="single" if dtype == "float32" else "double")
    dj = dj.with_timetables()
    dj = dj.with_external_ics(pos=pos, vel=vel)

    X, P, a_hist = dj.run_nbody(
        a_ini=a_ini_value,
        a_end=a_end,
        n_steps=n_steps,
        res_pm=res_pm,
        stepper=stepper,
        method=method,
        collect_all=collect_all,
        return_all_a=collect_all,
        ic_method="none",
        convert_to_numpy=True,
    )

    print(f"Simulation complete -> returned positions {X.shape}, velocities {P.shape}")
    return dj, np.asarray(X), np.asarray(P), np.asarray(a_hist)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiscoDJ using PKDGRAV initial conditions.")
    parser.add_argument(
        "--ic-file",
        type=Path,
        required=True,
        help="Path to the converted CosmoML.00000.npz",
    )
    parser.add_argument("--a-end", type=float, default=1.0, help="Final scale factor (default: 1.0)")
    parser.add_argument("--a-ini", type=float, required=True, help="Initial scale factor for the IC snapshot.")
    parser.add_argument("--boxsize", type=float, required=True, help="Simulation box size in Mpc/h.")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of integration steps (default: 10)")
    parser.add_argument("--res-pm", type=int, default=512, help="PM grid resolution (default: 512)")
    parser.add_argument(
        "--res",
        type=int,
        default=None,
        help="Particle mesh resolution per axis for DiscoDJ (default: inferred from IC count).",
    )
    parser.add_argument(
        "--stepper",
        choices=("bullfrog", "fastpm", "symplectic"),
        default="bullfrog",
        help="Time integrator (default: bullfrog)",
    )
    parser.add_argument("--method", choices=("pm", "nufftpm"), default="pm", help="Force solver (default: pm)")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Numerical precision for the ICs (default: float32)",
    )
    parser.add_argument(
        "--collect-all",
        action="store_true",
        help="Keep trajectories for every step (higher memory).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a 2D density slice from the final particle snapshot.",
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=2,
        help="Axis normal for the slice (0->x,1->y,2->z).",
    )
    parser.add_argument(
        "--slice-center",
        type=float,
        default=None,
        help="Slice center in Mpc/h (default: boxsize/2).",
    )
    parser.add_argument(
        "--slice-thickness",
        type=float,
        default=5.0,
        help="Slice thickness in Mpc/h (default: 5).",
    )
    parser.add_argument(
        "--slice-grid",
        type=int,
        default=256,
        help="2D histogram resolution for the slice plot (default: 256).",
    )
    parser.add_argument(
        "--save-final",
        type=Path,
        default=None,
        help="Optional path to store the final particle positions/velocities as NPZ.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save outputs (plots, final snapshots). Default is a 'outputs' folder next to this script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dj, X, P, a_hist = run_simulation(
        ic_path=args.ic_file,
        boxsize=args.boxsize,
        a_ini_value=args.a_ini,
        a_end=args.a_end,
        n_steps=args.n_steps,
        res_pm=args.res_pm,
        res=args.res,
        stepper=args.stepper,
        method=args.method,
        collect_all=args.collect_all,
        dtype=args.dtype,
    )

    if args.save_final is not None:
        args.save_final.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_final,
            pos=latest_state(X),
            vel=latest_state(P),
            a_hist=a_hist,
        )
        print(f"Saved final snapshot to {args.save_final}")

    if args.plot:
        plot_density_slice(
            positions=X,
            boxsize=dj.boxsize,
            slice_axis=args.slice_axis,
            slice_center=args.slice_center,
            slice_thickness=args.slice_thickness,
            grid=args.slice_grid,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
