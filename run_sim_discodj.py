from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from discodj import DiscoDJ


@dataclass
class InitialConditions:
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray | None
    a_ini: float
    boxsize: float
    res: int
    source: Path

def latest_state(array: np.ndarray) -> np.ndarray:
    """Return the last time slice if the array has a leading time axis."""
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[-1]
    return arr


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


def load_npz_ic(npz_path: Path, dtype: np.dtype = np.float32) -> InitialConditions:
    """Load the converted NPZ bundle and return flattened arrays."""
    npz_path = npz_path.expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"IC archive not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    pos = np.asarray(data["pos"], dtype=dtype)
    vel = np.asarray(data["vel"], dtype=dtype)
    mass = np.asarray(data["mass"], dtype=dtype) if "mass" in data.files else None
    a_ini = float(data.get("a_ini", 0.01))
    boxsize = float(data.get("boxsize", 900.0))
    n_particles = pos.shape[0]
    res = infer_resolution(n_particles)
    return InitialConditions(pos=pos, vel=vel, mass=mass, a_ini=a_ini, boxsize=boxsize, res=res, source=npz_path)


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


def run_simulation(
    ic_path: Path,
    a_end: float,
    n_steps: int,
    res_pm: int,
    stepper: str,
    method: str,
    collect_all: bool,
    dtype: str,
) -> Tuple[DiscoDJ, np.ndarray, np.ndarray, np.ndarray]:
    dtype_np = np.float32 if dtype == "float32" else np.float64
    ic = load_npz_ic(ic_path, dtype=dtype_np)

    print(
        f"Loaded ICs from {ic.source} -> res={ic.res}^3, a_ini={ic.a_ini:.5f}, boxsize={ic.boxsize} Mpc/h"
    )

    dj = DiscoDJ(dim=3, res=ic.res, boxsize=ic.boxsize, precision="single" if dtype == "float32" else "double")
    dj = dj.with_timetables()
    dj = dj.with_external_ics(pos=ic.pos, vel=ic.vel)

    X, P, a_hist = dj.run_nbody(
        a_ini=ic.a_ini,
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


def plot_density_slice(
    positions: np.ndarray,
    boxsize: float,
    slice_axis: int = 2,
    slice_center: float | None = None,
    slice_thickness: float = 5.0,
    grid: int = 256,
    output_dir: Path | None = None,
) -> Path:
    """Bin a thin slab of particles onto a 2D grid and plot the density contrast."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Matplotlib is required for plotting. Install it via 'conda install matplotlib' or rerun without --plot."
        ) from exc

    # Accept several common shapes from DiscoDJ:
    # - (N, 3)
    # - (T, N, 3)
    # - (nx, ny, nz, 3)
    # - (T, nx, ny, nz, 3)
    arr = np.asarray(positions)
    if arr.ndim == 2 and arr.shape[1] == 3:
        pos = arr
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        # (T, N, 3) -> take last snapshot
        pos = arr[-1]
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        # (nx, ny, nz, 3) -> flatten spatial grid
        pos = arr.reshape(-1, 3)
    elif arr.ndim == 5 and arr.shape[-1] == 3:
        # (T, nx, ny, nz, 3) -> take last snapshot and flatten
        pos = arr[-1].reshape(-1, 3)
    else:
        raise ValueError(f"Expected positions with final dimension 3, got {arr.shape}")

    center = slice_center if slice_center is not None else boxsize / 2.0
    half = slice_thickness / 2.0
    mask = (pos[:, slice_axis] >= center - half) & (pos[:, slice_axis] <= center + half)
    slab = pos[mask]
    if slab.size == 0:
        raise ValueError("Slice selection yielded zero particles. Adjust slice thickness/center.")

    axes = [0, 1, 2]
    axes.remove(slice_axis)
    hist, xedges, yedges = np.histogram2d(
        slab[:, axes[0]],
        slab[:, axes[1]],
        bins=grid,
        range=[[0.0, boxsize], [0.0, boxsize]],
    )
    density = hist / np.mean(hist) - 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    extent = (0, boxsize, 0, boxsize)
    norm = SymLogNorm(linthresh=1e-3, linscale=1.0, vmin=density.min(), vmax=density.max())
    im = ax.imshow(density.T, origin="lower", cmap="magma", extent=extent, norm=norm)
    cbar = fig.colorbar(im, ax=ax, label="δ (symlog)")
    ax.set_xlabel("x [Mpc/h]")
    ax.set_ylabel("y [Mpc/h]")
    ax.set_title(
        f"Density slice @ axis {slice_axis}, center={center:.1f} Mpc/h, thickness={slice_thickness:.1f}"
    )
    plt.tight_layout()

    if output_dir is None:
        output_dir = Path(__file__).with_name("outputs").joinpath("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"density_slice_axis{slice_axis}_center{center:.1f}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved density slice to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiscoDJ using PKDGRAV initial conditions.")
    parser.add_argument(
        "--ic-file",
        type=Path,
        required=True,
        help="Path to the converted CosmoML.00000.npz produced by tools/convert_pkdgrav_ic.py",
    )
    parser.add_argument("--a-end", type=float, default=1.0, help="Final scale factor (default: 1.0)")
    parser.add_argument("--n-steps", type=int, default=10, help="Number of integration steps (default: 10)")
    parser.add_argument("--res-pm", type=int, default=512, help="PM grid resolution (default: 512)")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dj, X, P, a_hist = run_simulation(
        ic_path=args.ic_file,
        a_end=args.a_end,
        n_steps=args.n_steps,
        res_pm=args.res_pm,
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
            grid=args.slice_grid
        )


if __name__ == "__main__":
    main()
