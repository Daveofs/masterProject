from pathlib import Path
import numpy as np


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
    selected_snapshot = None
    if arr.ndim == 2 and arr.shape[1] == 3:
        pos = arr
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        # (T, N, 3) -> prefer latest snapshot with finite entries
        finite_snapshots = np.isfinite(arr).all(axis=(1, 2))
        if finite_snapshots.any():
            selected_snapshot = int(np.flatnonzero(finite_snapshots)[-1])
            pos = arr[selected_snapshot]
            if selected_snapshot != arr.shape[0] - 1:
                print(
                    f"Warning: final snapshot is non-finite; using snapshot index {selected_snapshot} instead."
                )
        else:
            pos = arr[-1]
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        # (nx, ny, nz, 3) -> flatten spatial grid
        pos = arr.reshape(-1, 3)
    elif arr.ndim == 5 and arr.shape[-1] == 3:
        # (T, nx, ny, nz, 3) -> prefer latest snapshot with finite entries and flatten
        finite_snapshots = np.isfinite(arr).all(axis=(1, 2, 3, 4))
        if finite_snapshots.any():
            selected_snapshot = int(np.flatnonzero(finite_snapshots)[-1])
            pos = arr[selected_snapshot].reshape(-1, 3)
            if selected_snapshot != arr.shape[0] - 1:
                print(
                    f"Warning: final snapshot is non-finite; using snapshot index {selected_snapshot} instead."
                )
        else:
            pos = arr[-1].reshape(-1, 3)
    else:
        raise ValueError(f"Expected positions with final dimension 3, got {arr.shape}")

    finite_mask = np.isfinite(pos).all(axis=1)
    pos = pos[finite_mask]

    center = slice_center if slice_center is not None else boxsize / 2.0
    half = slice_thickness / 2.0
    if pos.size == 0:
        print(
            f"Warning: all particle positions are non-finite (shape {arr.shape}). Saving an empty density map."
        )
        density = np.zeros((grid, grid), dtype=np.float64)
        used_empty_fallback = True
    else:
        mask = (pos[:, slice_axis] >= center - half) & (pos[:, slice_axis] <= center + half)
        slab = pos[mask]
        if slab.size == 0:
            print(
                "Warning: slice selection yielded zero particles; using all finite particles for 2D projection."
            )
            slab = pos
        axes = [0, 1, 2]
        axes.remove(slice_axis)
        hist, _, _ = np.histogram2d(
            slab[:, axes[0]],
            slab[:, axes[1]],
            bins=grid,
            range=[[0.0, boxsize], [0.0, boxsize]],
        )
        mean_hist = np.mean(hist)
        if mean_hist > 0.0:
            density = hist / mean_hist - 1.0
        else:
            density = np.zeros_like(hist)
        used_empty_fallback = False

    fig, ax = plt.subplots(figsize=(6, 6))
    extent = (0, boxsize, 0, boxsize)
    if used_empty_fallback or np.allclose(density, density.flat[0]):
        im = ax.imshow(density.T, origin="lower", cmap="magma", extent=extent)
    else:
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