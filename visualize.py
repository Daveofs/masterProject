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