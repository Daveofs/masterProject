from pathlib import Path
import numpy as np


def plot_shells(
    npz_path: str | Path,
    z_bin: int = 5,
    nside: int = 512,
    output_dir: Path | None = None,
    plot_logarithmic: bool = False,
    name: str | None = None,
) -> Path:
    """Load a shell file (.npz or .fits) and plot it as a HEALPix map."""
    try:
        import healpy as hp
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "healpy (and matplotlib) is required for shell plotting. Install via 'conda install healpy matplotlib'."
        ) from exc

    npz_path = Path(npz_path)
    if npz_path.suffix.lower() == ".fits":
        m = np.asarray(hp.read_map(npz_path, nest=False, dtype=np.float32, verbose=False), dtype=float)
        if not hp.isnpixok(int(m.size)):
            raise ValueError(f"FITS map has invalid HEALPix size: {m.size}")
        nside = hp.npix2nside(int(m.size))
    else:
        data = np.load(npz_path, allow_pickle=False)

        if "b" in data:
            b = np.asarray(data["b"])
        elif "shells" in data:
            b = np.asarray(data["shells"])
        else:
            raise KeyError(f"Expected key 'b' (or 'shells') in {npz_path}, found keys: {list(data.keys())}")

        if b.ndim != 2:
            raise ValueError(f"Expected 'b' to be 2D [Npix, Nz], got shape {b.shape}")

        if "pix" in data:
            pix = np.asarray(data["pix"], dtype=np.int64)
            if b.shape[0] == pix.shape[0]:
                if z_bin < 0 or z_bin >= b.shape[1]:
                    raise IndexError(f"z_bin={z_bin} out of range for shape {b.shape}")
                m = np.zeros(hp.nside2npix(nside), dtype=float)
                m[pix] = b[:, z_bin]
            elif b.shape[1] == pix.shape[0]:
                if z_bin < 0 or z_bin >= b.shape[0]:
                    raise IndexError(f"z_bin={z_bin} out of range for shape {b.shape}")
                m = np.zeros(hp.nside2npix(nside), dtype=float)
                m[pix] = b[z_bin, :]
            else:
                raise ValueError(
                    f"Could not align 'pix' (len={pix.shape[0]}) with shell array shape {b.shape}."
                )
        else:
            npix0_valid = hp.isnpixok(int(b.shape[0]))
            npix1_valid = hp.isnpixok(int(b.shape[1]))
            if not (npix0_valid or npix1_valid):
                raise ValueError(
                    f"No 'pix' key and neither dimension of shell array {b.shape} is a valid HEALPix npix."
                )

            if npix1_valid:
                inferred_nside = hp.npix2nside(int(b.shape[1]))
                if z_bin < 0 or z_bin >= b.shape[0]:
                    raise IndexError(f"z_bin={z_bin} out of range for shape {b.shape}")
                nside = inferred_nside
                m = np.asarray(b[z_bin, :], dtype=float)
            else:
                inferred_nside = hp.npix2nside(int(b.shape[0]))
                if z_bin < 0 or z_bin >= b.shape[1]:
                    raise IndexError(f"z_bin={z_bin} out of range for shape {b.shape}")
                nside = inferred_nside
                m = np.asarray(b[:, z_bin], dtype=float)


    if output_dir is None:
        output_dir = Path(__file__).with_name("outputs").joinpath("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if name is not None and len(name.strip()) == 0:
        raise ValueError("name must be a non-empty string when provided")

    if plot_logarithmic:
        default_name = f"shell_mollview_symlog_zbin{z_bin}_nside{nside}.png"
        file_name = name if name is not None else default_name
        if not file_name.lower().endswith(".png"):
            file_name = f"{file_name}.png"
        out_path = output_dir / file_name
        projected = hp.mollview(
            m,
            nest=False,
            cbar=False,
            notext=True,
            title="",
            return_projected_map=True,
        )
        plt.close()

        valid = np.isfinite(projected) & (projected > hp.UNSEEN / 10.0)
        vals = projected[valid]
        if vals.size == 0:
            raise ValueError("Projected shell map has no valid pixels for SymLogNorm plot.")

        linthresh = max(float(np.nanpercentile(np.abs(vals), 5)), 1e-8)
        display = np.full_like(projected, np.nan, dtype=float)
        display[valid] = projected[valid]

        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="white", alpha=1.0)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            display,
            origin="lower",
            cmap=cmap,
            norm=SymLogNorm(
                linthresh=linthresh,
                linscale=1.0,
                vmin=np.nanmin(vals),
                vmax=np.nanmax(vals),
            ),
        )
        ax.set_axis_off()
        ax.set_title(f"z-bin {z_bin}, nside {nside}, (SymLogNorm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        default_name = f"shell_mollview_zbin{z_bin}_nside{nside}.png"
        file_name = name if name is not None else default_name
        if not file_name.lower().endswith(".png"):
            file_name = f"{file_name}.png"
        out_path = output_dir / file_name
        hp.mollview(m, nest=False, title=f"z-bin {z_bin}, nside {nside}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved shell plot to {out_path}")
    return out_path


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

    pos = np.asarray(positions)
    pos = pos[-1] # use final snapshot
  
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