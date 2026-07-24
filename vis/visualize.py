from pathlib import Path
import numpy as np


def plot_shells(
    npz_path: str | Path | list[str | Path],
    z_bin: int = 5,
    nside: int = 512,
    output_dir: Path | None = None,
    plot_logarithmic: bool = False,
    name: str | None = None,
    normalize: bool = True,
    vmin: float | None = -1.0,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
) -> Path:
    """Load a shell file (.npz or .fits) and plot it as a HEALPix map.

    npz_path: a single .npz/.fits path, or a list of .fits paths whose
        pixel counts are summed before plotting.  Use a list to combine
        several thin pkdgrav shells so their total redshift span matches
        a wider CosmoGrid compressed shell.
    normalize: if True (default), convert raw particle counts to overdensity
        delta = count/<count> - 1 before plotting.  This makes maps from
        different sources (different shell thicknesses, different particle
        counts) directly comparable on the same colour scale.
    """
    try:
        import healpy as hp
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "healpy (and matplotlib) is required for shell plotting. Install via 'conda install healpy matplotlib'."
        ) from exc

    requested_nside = int(nside)
    if not hp.isnsideok(requested_nside):
        raise ValueError(f"Requested nside is invalid: {requested_nside}")

    # ------------------------------------------------------------------ #
    # Accept a list of FITS paths → load and sum them pixel-by-pixel.     #
    # ------------------------------------------------------------------ #
    if isinstance(npz_path, list):
        paths = [Path(p) for p in npz_path]
        if not all(p.suffix.lower() == ".fits" for p in paths):
            raise ValueError("When passing a list, all entries must be .fits files.")
        maps = []
        for p in paths:
            mi = hp.read_map(p, nest=False, dtype=np.float32)
            if not hp.isnpixok(int(mi.size)):
                raise ValueError(f"FITS map has invalid HEALPix size: {mi.size}")
            maps.append(mi)
        sizes = {mi.size for mi in maps}
        if len(sizes) != 1:
            raise ValueError(f"FITS maps have inconsistent sizes: {sizes}")
        m = np.sum(maps, axis=0, dtype=np.float32)
        current_nside = hp.npix2nside(int(m.size))
    else:
        npz_path = Path(npz_path)
        if npz_path.suffix.lower() == ".fits":
            m = hp.read_map(npz_path, nest=False, dtype=np.float32)
            if not hp.isnpixok(int(m.size)):
                raise ValueError(f"FITS map has invalid HEALPix size: {m.size}")
            current_nside = hp.npix2nside(int(m.size))
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
                # Infer the native nside from the max pixel index so that ud_grade can
                # correctly convert to requested_nside rather than scattering sparse
                # pixels into a wrongly-sized map.
                native_npix = int(pix.max()) + 1
                # round up to the next valid HEALPix npix
                native_nside = 1
                while hp.nside2npix(native_nside) < native_npix:
                    native_nside *= 2
                if b.shape[0] == pix.shape[0]:
                    if z_bin < 0 or z_bin >= b.shape[1]:
                        raise IndexError(f"z_bin={z_bin} out of range for shape {b.shape}")
                    m = np.zeros(hp.nside2npix(native_nside), dtype=np.float32)
                    m[pix] = b[:, z_bin]
                    current_nside = native_nside
                elif b.shape[1] == pix.shape[0]:
                    if z_bin < 0 or z_bin >= b.shape[0]:
                        raise IndexError(f"z_bin={z_bin} out of range for shape {b.shape}")
                    m = np.zeros(hp.nside2npix(native_nside), dtype=np.float32)
                    m[pix] = b[z_bin, :]
                    current_nside = native_nside
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
                    current_nside = inferred_nside
                    m = np.asarray(b[z_bin, :], dtype=np.float32)
                else:
                    inferred_nside = hp.npix2nside(int(b.shape[0]))
                    if z_bin < 0 or z_bin >= b.shape[1]:
                        raise IndexError(f"z_bin={z_bin} out of range for shape {b.shape}")
                    current_nside = inferred_nside
                    m = np.asarray(b[:, z_bin], dtype=np.float32)

    if current_nside != requested_nside:
        m = hp.ud_grade(
            m,
            nside_out=requested_nside,
            order_in="RING",
            order_out="RING",
        )
    nside = requested_nside

    if normalize:
        # Use formula for matter overdensity (see also wiki)
        mean = np.mean(m)
        if mean <= 0:
            raise ValueError("Shell map has non-positive mean; cannot normalize to overdensity.")
        m = m / mean - 1.0


    if output_dir is None:
        output_dir = Path(__file__).with_name("outputs").joinpath("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if name is not None and len(name.strip()) == 0:
        raise ValueError("name must be a non-empty string when provided")

    if plot_logarithmic:
        default_name = f"shell_mollview_log_zbin{z_bin}_nside{nside}.png"
        file_name = name if name is not None else default_name
        if not file_name.lower().endswith(".png"):
            file_name = f"{file_name}.png"
        out_path = output_dir / file_name
        log_m = np.log10(1.01 + m)
        hp.mollview(log_m, nest=False, title=f"z-bin {z_bin}, nside {nside}, log10(1.01+δ)",
                    xsize=3000, min=vmin, max=vmax, cmap=cmap)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        default_name = f"shell_mollview_zbin{z_bin}_nside{nside}.png"
        file_name = name if name is not None else default_name
        if not file_name.lower().endswith(".png"):
            file_name = f"{file_name}.png"
        out_path = output_dir / file_name
        hp.mollview(m, nest=False, title=f"z-bin {z_bin}, nside {nside}", xsize=3000,
                    min=vmin, max=vmax, cmap=cmap)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
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
    input_file: str | Path | None = None,
    output_dir: Path | None = None,
    vmin: float | None = -1.0,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
    auto_scale: bool = False,
) -> Path:
    """Bin a thin slab of particles onto a 2D grid and plot the density contrast.

    auto_scale: if True, ignore vmin/vmax and stretch the color scale to this
        slice's own min/max instead. Use for near-uniform fields (e.g. initial
        conditions) whose true contrast is too small to show up against the
        shared (vmin, vmax) used for structure-formed snapshots/shells — the
        output is then no longer color-comparable to those other plots.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Matplotlib is required for plotting. Install it via 'conda install matplotlib' or rerun without --plot."
        ) from exc

    pos = np.asarray(positions)

    if not (0 <= int(slice_axis) <= 2):
        raise ValueError(f"slice_axis must be 0, 1, or 2; got {slice_axis}")

    # Normalize to [N, 3] to support common layouts like [N,3], [T,N,3], [N,3,T], [3,N].
    if pos.ndim < 2:
        raise ValueError(f"Expected positions with at least 2 dimensions, got shape {pos.shape}")

    if pos.shape[-1] == 3:
        pass
    elif pos.ndim >= 2 and pos.shape[1] == 3:
        pos = np.moveaxis(pos, 1, -1)
    elif pos.shape[0] == 3:
        pos = np.moveaxis(pos, 0, -1)
    else:
        raise ValueError(
            f"Could not identify coordinate axis of length 3 in positions with shape {pos.shape}"
        )

    if pos.ndim == 3:
        pos = pos[-1]  # use final snapshot/time slice
        print("Using final snapshot")
    elif pos.ndim > 3:
        pos = pos.reshape(-1, 3)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected normalized positions shape [N,3], got {pos.shape}")
  
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

    # δ = count/<count> - 1, same overdensity definition used by plot_shells().
    # Plot log10(1.01+δ) rather than raw δ: δ itself can run into the hundreds
    # in cluster cores, which would blow past any (vmin, vmax) shared with the
    # shell maps and clip almost all real structure to a single flat color.
    # The log transform compresses that range enough that IC, snapshot, and
    # shell plots can all share the same (vmin, vmax, cmap).
    density = hist / np.mean(hist) - 1.0
    log_density = np.log10(1.01 + density)

    if output_dir is None:
        output_dir = Path(__file__).with_name("outputs").joinpath("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_file is not None:
        try:
            base = Path(input_file).stem
        except Exception:
            base = None
    else:
        base = None

    if base:
        file_name = f"{base}_density_slice_axis{slice_axis}_center{center:.1f}.png"
    else:
        file_name = f"density_slice_axis{slice_axis}_center{center:.1f}.png"

    out_path = output_dir / file_name

    # Same quantity (log10(1.01+δ)), same (vmin, vmax), same colormap as
    # plot_shells(plot_logarithmic=True) so the two plot types are directly
    # comparable -- unless auto_scale overrides that for a near-uniform field.
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=None if auto_scale else vmin, vmax=None if auto_scale else vmax, clip=True)
    rgba = plt.get_cmap(cmap)(norm(log_density.T))
    plt.imsave(out_path, rgba, origin="lower")
    print(f"Saved density slice to {out_path}")
    return out_path