import argparse
import math
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MAS_library as MASL
import numpy as np
import Pk_library as PKL


def read_tipsy_dark(snapshot_path: Path) -> np.ndarray:
    header_dtype = np.dtype([
        ("time", ">f8"),
        ("nBodies", ">u4"),
        ("nDim", ">u4"),
        ("nSph", ">u4"),
        ("nDark", ">u4"),
        ("nStar", ">u4"),
        ("pad", ">u4"),
    ])
    dark_dtype = np.dtype([
        ("mass", ">f4"),
        ("pos", ">f4", (3,)),
        ("vel", ">f4", (3,)),
        ("eps", ">f4"),
        ("phi", ">f4"),
    ])

    with snapshot_path.open("rb") as handle:
        header = np.fromfile(handle, dtype=header_dtype, count=1)[0]
        n_sph = int(header["nSph"])
        n_dark = int(header["nDark"])
        handle.seek(header_dtype.itemsize + n_sph * 48)
        dark = np.fromfile(handle, dtype=dark_dtype, count=n_dark)

    return dark
    


def load_positions(snapshot_path: Path, default_boxsize: float) -> Tuple[np.ndarray, float]:
    if snapshot_path.suffix.lower() == ".npz":
        with np.load(snapshot_path, allow_pickle=False) as data:
            boxsize = float(data["boxsize"]) if "boxsize" in data else default_boxsize

            if "pos" in data:
                positions = np.asarray(data["pos"], dtype=np.float32)
            elif "psi" in data:
                psi = np.asarray(data["psi"], dtype=np.float32)
                if psi.ndim != 4 or psi.shape[-1] != 3:
                    raise ValueError(
                        f"Expected psi with shape (Nx, Ny, Nz, 3), got {psi.shape} from {snapshot_path}"
                    )

                nx, ny, nz = psi.shape[:3]
                ix, iy, iz = np.mgrid[0:nx, 0:ny, 0:nz]
                q = np.stack(
                    [
                        ix.astype(np.float32) * (boxsize / nx),
                        iy.astype(np.float32) * (boxsize / ny),
                        iz.astype(np.float32) * (boxsize / nz),
                    ],
                    axis=-1,
                )
                positions = ((q + psi).reshape(-1, 3)) % boxsize
            else:
                array_keys = [key for key in data.files if data[key].ndim >= 2 and data[key].shape[-1] == 3]
                if not array_keys:
                    raise KeyError(
                        f"NPZ file {snapshot_path} does not contain 'pos', 'psi', or any (..., 3) array. Keys: {list(data.files)}"
                    )
                positions = np.asarray(data[array_keys[0]], dtype=np.float32).reshape(-1, 3)

        return positions % boxsize, boxsize

    dark = read_tipsy_dark(snapshot_path)
    positions = ((dark["pos"].astype(np.float32) + 0.5) * default_boxsize) % default_boxsize
    return positions, default_boxsize


def get_pk(positions: np.ndarray, boxsize: float, ngrid: int, threads: int) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {positions.shape}")

    delta = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    MASL.MA(positions, delta, boxsize, MAS="CIC")
    delta /= np.mean(delta)
    delta -= 1.0

    pk = PKL.Pk(delta, boxsize, axis=0, MAS="CIC", threads=threads)
    return pk.k3D, pk.Pk[:, 0]


def write_pk_table(out_path: Path, k_values: np.ndarray, pk_values: np.ndarray) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# k_h_over_Mpc Pk_Mpch3\n")
        for k_value, pk_value in zip(k_values, pk_values):
            handle.write(f"{k_value:.8e} {pk_value:.8e}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare P(k) from two snapshots (.npz or PKDGRAV Tipsy).")
    parser.add_argument("--snapshot-a", required=True, help="First snapshot path (.npz or CosmoML.00000-style Tipsy)")
    parser.add_argument("--snapshot-b", required=True, help="Second snapshot path (.npz or CosmoML.00000-style Tipsy)")
    parser.add_argument("--out-dir", required=True, help="Output directory for the plot and tabulated spectra")
    parser.add_argument("--lbox", type=float, default=900.0, help="Simulation box size in Mpc/h")
    parser.add_argument("--ngrid", type=int, default=512, help="FFT grid size used for the density mesh")
    parser.add_argument("--threads", type=int, default=4, help="Thread count passed to Pk_library")
    parser.add_argument("--label-a", default="Snapshot A", help="Legend label for the first snapshot")
    parser.add_argument("--label-b", default="Snapshot B", help="Legend label for the second snapshot")
    parser.add_argument("--title", default="Power Spectrum Comparison", help="Plot title")
    parser.add_argument("--output-name", default="pk_snapshot_comparison", help="Base name for output files")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    snapshot_a = Path(args.snapshot_a).expanduser().resolve()
    snapshot_b = Path(args.snapshot_b).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (snapshot_a, snapshot_b):
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot not found: {path}")

    positions_a, boxsize_a = load_positions(snapshot_a, args.lbox)
    positions_b, boxsize_b = load_positions(snapshot_b, args.lbox)

    if not math.isclose(boxsize_a, boxsize_b, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Box size mismatch between snapshots: {boxsize_a} vs {boxsize_b}")

    print(f"Loaded {snapshot_a.name}: {positions_a.shape[0]} particles, Lbox={boxsize_a}")
    print(f"Loaded {snapshot_b.name}: {positions_b.shape[0]} particles, Lbox={boxsize_b}")
    print(f"Computing P(k) on a {args.ngrid}^3 CIC mesh")

    k_a, pk_a = get_pk(positions_a, boxsize_a, args.ngrid, args.threads)
    k_b, pk_b = get_pk(positions_b, boxsize_b, args.ngrid, args.threads)

    base = out_dir / args.output_name
    table_a = base.with_name(f"{base.name}_a.txt")
    table_b = base.with_name(f"{base.name}_b.txt")
    write_pk_table(table_a, k_a, pk_a)
    write_pk_table(table_b, k_b, pk_b)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k_a, pk_a, label=args.label_a, marker="o", markersize=3, linewidth=1.2)
    ax.loglog(k_b, pk_b, label=args.label_b, marker="s", markersize=3, linewidth=1.2)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k) [(Mpc/h)^3]")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = base.with_suffix(".png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {png_path}")
    print(f"Saved spectra to {table_a} and {table_b}")


if __name__ == "__main__":
    main()



