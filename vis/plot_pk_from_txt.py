import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np



def load_pk_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got shape {data.shape}")

    k = data[:, 0]
    pk = data[:, 1]

    valid = np.isfinite(k) & np.isfinite(pk) & (k > 0.0) & (pk > 0.0)
    if not np.any(valid):
        raise ValueError(f"No valid positive finite k, P(k) values in {path}")

    return k[valid], pk[valid]


def prepare_ratio(
    k_num: np.ndarray,
    pk_num: np.ndarray,
    k_den: np.ndarray,
    pk_den: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Fast path when both spectra share the same k-grid.
    if k_num.shape == k_den.shape and np.allclose(k_num, k_den, rtol=1e-10, atol=0.0):
        ratio = pk_num / pk_den
        return k_num, ratio

    # Otherwise interpolate denominator onto numerator grid inside overlap.
    lo = max(k_num.min(), k_den.min())
    hi = min(k_num.max(), k_den.max())
    mask = (k_num >= lo) & (k_num <= hi)
    if not np.any(mask):
        raise ValueError("No overlapping k-range between the two spectra")

    k_common = k_num[mask]
    pk_den_interp = np.interp(k_common, k_den, pk_den)
    ratio = pk_num[mask] / pk_den_interp
    return k_common, ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot P(k) tables with ratio panel (A/C). B is optional.")
    parser.add_argument("--pk-a", required=True, help="Path to numerator spectrum txt")
    parser.add_argument("--pk-b", required=False, help="Path to optional extra spectrum txt (plotted only)")
    parser.add_argument("--pk-c", required=True, help="Path to denominator spectrum txt (required)")
    parser.add_argument("--label-a", default="Spectrum A", help="Legend label for spectrum A")
    parser.add_argument("--label-b", default="Spectrum B", help="Legend label for spectrum B")
    parser.add_argument("--label-c", default="Spectrum C", help="Legend label for spectrum C")
    parser.add_argument("--output", required=True, help="Output PNG path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pk_a_path = Path(args.pk_a).expanduser().resolve()
    pk_c_path = Path(args.pk_c).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load required spectra A and C (C is the denominator)
    k_a, pk_a = load_pk_table(pk_a_path)
    k_c, pk_c = load_pk_table(pk_c_path)

    # Optional B spectrum
    k_b = None
    pk_b = None
    if args.pk_b:
        pk_b_path = Path(args.pk_b).expanduser().resolve()
        k_b, pk_b = load_pk_table(pk_b_path)

    # Prepare primary ratio: A / C
    k_ratio_ac, ratio_ac = prepare_ratio(k_a, pk_a, k_c, pk_c)

    # If B provided, try to prepare B / C ratio as well (may fail if no overlap)
    k_ratio_bc = None
    ratio_bc = None
    if k_b is not None:
        try:
            k_ratio_bc, ratio_bc = prepare_ratio(k_b, pk_b, k_c, pk_c)
        except ValueError:
            k_ratio_bc = None
            ratio_bc = None

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.loglog(k_a, pk_a, label=args.label_a, marker="o", markersize=3, linewidth=1.2)
    ax1.loglog(k_c, pk_c, label=args.label_c, marker="^", markersize=3, linewidth=1.2)
    if k_b is not None:
        ax1.loglog(k_b, pk_b, label=args.label_b, marker="s", markersize=3, linewidth=1.2)
    ax1.set_xlabel("k [h/Mpc]")
    ax1.set_ylabel("P(k) [(Mpc/h)^3]")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()
    ax2.plot(k_ratio_ac, ratio_ac, label=f"{args.label_a} / {args.label_c}", marker="o", markersize=3, linewidth=1.2)
    if k_ratio_bc is not None:
        ax2.plot(k_ratio_bc, ratio_bc, label=f"{args.label_b} / {args.label_c}", marker="s", markersize=3, linewidth=1.2)
    ax2.axhline(1.0, color="k", lw=0.8, linestyle="--")
    ax2.set_xlabel("k [h/Mpc]")
    ax2.set_ylabel("P(k) ratio")
    ax2.set_xscale("log")
    ax2.set_ylim(0.9, 1.1)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
