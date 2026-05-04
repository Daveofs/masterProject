"""compare_lightcone_shells.py
=============================
Compare DISCO-DJ lightcone particles (NPZ snapshot), DISCO-DJ pre-built
HEALPix shells, and CosmoGrid compressed_shells for matching redshift bins.

The lightcone NPZ must have:
  S        : (n_gpu, N_part, 5) float32
               columns ['1+z', 'X', 'Y', 'Z', 'p_rad']
               where X,Y,Z are comoving coords FROM the observer,
               and 1+z == 0 means the particle never crossed the lightcone.

Shell NPZ files (--disco-shells and --cosmo-shells) must have:
  shells     : (N_shells, N_pix)  HEALPix RING maps
  shell_info : structured array with fields lower_z, upper_z, shell_id

Usage
-----
python compare_lightcone_shells.py \\
    --lightcone    /path/to/lightcone_multigpu.npz \\
    --disco-shells /path/to/shells_nside=2048.npz \\
    --cosmo-shells /path/to/compressed_shells.npz \\
    --z-bin 10 \\
    --nside 512 \\
    --output-dir /path/to/output/ \\
    [--all-bins] \\
    [--plot-logarithmic] \\
    [--vmin -1.0 --vmax 1.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# HEALPix shell builder
# ---------------------------------------------------------------------------

def particles_to_healpix(
    xyz: np.ndarray,
    nside: int,
) -> np.ndarray:
    """Project (N,3) unit-direction vectors to a HEALPix RING map (counts)."""
    import healpy as hp
    npix = hp.nside2npix(nside)
    norms = np.linalg.norm(xyz, axis=1)
    valid = norms > 0
    x = xyz[valid, 0] / norms[valid]
    y = xyz[valid, 1] / norms[valid]
    z = xyz[valid, 2] / norms[valid]
    pix = hp.vec2pix(nside, x, y, z, nest=False)
    m = np.bincount(pix, minlength=npix).astype(np.float32)
    return m


def to_overdensity(m: np.ndarray) -> np.ndarray:
    mean = float(np.mean(m))
    if mean <= 0:
        raise ValueError("Map has non-positive mean; cannot convert to overdensity.")
    return m / mean - 1.0


# ---------------------------------------------------------------------------
# Multi-panel plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    m_disco_lc: np.ndarray | None,
    m_disco_shells: np.ndarray | None,
    m_cosmo: np.ndarray | None,
    nside: int,
    lower_z: float,
    upper_z: float,
    z_bin: int,
    output_dir: Path,
    plot_logarithmic: bool = False,
    plot_counts: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    name: str | None = None,
) -> Path:
    import healpy as hp
    import matplotlib.pyplot as plt

    def prepare(m: np.ndarray) -> np.ndarray:
        m = m.astype(np.float32)
        if plot_counts:
            if plot_logarithmic:
                return np.log10(m)
            return m
        m = to_overdensity(m)
        if plot_logarithmic:
            m = np.log10(1.01 + m)
        return m

    panels = []
    disco_prep = None
    cosmo_prep = None
    if m_disco_lc is not None:
        panels.append((prepare(m_disco_lc), "DISCO-DJ lightcone (particles)"))
    if m_disco_shells is not None:
        disco_prep = prepare(m_disco_shells)
        panels.append((disco_prep, "DISCO-DJ shells"))
    if m_cosmo is not None:
        cosmo_prep = prepare(m_cosmo)
        panels.append((cosmo_prep, "CosmoGrid compressed_shells"))

    # If no lightcone is provided, also show the difference between DISCO and Cosmo
    if m_disco_lc is None and disco_prep is not None and cosmo_prep is not None:
        panels.append((disco_prep - cosmo_prep, "DISCO - Cosmo (disco - cosmo)"))

    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("No maps to plot.")

    title_suffix = f"z=[{lower_z:.4f},{upper_z:.4f}]  (bin {z_bin})"
    if plot_counts and plot_logarithmic:
        scale_label = "log₁₀(counts)"
    elif plot_counts:
        scale_label = "counts"
    elif plot_logarithmic:
        scale_label = "log₁₀(1.01+δ)"
    else:
        scale_label = "δ"

    fig = plt.figure(figsize=(9 * n_panels, 6))

    for i, (m, label) in enumerate(panels):
        hp.mollview(
            m,
            fig=fig,
            sub=(1, n_panels, i + 1),
            nest=False,
            title=f"{label}\n{title_suffix}",
            xsize=2000,
            min=vmin,
            max=vmax,
            unit=scale_label,
            notext=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_counts and plot_logarithmic:
        tag = "counts_log"
    elif plot_counts:
        tag = "counts"
    elif plot_logarithmic:
        tag = "log"
    else:
        tag = "lin"
    default_name = f"compare_zbin{z_bin}_nside{nside}_{tag}.png"
    fname = (name if name else default_name)
    if not fname.lower().endswith(".png"):
        fname += ".png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot → {out_path}")
    return out_path


def plot_separate(
    m_disco_lc: np.ndarray | None,
    m_disco_shells: np.ndarray | None,
    m_cosmo: np.ndarray | None,
    nside: int,
    lower_z: float,
    upper_z: float,
    z_bin: int,
    output_dir: Path,
    plot_logarithmic: bool = False,
    plot_counts: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Path]:
    """Save each map as a separate PNG file."""
    import healpy as hp
    import matplotlib.pyplot as plt

    def prepare(m: np.ndarray) -> np.ndarray:
        m = m.astype(np.float32)
        if plot_counts:
            if plot_logarithmic:
                return np.log10(m)
            return m
        m = to_overdensity(m)
        if plot_logarithmic:
            m = np.log10(1.01 + m)
        return m

    panels = []
    disco_prep = None
    cosmo_prep = None
    if m_disco_lc is not None:
        panels.append((prepare(m_disco_lc), "disco_lc", "DISCO-DJ lightcone (particles)"))
    if m_disco_shells is not None:
        disco_prep = prepare(m_disco_shells)
        panels.append((disco_prep, "disco_shells", "DISCO-DJ shells"))
    if m_cosmo is not None:
        cosmo_prep = prepare(m_cosmo)
        panels.append((cosmo_prep, "cosmo_shells", "CosmoGrid compressed_shells"))

    # If no lightcone is provided, also save the difference between DISCO and Cosmo
    if m_disco_lc is None and disco_prep is not None and cosmo_prep is not None:
        panels.append((disco_prep - cosmo_prep, "disco_minus_cosmo", "DISCO-DJ - CosmoGrid (disco - cosmo)"))

    if not panels:
        raise ValueError("No maps to plot.")

    title_suffix = f"z=[{lower_z:.4f},{upper_z:.4f}]  (bin {z_bin})"
    if plot_counts and plot_logarithmic:
        scale_label = "log₁₀(counts)"
    elif plot_counts:
        scale_label = "counts"
    elif plot_logarithmic:
        scale_label = "log₁₀(1.01+δ)"
    else:
        scale_label = "δ"
    if plot_counts and plot_logarithmic:
        tag = "counts_log"
    elif plot_counts:
        tag = "counts"
    elif plot_logarithmic:
        tag = "log"
    else:
        tag = "lin"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    for m, slug, label in panels:
        fig = plt.figure(figsize=(9, 6))
        hp.mollview(
            m,
            fig=fig,
            nest=False,
            title=f"{label}\n{title_suffix}",
            xsize=2000,
            min=vmin,
            max=vmax,
            unit=scale_label,
            notext=False,
        )
        fname = f"{slug}_zbin{z_bin}_nside{nside}_{tag}.png"
        out_path = output_dir / fname
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved separate plot → {out_path}")
        out_paths.append(out_path)
    return out_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_shell_npz(path: Path, nside: int, zb: int) -> np.ndarray:
    """Load one z-bin from a shells NPZ and ud_grade to the requested nside."""
    import healpy as hp
    d = np.load(path, allow_pickle=False)
    m = d["shells"][zb].astype(np.float32)
    src_nside = hp.npix2nside(m.size)
    if src_nside != nside:
        m = hp.ud_grade(m, nside_out=nside, order_in="RING", order_out="RING")
    return m


def run(args: argparse.Namespace) -> None:
    import healpy as hp

    # ── Load lightcone (optional) ───────────────────────────────────────────
    S = None
    if args.lightcone is not None:
        print(f"Loading lightcone: {args.lightcone}")
        lc = np.load(args.lightcone, allow_pickle=False)
        S = lc["S"]                            # (n_gpu, N, 5)
        if S.ndim == 3:
            S = S.reshape(-1, S.shape[-1])     # (n_gpu*N, 5)
        elif S.ndim != 2:
            raise ValueError(f"Unexpected S shape: {S.shape}")

    # ── Load DISCO-DJ shells (optional) ─────────────────────────────────────
    disco_shells_data = None
    if args.disco_shells is not None:
        print(f"Loading DISCO-DJ shells: {args.disco_shells}")
        disco_shells_data = np.load(args.disco_shells, allow_pickle=False)
        _dn = hp.npix2nside(disco_shells_data["shells"].shape[1])
        print(f"  disco_shells: {disco_shells_data['shells'].shape[0]} shells, nside={_dn}")

    # ── Load CosmoGrid compressed shells (optional) ──────────────────────────
    cosmo_shells_data = None
    if args.cosmo_shells is not None:
        print(f"Loading CosmoGrid compressed shells: {args.cosmo_shells}")
        cosmo_shells_data = np.load(args.cosmo_shells, allow_pickle=False)
        _cn = hp.npix2nside(cosmo_shells_data["shells"].shape[1])
        print(f"  compressed_shells: {cosmo_shells_data['shells'].shape[0]} shells, nside={_cn}")

    # Use whichever shell file is available to determine z-bin metadata
    ref_data = cosmo_shells_data if cosmo_shells_data is not None else disco_shells_data
    if ref_data is None and S is None:
        raise ValueError("At least one of --lightcone, --disco-shells, or --cosmo-shells must be provided.")
    if ref_data is None:
        raise ValueError("At least one of --disco-shells or --cosmo-shells must be provided for z-bin metadata.")

    shell_info = ref_data["shell_info"]
    n_shells = ref_data["shells"].shape[0]

    # ── Determine z-bins to process ─────────────────────────────────────────
    if args.all_bins:
        z_bins = list(range(n_shells))
    else:
        z_bins = [args.z_bin]
        if args.z_bin < 0 or args.z_bin >= n_shells:
            raise IndexError(f"--z-bin {args.z_bin} out of range [0, {n_shells-1}]")

    nside = args.nside
    output_dir = Path(args.output_dir)

    for zb in z_bins:
        lower_z = float(shell_info["lower_z"][zb])
        upper_z = float(shell_info["upper_z"][zb])
        print(f"\nProcessing z-bin {zb}: z=[{lower_z:.4f}, {upper_z:.4f}]")

        # ── Build lightcone particle map ────────────────────────────────────
        m_disco_lc = None
        if S is not None:
            z_vals = S[:, 0] - 1.0   # convert 1+z -> z
            crossed = S[:, 0] != 0.0
            in_shell = crossed & (z_vals >= lower_z) & (z_vals < upper_z)
            pts = S[in_shell]
            print(f"  Lightcone particles in shell: {pts.shape[0]:,}")
            if pts.shape[0] == 0:
                print(f"  WARNING: no lightcone particles in z-bin {zb}, skipping lightcone panel.")
            else:
                xyz = pts[:, [1, 2, 3]].astype(np.float32)
                m_disco_lc = particles_to_healpix(xyz, nside)

        # ── DISCO-DJ shells map ─────────────────────────────────────────────
        m_disco_shells = None
        if disco_shells_data is not None:
            m_disco_shells = _load_shell_npz(args.disco_shells, nside, zb)

        # ── CosmoGrid shells map ────────────────────────────────────────────
        m_cosmo = None
        if cosmo_shells_data is not None:
            m_cosmo = _load_shell_npz(args.cosmo_shells, nside, zb)

        if m_disco_lc is None and m_disco_shells is None and m_cosmo is None:
            print(f"  No data for z-bin {zb}, skipping.")
            continue

        # ── Plot ─────────────────────────────────────────────────────────────
        plot_fn = plot_separate if args.separate else plot_comparison
        plot_fn(
            m_disco_lc=m_disco_lc,
            m_disco_shells=m_disco_shells,
            m_cosmo=m_cosmo,
            nside=nside,
            lower_z=lower_z,
            upper_z=upper_z,
            z_bin=zb,
            output_dir=output_dir,
            plot_logarithmic=args.plot_logarithmic,
            plot_counts=args.plot_counts,
            vmin=args.vmin,
            vmax=args.vmax,
        )

    print("\nDone.")


# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lightcone",    type=Path, default=None,
                   help="Path to lightcone NPZ file (contains S array; optional)")
    p.add_argument("--disco-shells", type=Path, default=None,
                   help="Path to DISCO-DJ pre-built shells NPZ (optional)")
    p.add_argument("--cosmo-shells", type=Path, default=None,
                   help="Path to compressed_shells.npz (CosmoGrid format; optional)")
    p.add_argument("--z-bin",        type=int, default=10,
                   help="z-bin index to visualize (0-based, default: 10)")
    p.add_argument("--all-bins",     action="store_true",
                   help="Process all z-bins (overrides --z-bin)")
    p.add_argument("--nside",        type=int, default=512,
                   help="Output HEALPix nside (default: 512)")
    p.add_argument("--output-dir",   type=Path,
                   default=Path("/capstor/scratch/cscs/damrein/outputs/plots/shells"),
                   help="Directory to write comparison plots")
    p.add_argument("--plot-logarithmic", action="store_true",
                   help="Plot log10(1.01 + delta) instead of delta")
    p.add_argument("--plot-counts", action="store_true",
                   help="Plot raw counts instead of overdensity (1+δ)")
    p.add_argument("--vmin", type=float, default=None,
                   help="Colorbar minimum")
    p.add_argument("--vmax", type=float, default=None,
                   help="Colorbar maximum")
    p.add_argument("--separate", action="store_true",
                   help="Save each map as a separate PNG instead of a single side-by-side panel")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run(args)
