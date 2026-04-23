#!/usr/bin/env python3
"""
validate_shell_builder.py
=========================
Validate LightconeShellBuilder against pkdgrav3's own on-the-fly lightcone.

Algorithm
---------
For each consecutive pkdgrav3 Tipsy snapshot pair (step i-1, step i):

  1. Read both Tipsy files; apply +L/2 mod L shift to convert pkdgrav3's
     corner-origin coordinates to DISCO-DJ's box-centre-origin convention.
  2. Compute r_lo = chi(a_curr), r_hi = chi(a_prev) using DISCO's chi_of_a.
  3. Call LightconeShellBuilder.accumulate_shell_jax() with those boundaries.
  4. Compare to the FITS shell produced by pkdgrav3's built-in lightcone
     accumulator for the SAME step (from shell_collector.py output).

The two methods use the same set of particles in the same positions.
Any difference is due to:
  - Linear endpoint interpolation in LightconeShellBuilder (vs pkdgrav3's
    exact sub-step crossing position tracking).
  - Tiny chi_of_a differences at shell boundaries (negligible).

Expected result
---------------
  ratio  (built mean / pkdgrav mean) ~ 1.000 ± 0.005 per step
  xcorr  (pixel cross-correlation)   > 0.990 per step

Large deviations (ratio ≫ 1.01 or xcorr < 0.95) indicate a bug in the
shell builder (wrong observer position, missing replicas, boundary error).

Usage
-----
python validate_shell_builder.py \\
    --snap-dir   /capstor/scratch/cscs/damrein/outputs/pkdgrav_validation/ \\
    --fits-dir   /capstor/scratch/cscs/damrein/outputs/pkdgrav_validation/ \\
    --params-yml /capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000001/run_0/params.yml \\
    --output-dir /capstor/scratch/cscs/damrein/outputs/plots/shell_builder_validation/ \\
    [--snap-prefix CosmoML_val] [--nside 512] [--z-max 3.5] [--no-gpu]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from time import time

import numpy as np

# ---------------------------------------------------------------------------
# Add DISCO-DJ scripts to sys.path (LightconeShellBuilder lives there)
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_disco_dir  = _script_dir.parent / "disco"
if str(_disco_dir) not in sys.path:
    sys.path.insert(0, str(_disco_dir))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--snap-dir",    type=Path, required=True,
                   help="Directory containing Tipsy snapshots and the .log file")
    p.add_argument("--snap-prefix", type=str,  default="CosmoML_val",
                   help="Filename prefix of snapshots / log (default: CosmoML_val)")
    p.add_argument("--fits-dir",    type=Path, default=None,
                   help="Directory with FITS reference shells from shell_collector.py "
                        "(default: same as --snap-dir)")
    p.add_argument("--params-yml",  type=Path, required=True,
                   help="Path to CosmoGridV1 params.yml (cosmological parameters)")
    p.add_argument("--boxsize",     type=float, default=900.0,
                   help="Box side length [Mpc/h] (default: 900.0)")
    p.add_argument("--nside",       type=int,   default=512,
                   help="HEALPix Nside for comparison maps (default: 512 for speed; "
                        "use 2048 for full resolution)")
    p.add_argument("--output-dir",  type=Path,  required=True,
                   help="Output directory for the validation plot and NPZ summary")
    p.add_argument("--z-max",       type=float, default=3.5,
                   help="Maximum lightcone redshift (default: 3.5)")
    p.add_argument("--no-gpu",      action="store_true",
                   help="Use CPU accumulate_shell() instead of GPU accumulate_shell_jax()")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    fits_dir = args.fits_dir or args.snap_dir

    # ── JAX initialisation (must happen before any jax-dependent imports) ──
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu" if not args.no_gpu else "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")

    import jax
    jax.config.update("jax_enable_x64", False)

    import healpy as hp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    from discodj import DiscoDJ
    from build_lightcone_shells import make_chi_of_a, LightconeShellBuilder
    from read_tipsy_file import read_tipsy

    # ── Cosmology → chi_of_a (same function as the DISCO simulation uses) ──
    with open(args.params_yml) as f:
        p_yml = yaml.safe_load(f)

    cosmo_dict = dict(
        Omega_c = float(p_yml["O_cdm"]),
        Omega_b = float(p_yml["Ob"]),
        h       = float(p_yml["H0"]) / 100.0,
        sigma8  = float(p_yml["s8"]),
        n_s     = float(p_yml["ns"]),
        Omega_k = 0.0,
        w0      = float(p_yml.get("w0", -1.0)),
        wa      = float(p_yml.get("wa",  0.0)),
    )
    print(f"Cosmology: {cosmo_dict}")

    # Minimal DiscoDJ (res=2 → 8 particles, only used to get cosmo object)
    dj_tmp = DiscoDJ(dim=3, res=2, boxsize=args.boxsize, cosmo=cosmo_dict)
    chi_of_a, _ = make_chi_of_a(dj_tmp.cosmo)

    chi_max_lc = float(chi_of_a(1.0 / (1.0 + args.z_max)))
    print(f"chi(z_max={args.z_max}) = {chi_max_lc:.1f} Mpc/h\n")

    # ── LightconeShellBuilder (unchanged from production code) ─────────────
    builder = LightconeShellBuilder(
        boxsize   = args.boxsize,
        chi_of_a  = chi_of_a,
        nside     = args.nside,
        z_min     = 0.0,
        z_max     = args.z_max,
        interpolate = True,
    )

    # ── Read pkdgrav3 log file (one redshift per step) ─────────────────────
    log_file = args.snap_dir / f"{args.snap_prefix}.log"
    if not log_file.exists():
        raise FileNotFoundError(f"pkdgrav3 log not found: {log_file}")

    log_data = np.genfromtxt(str(log_file))
    z_output = log_data[:, 1]          # column 1 = redshift at each step
    n_steps  = len(z_output) - 1
    print(f"Log: {log_file.name}  n_steps={n_steps}  "
          f"z_range=[{z_output[-1]:.4f},{z_output[0]:.4f}]")

    # ── Discover snapshot files ────────────────────────────────────────────
    snap_files: dict[int, Path] = {
        int(p.name.rsplit(".", 1)[-1]): p
        for p in sorted(args.snap_dir.glob(f"{args.snap_prefix}.[0-9][0-9][0-9][0-9][0-9]"))
    }
    if len(snap_files) < 2:
        raise FileNotFoundError(
            f"Need ≥2 Tipsy snapshots matching {args.snap_prefix}.NNNNN "
            f"in {args.snap_dir}"
        )
    print(f"Snapshots found: {len(snap_files)}  "
          f"(steps {min(snap_files)}–{max(snap_files)})\n")

    # Build FITS index: (z_hi_str, z_lo_str) → Path
    fits_index: dict[tuple[str, str], Path] = {}
    for fp in fits_dir.glob(f"{args.snap_prefix}-shell_z-high=*_z-low=*.fits"):
        m = re.search(r"z-high=([0-9.e+\-]+)_z-low=([0-9.e+\-]+)", fp.name)
        if m:
            fits_index[(m.group(1), m.group(2))] = fp
    print(f"FITS reference shells found: {len(fits_index)}\n")

    # ── Main loop: stream through snapshot pairs ───────────────────────────
    npix     = hp.nside2npix(args.nside)
    results  = []
    pos_prev = None
    a_prev   = None

    for step_i in range(n_steps + 1):
        if step_i not in snap_files:
            print(f"Step {step_i}: snapshot file not found — stopping.")
            break

        # Load snapshot (streaming: release memory after sliding the window)
        t_read = time()
        p_data, hdr = read_tipsy(str(snap_files[step_i]), args.boxsize)
        a_curr = float(hdr['a'])
        pos_curr = np.stack([p_data['x'], p_data['y'], p_data['z']], axis=1)
        del p_data

        # pkdgrav3 corner-origin → DISCO box-centre-origin (same as _load_external_ics)
        pos_curr = (pos_curr + args.boxsize / 2.0) % args.boxsize
        t_read = time() - t_read

        if step_i == 0:
            print(f"Step {step_i:3d}: a={a_curr:.6f}  z={1/a_curr-1:.4f}  "
                  f"read in {t_read:.1f}s  [IC / no comparison]")
            pos_prev = pos_curr
            a_prev   = a_curr
            continue

        z_hi = float(z_output[step_i - 1])   # upper z → larger r → earlier time
        z_lo = float(z_output[step_i])        # lower z → smaller r → later time
        r_hi = float(chi_of_a(1.0 / (1.0 + z_hi)))
        r_lo = float(chi_of_a(1.0 / (1.0 + z_lo)))

        print(f"Step {step_i:3d}: z=[{z_lo:.4f},{z_hi:.4f}]  r=[{r_lo:.1f},{r_hi:.1f}]  "
              f"read {t_read:.1f}s", end="", flush=True)

        # Skip steps entirely outside the lightcone redshift range
        if r_lo >= chi_max_lc:
            print("  [pre-lightcone]")
            pos_prev = pos_curr
            a_prev   = a_curr
            continue

        # ── Find matching FITS reference ──────────────────────────────────
        fits_path = _find_fits(fits_index, z_hi, z_lo, args.snap_prefix, fits_dir)
        if fits_path is None:
            print("  [no FITS match — skipping comparison]")
            pos_prev = pos_curr
            a_prev   = a_curr
            continue

        # Load and optionally degrade the reference map
        m_ref_raw = hp.read_map(str(fits_path), field=0, dtype=np.float32, verbose=False)
        nside_ref = hp.npix2nside(len(m_ref_raw))
        if nside_ref != args.nside:
            m_ref = hp.ud_grade(m_ref_raw, nside_out=args.nside, order_in='RING')
        else:
            m_ref = m_ref_raw

        # ── Run LightconeShellBuilder (unchanged from production) ─────────
        t_build = time()
        if args.no_gpu:
            m_built = builder.accumulate_shell(
                pos_prev, pos_curr, a_prev, a_curr,
                r_lo_override=r_lo, r_hi_override=r_hi,
            )
        else:
            # Pass numpy arrays directly — accumulate_shell_jax handles the
            # reshape/cast internally; no explicit jnp.asarray needed.
            m_built = builder.accumulate_shell_jax(
                pos_prev, pos_curr, a_prev, a_curr,
                r_lo_override=r_lo, r_hi_override=r_hi,
            )
        t_build = time() - t_build

        # ── Per-step statistics ───────────────────────────────────────────
        b_mean = float(m_built.mean())
        r_mean = float(m_ref.mean())
        ratio  = b_mean / r_mean if r_mean > 0 else float("nan")
        xcorr  = float("nan")
        if b_mean > 0 and r_mean > 0:
            b_od  = m_built.astype(np.float32) / b_mean - 1.0
            r_od  = m_ref.astype(np.float32)   / r_mean - 1.0
            xcorr = float(np.corrcoef(b_od.ravel(), r_od.ravel())[0, 1])

        print(f"  built={b_mean:.4f}  ref={r_mean:.4f}  "
              f"ratio={ratio:.4f}  xcorr={xcorr:.4f}  build={t_build:.1f}s")

        results.append({
            "step":       step_i,
            "z_lo":       z_lo,   "z_hi":       z_hi,
            "r_lo":       r_lo,   "r_hi":       r_hi,
            "built_mean": b_mean, "ref_mean":   r_mean,
            "ratio":      ratio,  "xcorr":      xcorr,
        })

        # Slide window: release prev, promote curr to prev
        pos_prev = pos_curr
        a_prev   = a_curr

    # ── Summary ───────────────────────────────────────────────────────────
    if not results:
        print("\nNo steps were compared. "
              "Check that FITS files exist and --z-max covers the lightcone range.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratios = np.array([r["ratio"] for r in results])
    xcorrs = np.array([r["xcorr"] for r in results])

    print(f"\n{'='*65}")
    print(f"Shell builder validation: {len(results)} steps compared")
    print(f"  ratio  mean={np.nanmean(ratios):.4f}  std={np.nanstd(ratios):.4f}"
          f"  range=[{np.nanmin(ratios):.4f},{np.nanmax(ratios):.4f}]")
    print(f"  xcorr  mean={np.nanmean(xcorrs):.4f}  std={np.nanstd(xcorrs):.4f}"
          f"  range=[{np.nanmin(xcorrs):.4f},{np.nanmax(xcorrs):.4f}]")
    if np.nanmean(ratios) > 0.99 and np.nanmean(xcorrs) > 0.99:
        print("\n  RESULT: PASS — shell builder algorithm is correct.")
    else:
        print("\n  RESULT: WARNING — check plot for systematic deviations.")
    print("="*65)

    # ── Plot ──────────────────────────────────────────────────────────────
    z_his = np.array([r["z_hi"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(z_his, ratios, "b.-", lw=1, markersize=4)
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[0].fill_between([0, args.z_max], [0.99, 0.99], [1.01, 1.01],
                         alpha=0.15, color="green", label="±1%")
    axes[0].set_xlabel("Step upper redshift z_hi")
    axes[0].set_ylabel("LightconeShellBuilder / pkdgrav3  (mean counts)")
    axes[0].set_title(f"Mean count ratio per step  (nside={args.nside})")
    axes[0].set_ylim(0.85, 1.15)
    axes[0].legend()

    axes[1].plot(z_his, xcorrs, "r.-", lw=1, markersize=4)
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].fill_between([0, args.z_max], [0.99, 0.99], [1.00, 1.00],
                         alpha=0.15, color="green", label=">0.99")
    axes[1].set_xlabel("Step upper redshift z_hi")
    axes[1].set_ylabel("Pixel cross-correlation r")
    axes[1].set_title("Pixel-level cross-correlation per step")
    axes[1].set_ylim(0.85, 1.02)
    axes[1].legend()

    plt.tight_layout()
    out_plot = args.output_dir / "shell_builder_validation.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {out_plot}")

    # ── Save numerical results ────────────────────────────────────────────
    out_npz = args.output_dir / "shell_builder_validation.npz"
    arr = np.array([[r["step"], r["z_lo"], r["z_hi"], r["r_lo"], r["r_hi"],
                     r["built_mean"], r["ref_mean"], r["ratio"], r["xcorr"]]
                    for r in results])
    np.savez(out_npz,
             results=arr,
             columns=np.array(["step", "z_lo", "z_hi", "r_lo", "r_hi",
                                "built_mean", "ref_mean", "ratio", "xcorr"]))
    print(f"NPZ : {out_npz}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_fits(
    fits_index: dict[tuple[str, str], Path],
    z_hi: float,
    z_lo: float,
    prefix: str,
    fits_dir: Path,
) -> Path | None:
    """Return the FITS file whose z boundaries match (z_hi, z_lo) best."""
    # Try exact string match first (shell_collector formats floats as-is)
    for (zh_str, zl_str), fp in fits_index.items():
        try:
            if abs(float(zh_str) - z_hi) < 1e-4 and abs(float(zl_str) - z_lo) < 1e-4:
                return fp
        except ValueError:
            pass

    # Fallback: closest match
    best_path = None
    best_err  = 1.0      # reject if error > 0.01 in z
    for (zh_str, zl_str), fp in fits_index.items():
        try:
            err = abs(float(zh_str) - z_hi) + abs(float(zl_str) - z_lo)
            if err < best_err:
                best_err  = err
                best_path = fp
        except ValueError:
            pass

    if best_err < 1e-3:
        return best_path
    return None


if __name__ == "__main__":
    main()
