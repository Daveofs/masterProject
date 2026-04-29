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
import shutil
import healpy as hp

# ---------------------------------------------------------------------------
# Add local `masterProject/disco` and try to autodetect a sibling
# `DISCO-DJ` checkout. We need `DISCO-DJ/src` (package) and
# `DISCO-DJ/scripts` (helper modules like `utils_jens`) on sys.path so
# imports such as `from discodj import DiscoDJ` and `import utils_jens`
# work when running locally.
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
# local copy used by this repo (contains build_lightcone_shells, etc.)
_local_disco = _script_dir.parent / "disco"
if str(_local_disco) not in sys.path:
    sys.path.insert(0, str(_local_disco))

# Try to find a workspace DISCO-DJ checkout by walking ancestors and add
# its `scripts/` and `src/` folders to sys.path so helper modules are
# available (e.g. `utils_jens`). Searching ancestors is more robust
# than assuming a fixed number of parents.
_disco_dj_root = None
for _ancestor in _script_dir.resolve().parents:
    _candidate = _ancestor / "DISCO-DJ"
    if _candidate.exists():
        _disco_dj_root = _candidate
        break

if _disco_dj_root is not None:
    _disco_dj_scripts = _disco_dj_root / "scripts"
    _disco_dj_src = _disco_dj_root / "src"
    for _p in (_disco_dj_scripts, _disco_dj_src):
        if _p.exists() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))


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
    p.add_argument("--no-save-built", dest="save_built", action="store_false",
                   help="Do not save built shell FITS files and diffs (default: save built)")
    p.add_argument("--save-plots", action="store_true",
                   help="Save per-step comparison PNGs (built / ref / diff) for eye-check")
    p.set_defaults(save_built=True)
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

    # Build a lightweight chi(a) interpolant directly from cosmological parameters.
    # This avoids importing/instantiating DiscoDJ and its heavy timetables.
    def _make_chi_of_a_from_params(cosmo_params, n_table: int = 1000):
        Om_c = float(cosmo_params["Omega_c"])
        Om_b = float(cosmo_params["Omega_b"])
        h = float(cosmo_params["h"])
        Omega_k = float(cosmo_params.get("Omega_k", 0.0))
        w0 = float(cosmo_params.get("w0", -1.0))
        wa = float(cosmo_params.get("wa", 0.0))
        Om_m = Om_c + Om_b
        Om_de = 1.0 - Om_m - Omega_k

        def E(a):
            return np.sqrt(Om_m * a ** -3 + Omega_k * a ** -2 +
                           Om_de * a ** (-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a)))

        a_table = np.linspace(1e-6, 1.0, n_table)
        f = 1.0 / (E(a_table) * a_table ** 2)
        da = np.diff(a_table)
        mid = 0.5 * (f[:-1] + f[1:])
        # cumulative integral from a -> 1 using trapezoids (matches compute_conformalt)
        rev = np.concatenate(([0.0], (da * mid)[::-1]))
        conformal = -np.cumsum(rev)[::-1]
        chi_table = 2997.92458 * np.abs(conformal)

        def chi_of_a(a):
            a_arr = np.atleast_1d(a)
            chi_arr = np.interp(a_arr, a_table, chi_table, left=chi_table[0], right=0.0)
            return float(chi_arr) if np.isscalar(a) else chi_arr

        def a_of_chi(chi):
            chi_arr = np.atleast_1d(chi)
            a_arr = np.interp(chi_arr, chi_table[::-1], a_table[::-1], left=a_table[0], right=1.0)
            return float(a_arr) if np.isscalar(chi) else a_arr

        return chi_of_a, a_of_chi

    chi_of_a, _ = _make_chi_of_a_from_params(cosmo_dict, n_table=2000)

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
    # Filenames often have a trailing '.' before the extension (e.g. "..._z-low=0.0.fits").
    # Extract numeric substrings robustly so float conversion later never fails.
    def _extract_num(token: str) -> str:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", token)
        return m.group(0) if m else token.strip("., ")

    fits_index: dict[tuple[str, str], Path] = {}
    for fp in fits_dir.glob(f"{args.snap_prefix}-shell_z-high=*_z-low=*.fits"):
        m = re.search(r"z-high=([0-9.e+\-]+)_z-low=([0-9.e+\-]+)", fp.name)
        if m:
            zh_raw, zl_raw = m.group(1), m.group(2)
            zh = _extract_num(zh_raw)
            zl = _extract_num(zl_raw)
            fits_index[(zh, zl)] = fp
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
        # also read snapshot velocities (tipsy fields 'vx','vy','vz')
        vel_curr = np.stack([p_data['vx'], p_data['vy'], p_data['vz']], axis=1).astype(np.float32)
        del p_data

        # pkdgrav3 corner-origin → DISCO box-centre-origin (same as _load_external_ics)
        pos_curr = (pos_curr + args.boxsize / 2.0) % args.boxsize
        t_read = time() - t_read

        if step_i == 0:
            print(f"Step {step_i:3d}: a={a_curr:.6f}  z={1/a_curr-1:.4f}  "
                  f"read in {t_read:.1f}s  [IC / no comparison]")
            pos_prev = pos_curr
            # initialize previous velocities for the sliding window
            vel_prev = vel_curr
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
            vel_prev = vel_curr
            a_prev   = a_curr
            continue

        # ── Find matching FITS reference ──────────────────────────────────
        fits_path = _find_fits(fits_index, z_hi, z_lo, args.snap_prefix, fits_dir)
        if fits_path is None:
            print("  [no FITS match — skipping comparison]")
            pos_prev = pos_curr
            vel_prev = vel_curr
            a_prev   = a_curr
            continue

        # Load the reference map (may be high-res). We'll detect if the
        # reference encodes binary occupancy (0/1) at high resolution and
        # handle comparisons appropriately.
        m_ref_raw = hp.read_map(str(fits_path), field=0, dtype=np.float32, verbose=False)
        nside_ref = hp.npix2nside(len(m_ref_raw))

        # Cheap sampling test to decide if reference is binary (occupancy)
        is_ref_binary = False
        try:
            L = len(m_ref_raw)
            # sample up to 100k pixels evenly to avoid expensive full scans
            nsample = min(100000, L)
            idx = np.linspace(0, L - 1, num=nsample, dtype=int)
            sample = m_ref_raw[idx]
            if sample.max() <= 1.0 and sample.min() >= 0.0 and np.allclose(sample, np.rint(sample)):
                is_ref_binary = True
        except Exception:
            # If sampling fails, assume non-binary and continue
            is_ref_binary = False

        # Produce a coarse (args.nside) representation of the reference.
        if nside_ref != args.nside:
            m_ref_coarse = hp.ud_grade(m_ref_raw, nside_out=args.nside, order_in='RING')
        else:
            m_ref_coarse = m_ref_raw

        # If the reference is binary occupancy at high-res, convert the
        # coarse-averaged map into a per-coarse-pixel count of occupied
        # high-res pixels. This is not identical to per-particle counts,
        # but gives a comparable integer-like quantity.
        if is_ref_binary:
            scale = (nside_ref // args.nside) ** 2 if nside_ref >= args.nside else 1
            m_ref = (m_ref_coarse * float(scale)).astype(np.float32)
        else:
            m_ref = m_ref_coarse.astype(np.float32)

        # ── Run LightconeShellBuilder (unchanged from production) ─────────
        t_build = time()
        if args.no_gpu:
            # Convert snapshot velocities to builder units (match sim loader conversion)
            # Use a_prev/a_curr for prev/curr velocities respectively
            vfac_prev = (a_prev ** 2 / np.sqrt(8 * np.pi / 3.0)) * args.boxsize
            vfac_curr = (a_curr ** 2 / np.sqrt(8 * np.pi / 3.0)) * args.boxsize
            vel_prev_conv = vel_prev.astype(np.float32) * vfac_prev
            vel_curr_conv = vel_curr.astype(np.float32) * vfac_curr

            m_built = builder.accumulate_shell(
                pos_prev, pos_curr, a_prev, a_curr,
                r_lo_override=r_lo, r_hi_override=r_hi,
                vel_prev=vel_prev_conv, vel_curr=vel_curr_conv,
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

        # If the reference was detected as binary occupancy at high-res,
        # also report occupancy-based metrics (coarse occupancy vs built)
        if 'is_ref_binary' in locals() and is_ref_binary:
            try:
                ref_occ = (m_ref_coarse > 0.5)
                built_occ = (m_built > 0)
                ref_occ_count = int(np.count_nonzero(ref_occ))
                built_occ_count = int(np.count_nonzero(built_occ))
                occ_ratio = float(built_occ_count) / float(ref_occ_count) if ref_occ_count > 0 else float('nan')
                occ_xcorr = float(np.corrcoef(built_occ.ravel().astype(np.float32), ref_occ.ravel().astype(np.float32))[0, 1]) \
                    if (built_occ_count > 0 and ref_occ_count > 0) else float('nan')
                print(f"  [note] reference appears binary occupancy (nside={nside_ref})  "
                      f"occ_built={built_occ_count} occ_ref={ref_occ_count} "
                      f"occ_ratio={occ_ratio:.3f} occ_xcorr={occ_xcorr:.3f}")
            except Exception:
                pass

        # Save built / reference / diff maps and optional PNG for eye-check
        if args.save_built:
            built_dir = args.output_dir / "built_maps"
            ref_dir = args.output_dir / "ref_maps"
            diff_dir = args.output_dir / "diff_maps"
            built_dir.mkdir(parents=True, exist_ok=True)
            ref_dir.mkdir(parents=True, exist_ok=True)
            diff_dir.mkdir(parents=True, exist_ok=True)

            # Prefer the reference FITS' z-range tokens when naming saved files
            m_z = re.search(r"z-high=([0-9.eE+\-]+)_z-low=([0-9.eE+\-]+)", fits_path.name)
            if m_z:
                zh_raw, zl_raw = m_z.group(1), m_z.group(2)
                zh_str = _extract_num(zh_raw)
                zl_str = _extract_num(zl_raw)
            else:
                zh_str = f"{z_hi:.6f}".rstrip('0').rstrip('.')
                zl_str = f"{z_lo:.6f}".rstrip('0').rstrip('.')

            # Normalize negative-zero strings (keep positive 0.0)
            def _norm_zero(s: str) -> str:
                try:
                    v = float(s)
                except Exception:
                    return s
                return "0.0" if v == 0.0 else s

            zh_str = _norm_zero(zh_str)
            zl_str = _norm_zero(zl_str)

            # Write built map using the reference's z tokens for consistent labels
            built_fname = f"{args.snap_prefix}-built-shell_step={step_i:05d}_z-high={zh_str}_z-low={zl_str}.fits"
            built_path = built_dir / built_fname
            try:
                _safe_write_map(built_path, m_built)
            except Exception as e:
                print(f"  [warning] failed to save built FITS: {e}")

            # Copy reference FITS into output folder using a sanitized name
            # Always (re)write the sanitized copy so names remain consistent
            ref_copy_name = f"{args.snap_prefix}-ref-shell_step={step_i:05d}_z-high={zh_str}_z-low={zl_str}.fits"
            ref_copy = ref_dir / ref_copy_name
            try:
                shutil.copy(str(fits_path), str(ref_copy))
            except Exception as e:
                print(f"  [warning] failed to copy reference FITS: {e}")

            # Save diff map (built - ref) using the same z tokens
            try:
                diff = m_built.astype(np.float32) - m_ref.astype(np.float32)
                diff_fname = f"{args.snap_prefix}-diff-shell_step={step_i:05d}_z-high={zh_str}_z-low={zl_str}.fits"
                diff_path = diff_dir / diff_fname
                _safe_write_map(diff_path, diff)
            except Exception as e:
                print(f"  [warning] failed to save diff FITS: {e}")

            # Optional PNG side-by-side for quick visual inspection
            if args.save_plots:
                try:
                    fig = plt.figure(figsize=(12, 4))
                    hp.mollview(m_built, title=f"Built step {step_i}", sub=(1, 3, 1), min=None, max=None)
                    hp.mollview(m_ref,   title=f"Reference",      sub=(1, 3, 2), min=None, max=None)
                    hvmax = max(abs(diff).max(), 1e-9)
                    hp.mollview(diff,    title="Diff (built - ref)", sub=(1, 3, 3), min=-hvmax, max=hvmax, cmap='seismic')
                    outpng = args.output_dir / f"comparison_step_{step_i:05d}.png"
                    plt.savefig(str(outpng), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"  [warning] failed to write comparison PNG: {e}")

        results.append({
            "step":       step_i,
            "z_lo":       z_lo,   "z_hi":       z_hi,
            "r_lo":       r_lo,   "r_hi":       r_hi,
            "built_mean": b_mean, "ref_mean":   r_mean,
            "ratio":      ratio,  "xcorr":      xcorr,
        })

        # Slide window: release prev, promote curr to prev
        pos_prev = pos_curr
        vel_prev = vel_curr
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
    # Parse available FITS z-bounds into floats
    entries: list[tuple[float, float, Path]] = []
    for (zh_str, zl_str), fp in fits_index.items():
        try:
            zh = float(zh_str); zl = float(zl_str)
        except ValueError:
            continue
        entries.append((zh, zl, fp))

    # 1) Exact float match within tol
    tol_exact = 1e-4
    for zh, zl, fp in entries:
        if abs(zh - z_hi) < tol_exact and abs(zl - z_lo) < tol_exact:
            return fp

    # 2) FITS interval contained inside the snapshot pair interval
    #    (common case when shell_collector used different bin edges)
    for zh, zl, fp in entries:
        if zl >= z_lo - 1e-6 and zh <= z_hi + 1e-6:
            return fp

    # 3) Choose the FITS file with the largest overlap with [z_lo, z_hi]
    best_fp = None
    best_overlap = 0.0
    for zh, zl, fp in entries:
        overlap = min(zh, z_hi) - max(zl, z_lo)
        if overlap > best_overlap:
            best_overlap = overlap
            best_fp = fp
    if best_overlap > 0.0:
        return best_fp

    # 4) Fallback: closest (sum of absolute differences) within a small error
    best_path = None
    best_err = 1.0
    for zh, zl, fp in entries:
        err = abs(zh - z_hi) + abs(zl - z_lo)
        if err < best_err:
            best_err = err
            best_path = fp
    if best_err < 1e-3:
        return best_path

    return None


def _safe_write_map(path: Path, m: np.ndarray) -> None:
    """Write a HEALPix map to `path` handling older healpy APIs.

    Accepts `m` as a numpy array; ensures the file is written and
    overwrites existing files if necessary.
    """
    try:
        hp.write_map(str(path), m.astype(np.float32), overwrite=True)
    except TypeError:
        # Older healpy versions may not support `overwrite` kwarg
        if path.exists():
            path.unlink()
        hp.write_map(str(path), m.astype(np.float32))


if __name__ == "__main__":
    main()
