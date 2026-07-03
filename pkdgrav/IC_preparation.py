#!/usr/bin/env python3
"""IC preparation for all cosmo_*/run_0 directories.

For every cosmo_*/run_0 directory this script:
  0. Deletes all run_* directories except run_0
  1. Unzips param_files.tar.gz if cosmology.par is missing
  2. Patches cosmology.par (achTfFile, bClass, cosmological params,
     bWriteIC, bParaWrite, nSteps4)
  3. Patches baryonification_params.py (transfct path)

This is a faster, parallel rewrite of the old run_IC_preparation.sh loop.
The old script launched ~4 fresh Python interpreters per directory (each
re-importing numpy/h5py/classy) and ran the expensive CLASS computation
serially. Here everything runs in one interpreter and the per-directory
work is spread across processes, so the 48 CLASS solves run concurrently.

Patching is idempotent: each change is only applied if still needed.
"""

# Keep numeric libraries single-threaded: we parallelize across directories,
# so per-process BLAS/OpenMP threads would only oversubscribe the node.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import sys
import shutil
import tarfile
import multiprocessing as mp
from pathlib import Path

import numpy as np
import h5py
import yaml

COSMOGRID_DIR = Path("/capstor/scratch/cscs/damrein/cosmogridv1")
K_PIVOT = 0.05


# --------------------------------------------------------------------------
# Extraction helpers (translated from the shell here-docs)
# --------------------------------------------------------------------------
def extract_py_param(py_file: Path, var: str):
    """Mimic: grep '^\\s*var\\s*=' | head -1 | sed 's/.*=\\s*//' | tr -d ' \"'."""
    pat = re.compile(r"^\s*" + re.escape(var) + r"\s*=")
    for line in py_file.read_text().splitlines():
        if pat.search(line):
            value = line.rsplit("=", 1)[1]        # take text after the last '='
            return value.replace(" ", "").replace('"', "")
    return None


def extract_omega_0(params_file: Path):
    with params_file.open() as f:
        p = yaml.safe_load(f)
    omega_0 = float(p["O_cdm"]) + float(p.get("O_nu", 0.0)) + float(p["Ob"])
    return f"{omega_0:.12f}"


def extract_w_0(params_file: Path):
    with params_file.open() as f:
        p = yaml.safe_load(f)
    return f"{float(p['w0']):.12f}"


def _tophat_W(x):
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3


def _compute_sigma8(k_hMpc, Pk, R=8.0):
    x = k_hMpc * R
    W = _tophat_W(x)
    integrand = Pk * k_hMpc**2 * W**2 / (2 * np.pi**2)
    return np.sqrt(np.trapezoid(integrand, k_hMpc))


def extract_sigma8_and_omega_rad(hdf5_file: Path, params_file: Path):
    """Return (sigma8_str, omega_rad_str) or None on failure."""
    from classy import Class  # imported lazily inside each worker

    with params_file.open() as f:
        p = yaml.safe_load(f)

    A_s = float(p["As"])
    n_s = float(p["ns"])
    H0 = float(p["H0"])
    h = H0 / 100.0

    Omega_b = float(p["Ob"])
    Omega_cdm = float(p["O_cdm"])
    _mnu = p["m_nu"] if isinstance(p["m_nu"], list) else [float(p["m_nu"])]
    _w0 = float(p["w0"])
    _wa = float(p["wa"])

    # TODO: T_ncdm = (4/11)**(1/3) * (3.046/len(_mnu))**(1/4)
    T_ncdm = (4.0 / 11.0) ** (1.0 / 3.0) * (3.046 / 3) ** (1.0 / 4.0)

    with h5py.File(hdf5_file, "r") as f:
        k = f["perturbations/k"][:]
        a_pert = f["perturbations/a"][:]
        d_cb = f["perturbations/delta_cdm+b"][:]

    i_a0 = np.argmin(np.abs(a_pert - 1.0))
    delta_cb = d_cb[i_a0, :]

    P_prim = A_s * (k / K_PIVOT) ** (n_s - 1)
    Pk_hdf5_cb = (2 * np.pi**2 / k**3) * P_prim * delta_cb**2 * h**3
    k_hMpc = k / h

    sigma8 = _compute_sigma8(k_hMpc, Pk_hdf5_cb)

    cosmo_dict = {
        "H0": H0,
        "Omega_b": Omega_b,
        "Omega_cdm": Omega_cdm,
        "Omega_Lambda": 0.0,
        "w0_fld": _w0,
        "wa_fld": _wa,
        "N_ur": 0,
        "N_ncdm": 1,
        "deg_ncdm": 3,
        "m_ncdm": 0.02,
        "T_ncdm": T_ncdm,
        "A_s": A_s,
        "n_s": n_s,
        "k_pivot": K_PIVOT,
        "output": "mPk",
        "P_k_max_h/Mpc": k_hMpc.max() * 1.1,
        "z_pk": 0.0,
    }

    cosmo = Class()
    try:
        cosmo.set(cosmo_dict)
        cosmo.compute()
        omega_rad = cosmo.Omega_r()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()

    return f"{sigma8:.10e}", f"{omega_rad:.10e}"


# --------------------------------------------------------------------------
# Patching (translated from the shell sed logic)
# --------------------------------------------------------------------------
def patch_cosmology_par(par_file: Path, abs_class: Path, out_name: str,
                        bary_file: Path, params_file: Path, log):
    text = par_file.read_text()
    lines = text.splitlines()
    changed = False

    # achOutName = "..."
    new_lines = []
    for ln in lines:
        if re.match(r"^achOutName\s*=", ln):
            new_lines.append(f'achOutName = "{out_name}"')
            changed = True
        else:
            new_lines.append(ln)
    lines = new_lines

    # achClassFilename... -> achTfFile = "transfer_fiducial_cb.dat"
    if any("achClassFilename" in ln for ln in lines):
        new_lines = []
        for ln in lines:
            if ln.startswith("achClassFilename"):
                new_lines.append('achTfFile = "transfer_fiducial_cb.dat"')
                changed = True
            else:
                new_lines.append(ln)
        lines = new_lines

    # bClass: force to 0, inserting the key if absent
    if any(re.match(r"^bClass\s*=\s*1", ln) for ln in lines):
        lines = [re.sub(r"^bClass\s*=\s*1", "bClass = 0", ln) for ln in lines]
        changed = True
    elif not any(re.match(r"^bClass", ln) for ln in lines):
        new_lines = []
        inserted = False
        for ln in lines:
            new_lines.append(ln)
            if not inserted and ln.startswith("achTfFile"):
                new_lines.append("bClass = 0")
                inserted = True
        if not inserted:
            new_lines.append("bClass = 0")
        lines = new_lines
        changed = True

    # Remove obsolete parameters
    for rp in ("nGridLin", "achLinSpecies", "achPkSpecies",
               "b2", "b2LPT", "dTheta20", "dTheta2"):
        pat = re.compile(r"^" + re.escape(rp) + r"\s*=")
        kept = [ln for ln in lines if not pat.match(ln)]
        if len(kept) != len(lines):
            lines = kept
            changed = True

    # Cosmological parameter block
    if bary_file.is_file() and str(abs_class):
        vals = None
        try:
            sigma8_val, omega_rad = extract_sigma8_and_omega_rad(abs_class, params_file)
            h_val = extract_py_param(bary_file, "par.cosmo.h0")
            omega_0 = extract_omega_0(params_file)
            omega_b_val = extract_py_param(bary_file, "par.cosmo.Ob")
            ns_val = extract_py_param(bary_file, "par.cosmo.ns")
            w0_val = extract_w_0(params_file)
            vals = (sigma8_val, omega_rad, h_val, omega_0,
                    omega_b_val, ns_val, w0_val)
        except Exception as e:  # noqa: BLE001
            log.append(f"  Warning: failed to compute cosmo params: {e}")

        if vals and all(v not in (None, "") for v in vals):
            sigma8_val, omega_rad, h_val, omega_0, omega_b_val, ns_val, w0_val = vals

            # Scrub previous block (header + any of these keys)
            scrub = re.compile(
                r"^\s*(# Cosmological parameters|"
                r"(dOmegaRad|h|dOmega0|dOmegaDE|dOmegab|dSpectral|dSigma8|w0|wa)\s*=)"
            )
            lines = [ln for ln in lines if not scrub.match(ln)]
            # Drop trailing blank lines so re-runs don't accumulate whitespace
            while lines and lines[-1].strip() == "":
                lines.pop()

            omega_de = f"{1 - float(omega_0):.10g}"
            lines += [
                "",
                "# Cosmological parameters",
                f"dOmegaRad        = {omega_rad}",
                f"h                = {h_val}",
                f"dOmega0          = {omega_0}",
                f"dOmegaDE          = {omega_de}",
                f"dOmegab          = {omega_b_val}",
                f"dSpectral        = {ns_val}",
                f"dSigma8          = {sigma8_val}",
                f"w0               = {w0_val}",
                "wa               = 0.0",
            ]
            changed = True

    # Comment out nSteps4
    new_lines = []
    for ln in lines:
        if re.match(r"^nSteps4", ln):
            new_lines.append("#" + ln)
            changed = True
        else:
            new_lines.append(ln)
    lines = new_lines

    # bWriteIC = 0 -> 1
    if any(re.search(r"bWriteIC\s*=\s*0", ln) for ln in lines):
        lines = [re.sub(r"bWriteIC.*", "bWriteIC         = 1", ln)
                 if re.search(r"bWriteIC\s*=\s*0", ln) else ln for ln in lines]
        changed = True

    # bParaWrite = 0 -> 1
    if any(re.search(r"bParaWrite\s*=\s*0", ln) for ln in lines):
        lines = [re.sub(r"bParaWrite.*", "bParaWrite       = 1", ln)
                 if re.search(r"bParaWrite\s*=\s*0", ln) else ln for ln in lines]
        changed = True

    if changed:
        par_file.write_text("\n".join(lines) + "\n")
        log.append("  Patched cosmology.par")


def patch_baryonification_params(py_file: Path, abs_class: Path, log):
    old = 'transfct        = "class_processed.hdf5"'
    text = py_file.read_text()
    if old in text:
        py_file.write_text(text.replace(old, f'transfct        = "{abs_class}"'))
        log.append("  Patched baryonification_params.py")


# --------------------------------------------------------------------------
# Per-directory driver
# --------------------------------------------------------------------------
def process_cosmo_dir(cosmo_dir: Path):
    log = [f"Processing {cosmo_dir}/run_0/..."]

    # Step 0: delete run_* except run_0
    for run_dir in sorted(cosmo_dir.glob("run_*")):
        if run_dir.is_dir() and run_dir.name != "run_0":
            log.append(f"Deleting {run_dir}/...")
            shutil.rmtree(run_dir)

    run_dir = cosmo_dir / "run_0"
    if not run_dir.is_dir():
        return log

    tarball = run_dir / "param_files.tar.gz"
    cos_file = run_dir / "cosmology.par"
    params_file = run_dir / "params.yml"
    bary_file = run_dir / "baryonification_params.py"
    abs_class = run_dir / "class_processed.hdf5"

    cosmo_id = cosmo_dir.name
    tag = cosmo_id[len("cosmo_"):] if cosmo_id.startswith("cosmo_") else cosmo_id
    out_name = f"CosmoML_{tag}_run_0"

    if not tarball.is_file():
        log.append(f"  Warning: {tarball} not found. Skipping.")
        return log

    # Step 1: unzip if not yet extracted
    if not cos_file.is_file():
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(run_dir)
        log.append(f"  Unzipped: {run_dir}")

    # Step 2: patch cosmology.par
    if cos_file.is_file():
        patch_cosmology_par(cos_file, abs_class, out_name, bary_file, params_file, log)

    # Step 3: patch baryonification_params.py
    if bary_file.is_file():
        patch_baryonification_params(bary_file, abs_class, log)

    return log


def main():
    if not COSMOGRID_DIR.is_dir():
        print(f"Error: Directory {COSMOGRID_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    cosmo_dirs = sorted(d for d in COSMOGRID_DIR.glob("cosmo_*") if d.is_dir())
    if not cosmo_dirs:
        print("No cosmo_* directories found.")
        return

    # One worker per directory, capped by the cores we're allowed to use.
    # Under SLURM the batch step's cgroup may hide the full node, so prefer an
    # explicit budget: IC_PREP_WORKERS, else SLURM's allocation, else os count.
    if os.environ.get("IC_PREP_WORKERS"):
        core_budget = int(os.environ["IC_PREP_WORKERS"])
    elif os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS_PER_NODE"):
        core_budget = (int(os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
                       * int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    else:
        core_budget = os.cpu_count() or 1
    n_workers = max(1, min(len(cosmo_dirs), core_budget))
    print(f"Preparing {len(cosmo_dirs)} directories using {n_workers} workers...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for log in pool.imap_unordered(process_cosmo_dir, cosmo_dirs):
            print("\n".join(log), flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
