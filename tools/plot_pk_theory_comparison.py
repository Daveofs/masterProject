"""
plot_pk_theory_comparison.py
=====================
Compare the linear matter power spectrum P(k) at z=0 reconstructed from a
CLASS HDF5 output file against the direct CLASS theory prediction.

Usage
-----
python plot_pk_theory_comparison.py \\
    [--params   /path/to/params.yml] \\
    [--hdf5     /path/to/class_processed.hdf5] \\
    [--out-dir  /path/to/output_dir]

Defaults:
    --params   /capstor/scratch/cscs/damrein/cosmogridv1_fiducial/run_0000/params.yml
    --hdf5     /capstor/scratch/cscs/damrein/cosmogridv1_fiducial/run_0000/class_processed.hdf5
    --out-dir  /capstor/scratch/cscs/damrein/outputs/plots
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import sys
import types
from pathlib import Path as _Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Bootstrap discodj cosmology without triggering the full package stack ──
# discodj/__init__.py pulls in the nbody stack which requires scripts/ on
# sys.path. We only need the cosmology subpackage, so we register lightweight
# stub packages for 'discodj' and its sub-packages before any import touches
# __init__.py. The real module files are then loaded normally by Python.
_DISCODJ_SRC = _Path("/users/damrein/DISCO-DJ/src")
sys.path.insert(0, str(_DISCODJ_SRC))

def _stub_pkg(dotted_name: str, fs_path: _Path) -> types.ModuleType:
    """Register a package stub that skips __init__.py execution."""
    pkg = types.ModuleType(dotted_name)
    pkg.__path__ = [str(fs_path)]
    pkg.__package__ = dotted_name
    sys.modules.setdefault(dotted_name, pkg)
    return pkg

_stub_pkg("discodj",            _DISCODJ_SRC / "discodj")
_stub_pkg("discodj.core",       _DISCODJ_SRC / "discodj" / "core")
_stub_pkg("discodj.cosmology",  _DISCODJ_SRC / "discodj" / "cosmology")

from discodj.cosmology.cosmology import Cosmology as DiscoCosmology
from discodj.cosmology.transfer_functions import eisenstein_hu
from discodj.cosmology.cosmo_utils import get_sigma8_squared_from_Pk

import jax.numpy as jnp
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from classy import Class

# ── Defaults ──────────────────────────────────────────────────────────────────
_FIDUCIAL_DIR = Path("/capstor/scratch/cscs/damrein/cosmogridv1_fiducial/run_0000")
DEFAULT_PARAMS  = _FIDUCIAL_DIR / "params.yml"
DEFAULT_HDF5    = _FIDUCIAL_DIR / "class_processed.hdf5"
DEFAULT_OUT_DIR = Path("/capstor/scratch/cscs/damrein/outputs/plots")

K_PIVOT = 0.05  # Mpc^-1 (fixed CLASS convention)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot P(k): HDF5 reconstruction vs CLASS theory")
    p.add_argument("--params",  default=str(DEFAULT_PARAMS),  help="Path to params.yml")
    p.add_argument("--hdf5",    default=str(DEFAULT_HDF5),    help="Path to class_processed.hdf5")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for the PNG")
    return p.parse_args()


# ── Load parameters from params.yml ───────────────────────────────────────────
def load_params(params_yml: str) -> dict:
    with open(params_yml) as fh:
        p = yaml.safe_load(fh)

    H0      = float(p["H0"])
    h       = H0 / 100.0
    Omega_b = float(p["Ob"])
    Omega_m = float(p["Om"])
    m_nu    = float(p["m_nu"])          # mass per species in eV (3 degenerate)
    A_s     = float(p["As"])
    n_s     = float(p["ns"])
    w0      = float(p.get("w0", -1.0))
    wa      = float(p.get("wa",  0.0))

    sigma8 = float(p["s8"])

    # Derived quantities
    Omega_nu  = float(p["O_nu"])        # already computed in params.yml
    Omega_cdm = float(p["O_cdm"])       # already computed in params.yml

    # T_ncdm: 3 degenerate neutrinos, effective N_eff=3.046
    T_ncdm = (4.0 / 11.0) ** (1.0 / 3.0) * (3.046 / 3.0) ** (1.0 / 4.0)

    return dict(
        H0=H0, h=h,
        Omega_b=Omega_b, Omega_m=Omega_m,
        Omega_cdm=Omega_cdm, Omega_nu=Omega_nu,
        m_nu=m_nu, T_ncdm=T_ncdm,
        A_s=A_s, n_s=n_s,
        sigma8=sigma8,
        w0=w0, wa=wa,
    )


# ── Load HDF5 perturbations & background ──────────────────────────────────────
def load_hdf5(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        k        = f["perturbations/k"][:]               # Mpc^-1, shape (n_k,)
        a_pert   = f["perturbations/a"][:]               # shape (n_a,)
        d_cb     = f["perturbations/delta_cdm+b"][:]     # shape (n_a, n_k)
        d_ncdm   = f["perturbations/delta_ncdm[0]"][:]   # shape (n_a, n_k)

        a_bg     = f["background/a"][:]
        rho_cb   = f["background/rho_cdm+b"][:]
        rho_ncdm = f["background/rho_ncdm[0]"][:]

    return k, a_pert, d_cb, d_ncdm, a_bg, rho_cb, rho_ncdm


# ── Reconstruct P(k) from HDF5 ────────────────────────────────────────────────
def reconstruct_pk(k, a_pert, d_cb, d_ncdm, a_bg, rho_cb, rho_ncdm, params: dict):
    rho_cb_interp   = interp1d(a_bg, rho_cb,   kind="linear")(a_pert)
    rho_ncdm_interp = interp1d(a_bg, rho_ncdm, kind="linear")(a_pert)

    i_a0 = np.argmin(np.abs(a_pert - 1.0))
    print(f"Using a = {a_pert[i_a0]:.6f} (index {i_a0}) for z=0")

    rho_cb_0   = rho_cb_interp[i_a0]
    rho_ncdm_0 = rho_ncdm_interp[i_a0]
    rho_m      = rho_cb_0 + rho_ncdm_0

    delta_m = (rho_cb_0 * d_cb[i_a0, :] + rho_ncdm_0 * d_ncdm[i_a0, :]) / rho_m

    h     = params["h"]
    A_s   = params["A_s"]
    n_s   = params["n_s"]

    P_prim  = A_s * (k / K_PIVOT) ** (n_s - 1)
    Pk_hdf5 = (2 * np.pi**2 / k**3) * P_prim * delta_m**2 * h**3  # (Mpc/h)^3
    k_hMpc  = k / h

    return k_hMpc, Pk_hdf5


# ── DISCO-DJ: Eisenstein-Hu transfer function + sigma8 normalisation ────────
# Uses discodj.cosmology directly — same code path as
# DiscoDJ.with_linear_ps(transfer_function="Eisenstein-Hu").

def compute_discodj_pk(k_hMpc: np.ndarray, params: dict) -> np.ndarray:
    """Compute DISCO-DJ's linear P(k) using discodj.cosmology directly.

    Mirrors DiscoDJ.with_linear_ps(transfer_function='Eisenstein-Hu'):
      Pk = k^n_s * T_EH(k)^2  then rescaled to sigma8.
    """
    cosmo = DiscoCosmology(
        Omega_c = params["Omega_cdm"],
        Omega_b = params["Omega_b"],
        h       = params["h"],
        sigma8  = params["sigma8"],
        n_s     = params["n_s"],
        Omega_k = 0.0,
        w0      = params["w0"],
        wa      = params["wa"],
    )

    k_jnp = jnp.asarray(k_hMpc)

    # Transfer function — discodj.cosmology.transfer_functions.eisenstein_hu
    T = np.asarray(eisenstein_hu(cosmo, k_jnp))

    # Primordial spectrum: P ∝ k^n_s (mirrors DiscoDJ.compute_primordial_ps)
    Pk = k_hMpc ** params["n_s"] * T ** 2

    # Rescale to sigma8 — discodj.cosmology.cosmo_utils.get_sigma8_squared_from_Pk
    sigma8_sq = float(get_sigma8_squared_from_Pk(k_jnp, jnp.asarray(Pk), with_jax=True))
    Pk *= params["sigma8"] ** 2 / sigma8_sq
    return Pk


# ── CLASS theory prediction ───────────────────────────────────────────────────
def compute_class_pk(k_Mpc, k_hMpc, params: dict):
    cosmo = Class()
    cosmo.set({
        "H0"            : params["H0"],
        "Omega_b"       : params["Omega_b"],
        "Omega_cdm"     : params["Omega_cdm"],
        "Omega_Lambda"  : 0,
        "w0_fld"        : params["w0"],
        "wa_fld"        : params["wa"],
        "N_ur"          : 0,
        "N_ncdm"        : 1,
        "deg_ncdm"      : 3,
        "m_ncdm"        : params["m_nu"],
        "T_ncdm"        : params["T_ncdm"],
        "A_s"           : params["A_s"],
        "n_s"           : params["n_s"],
        "k_pivot"       : K_PIVOT,
        "output"        : "mPk",
        "P_k_max_h/Mpc" : k_hMpc.max() * 1.1,
        "z_pk"          : 0.0,
    })
    cosmo.compute()

    h = params["h"]
    Pk_class = np.array([cosmo.pk_lin(ki, 0.0) * h**3 for ki in k_Mpc])  # (Mpc/h)^3

    cosmo.struct_cleanup()
    cosmo.empty()

    return Pk_class


# ── Plot ──────────────────────────────────────────────────────────────────────
def make_plot(k_hMpc, Pk_class, Pk_hdf5, Pk_discodj, params: dict, out_path: Path):
    m_nu_sum = params["m_nu"] * 3
    w0       = params["w0"]
    sigma8   = params["sigma8"]

    fig = plt.figure(figsize=(9, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Main panel
    ax1.loglog(k_hMpc, Pk_class,   color="steelblue",   lw=2,   label="CLASS (theory)")
    ax1.loglog(k_hMpc, Pk_hdf5,    color="tomato",       lw=1.5, ls="--", label="HDF5 reconstruction")
    ax1.loglog(k_hMpc, Pk_discodj, color="forestgreen",  lw=1.5, ls=":",  label=rf"DISCO-DJ (E-H, $\sigma_8={sigma8}$)")
    ax1.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_title(
        rf"Linear matter $P(k)$ at $z=0$ — $\sum m_\nu={m_nu_sum:.2f}\,$eV, $w_0={w0}$",
        fontsize=12,
    )
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, which="both", alpha=0.2)

    # Ratio panel — both HDF5 and DISCO-DJ relative to CLASS
    ax2.semilogx(k_hMpc, Pk_hdf5    / Pk_class, color="tomato",      lw=1.5, label="HDF5 / CLASS")
    ax2.semilogx(k_hMpc, Pk_discodj / Pk_class, color="forestgreen", lw=1.5, ls=":", label="DISCO-DJ / CLASS")
    ax2.axhline(1.0,  color="gray", lw=1,   ls="--")
    ax2.axhline(1.05, color="gray", lw=0.7, ls=":")
    ax2.axhline(0.95, color="gray", lw=0.7, ls=":")
    ax2.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$", fontsize=13)
    ax2.set_ylabel(r"ratio / CLASS",          fontsize=11)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, which="both", alpha=0.2)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Pk_comparison.png"

    print(f"Loading parameters from {args.params}")
    params = load_params(args.params)
    print(
        f"  H0={params['H0']}, Omega_b={params['Omega_b']}, Omega_cdm={params['Omega_cdm']}, "
        f"Omega_nu={params['Omega_nu']}, m_nu={params['m_nu']} eV, "
        f"A_s={params['A_s']:.4e}, n_s={params['n_s']}, w0={params['w0']}, wa={params['wa']}"
    )

    print(f"Loading HDF5 from {args.hdf5}")
    k, a_pert, d_cb, d_ncdm, a_bg, rho_cb, rho_ncdm = load_hdf5(args.hdf5)

    k_hMpc, Pk_hdf5 = reconstruct_pk(k, a_pert, d_cb, d_ncdm, a_bg, rho_cb, rho_ncdm, params)

    print("Running CLASS ...")
    Pk_class = compute_class_pk(k, k_hMpc, params)

    print("Computing DISCO-DJ E-H P(k) ...")
    Pk_discodj = compute_discodj_pk(k_hMpc, params)

    make_plot(k_hMpc, Pk_class, Pk_hdf5, Pk_discodj, params, out_path)


if __name__ == "__main__":
    main()
