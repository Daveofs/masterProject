"""Weak-lensing convergence (kappa) map diagnostic: reduces a full lightcone (every
shell within [zi, zf]) into ONE kappa map via UFalcon, for low/corrected/high side by
side, across held-out cosmologies -- the same low-vs-corrected-vs-high comparison
every other diagnostic in this project makes, but at the level UFalcon's downstream
science actually consumes: one kappa map per cosmology, not per shell.

Generalizes /users/damrein/masterProject/vis/weak_lensing_ufalcon.py, which hardcoded
ONE example cosmology's params (and, checked directly: those hardcoded params did not
even match the cosmology whose lightcone that script actually loaded). Here every
cosmology's own params.yml is read live -- "further values" beyond H0/Om/Ob/s8/ns/w0
means z_low/z_high (each run's own shell_info) and the astropy Cosmology object built
from those params, all needed before UFalcon's construct_map_cosmogrid can run.

Shared by both unet/apply_flow.py and transfer/apply_transfer.py -- see
each script's --kappa-* section. Both now run in the SAME venv (sphereflow) so this
module (and UFalcon) only needs to be installed once, not duplicated across a
uenv/venv split and a separate conda env.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import healpy as hp
import yaml
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import scipy.integrate

# UFalcon 2.0.0's probe_weights.py calls the long-removed scipy.integrate.simps
# (renamed to `simpson` in scipy >= 1.14, and fully removed by the scipy version
# this project's shared venv uses, 1.18) -- patch it back rather than pin an old
# scipy for the whole shared env just for this one call.
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

from UFalcon import construct_maps

# CosmoGridV1 constants: box size and particle count are FIXED across every
# grid+fiducial cosmology actually present under this project's --data-root
# (verified directly against CosmoGridV1_metainfo.h5: box_size_Mpc_over_h=900.0,
# n_particles=832**3 for all 2517 grid/fiducial rows) -- NOT read per-cosmology.
BOX_SIZE_MPC_OVER_H = 900.0
N_PARTICLES = 832 ** 3
# Neff and m_nu are not stored in params.yml (checked: absent from every params.yml
# key set) -- m_nu matches the constant value params.yml DOES store (0.02 eV across
# every cosmology checked); Neff matches weak_lensing_ufalcon.py's own hardcoded value.
NEFF = 3.046
M_NU_EV = 0.02
TCMB0_K = 2.7255


def load_cosmo_yaml(run_dir) -> dict:
    """params.yml for one cosmology/run -- H0, Om, Ob, O_cdm, O_nu, s8, ns, w0, ...
    (see transfer_function.py's load_cosmo, which reads the same file for its own
    COSMO_KEYS subset; this reads the whole file since build_astropy_cosmo below
    needs O_nu too, which COSMO_KEYS does not include)."""
    with open(Path(run_dir) / "params.yml") as f:
        return yaml.safe_load(f)


def build_astropy_cosmo(params: dict) -> FlatLambdaCDM:
    """FlatLambdaCDM for ONE cosmology's own params.yml values -- Om0 excludes the
    neutrino contribution (Om - O_nu), same convention weak_lensing_ufalcon.py's
    (hardcoded) example used."""
    return FlatLambdaCDM(H0=params["H0"], Om0=params["Om"] - params["O_nu"],
                         Neff=NEFF, Ob0=params["Ob"], m_nu=M_NU_EV * u.eV,
                         Tcmb0=TCMB0_K)


def usable_shell_mask(lower_z: np.ndarray, upper_z: np.ndarray,
                      zi: float, zf: float) -> np.ndarray:
    """Shells UFalcon's construct_kappa_map will actually integrate over (it skips
    shells outside [zi, zf] internally, per-shell) -- filter BEFORE gathering shells
    (e.g. before the flow pipeline's expensive per-patch full-sky reconstruction) so
    no work is wasted on shells the kappa map would discard anyway."""
    return (upper_z > zi) & (lower_z < zf)


def kappa_map(maps: np.ndarray, lower_z: np.ndarray, upper_z: np.ndarray,
             cosmo_params: dict, n_of_z_path, nside: int = 128,
             zi: float = 0.0, zf: float = 1.05, fast_mode: bool = True,
             fast_mode_num_points_1d: int = 13, fast_mode_num_points_2d: int = 512
             ) -> np.ndarray:
    """One kappa map from a stack of full-sky shells (maps: (n_shells, npix) raw
    projected particle counts, already restricted to usable_shell_mask's selection),
    this cosmology's own params.yml-derived cosmology, and the n(z) at n_of_z_path.
    boxsize is converted from the fixed CosmoGridV1 box (Mpc/h) to Gpc using THIS
    cosmology's own H0 (same conversion weak_lensing_ufalcon.py's example used)."""
    cosmo = build_astropy_cosmo(cosmo_params)
    boxsize_gpc = (BOX_SIZE_MPC_OVER_H / 1000.0) / (cosmo_params["H0"] / 100.0)
    cls = construct_maps.construct_map_cosmogrid(
        maps=maps, z_low=lower_z, z_high=upper_z, nside=nside, boxsize=boxsize_gpc,
        cosmo=cosmo, n_particles=N_PARTICLES, zi=zi, zf=zf)
    return cls.construct_kappa_map(n_of_z=str(n_of_z_path), shift_nz=0.0, IA=None,
                                   fast_mode=fast_mode,
                                   fast_mode_num_points_1d=fast_mode_num_points_1d,
                                   fast_mode_num_points_2d=fast_mode_num_points_2d)


def kappa_cl(kappa: np.ndarray, lmax: int) -> np.ndarray:
    """Angular power spectrum of a kappa map. Unlike full_sky.od_cl (which computes
    an overdensity from a raw-COUNTS map first), a kappa map already IS the field of
    interest -- hp.anafast applies directly, no transform."""
    return hp.anafast(kappa.astype(np.float64), lmax=lmax)


# ---------------------------------------------------------------------------
# Cross-pipeline comparison support
# ---------------------------------------------------------------------------

_MOMENT_KEYS = ("variance", "skewness", "excess_kurtosis")


def save_kappa_moment_summary(out_path, cosmo_labels, mom_low, mom_corr, mom_high,
                              method_label: str, tag: str):
    """Persist the per-cosmology kappa-map moments behind kappa_moments_scatter.

    Every pipeline recomputes these from scratch inside its own --kappa block and
    then throws them away into a PNG, which makes a cross-pipeline comparison
    impossible without re-running all three full-sky reconstructions. Writing the
    three arrays out here is what lets vis/plot_method_comparison.py put transfer,
    unet-flow and diffusion on one axes afterwards, from cheap files.
    """
    import numpy as np
    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pack = lambda moms: np.array([[float(m[k]) for k in _MOMENT_KEYS] for m in moms])
    np.savez(out_path,
             cosmo_labels=np.array([str(c) for c in cosmo_labels]),
             moment_keys=np.array(_MOMENT_KEYS),
             mom_low=pack(mom_low), mom_corr=pack(mom_corr), mom_high=pack(mom_high),
             method_label=str(method_label), tag=str(tag))
    print(f"[weak_lensing] kappa moment summary ({len(cosmo_labels)} cosmologies, "
          f"{tag}) -> {out_path}", flush=True)
    return out_path
