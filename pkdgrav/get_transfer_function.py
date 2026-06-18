import re, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from classy import Class
import argparse


def get_class_transfer_function(k, Pk_hdf5, A_s, n_s, k_pivot, h):
    """
    Extract CLASS-format transfer function from HDF5 power spectrum
    
    Parameters:
    -----------
    k : array
        Wavenumber in h/Mpc
    Pk_hdf5 : array
        Matter power spectrum from HDF5 in (Mpc/h)^3
    A_s : float
        Scalar amplitude at k_pivot
    n_s : float
        Scalar spectral index
    k_pivot : float
        Pivot scale in 1/Mpc
    h : float
        Hubble parameter H0 / 100
    
    Returns:
    --------
    k : array
        Wavenumber in h/Mpc
    tf : array
        -T_tot/k^2 in CLASS format
    """
    
    # Reconstruct primary power spectrum
    P_prim = A_s * (k / k_pivot)**(n_s - 1)
    
    # Extract Δm from your equation:
    # Pk_hdf5 = (2π²/k³) × P_prim × Δm² × h³
    # Therefore: Δm = sqrt(Pk_hdf5 × k³ / (2π² × P_prim × h³))
    
    delta_m_sq = Pk_hdf5 * k**3 / (2 * np.pi**2 * P_prim * h**3)
    delta_m = np.sqrt(delta_m_sq)
    
    # CLASS defines: T_tot(k) = Δm(k)
    # And outputs: -T_tot/k^2 = -Δm/k^2
    # (The minus sign is convention; the /k^2 removes k-dependence for plotting)
    
    tf = delta_m / k**2
    
    return k, tf


def save_class_transfer_function(k, tf, filename='transfer_function.txt', label='tot'):
    """
    Save transfer function in CLASS format
    """
    header = f"k (h/Mpc)    -T_{label}/k2"
    np.savetxt(filename, np.column_stack([k, tf]),
               header=header, comments='#', fmt='%.8e')
    

parser = argparse.ArgumentParser(description='Extract CLASS-format transfer function from HDF5 power spectrum')
parser.add_argument('--param_dir', type=str, help='Path to the param dir used to start the sim', required=True)

# parse
args = parser.parse_args()

# ── 0. Parameter files ────────────────────────────────────────────────────────
PARAM_DIR = args.param_dir
# PARAM_DIR = "param_files_fiducial"
# PARAM_DIR = "param_files_cosmo_flamingo"
# PARAM_DIR = "param_files_grid00001"

def _parse_cosmology_par(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            m = re.match(r'(\w+)\s*=\s*(.+)', line)
            if m:
                params[m.group(1)] = m.group(2).strip()
    return params

def _parse_concept_params(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()

    def _get_float(pattern):
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"Pattern {pattern!r} not found in {path}")
        return float(m.group(1))

    def _get_list(pattern):
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"Pattern {pattern!r} not found in {path}")
        return [float(x) for x in m.group(1).split(',')]

    return {
        'H0'            : _get_float(r'H0\s*=\s*([\d.]+)'),
        'Omega_b'       : _get_float(r'Ωb\s*=\s*([\d.]+)'),
        'Omega_cdm_base': _get_float(r'Ωcdm\s*=\s*([\d.]+)'),  # Ωcdm + Ων before neutrino subtraction
        '_mnu'          : _get_list(r'_m[νv]\s*=\s*\[([\d.,\s]+)\]'),
        '_w0'           : _get_float(r'_w0\s*=\s*([-\d.]+)'),
        '_wa'           : _get_float(r'_wa\s*=\s*([-\d.]+)'),
    }

_cospar = _parse_cosmology_par(f"{PARAM_DIR}/cosmology.par")
_conpar = _parse_concept_params(f"{PARAM_DIR}/concept.params")
print('_conpar:',_conpar)
print('_cospar:',_cospar)

# ── 1. Parameters ─────────────────────────────────────────────────────────────
A_s     = float(_cospar['dNormalization'])
n_s     = float(_cospar['dSpectral'])

H0      = _conpar['H0']
h       = H0 / 100.0
Omega_b = _conpar['Omega_b']
Omega_m = _conpar['Omega_cdm_base'] + Omega_b   # Ωcdm_base + Ωb
_mnu    = _conpar['_mnu']
_w0     = _conpar['_w0']
_wa     = _conpar['_wa']

Omega_nu  = sum(_mnu) / (93.14 * h**2)
Omega_cdm = Omega_m - Omega_b - Omega_nu
T_ncdm    = (4/11)**(1/3) * (3.046/len(_mnu))**(1/4)

k_pivot = 0.05   # Mpc^-1

# ── 2. Load HDF5 ──────────────────────────────────────────────────────────────
HDF5_FILE = f"{PARAM_DIR}/class_processed.hdf5"

with h5py.File(HDF5_FILE, "r") as f:
    k        = f["perturbations/k"][:]            # Mpc^-1, shape (n_k,)
    a_pert   = f["perturbations/a"][:]            # shape (n_a,)
    d_cb     = f["perturbations/delta_cdm+b"][:]  # shape (n_a, n_k)
    # d_cb     = f["perturbations/delta_cdm"][:] + f["perturbations/delta_b"][:]  # shape (n_a, n_k)
    d_ncdm   = f["perturbations/delta_ncdm[0]"][:] 

    a_bg     = f["background/a"][:]
    rho_cb   = f["background/rho_cdm+b"][:]
    # rho_cb   = f["background/rho_cdm"][:] + f["background/rho_b"][:]
    rho_ncdm = f["background/rho_ncdm[0]"][:]

# ── 3. Reconstruct P(k) from HDF5 ─────────────────────────────────────────────
rho_cb_interp   = interp1d(a_bg, rho_cb,   kind="linear")(a_pert)
rho_ncdm_interp = interp1d(a_bg, rho_ncdm, kind="linear")(a_pert)

i_a0 = np.argmin(np.abs(a_pert - 1.0))
print(f"Using a = {a_pert[i_a0]:.6f} (index {i_a0}) for z=0")

rho_cb_0   = rho_cb_interp[i_a0]
rho_ncdm_0 = rho_ncdm_interp[i_a0]
rho_m      = rho_cb_0 + rho_ncdm_0

delta_m  = (rho_cb_0 * d_cb[i_a0, :] + rho_ncdm_0 * d_ncdm[i_a0, :]) / rho_m
delta_cb = d_cb[i_a0, :]

# Full P(k) with primordial tilt and absolute normalization
P_prim     = A_s * (k / k_pivot)**(n_s - 1)
Pk_hdf5    = (2 * np.pi**2 / k**3) * P_prim * delta_m**2  * h**3  # (Mpc/h)^3
Pk_hdf5_cb = (2 * np.pi**2 / k**3) * P_prim * delta_cb**2 * h**3  # (Mpc/h)^3

k_hMpc  = k / h

cosmo_dict = {
    'H0'              : H0,
    'Omega_b'         : Omega_b,
    'Omega_cdm'       : Omega_cdm,
    'Omega_Lambda'    : 0.,
    # 'Omega_k'    : 0.0,
    'w0_fld'          : _w0,
    'wa_fld'          : _wa,
    'N_ur'            : 0,
    'N_ncdm'          : 1,
    'deg_ncdm'        : 3,
    'm_ncdm'          : 0.02,
    'T_ncdm'          : T_ncdm,
    
    # for flamingo
    # 'N_ur'            : 2.0328,
    # 'N_ncdm'          : 1,
    # 'deg_ncdm'        : 1,
    # 'm_ncdm'          : 0.06,
    
    'A_s'             : A_s,
    'n_s'             : n_s,
    'k_pivot'         : k_pivot,
    'output'          : 'mPk',
    'P_k_max_h/Mpc'   : k_hMpc.max() * 1.1,
    'z_pk'            : 0.0,
}
print("Cosmological parameters for CLASS:")
for key, value in cosmo_dict.items():
    print(f"  {key}: {value}")
# ── 4. CLASS theory prediction ─────────────────────────────────────────────────
cosmo = Class()
cosmo.set(cosmo_dict)
cosmo.compute()

Pk_class = np.array([cosmo.pk_lin(ki, 0.0) * h**3 for ki in k])  # (Mpc/h)^3

print(f"sigma8 from classy: {cosmo.sigma8()}")
print(f"sigma8_cb from classy: {cosmo.sigma8_cb()}")
print(f"Omega_lambda: {cosmo.Omega_Lambda()}")
print(f"Omega_b: {cosmo.Omega_b()}")
print(f"Omega_g: {cosmo.Omega_g()}")
print(f"Omega_r: {cosmo.Omega_r()}")
print(f"Omega_m: {cosmo.Omega_m()}")
print(f"Omega_m injected: {Omega_m}")

print(f"Neff: {cosmo.Neff()}")
print(f"Omega_cdm: {cosmo.Om_cdm(0)}")


cosmo.struct_cleanup()
cosmo.empty()

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Main panel
ax1.loglog(k_hMpc, Pk_class,    color="steelblue", lw=2,   label="CLASS (theory, total)")
ax1.loglog(k_hMpc, Pk_hdf5,     color="tomato",    lw=1.5, ls="--", label="HDF5 total")
ax1.loglog(k_hMpc, Pk_hdf5_cb,  color="seagreen",  lw=1.5, ls=":",  label="HDF5 cdm+b")
ax1.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$", fontsize=13)
ax1.legend(fontsize=11)
ax1.set_title(r"Linear matter $P(k)$ at $z=0$ — $\sum m_\nu=0.06\,$eV, $w_0=-1$", fontsize=12)
ax1.tick_params(labelbottom=False)
ax1.grid(True, which="both", alpha=0.2)


# # add P(k) from class_output_fiducial_00_pk.dat if PARAM_DIR is "param_files_fiducial"
# if PARAM_DIR == "param_files_fiducial":
#     data = np.loadtxt(f"{PARAM_DIR}/class_output_fiducial_04_pk.dat")
#     k_class_file = data[:, 0]  # Mpc/h
#     Pk_class_file = data[:, 1]  # (Mpc/h)^3
#     ax1.loglog(k_class_file, Pk_class_file, color="seagreen", lw=1.5, ls=":", label="CLASS (file)")
#     ax1.legend(fontsize=11)

#     # and ratio of HDF5 to CLASS file
#     ratio_file = Pk_hdf5 / np.interp(k_hMpc, k_class_file, Pk_class_file)
#     ax2.semilogx(k_hMpc, ratio_file, color="seagreen", lw=1.5, ls=":")


# Ratio panel
ratio    = Pk_hdf5    / Pk_class
ratio_cb = Pk_hdf5_cb / Pk_class
ax2.semilogx(k_hMpc, ratio,    color="mediumpurple", lw=1.5, label="total / CLASS")
ax2.semilogx(k_hMpc, ratio_cb, color="seagreen",     lw=1.5, ls=":", label="cdm+b / CLASS")
ax2.axhline(1.0,  color="gray", lw=1,   ls="--")
ax2.axhline(1.01, color="gray", lw=0.7, ls=":")
ax2.axhline(0.99, color="gray", lw=0.7, ls=":")
ax2.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$", fontsize=13)
ax2.set_ylabel(r"HDF5 / CLASS",           fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(0.95, 1.05)
ax2.grid(True, which="both", alpha=0.2)

plt.savefig(f"{PARAM_DIR}/Pk_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: Pk_comparison.png")


np.savetxt(f"{PARAM_DIR}/Pk_linear_concept.txt",np.c_[k_hMpc,Pk_class], header="k [h/Mpc]    P(k) [(Mpc/h)^3]")



k, tf = get_class_transfer_function(k_hMpc, Pk_hdf5, A_s, n_s, k_pivot, h)
save_class_transfer_function(k, tf, os.path.join(PARAM_DIR, 'transfer_fiducial.dat'), label='tot')

k, tf_cb = get_class_transfer_function(k_hMpc, Pk_hdf5_cb, A_s, n_s, k_pivot, h)
save_class_transfer_function(k, tf_cb, os.path.join(PARAM_DIR, 'transfer_fiducial_cb.dat'), label='cb')

