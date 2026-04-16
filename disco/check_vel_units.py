"""Diagnostic: verify pkdgrav tipsy velocity unit convention.

Compares the canonical momentum P derived from:
  (a) Tipsy IC file using the current v_factor formula
  (b) DISCO-DJ's own 1LPT computed from the same white noise / delta field

Run interactively (no JAX multi-process needed):
  python check_vel_units.py
"""

import sys
import numpy as np
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
IC_FILE   = Path("/capstor/scratch/cscs/damrein/outputs/ICs/cosmo_000001/run_0/CosmoML.00000")
LBOX      = 900.0        # Mpc/h
A_INI     = 0.01         # z=99
N_SAMPLE  = 200_000       # particles to load (enough for statistics)

# cosmo_000001 cosmology (params.yml)
H0      = 73.0
OmegaM  = 0.3
OmegaB  = 0.045
OmegaC  = 0.253788765179
sigma8  = 0.9
ns      = 0.97
w0      = -1.1665
wa      = 0.0

# ── Load tipsy velocities ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from read_tipsy_file import read_tipsy

print(f"Loading {N_SAMPLE} particles from {IC_FILE} …")
p, hdr = read_tipsy(IC_FILE, LBOX)
npart   = len(p["vx"])
rng     = np.random.default_rng(42)
idx     = rng.choice(npart, size=min(N_SAMPLE, npart), replace=False)

vx_raw = p["vx"][idx]
vy_raw = p["vy"][idx]
vz_raw = p["vz"][idx]
v_rms_tipsy = np.sqrt(np.mean(vx_raw**2 + vy_raw**2 + vz_raw**2))
print(f"  v_rms (tipsy raw units) = {v_rms_tipsy:.6e}")

# ── Candidates for v_factor → P [Mpc/h] ─────────────────────────────────────
# In standard N-body / tipsy units:  v_unit = Lbox_phys * H0_phys / sqrt(8π/3)
# where Lbox_phys [Mpc] = Lbox [Mpc/h] / h,  H0_phys [km/s/Mpc]
h    = H0 / 100.0
sqfac = np.sqrt(8 * np.pi / 3)

# Physical interpretation of v_tipsy depends on what pkdgrav stores:
# (A) physical peculiar velocity v_pec_phys → P = a * v_pec_phys / H0 = a * v_tipsy * Lbox / sqfac
# (B) pkdgrav canonical p = a² dxcom/dτ = a² (a dxcom/dt) = a³ dxcom/dt → in v_unit
#     v_pec_phys = v_tipsy * v_unit / a²  → P = v_tipsy * Lbox / (a * sqfac)
# (C) current formula (as in code): P = v_tipsy * a² * Lbox / sqfac
# (D) just v_pec / v_unit (v_pec = v_tipsy * v_unit): P = v_tipsy * Lbox / sqfac

candidates = {
    "A  (a * L/sqfac)           ": A_INI * LBOX / sqfac,
    "B  (L/(a * sqfac))         ": LBOX / (A_INI * sqfac),
    "C  (a² * L/sqfac) [current]": A_INI**2 * LBOX / sqfac,
    "D  (L/sqfac)               ": LBOX / sqfac,
}
print("\nP_rms for each v_factor candidate [Mpc/h]:")
for label, vf in candidates.items():
    p_rms = v_rms_tipsy * vf
    print(f"  Factor {label} = {vf:.4f}  ->  P_rms = {p_rms:.6e} Mpc/h")

# ── Numerical Fplus(a_ini) using scipy ──────────────────────────────────────
print("\nComputing Fplus(a_ini) numerically (scipy) …")
from scipy.integrate import quad

def E_sq(a, Om=OmegaM, w0_=w0, wa_=wa):
    """Dimensionless Hubble rate squared E^2(a) for w0-wa dark energy."""
    Ode0 = 1.0 - Om        # flat universe (ignore Omega_k, Omega_b treated as part of Om)
    # de w(a) = w0 + wa*(1-a)
    return Om * a**-3 + Ode0 * a**(-3*(1+w0_+wa_)) * np.exp(-3*wa_*(1-a))

def growth_integral(a):
    """Integral for D+(a) = 5/2*Om*E(a)*I(a), I(a)=int_0^a da'/(a'*E(a'))^3."""
    def integrand(ap):
        return 1.0 / (ap**3 * E_sq(ap)**1.5)   # 1/(a'^3 * E(a')^3)
    return quad(integrand, 1e-6, a, limit=200)[0]

# Growth factor (unnormalized)
I_ai = growth_integral(A_INI)
I_1  = growth_integral(1.0)
E_ai = np.sqrt(E_sq(A_INI))
E_1  = np.sqrt(E_sq(1.0))
Dplus_ai_unnorm = E_ai * I_ai
Dplus_1_unnorm  = E_1  * I_1

# Normalize so D+(1) = 1
norm = Dplus_1_unnorm
Dplus_ai = Dplus_ai_unnorm / norm

# D'+(a) = dD+/da  (finite difference)
da = 1e-5 * A_INI
Dplus_ai_p = A_INI * np.sqrt(E_sq(A_INI)) * I_ai  # unnorm, same as Dplus_ai_unnorm
Dplus_ai_m_unnorm = np.sqrt(E_sq(A_INI - da)) * growth_integral(A_INI - da)
Dplus_ai_p_unnorm = np.sqrt(E_sq(A_INI + da)) * growth_integral(A_INI + da)
Dplusdiff = (Dplus_ai_p_unnorm - Dplus_ai_m_unnorm) / (2*da) / norm   # normalized D'+(a)

Fplus_ai = A_INI**3 * E_ai * Dplusdiff
print(f"  D+(a_ini) = {Dplus_ai:.6e}   D+(1) = 1 (normalized)")
print(f"  E(a_ini)  = {E_ai:.6e}")
print(f"  D'+  (a_ini) = {Dplusdiff:.6e}")
print(f"  Fplus(a_ini) = {Fplus_ai:.6e}")

# ── Estimate sigma_Pi = sigma_s1 ────────────────────────────────────────────
# Pi = dPsi/dD+ = s_1 [Mpc/h], invariant of time in linear theory.
# sigma_s1 ~ sigma_Psi_today (since D+ normalized to 1 at a=1).
# A crude estimate from sigma_8=0.9: sigma_Psi_today ~ few Mpc/h.
# Computation: sigma_Psi^2 = int P_lin(k)/k^2 * (dk/2pi^2)
# For Eisenstein-Hu P(k) ~ k^ns * T^2(k), this is O(5-10) Mpc/h.
# We'll use the fact that v_tipsy should correspond to physical peculiar velocity
# and work backwards: v_pec [km/s] = H0 * Fplus(a) * sigma_s1 / a
# For sigma_s1 ~ 5 Mpc/h, v_pec ~ H0*Fplus(0.01)*5 / 0.01
for sigma_s1_guess in [3.0, 5.0, 8.0, 12.0]:  # Mpc/h
    P_rms_est = Fplus_ai * sigma_s1_guess
    ideal_vf  = P_rms_est / v_rms_tipsy
    v_pec_km_s = H0 * sigma_s1_guess * Fplus_ai / A_INI  # very rough
    print(f"  sigma_s1={sigma_s1_guess:.0f} Mpc/h -> P_rms={P_rms_est:.3e} -> v_pec~{v_pec_km_s:.1f} km/s -> ideal v_factor={ideal_vf:.4e}")

print("\n--- Ratio of v_factor candidates to Fplus ---")
for label, vf in candidates.items():
    print(f"  {label}: vf={vf:.4e}   vf/Fplus={vf/Fplus_ai:.4f}")

# Physical interpretation: if pkdgrav stores physical peculiar velocity in v_unit,
# then v_pec_phys = v_tipsy * v_unit, and P = a * v_pec_phys / (H0 * Lbox_h * h)
# For P [Mpc/h], H0 in km/s/Mpc: P = a * v_tipsy * Lbox_h_Mpc * H0_Mpc / (H0_Mpc * sqfac) = FORMULA_A
h_factor = H0 / 100.0
print(f"\n  h = {h_factor:.4f}, H0 = {H0} km/s/Mpc")
print(f"  v_rms * Lbox_phys_Mpc * H0 / sqfac = {v_rms_tipsy * LBOX/h_factor * H0 / sqfac:.4e} km/s")
print(f"  (this should be ~v_pec_phys_rms if pkdgrav stores in v_unit = Lbox_phys * H0 / sqfac)")
print(f"  Expected v_pec_phys at z=99: ~1-5 km/s (linear, sigma_8=0.9)")
v_pec_from_factorA = v_rms_tipsy * LBOX/h_factor * H0 / sqfac
print(f"  Factor A implies v_pec_phys_rms = {v_pec_from_factorA:.2f} km/s")
v_pec_from_factorC = v_rms_tipsy * LBOX/h_factor * H0 / sqfac / (A_INI)  # formula C tipsy stores a^2 * vpec / v_unit
print(f"  Factor C implies v_pec_phys_rms = {v_rms_tipsy * LBOX * H0 / sqfac / h_factor / A_INI**2:.2f} km/s  (if tipsy=a^2*vpec/vunit)")

