# ---------------------------------------------------------------------------
# shell_builder/utils.py
#
# Utility functions shared across the shell_builder package:
#   - make_chi_of_a         : comoving distance interpolator from DiscoDJ cosmo
#   - build_replica_offsets : PKD-style periodic replica tile table
#   - vec2pix_ring_jax      : pure-JAX HEALPix RING pixel index
# ---------------------------------------------------------------------------

import numpy as np


# ---------------------------------------------------------------------------
# Comoving distance helper
# ---------------------------------------------------------------------------

def make_comoving_distance_fn(cosmo, n_table: int = 2000):
    """
    Return (chi_of_a, a_of_chi) interpolators from a DiscoDJ cosmology object.

    chi_of_a(a)   -> comoving distance [Mpc/h]
    a_of_chi(chi) -> scale factor

    Uses DiscoDJ's conformal time:  chi(a) = c * |eta(a) - eta(1)|
    where c = 299792.458 * h  [km/s * h].
    """
    from scipy.interpolate import interp1d

    h = float(cosmo.h)
    c_kmsh = 2.99792458e5 * h  # km/s · h  →  Mpc/h when × conformal time

    a_table   = np.linspace(1e-4, 1.0, n_table)
    eta_table = np.array([float(cosmo.a_to_conformalt(float(a))) for a in a_table])
    chi_table = c_kmsh * np.abs(eta_table)  # Mpc/h  (eta ≤ 0 for a < 1)

    chi_of_a = interp1d(a_table, chi_table, kind='linear',
                        bounds_error=False, fill_value=(chi_table[0], 0.0))
    a_of_chi = interp1d(chi_table[::-1], a_table[::-1], kind='linear',
                        bounds_error=False, fill_value=(1e-4, 1.0))
    return chi_of_a, a_of_chi


# ---------------------------------------------------------------------------
# Replica offset table  (mirrors pkdgrav3's initLightConeOffsets)
# ---------------------------------------------------------------------------

def compute_replica_offsets(chi_max_Mpch: float, boxsize: float) -> np.ndarray:
    """
    Return (N_rep, 3) integer tile indices whose nearest corner to the observer
    (placed at the box centre) is within ``chi_max_Mpch``.

    Multiply the returned array by ``boxsize`` to get offsets in Mpc/h.
    """
    n_max = int(np.ceil(chi_max_Mpch / boxsize)) + 1
    offsets = []
    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            for nz in range(-n_max, n_max + 1):
                cx, cy, cz = nx * boxsize, ny * boxsize, nz * boxsize
                # Nearest point of this tile to the observer (at origin after centring)
                nearest_sq = (max(0.0, abs(cx) - boxsize / 2) ** 2 +
                              max(0.0, abs(cy) - boxsize / 2) ** 2 +
                              max(0.0, abs(cz) - boxsize / 2) ** 2)
                if nearest_sq <= chi_max_Mpch ** 2:
                    offsets.append((nx, ny, nz))
    return np.array(offsets, dtype=np.float32)  # shape (N_rep, 3)


# ---------------------------------------------------------------------------
# JAX-native GPU HEALPix helper
# ---------------------------------------------------------------------------

def vec2pix_ring_jax(nside: int, x, y, z):
    """
    Pure-JAX HEALPix RING-scheme pixel index for a batch of unit vectors.

    ``x``, ``y``, ``z`` are JAX arrays of the same shape; ``nside`` must be a
    power of 2.  Implements the healpix_bare C algorithm (loc2hpd → hpd2ring).

    Reference: https://github.com/ntessore/healpix (BSD-3-Clause).
    """
    import jax.numpy as jnp

    # Normalise to unit sphere
    norm      = jnp.sqrt(x * x + y * y + z * z)
    safe_norm = jnp.where(norm > 0.0, norm, jnp.ones_like(norm))
    x = x / safe_norm;  y = y / safe_norm;  z = z / safe_norm

    za  = jnp.abs(z)
    s   = jnp.sqrt(jnp.maximum(1.0 - z * z, 0.0))
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0.0, phi + 2.0 * jnp.pi, phi)
    tt  = phi * (2.0 / jnp.pi)

    _jrll = jnp.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=jnp.int32)
    _jpll = jnp.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=jnp.int32)

    # Equatorial region  |z| ≤ 2/3
    temp1e = 0.5 + tt
    temp2e = z * 0.75
    jp_fe  = temp1e - temp2e
    jm_fe  = temp1e + temp2e
    ifp    = jnp.floor(jp_fe).astype(jnp.int32)
    ifm    = jnp.floor(jm_fe).astype(jnp.int32)
    xe     = (jm_fe - ifm.astype(jnp.float32)) * nside
    ye     = (1.0 + ifp.astype(jnp.float32) - jp_fe) * nside
    fe     = jnp.where(ifp == ifm,  ifp | 4,
             jnp.where(ifp <  ifm,  ifp,
                                    ifm + 8))

    # Polar regions  |z| > 2/3
    ntt_p  = jnp.minimum(jnp.floor(tt).astype(jnp.int32),
                         jnp.full_like(jnp.floor(tt).astype(jnp.int32), 3))
    tp_p   = tt - ntt_p.astype(jnp.float32)
    tmp_p  = s / jnp.sqrt(jnp.maximum((1.0 + za) / 3.0, 1e-30))
    jp_p   = jnp.minimum(tp_p * tmp_p, 1.0)
    jm_p   = jnp.minimum((1.0 - tp_p) * tmp_p, 1.0)
    jp_p2  = jnp.where(z >= 0.0, 1.0 - jm_p, jp_p)
    jm_p2  = jnp.where(z >= 0.0, 1.0 - jp_p, jm_p)
    xp     = jp_p2 * nside
    yp     = jm_p2 * nside
    fp     = jnp.where(z >= 0.0, ntt_p, ntt_p + 8)

    # Select region
    is_eq  = za <= 2.0 / 3.0
    hx_f   = jnp.where(is_eq, xe, xp)
    hy_f   = jnp.where(is_eq, ye, yp)
    hf     = jnp.where(is_eq, fe, fp).astype(jnp.int32)
    hpd_x  = jnp.minimum(jnp.floor(hx_f).astype(jnp.int32), nside - 1)
    hpd_y  = jnp.minimum(jnp.floor(hy_f).astype(jnp.int32), nside - 1)

    # hpd → ring pixel
    jrll_v = jnp.take(_jrll, hf)
    jpll_v = jnp.take(_jpll, hf)
    jr     = jrll_v * nside - hpd_x - hpd_y - 1
    nl4    = 4 * nside
    npix_  = 12 * nside * nside

    def _wrap(jp):
        return jnp.where(jp > nl4, jp - nl4,
               jnp.where(jp < 1,   jp + nl4, jp))

    jpn  = _wrap((jpll_v * jr + hpd_x - hpd_y + 1) // 2)
    ipn  = 2 * jr * (jr - 1) + jpn - 1

    ksh  = (jr + nside) & 1
    jpe  = _wrap((jpll_v * nside + hpd_x - hpd_y + 1 + ksh) // 2)
    ipe  = 2 * nside * (nside - 1) + (jr - nside) * nl4 + jpe - 1

    jr_s = nl4 - jr
    jps  = _wrap((jpll_v * jr_s + hpd_x - hpd_y + 1) // 2)
    ips  = npix_ - 2 * (jr_s + 1) * jr_s + jps - 1

    return jnp.where(jr < nside, ipn,
           jnp.where(jr > 3 * nside, ips, ipe))
