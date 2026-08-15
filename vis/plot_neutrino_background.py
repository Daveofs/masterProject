#!/usr/bin/env python3
"""Reproduce the two neutrino background figures of the thesis introduction with CLASS.

Replaces the scanned/schematic versions of report Figs. 1.1 and 1.2 (which were
reproductions of Lesgourgues & Pastor, "Massive neutrinos and cosmology") by
figures computed directly with the Einstein-Boltzmann solver, so every curve is
a real solution of the linear perturbation equations rather than a redraw.

Two figures, selectable with --figure. Each reference figure had two panels; the
thesis uses only one of them, so that one is the default and --both-panels adds
the other back (writing to its own filename, so the two never overwrite):

  density   -> neutrino_densities.png   (report Fig. 1.1)
            Omega_i = rho_i / rho_tot vs a, for photons, cdm, baryons, Lambda and
            the three neutrino mass states, with matter-radiation equality marked.
            Each massive state tracks the photons while relativistic and turns
            over to a matter-like dilution at its non-relativistic transition.
            One CLASS run with three separate ncdm species (the mass states).
            --both-panels also draws rho_i^(1/4) [eV], the reference figure's
            left panel, into neutrino_densities_rho_omega.png.

  neutrinos -> neutrino_effect_pk.png   (report Fig. 1.2)
            The linear matter power spectrum P(k) [(Mpc/h)^3] at z=0 -- the
            3D matter spectrum, NOT a CMB or angular spectrum. Three CLASS runs
            sharing a primordial spectrum, omega_b and the TOTAL omega_m: no
            neutrinos at all, massless neutrinos (f_nu=0), and f_nu =
            omega_nu/omega_m = 0.1. Because only the neutrino sector differs, the
            offsets between the curves isolate two effects: a shifted turnover
            (matter-radiation equality moves) and small-scale suppression
            (massive neutrinos free-stream instead of clustering).
            --both-panels also draws the CMB temperature spectrum
            l(l+1)C_l/2pi [uK^2] into neutrino_effect_cmb_pk.png.

Everything meant to be tuned lives in the CONFIG block below: cosmological
parameters, the neutrino mass states, f_nu, plotting ranges, and the style.
The scale factor is plotted as `a`, using the usual convention a_0 = 1.

Usage
-----
    # both figures, default output dir
    /users/damrein/miniforge3/bin/python plot_neutrino_background.py

    # one figure, custom destination / masses
    /users/damrein/miniforge3/bin/python plot_neutrino_background.py \
        --figure density --out-dir <dir> --m-ncdm 0,0.009,0.05

    # heavier neutrino fraction in the right-hand figure
    /users/damrein/miniforge3/bin/python plot_neutrino_background.py \
        --figure neutrinos --f-nu 0.05

CLASS lives in the miniforge BASE environment (`classy` imports there directly);
no conda activate is needed if the interpreter above is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===========================================================================
# CONFIG -- everything intended for tuning
# ===========================================================================

# --- fiducial cosmology (CosmoGridV1-like; see grid params.yml / cosmology.par) ---
H0 = 67.5
OMEGA_B = 0.022_2          # omega_b = Omega_b h^2
OMEGA_CDM = 0.120          # omega_cdm = Omega_cdm h^2
A_S = 2.1e-9
N_S = 0.9649
TAU_REIO = 0.0543
T_CMB = 2.7255             # K
# Primordial helium fraction, held FIXED across the three comparison models of the
# "neutrinos" figure. CLASS would otherwise predict it from BBN as a function of
# N_eff, which (a) fails outright for the no-neutrino model, whose Delta N_eff =
# -3.046 lies outside the BBN interpolation table, and (b) would make the three
# curves differ through the helium fraction as well as through the neutrinos.
Y_HE = 0.2454

# --- figure "density": the three neutrino mass states [eV] ---
# 0 stands for a massless state; it is passed to CLASS as MASSLESS_EV.
M_NCDM = (0.0, 0.009, 0.05)
MASSLESS_EV = 1e-9         # CLASS stand-in for an exactly massless ncdm state
A_MIN_DENSITY = 3e-10      # left edge of the scale-factor axis

# --- figure "neutrinos": the neutrino fraction of the middle/right model ---
F_NU = 0.1                 # f_nu = omega_nu / omega_m
L_MAX = 1500
K_MIN_HMPC = 3e-4          # h/Mpc
K_MAX_HMPC = 0.7           # h/Mpc
N_K = 400

# --- physical constants ---
RHO_CRIT0_EV4_PER_H2 = 8.0996e-11   # rho_crit,0 / h^2 in eV^4
K_B_EV_PER_K = 8.617333262e-5       # eV/K
T_NU_OVER_T_CMB = (4.0 / 11.0) ** (1.0 / 3.0)

# --- style: hues from the validated categorical palette (references/palette.md).
# Each species additionally carries its own dash pattern, and the curves are
# labelled directly on the axes, so identity never rests on colour alone -- the
# required mitigation here, since three of these series are warm hues that
# converge under red-green colour-vision deficiency.
C_CDM = "#2a78d6"      # blue
C_B = "#e87ba4"        # magenta   (kept well away from cdm's blue: the two run
                       #            as close parallel lines in the left panel)
C_G = "#008300"        # green
C_LAMBDA = "#eb6834"   # orange
C_NU = "#e34948"       # red -- the subject of both figures
C_REF = "#52514e"      # neutral ink for the "no neutrinos" reference case

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def _style_axes(ax, keep_top=False):
    """Recessive grid and spines, so the data carries the ink."""
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(keep_top)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)


def find_a_eq(bg: dict) -> float:
    """Scale factor of matter-radiation equality, from CLASS's own budget columns.

    Omega_m(z) and Omega_r(z) are used rather than a hand-rolled sum of the rho
    columns because CLASS already splits each massive-neutrino species between the
    two budgets according to its instantaneous p/rho -- which is exactly the
    subtlety this figure is about, and which a fixed 'ncdm counts as matter'
    assignment would get wrong around each state's non-relativistic transition.
    Solved by linear interpolation of ln(Omega_m/Omega_r) in ln a.
    """
    a = 1.0 / (1.0 + bg["z"])
    order = np.argsort(a)
    a = a[order]
    ratio = np.log(bg["Omega_m(z)"][order] / bg["Omega_r(z)"][order])
    # the last sign change before today is the equality crossing
    sign_change = np.where(np.diff(np.sign(ratio)) != 0)[0]
    if len(sign_change) == 0:
        raise RuntimeError("no matter-radiation equality found in the background")
    i = sign_change[0]
    ln_a = np.interp(0.0, ratio[i:i + 2], np.log(a[i:i + 2]))
    return float(np.exp(ln_a))


def find_a_eq_no_nu(bg: dict) -> float:
    """Scale factor matter-radiation equality would occur at if neutrinos
    contributed to neither budget -- (cdm+b) against (g) alone, from the same
    per-species rho columns the density figure already plots. Not a CLASS run
    with neutrinos switched off (that would also change the expansion history
    through N_eff); a synthetic ratio of the real background's own species,
    isolating what the three neutrino states' presence shifts a_eq by.
    """
    a = 1.0 / (1.0 + bg["z"])
    order = np.argsort(a)
    a = a[order]
    rho_m = bg["(.)rho_cdm"][order] + bg["(.)rho_b"][order]
    rho_r = bg["(.)rho_g"][order]
    ratio = np.log(rho_m / rho_r)
    sign_change = np.where(np.diff(np.sign(ratio)) != 0)[0]
    if len(sign_change) == 0:
        raise RuntimeError("no photon/cdm+b crossing found in the background")
    i = sign_change[0]
    ln_a = np.interp(0.0, ratio[i:i + 2], np.log(a[i:i + 2]))
    return float(np.exp(ln_a))


def _mark_a_eq(ax, a_eq, label_y=None, color=MUTED, label=r"$a_{\rm eq}$",
               ls=(0, (4, 3))):
    """Vertical marker at matter-radiation equality. color/label/ls let a second
    call draw the no-neutrino comparison line distinctly from the real a_eq."""
    ax.axvline(a_eq, color=color, lw=0.9, ls=ls, zorder=1.5)
    ymin, ymax = ax.get_ylim()
    y = label_y if label_y is not None else ymax / 2.2
    ax.annotate(label, xy=(a_eq, y), xytext=(3, 0),
                textcoords="offset points", color=color, fontsize=9,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def _add_temperature_axis(ax, t_nu_today, label):
    """Top axis showing the neutrino temperature T_nu = T_nu,0 / a.

    Built as a twin axis with explicitly inverted limits rather than
    `secondary_xaxis(functions=...)`: the mapping a -> T is *decreasing*, and the
    secondary-axis helper lays its ticks out in ascending order regardless, which
    silently renders the temperature scale backwards (hot end at late times).
    """
    a_lo, a_hi = ax.get_xlim()
    axt = ax.twiny()
    axt.set_xscale("log")
    # decreasing limits: high temperature at small a (left), low at a = 1 (right)
    axt.set_xlim(t_nu_today / a_lo, t_nu_today / a_hi)
    axt.set_xlabel(label)
    axt.tick_params(labelsize=9)
    axt.grid(False)
    for side in ("left", "right", "bottom"):
        axt.spines[side].set_visible(False)
    axt.spines["top"].set_linewidth(0.8)
    return axt


# ===========================================================================
# CLASS helpers
# ===========================================================================

def _base_params() -> dict:
    return {
        "H0": H0,
        "omega_b": OMEGA_B,
        "A_s": A_S,
        "n_s": N_S,
        "tau_reio": TAU_REIO,
        "T_cmb": T_CMB,
    }


def run_background(m_ncdm=M_NCDM) -> tuple[dict, float]:
    """One CLASS run carrying each neutrino mass state as its own ncdm species.

    Returns (background dict, h). Separate species (rather than one degenerate
    one) is what lets the three states be drawn as three curves.
    """
    from classy import Class

    masses = [MASSLESS_EV if m <= 0 else float(m) for m in m_ncdm]
    params = _base_params()
    params.update({
        "output": "mPk",
        "omega_cdm": OMEGA_CDM,
        "N_ur": 0.0,
        "N_ncdm": len(masses),
        "m_ncdm": ",".join(f"{m:.10g}" for m in masses),
        "deg_ncdm": ",".join("1" for _ in masses),
        "T_ncdm": ",".join(f"{T_NU_OVER_T_CMB:.10f}" for _ in masses),
        "P_k_max_h/Mpc": 1.0,
        "z_max_pk": 0.0,
        # the background must be integrated far enough back to show the
        # relativistic era of every mass state
        "a_ini_over_a_today_default": min(A_MIN_DENSITY, 1e-10),
    })

    cosmo = Class()
    try:
        cosmo.set(params)
        cosmo.compute(level=["background"])
        bg = {k: np.asarray(v, dtype=float) for k, v in cosmo.get_background().items()}
        h = cosmo.h()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()
    return bg, h


def run_spectra(case: str, f_nu: float = F_NU, want_cmb: bool = False,
                pk_species: str = "m") -> dict:
    """CMB and matter spectra for one of the three comparison models.

    All three share omega_b, the TOTAL matter density omega_m, and the primordial
    spectrum; they differ only in how the neutrino sector is treated:

      'none'     -- no neutrinos at all (no ncdm, no ultra-relativistic species)
      'massless' -- the standard three massless neutrinos, f_nu = 0
      'massive'  -- three degenerate massive states carrying omega_nu = f_nu * omega_m

    Holding omega_m fixed while moving density between cdm and neutrinos is what
    makes the comparison a statement about neutrinos rather than about Omega_m.

    pk_species selects WHICH linear power spectrum is returned -- a real physical
    choice, not a convention:
      'm'  -- total matter, delta_m = cdm + baryons + massive neutrinos. This is
              the quantity the reference figure shows and the one Eq. (1.2) of the
              report defines.
      'cb' -- cdm + baryons only. Massive neutrinos free-stream and do not cluster
              on small scales, so P_cb is suppressed LESS than P_m there; this is
              also the spectrum the thesis' own initial conditions are built from
              (the tabulated delta_cb of Sec. 2.2.1), which is why it is offered.
    """
    from classy import Class

    omega_m = OMEGA_B + OMEGA_CDM
    params = _base_params()
    params.update({
        "P_k_max_h/Mpc": max(2.0, 2.0 * K_MAX_HMPC),
        "z_max_pk": 0.0,
        "YHe": Y_HE,
    })
    # the lensed CMB spectra are by far the expensive part of the run, so they are
    # only requested when the CMB panel is actually drawn
    if want_cmb:
        params.update({"output": "tCl,pCl,lCl,mPk", "lensing": "yes",
                       "l_max_scalars": L_MAX + 500})
    else:
        params.update({"output": "mPk"})

    if case == "none":
        params.update({"N_ur": 0.0, "N_ncdm": 0, "omega_cdm": omega_m - OMEGA_B})
    elif case == "massless":
        params.update({"N_ur": 3.046, "N_ncdm": 0, "omega_cdm": omega_m - OMEGA_B})
    elif case == "massive":
        omega_nu = f_nu * omega_m
        # 93.14 eV is the standard omega_nu -> sum(m_nu) conversion
        m_each = omega_nu * 93.14 / 3.0
        params.update({
            "N_ur": 0.00641,          # the small residual of the 3.046 budget
            "N_ncdm": 1,
            "deg_ncdm": 3,            # three degenerate states
            "m_ncdm": f"{m_each:.10g}",
            "T_ncdm": f"{T_NU_OVER_T_CMB:.10f}",
            "omega_cdm": omega_m - OMEGA_B - omega_nu,
        })
        print(f"    f_nu={f_nu}: omega_nu={omega_nu:.5f}, "
              f"m_nu={m_each:.4f} eV each (sum {3*m_each:.4f} eV)")
    else:
        raise ValueError(f"unknown case {case!r}")

    cosmo = Class()
    try:
        cosmo.set(params)
        cosmo.compute()
        h = cosmo.h()
        ell = dl_tt = None
        if want_cmb:
            cl = cosmo.lensed_cl(L_MAX)
            ell = cl["ell"]
            # CLASS returns C_l dimensionless (in units of T_cmb^2) -> uK^2
            norm = (T_CMB * 1e6) ** 2
            dl_tt = ell * (ell + 1) * cl["tt"] * norm / (2 * np.pi)

        k_h = np.logspace(np.log10(K_MIN_HMPC), np.log10(K_MAX_HMPC), N_K)
        # the *_lin getters are used explicitly rather than pk()/pk_cb(), which
        # would silently return the non-linear spectrum if non_linear were ever
        # switched on in the CONFIG block above
        pk_of = cosmo.pk_lin if pk_species == "m" else cosmo.pk_cb_lin
        # these take k in 1/Mpc and return (Mpc)^3
        pk = np.array([pk_of(k * h, 0.0) for k in k_h]) * h ** 3
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()

    return {"ell": ell, "dl_tt": dl_tt, "k_h": k_h, "pk": pk, "h": h}


# ===========================================================================
# Figure 1: densities and density parameters
# ===========================================================================

def figure_density(out_path: Path, m_ncdm=M_NCDM, both_panels: bool = False):
    """Omega_i(a) for every species. With both_panels=True the rho_i^(1/4) panel
    of the original two-panel reference figure is drawn alongside it.

    The marked a_eq is the SIMPLIFIED photon-vs-(cdm+b) crossing (find_a_eq_no_nu),
    not the true equality epoch of the realized background (find_a_eq, still
    computed and logged for comparison) -- the two differ by ~40% in a here, since
    neutrinos are a real part of the radiation budget at this epoch. Labelled
    plainly as a_eq on the figure by request."""
    print("[density] running CLASS background ...")
    bg, h = run_background(m_ncdm)
    a_eq = find_a_eq(bg)
    print(f"[density] matter-radiation equality at a_eq = {a_eq:.4g} "
          f"(z_eq = {1/a_eq - 1:.0f})")
    a_eq_no_nu = find_a_eq_no_nu(bg)
    shift_pct = 100.0 * (a_eq_no_nu - a_eq) / a_eq
    print(f"[density] photon/cdm+b crossing WITHOUT neutrinos (what the figure "
          f"marks as a_eq) at a_eq_no_nu = {a_eq_no_nu:.4g} (z = {1/a_eq_no_nu - 1:.0f}), "
          f"{shift_pct:+.1f}% in a relative to the TRUE a_eq above")

    z = bg["z"]
    a = 1.0 / (1.0 + z)
    order = np.argsort(a)
    a = a[order]

    rho_crit0_ev4 = RHO_CRIT0_EV4_PER_H2 * h ** 2
    rho_tot = bg["(.)rho_tot"][order]
    # CLASS densities are in its own units; normalise by today's total (= critical
    # density for a flat model) and rescale to eV^4.
    rho_tot_today = rho_tot[-1]

    def to_ev4(key):
        return bg[key][order] / rho_tot_today * rho_crit0_ev4

    def frac(key):
        return bg[key][order] / rho_tot

    species = [
        ("cdm", r"cdm", "(.)rho_cdm", C_CDM, (0, (1, 1.6)), 1.6),
        ("b", r"b", "(.)rho_b", C_B, (0, (1, 2.6)), 1.5),
        ("g", r"$\gamma$", "(.)rho_g", C_G, (0, (5, 2)), 1.6),
        ("lambda", r"$\Lambda$", "(.)rho_lambda", C_LAMBDA, (0, (6, 1.6, 1, 1.6)), 1.6),
    ]
    nu_widths = (0.9, 1.5, 2.3)   # nu_1 thin ... nu_3 thick, as in the original
    nu_keys = [f"(.)rho_ncdm[{i}]" for i in range(len(m_ncdm))]

    if both_panels:
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
        ax_omega = axes[1]
    else:
        fig, ax_omega = plt.subplots(1, 1, figsize=(6.6, 4.9))
        axes = [ax_omega]

    # ---------------- optional: rho^(1/4) in eV ----------------
    if both_panels:
        ax = axes[0]
        for _, lab, key, col, dash, lw in species:
            ax.plot(a, to_ev4(key) ** 0.25, color=col, ls=dash, lw=lw, label=lab)
        for i, key in enumerate(nu_keys):
            ax.plot(a, to_ev4(key) ** 0.25, color=C_NU, ls="-", lw=nu_widths[i],
                    label=rf"$\nu_{{{i+1}}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(A_MIN_DENSITY, 1.0)
        ax.set_ylim(1e-4, 1e7)
        ax.set_xlabel(r"$a$")   # a_0 = 1 by the usual convention
        ax.set_ylabel(r"$\rho_i^{1/4}$   [eV]")
        _style_axes(ax, keep_top=True)
        _mark_a_eq(ax, a_eq_no_nu)   # same simplified crossing as the Omega_i panel below
        t_nu0_ev = T_NU_OVER_T_CMB * T_CMB * K_B_EV_PER_K
        _add_temperature_axis(ax, t_nu0_ev, r"$T_\nu$   [eV]")

    # ---------------- Omega_i ----------------
    ax = ax_omega
    for _, lab, key, col, dash, lw in species:
        ax.plot(a, frac(key), color=col, ls=dash, lw=lw, label=lab)
    for i, key in enumerate(nu_keys):
        ax.plot(a, frac(key), color=C_NU, ls="-", lw=nu_widths[i],
                label=rf"$\nu_{{{i+1}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(A_MIN_DENSITY, 1.0)
    ax.set_ylim(1e-4, 1.6)
    ax.set_xlabel(r"$a$")   # a_0 = 1 by the usual convention
    ax.set_ylabel(r"$\Omega_i$")
    _style_axes(ax, keep_top=True)
    # Marked line is the SIMPLIFIED two-species crossing (photon vs cdm+b), not the
    # true equality epoch of this cosmology's realized background -- those differ by
    # ~40% in a (find_a_eq's CLASS-budget value logged above), since neutrinos are a
    # real part of the radiation budget at this epoch. Labelled plainly as a_eq by
    # request; see find_a_eq's docstring for the exact definition it is NOT using here.
    _mark_a_eq(ax, a_eq_no_nu)

    t_nu0_k = T_NU_OVER_T_CMB * T_CMB
    _add_temperature_axis(ax, t_nu0_k, r"$T_\nu$   [K]")

    # the dash pattern in each swatch repeats the on-axis identity, so the curves
    # stay separable without relying on hue (three of these are warm hues, which
    # converge under red-green colour-vision deficiency)
    handles = [Line2D([], [], color=c, ls=d, lw=w, label=l)
               for _, l, _, c, d, w in species]
    handles += [Line2D([], [], color=C_NU, ls="-", lw=nu_widths[i],
                       label=rf"$\nu_{{{i+1}}}$  ($m={m_ncdm[i]:g}$ eV)")
                for i in range(len(m_ncdm))]
    # a light frame here (rather than the frameless default) only because the
    # lower-left corner is crossed by the rising cdm and b curves
    (axes[0] if both_panels else ax_omega).legend(
        handles=handles, loc="lower left", ncol=1,
        handlelength=2.8, borderaxespad=0.8, labelspacing=0.35,
        frameon=True, facecolor="white", framealpha=0.88, edgecolor=GRID)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[density] wrote {out_path}")


# ===========================================================================
# Figure 2: the effect of neutrinos on the CMB and on P(k)
# ===========================================================================

def figure_neutrinos(out_path: Path, f_nu: float = F_NU, both_panels: bool = False,
                     pk_species: str = "m"):
    """P(k) for the three neutrino treatments. With both_panels=True the CMB
    temperature-spectrum panel of the original reference figure is drawn too.
    pk_species: 'm' (total matter) or 'cb' (cdm+baryons) -- see run_spectra."""
    models = [
        ("none", r"no $\nu$'s", C_REF, (0, (1.5, 1.8)), 1.2),
        ("massless", r"$f_\nu = 0$", C_CDM, "-", 1.7),
        ("massive", rf"$f_\nu = {f_nu:g}$", C_LAMBDA, (0, (5, 2)), 1.7),
    ]

    results = {}
    for case, *_ in models:
        print(f"[neutrinos] running CLASS: {case} ...")
        results[case] = run_spectra(case, f_nu, both_panels, pk_species)

    if both_panels:
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
        ax_pk = axes[1]
    else:
        fig, ax_pk = plt.subplots(1, 1, figsize=(6.6, 4.9))
        axes = [ax_pk]

    # ---------------- optional: CMB temperature spectrum ----------------
    if both_panels:
        ax = axes[0]
        for case, lab, col, dash, lw in models:
            r = results[case]
            m = r["ell"] >= 2
            ax.plot(r["ell"][m], r["dl_tt"][m], color=col, ls=dash, lw=lw, label=lab)
        ax.set_xlim(2, L_MAX)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{TT} / 2\pi$   [$\mu$K$^2$]")
        _style_axes(ax)
        ax.legend(loc="upper right", handlelength=2.6)

    # ---------------- the matter power spectrum ----------------
    ax = ax_pk
    for case, lab, col, dash, lw in models:
        r = results[case]
        ax.plot(r["k_h"], r["pk"], color=col, ls=dash, lw=lw, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(K_MIN_HMPC, K_MAX_HMPC)
    ax.set_xlabel(r"$k$   [$h$/Mpc]")
    ax.set_ylabel(r"$P_{\rm m}(k)$   [(Mpc/$h$)$^3$]" if pk_species == "m"
                  else r"$P_{\rm cb}(k)$   [(Mpc/$h$)$^3$]")
    _style_axes(ax)
    ax.legend(loc="lower left", handlelength=2.6)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[neutrinos] wrote {out_path}")


# ===========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", choices=["density", "neutrinos", "all"], default="all")
    p.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/background")
    p.add_argument("--m-ncdm", default=None,
                   help="comma-separated neutrino masses [eV] for the density "
                        f"figure (default {','.join(str(m) for m in M_NCDM)})")
    p.add_argument("--f-nu", type=float, default=F_NU,
                   help=f"neutrino fraction omega_nu/omega_m (default {F_NU})")
    p.add_argument("--pk-species", choices=["m", "cb"], default="m",
                   help="which linear power spectrum the 'neutrinos' figure shows: "
                        "'m' = total matter incl. massive neutrinos (default, the "
                        "quantity the reference figure and report Eq. 1.2 define), "
                        "'cb' = cdm+baryons only (what the thesis' own ICs are built "
                        "from; less suppressed at large k, since neutrinos do not "
                        "cluster there)")
    p.add_argument("--both-panels", action="store_true",
                   help="draw the second panel of each original two-panel reference "
                        "figure as well: rho_i^(1/4) for 'density', the CMB "
                        "temperature spectrum for 'neutrinos'. Default: the single "
                        "panel each thesis figure uses (Omega_i, and P(k)).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_ncdm = M_NCDM
    if args.m_ncdm:
        m_ncdm = tuple(float(x) for x in args.m_ncdm.split(","))

    if args.figure in ("density", "all"):
        # distinct name per panel choice, so the two variants never overwrite
        # each other in the output directory
        stem = "neutrino_densities_rho_omega" if args.both_panels else "neutrino_densities"
        figure_density(out_dir / f"{stem}.png", m_ncdm, args.both_panels)
    if args.figure in ("neutrinos", "all"):
        stem = "neutrino_effect_cmb_pk" if args.both_panels else "neutrino_effect_pk"
        if args.pk_species != "m":
            stem += f"_{args.pk_species}"
        figure_neutrinos(out_dir / f"{stem}.png", args.f_nu, args.both_panels,
                         args.pk_species)


if __name__ == "__main__":
    main()
