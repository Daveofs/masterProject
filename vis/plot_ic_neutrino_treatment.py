#!/usr/bin/env python3
"""Why the backscaled ICs must use the CDM+baryon transfer function, and why the
neutrino sector must be declared to CLASS the way CONCEPT declares it.

Two independent mistakes are possible when rebuilding CosmoGridV1's linear input
outside CONCEPT, and this script separates them:

  1. WHICH FIELD seeds the particles. PkdGrav3's particle load represents CDM+baryons
     only; neutrinos never become particles. CosmoGridV1 can get away with the total
     matter field because bClass=1 evolves the neutrinos as their own linear species
     on a grid. Our backscaled setup (bClass=0 + achTfFile) has no such species, so
     seeding with delta_m instead of delta_cb puts the neutrino free-streaming
     suppression into the particle field -- where it does not belong.

  2. HOW THE NEUTRINOS ARE DECLARED to CLASS, which fixes Omega_r and hence the whole
     background expansion. Treating the three 0.02 eV states as massless (N_ur=3.046,
     N_ncdm=0) keeps their energy density scaling as a^-4 forever instead of turning
     into matter after they go non-relativistic. That changes Omega_r, the epoch of
     equality, and the growth normalisation.

Both are quantified against CONCEPT's own delta_cb from class_processed.hdf5, which
is the reference the pipeline actually has to reproduce.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_ic_neutrino_treatment.py \
        --run-dir /capstor/scratch/cscs/damrein/grid/cosmo_000176/run_0 \
        --out-dir /capstor/scratch/cscs/damrein/outputs/plots/ic
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from classy import Class

FS_AXIS, FS_TICK, FS_LEGEND = 16, 14, 13
K_PIVOT = 0.05
C_REF, C_FIX, C_OLD = "#52514e", "#28866A", "#B85F34"


def read_params(run_dir: Path):
    """Cosmology from params.yml (the grid's own record for this run)."""
    txt = (run_dir / "params.yml").read_text()
    def g(key):
        m = re.search(rf"^{key}:\s*([-\d.eE+]+)", txt, re.M)
        if not m:
            raise KeyError(f"{key} missing from {run_dir/'params.yml'}")
        return float(m.group(1))
    p = {k: g(k) for k in ("As", "H0", "O_cdm", "O_nu", "Ob", "Om", "m_nu", "ns", "w0", "wa")}
    p["h"] = p["H0"] / 100.0
    return p


def base_dict(p):
    return {"H0": p["H0"], "Omega_b": p["Ob"], "Omega_Lambda": 0.0,
            "w0_fld": p["w0"], "wa_fld": p["wa"], "A_s": p["As"], "n_s": p["ns"],
            "k_pivot": K_PIVOT, "output": "mPk", "z_pk": 0.0}


def run_class(cfg, k_hMpc, h, want_cb):
    cosmo = Class()
    try:
        cosmo.set({**cfg, "P_k_max_h/Mpc": float(k_hMpc.max() * 1.1)})
        cosmo.compute()
        # pk_cb_lin is the CDM+baryon spectrum -- the field the particles represent.
        # Falls back to the total spectrum for the massless model, where CLASS has no
        # ncdm species and the two are identical by construction.
        pk = []
        for kk in k_hMpc * h:
            try:
                pk.append((cosmo.pk_cb_lin(kk, 0.0) if want_cb else cosmo.pk_lin(kk, 0.0)) * h**3)
            except Exception:
                pk.append(cosmo.pk_lin(kk, 0.0) * h**3)
        info = {"Omega_r": cosmo.Omega_r(), "Omega_m": cosmo.Omega_m(),
                "Neff": cosmo.Neff(), "sigma8": cosmo.sigma8(),
                "z_eq": cosmo.Omega_m() / cosmo.Omega_r() - 1.0}
        return np.array(pk), info
    finally:
        cosmo.struct_cleanup(); cosmo.empty()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000001/run_0")
    ap.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/ic")
    ap.add_argument("--ratio-only", action="store_true",
                    help="drop the P(k) panel (all curves overlap on a six-decade axis)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir); out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = read_params(run_dir)
    h = p["h"]
    print(f"[ic] {run_dir.parent.name}: H0={p['H0']:.4f} Om={p['Om']:.6f} "
          f"Ob={p['Ob']:.6f} O_cdm={p['O_cdm']:.6f} O_nu={p['O_nu']:.8f} m_nu={p['m_nu']} eV/state")

    # --- CONCEPT's own delta_cb at a=1: the reference the pipeline must reproduce ---
    with h5py.File(run_dir / "class_processed.hdf5", "r") as f:
        k = f["perturbations/k"][:]                     # 1/Mpc
        a_pert = f["perturbations/a"][:]
        d_cb = f["perturbations/delta_cdm+b"][:]
    d0 = d_cb[int(np.argmin(np.abs(a_pert - 1.0))), :]
    k_hMpc = k / h
    P_prim = p["As"] * (k / K_PIVOT) ** (p["ns"] - 1)
    pk_ref = (2 * np.pi**2 / k**3) * P_prim * d0**2 * h**3

    # --- the two declarations of the same neutrino sector ---
    T_ncdm = (4 / 11) ** (1 / 3) * (3.046 / 3) ** (1 / 4)
    cfg_fix = {**base_dict(p), "Omega_cdm": p["Om"] - p["Ob"] - p["O_nu"],
               "N_ur": 0, "N_ncdm": 1, "deg_ncdm": 3, "m_ncdm": p["m_nu"], "T_ncdm": T_ncdm}
    # "old": the three states are declared massless, so they stay radiation forever and
    # their mass is silently absorbed into CDM to keep Omega_m fixed.
    cfg_old = {**base_dict(p), "Omega_cdm": p["Om"] - p["Ob"], "N_ur": 3.046, "N_ncdm": 0}

    pk_fix, i_fix = run_class(cfg_fix, k_hMpc, h, want_cb=True)
    pk_old, i_old = run_class(cfg_old, k_hMpc, h, want_cb=False)
    # Same (correct) background, but seeded with the TOTAL matter field instead of
    # CDM+baryons -- the second, independent way to get this wrong.
    pk_tot, _ = run_class(cfg_fix, k_hMpc, h, want_cb=False)

    print("\n[ic] background consequences of the declaration")
    print(f"  {'':22s} {'Omega_r':>12s} {'N_eff':>8s} {'z_eq':>9s} {'sigma8':>8s}")
    for nm, i in (("massive (CONCEPT-like)", i_fix), ("massless (old)", i_old)):
        print(f"  {nm:22s} {i['Omega_r']:12.6e} {i['Neff']:8.3f} {i['z_eq']:9.1f} {i['sigma8']:8.4f}")
    print(f"  Omega_r ratio old/fixed: {i_old['Omega_r']/i_fix['Omega_r']:.4f}")

    r_fix, r_old, r_tot = pk_fix / pk_ref, pk_old / pk_ref, pk_tot / pk_ref
    k_box = 2 * np.pi / 900.0   # fundamental mode of the 900 Mpc/h CosmoGridV1 box
    def band(r, lo, hi):
        m = (k_hMpc >= lo) & (k_hMpc <= hi)
        return np.median(r[m]) if m.any() else np.nan
    print("\n[ic] P(k) relative to CONCEPT delta_cb, by k band [h/Mpc]")
    print(f"  {'band':>20s} {'cb+massive':>11s} {'massless':>10s} {'total-matter':>13s}")
    for lo, hi in [(3e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, 10.0)]:
        print(f"  {lo:8.1e}-{hi:8.1e}: {band(r_fix,lo,hi):11.4f} {band(r_old,lo,hi):10.4f} "
              f"{band(r_tot,lo,hi):13.4f}")
    print(f"  (box fundamental mode k = 2pi/900 = {k_box:.4f} h/Mpc -- everything to the "
          f"LEFT of this is outside the simulation entirely)")

    # --- why the error MOVES to large scales in the actual pipeline -----------
    # cosmology.par carries dSigma8 / dNormalization ("calculated from sigma_8"), so
    # PkdGrav3 fixes the IC amplitude by sigma_8, not by the large-scale amplitude.
    # sigma_8 is an integral dominated by k ~ 0.1-0.2 h/Mpc, i.e. by exactly the small
    # scales where these spectra differ. Matching it therefore RESCALES the whole
    # spectrum, converting a small-scale shape error into a large-scale amplitude
    # error -- which is how a free-streaming effect shows up as a "3% wrong large
    # scale P(k)".
    def sigma8_of(pk):
        R = 8.0
        x = k_hMpc * R
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
        return np.sqrt(np.trapezoid(k_hMpc**2 * pk * W**2 / (2 * np.pi**2), k_hMpc))
    s8_ref, s8_fix = sigma8_of(pk_ref), sigma8_of(pk_fix)
    s8_old, s8_tot = sigma8_of(pk_old), sigma8_of(pk_tot)
    print("\n[ic] sigma_8 of each spectrum, and the large-scale offset AFTER matching it")
    print(f"  {'':28s} {'sigma_8':>9s} {'/ref':>8s} {'renorm P':>9s} {'-> P_large':>11s}")
    for nm, s8, r in (("CONCEPT delta_cb (ref)", s8_ref, None),
                      ("massive nu, delta_cb", s8_fix, r_fix),
                      ("massless nu", s8_old, r_old),
                      ("total matter delta_m", s8_tot, r_tot)):
        if r is None:
            print(f"  {nm:28s} {s8:9.5f} {1.0:8.4f} {'--':>9s} {'--':>11s}")
            continue
        f = (s8_ref / s8) ** 2          # factor applied to P when sigma_8 is forced
        large = np.median(r[(k_hMpc > k_box) & (k_hMpc < 3e-2)]) * f
        print(f"  {nm:28s} {s8:9.5f} {s8/s8_ref:8.4f} {f:9.4f} {large:11.4f}")

    # Saturated small-scale offset, quoted in the legends the same way
    # plot_backscaling_vs_concept.py and plot_omega_rad_effect.py quote theirs. Taken
    # over k in [0.1, 10] h/Mpc, where the free-streaming suppression has plateaued --
    # not over the in-box average, which would mix the plateau with the rising part.
    def sat(r):
        m = (k_hMpc > 0.1) & (k_hMpc < 10.0)
        return 100.0 * (np.median(r[m]) - 1.0)
    p_fix, p_old, p_tot = sat(r_fix), sat(r_old), sat(r_tot)
    print(f"\n[ic] saturated offset over k in [0.1, 10] h/Mpc:")
    print(f"  massive nu + delta_cb (used) : {p_fix:+.2f}%")
    print(f"  massless nu                  : {p_old:+.2f}%")
    print(f"  total matter delta_m         : {p_tot:+.2f}%")

    # ------------------------------------------------------------------ figure
    # Ratio ONLY. The P_cb(k) panel was dropped: on a six-decade log axis all four
    # spectra lie on top of each other, so it carried no information the ratio does
    # not carry better -- the whole point is a few-percent difference.
    if a.ratio_only:
        fig, axr = plt.subplots(figsize=(9, 5.0))
        ax = None
    else:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 8.2), sharex=True,
                                      gridspec_kw={"height_ratios": [2.0, 1.15],
                                                   "hspace": 0.07})
        ax.loglog(k_hMpc, pk_ref, "-", color=C_REF, lw=2.6, label="CONCEPT $\\delta_{cb}$ (reference)")
        ax.loglog(k_hMpc, pk_old, ":", color=C_OLD, lw=2.2, label=f"massless $\\nu$ ({p_old:+.1f}%)")
        ax.loglog(k_hMpc, pk_tot, "-.", color="#7A51C6", lw=1.9, label=f"total matter $\\delta_m$ ({p_tot:+.1f}%)")
        ax.loglog(k_hMpc, pk_fix, "--", color=C_FIX, lw=2.2, label=f"massive $\\nu$, $\\delta_{{cb}}$ ({p_fix:+.1f}%)")
        ax.axvspan(k_hMpc.min(), k_box, color="0.85", alpha=0.45, lw=0, zorder=0)
        ax.axvline(k_box, color="0.45", ls="-", lw=1.1, zorder=1)
        ax.set_ylabel(r"$P(k)\ \ [(\mathrm{Mpc}/h)^3]$", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(fontsize=FS_AXIS, loc="lower left", framealpha=1.0,
                  borderpad=0.4, labelspacing=0.35, handlelength=2.0, handletextpad=0.5)
        ax.grid(alpha=0.25, which="both", lw=0.5); ax.set_axisbelow(True)

    axr.semilogx(k_hMpc, r_old, ":", color=C_OLD, lw=2.2,
                 label=f"massless $\\nu$ ({p_old:+.1f}%)")
    axr.semilogx(k_hMpc, r_tot, "-.", color="#7A51C6", lw=1.9,
                 label=f"total matter $\\delta_m$ ({p_tot:+.1f}%)")
    axr.semilogx(k_hMpc, r_fix, "--", color=C_FIX, lw=2.2,
                 label=f"massive $\\nu$, $\\delta_{{cb}}$ ({p_fix:+.1f}%)")

    axr.axhline(1.0, color="k", ls="--", lw=0.9)
    for y in (0.99, 1.01):
        axr.axhline(y, color="0.6", ls=":", lw=0.8)
    # Shade the region the 900 Mpc/h box cannot represent -- the common ~2.6% rise
    # there is a horizon-scale gauge difference, not a reconstruction error, and
    # without this it reads as a failure of the correct configuration too.
    axr.axvspan(k_hMpc.min(), k_box, color="0.85", alpha=0.45, lw=0, zorder=0)
    axr.axvline(k_box, color="0.45", ls="-", lw=1.1, zorder=1)
    # Horizontal, centred in the shaded band along the bottom of the panel.
    axr.text(np.sqrt(k_hMpc.min() * k_box), 0.035, "outside the box",
             fontsize=FS_LEGEND - 1, color="0.35", va="bottom", ha="center",
             transform=axr.get_xaxis_transform())

    axr.set_xlabel(r"$k\ \ [h/\mathrm{Mpc}]$", fontsize=FS_AXIS)
    axr.set_ylabel(r"$P(k)\,/\,P^{\textsc{concept}}_{cb}(k)$"
                   if plt.rcParams["text.usetex"] else "$P(k)$ / CONCEPT $P_{cb}(k)$",
                   fontsize=FS_AXIS)
    axr.tick_params(labelsize=FS_TICK)
    # Smaller and tighter: at full size the box reached across to the rising
    # massless-nu curve. Kept opaque so the shaded band behind it stays legible.
    if ax is None:
        axr.legend(fontsize=FS_AXIS, loc="upper left", framealpha=1.0,
                   borderpad=0.4, labelspacing=0.35, handlelength=2.0,
                   handletextpad=0.5, borderaxespad=0.5)
    axr.grid(alpha=0.25, which="both", lw=0.5); axr.set_axisbelow(True)
    axr.set_xlim(k_hMpc.min(), k_hMpc.max())
    # y-range from the IN-BOX modes only. The horizon-scale excursion in the shaded
    # region reaches +21% on some cosmologies and, if allowed to set the limits,
    # compresses the few-percent differences that are the entire point of the figure
    # into a couple of pixels. It is clipped instead -- it is shaded and labelled as
    # a region the simulation does not contain.
    inbox = k_hMpc > k_box
    fin = np.concatenate([r[inbox][np.isfinite(r[inbox])] for r in (r_fix, r_old, r_tot)])
    lo_v, hi_v = fin.min(), fin.max()
    pad = max(hi_v - lo_v, 2e-2) * 0.18
    axr.set_ylim(min(lo_v - pad, 0.985), max(hi_v + pad, 1.015))

    out = out_dir / f"ic_neutrino_treatment_{run_dir.parent.name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\n[ic] -> {out}")


if __name__ == "__main__":
    main()
