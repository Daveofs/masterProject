#!/usr/bin/env python3
"""Where the wall-clock of a production DISCO-CUSTOM lightcone actually goes.

One lightcone is ONE simulation sharded across 8 GPUs (2 Alps nodes x 4 GH200,
srun --ntasks=8): the .err log shows sharding={devices=[8]<=[8]} on jit(run_nbody),
so all 8 tasks execute the same step collectively and each prints its own view of it.
The 8 copies of every timing line are therefore 8 reports of ONE step, not 8 steps --
they are averaged, not summed.

Each of the 69 integration steps logs

    running run_nbody_step: <t> s                      <- the PM force solve
    [jax_shell] gather/cast <t>s  r_step=[a,b] ...     <- positions -> device
    [jax_shell] r_step=[a,b] ... gpu=<t>s replicas=n/N <- crossing test + pixelise

so the two costs of Sec.~\ref{sec:shell_builder} separate per step. The step grid IS
the shell grid (Eq. step_grid), so "step" and "shell" are one axis, and r_step gives
the comoving radius directly -- converted to redshift through the run's shell table.

Two things the figure has to say honestly at once:

  (a) PER STEP, the N-body time is flat -- the PM solve does the same FFT every step
      wherever the lightcone front is -- while the shell-builder time falls by a
      factor ~5 from the far shells to the near ones, tracking the number of periodic
      replicas surviving the reject of Eq. replica_bounds. Integrated, the two are
      comparable: building the lightcone is not a free by-product of the integration.

  (b) These internal timers cover only ~44% of the run. The rest is framework
      overhead and untimed host work between the timed regions. Panel (b) therefore
      shows the FULL per-cosmology budget, so the timed kernels are never mistaken
      for the whole cost. Reported production cost is the full budget, not the sum
      of the timers.

Usage
-----
    /users/damrein/miniforge3/bin/python plot_production_timing.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS_AXIS, FS_TICK, FS_LEGEND = 14, 12, 11
C_NBODY, C_SHELL, C_GATHER, C_REPL = "#B85F34", "#3F63A6", "#8E9BC4", "#5C6480"
C_SETUP, C_REST, C_IC, C_PRE = "#C9CEDC", "#E3E6EF", "#28866A", "#E0A87C"

RE_NB = re.compile(r"running run_nbody_step:\s*([\d.]+)\s*s")
RE_GC = re.compile(r"\[jax_shell\]\s+gather/cast\s+([\d.]+)s\s+r_step=\[([\d.]+),([\d.]+)\]")
RE_SH = re.compile(r"\[jax_shell\]\s+r_step=\[([\d.]+),([\d.]+)\].*?gpu=([\d.]+)s\s+"
                   r"replicas=(\d+)/(\d+)\s+placed=([\d,]+)")
RE_START = re.compile(r"\[([\d\-T:+]+)\] Starting DISCO-CUSTOM \(cosmo_(\d+)")
RE_FIN = re.compile(r"\[([\d\-T:+]+)\] Finished \(exit=(\d+)")
RE_IC0 = re.compile(r"\[([\d\-T:+]+)\] Converting tipsy IC")
RE_IC1 = re.compile(r"\[([\d\-T:+]+)\] IC conversion complete")
RE_W = re.compile(r"^W\d{4} (\d{2}:\d{2}:\d{2})\.")
RE_PRE = re.compile(r"running run_nbody_prestep:\s*([\d.]+)\s*s")
RE_REORD = re.compile(r"running initial_reorder:\s*([\d.]+)\s*s")
RE_LOAD = re.compile(r"load data from IC file:\s*([\d.]+)\s*s")
RE_COMP = re.compile(r"compiling [a-z_]+:\s*([\d.]+)\s*(s|ms)")


def parse_steps(path: Path):
    """-> {(r_lo, r_hi): {...}} with one entry per task per step. A step block is
    flushed on its trailing 'res_pm' line, which delimits one task's step in the
    interleaved log."""
    recs, cur = {}, None
    for line in path.read_text(errors="replace").splitlines():
        m = RE_NB.search(line)
        if m:
            cur = {"nb": float(m.group(1)), "gc": 0.0, "gpu": 0.0,
                   "rep": None, "placed": 0, "key": None}
            continue
        if cur is None:
            continue
        m = RE_GC.search(line)
        if m:
            cur["gc"] += float(m.group(1))
            cur["key"] = (float(m.group(2)), float(m.group(3)))
            continue
        m = RE_SH.search(line)
        if m:
            cur["gpu"] += float(m.group(3)); cur["rep"] = int(m.group(4))
            cur["placed"] += int(m.group(6).replace(",", ""))
            cur["key"] = (float(m.group(1)), float(m.group(2)))
            continue
        if line.startswith("res_pm") and cur["key"] is not None:
            d = recs.setdefault(cur["key"],
                                {"nb": [], "gc": [], "gpu": [], "rep": [], "placed": []})
            for k in ("nb", "gc", "gpu", "rep", "placed"):
                d[k].append(cur[k])
            cur = None
    return recs



_ZCACHE: dict = {}


def shell_z(out_path: Path, grid: str, keys):
    """Mid-shell redshifts for the cosmology this log belongs to, ordered like
    `keys` (near -> far). Falls back to NaN if the run's shell table is missing."""
    m = re.search(r"Starting DISCO-CUSTOM \(cosmo_(\d+)", out_path.read_text(errors="replace")[:4000])
    if not m:
        return [np.nan] * len(keys)
    tag = m.group(1)
    if tag not in _ZCACHE:
        p = Path(grid) / f"cosmo_{tag}" / "run_0" / "compressed_shells.npz"
        if not p.exists():
            _ZCACHE[tag] = None
        else:
            si = np.load(p, allow_pickle=True)["shell_info"]
            _ZCACHE[tag] = (si["upper_com"].astype(float),
                            0.5 * (si["lower_z"].astype(float) + si["upper_z"].astype(float)))
    if _ZCACHE[tag] is None:
        return [np.nan] * len(keys)
    hi_c, zmid = _ZCACHE[tag]
    return [zmid[int(np.argmin(np.abs(hi_c - t[1])))] for t in keys]


def parse_budget(out_path: Path, ntask: int = 8):
    """Per-run wall-clock budget. Timed quantities are summed over steps and divided
    by the number of tasks, since every task logs the same collective step."""
    t = out_path.read_text(errors="replace")
    s, f = RE_START.search(t), RE_FIN.search(t)
    if not (s and f):
        return None
    P = dt.datetime.fromisoformat
    t0, t1 = P(s.group(1)), P(f.group(1))
    i0, i1 = RE_IC0.search(t), RE_IC1.search(t)
    setup = np.nan
    err = out_path.with_suffix(".err")
    if err.exists():
        for line in err.read_text(errors="replace").splitlines():
            if "jit(run_nbody)" in line and (m := RE_W.search(line)):
                hh = dt.datetime.strptime(m.group(1), "%H:%M:%S").time()
                setup = (t0.replace(hour=hh.hour, minute=hh.minute,
                                    second=hh.second, microsecond=0) - t0).total_seconds()
                break
    nb = sum(float(x) for x in RE_NB.findall(t)) / ntask
    gc = sum(float(m[0]) for m in RE_GC.findall(t)) / ntask
    sh = sum(float(m[2]) for m in RE_SH.findall(t)) / ntask
    pre = sum(float(x) for x in RE_PRE.findall(t)) / ntask
    comp = sum(float(v) / (1000 if u == "ms" else 1) for v, u in RE_COMP.findall(t)) / ntask
    boot = (sum(float(x) for x in RE_REORD.findall(t))
            + sum(float(x) for x in RE_LOAD.findall(t))) / ntask + comp
    return dict(total=(t1 - t0).total_seconds(), exit=int(f.group(2)),
                ic=(P(i1.group(1)) - P(i0.group(1))).total_seconds() if (i0 and i1) else np.nan,
                setup=setup, nb=nb, gc=gc, sh=sh, pre=pre, boot=boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="/capstor/scratch/cscs/damrein/outputs/logs/disco_custom")
    ap.add_argument("--job", default="2746372")
    ap.add_argument("--detail-task", default=None,
                   help="restrict panel (a) to one array task; default: average all")
    ap.add_argument("--n-shells", type=int, default=69)
    ap.add_argument("--grid", default="/capstor/scratch/cscs/damrein/grid")
    ap.add_argument("--ntask", type=int, default=8, help="GPUs the run is sharded over")
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--out-dir", default="/capstor/scratch/cscs/damrein/outputs/plots/production")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-step curves, averaged over cosmologies -----------------------------
    # Steps are aggregated by SHELL INDEX, not by comoving radius: every cosmology
    # has the same 69 shells but at its own chi(z), so r_step is not comparable
    # across runs while the shell index is.
    files = sorted(glob.glob(f"{a.log_dir}/disco_gen_{a.job}_*.out"))
    if a.detail_task is not None:
        files = [f for f in files if f.endswith(f"_{a.detail_task}.out")]
    NB, SH, RP, ZZ = [], [], [], []
    for k, f in enumerate(files):
        recs = parse_steps(Path(f))
        if len(recs) != a.n_shells:
            continue
        keys = sorted(recs, key=lambda t: t[1])          # near -> far == shell 0..S-1
        NB.append([np.mean(recs[t]["nb"]) for t in keys])
        SH.append([np.mean(recs[t]["gc"]) + np.mean(recs[t]["gpu"]) for t in keys])
        RP.append([np.mean(recs[t]["rep"]) for t in keys])
        ZZ.append([zc for zc in shell_z(Path(f), a.grid, keys)])
        if (k + 1) % 25 == 0:
            print(f"  parsed {k+1}/{len(files)} logs", flush=True)
    NB, SH, RP, ZZ = map(np.asarray, (NB, SH, RP, ZZ))
    print(f"[timing] per-step curves averaged over {len(NB)} cosmologies")

    z = np.nanmean(ZZ, axis=0)
    nb, shell, rep = NB.mean(0), SH.mean(0), RP.mean(0)
    nb_lo, nb_hi = np.percentile(NB, [16, 84], axis=0)
    sh_lo, sh_hi = np.percentile(SH, [16, 84], axis=0)
    ntask = 8

    print(f"  N-body   per step {nb.mean():5.2f} s   total {nb.sum():6.1f} s")
    print(f"  shell    per step {shell.mean():5.2f} s ({shell.max():.1f} far -> {shell.min():.1f} near)"
          f"   total {shell.sum():6.1f} s  = {shell.sum()/nb.sum():.2f}x N-body")
    print(f"  replicas {int(rep.max())} far -> {int(rep.min())} near")

    # ---- campaign budget -------------------------------------------------------
    B = [b for f in sorted(glob.glob(f"{a.log_dir}/disco_gen_{a.job}_*.out"))
         if (b := parse_budget(Path(f), a.ntask)) is not None]
    M = lambda k: float(np.nanmedian([b[k] for b in B]))
    tot, ic = M("total"), M("ic")
    t_nb, t_gc, t_sh, t_pre, t_boot = M("nb"), M("gc"), M("sh"), M("pre"), M("boot")
    timed = t_nb + t_gc + t_sh + t_pre + t_boot
    rest = tot - timed
    e2e = tot + ic
    print(f"\n[campaign] {len(B)} cosmologies, {sum(b['exit']==0 for b in B)} exited 0")
    print(f"  IC conv {ic:5.0f} s | load+JIT {t_boot:5.0f} s | prestep {t_pre:6.0f} s | "
          f"N-body {t_nb:6.0f} s | shell {t_gc+t_sh:6.0f} s | untimed {rest:6.0f} s "
          f"({100*timed/tot:.0f}% accounted)")
    print(f"  end-to-end per cosmology {e2e/60:5.1f} min  "
          f"= {a.nodes*e2e/3600:.2f} node-h = {a.ntask*e2e/3600:.2f} GPU-h")
    print(f"  campaign total {len(B)*e2e/3600:.0f} h wall = {a.nodes*len(B)*e2e/3600:.0f} node-h")

    # ---- figure ----------------------------------------------------------------
    fig = plt.figure(figsize=(9.0, 8.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.55, 0.85, 0.50], hspace=0.75)
    ax, axr, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    ax.fill_between(z, nb_lo, nb_hi, color=C_NBODY, alpha=0.18, lw=0)
    ax.fill_between(z, sh_lo, sh_hi, color=C_SHELL, alpha=0.18, lw=0)
    ax.plot(z, nb, "-o", color=C_NBODY, lw=2.2, ms=3.0, label="N-body step (PM force solve)")
    ax.plot(z, shell, "-o", color=C_SHELL, lw=2.2, ms=3.0, label="shell builder (total)")
    ax.set_ylabel("timed wall-clock\nper step  [s]", fontsize=FS_AXIS)
    ylo = min(nb_lo.min(), sh_lo.min()); yhi = max(nb_hi.max(), sh_hi.max())
    pad = 0.10 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + 2.4 * pad)
    ax.set_xlim(z.max() * 1.02, -0.05)
    ax.tick_params(labelsize=FS_TICK, labelbottom=False)
    ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEGEND, loc="upper center", ncol=2, framealpha=1.0,
              borderpad=0.4, columnspacing=1.2)
    ax.set_title(f"(a)  per step, mean over {len(NB)} cosmologies (band: 16-84th pctile)",
                 fontsize=FS_AXIS, loc="left", pad=8)

    axr.plot(z, rep, "-o", color=C_REPL, lw=2.0, ms=3.4)
    axr.set_ylabel("active box\nreplicas", fontsize=FS_AXIS)
    axr.set_xlabel("shell redshift  $z$   (integration runs right $\\to$ left)", fontsize=FS_AXIS)
    axr.tick_params(labelsize=FS_TICK)
    axr.grid(alpha=0.25, lw=0.5); axr.set_axisbelow(True)
    axr.set_xlim(z.max() * 1.02, -0.05); axr.set_ylim(0, rep.max() * 1.15)

    segs = [("IC conversion", ic, C_IC), ("load + JIT", t_boot, C_SETUP),
            ("pre-lightcone integration ($z\\!=\\!99\\to3.5$)", t_pre, C_PRE),
            ("N-body, on-lightcone", t_nb, C_NBODY),
            ("shell builder", t_gc + t_sh, C_SHELL),
            ("untimed (dispatch, host, I/O)", rest, C_REST)]
    left = 0.0
    for lab, val, col in segs:
        axb.barh(0, val / 60, left=left / 60, height=0.55, color=col,
                 edgecolor="white", lw=1.0)
        if val / e2e > 0.08:
            axb.text((left + val / 2) / 60, 0, f"{val/60:.1f}", ha="center", va="center",
                     fontsize=FS_LEGEND - 1, color="#1a1a1a")
        left += val
    axb.set_xlim(0, e2e / 60 * 1.01); axb.set_ylim(-0.55, 0.75)
    axb.set_yticks([]); axb.tick_params(labelsize=FS_TICK)
    axb.set_xlabel("wall-clock per cosmology  [min]", fontsize=FS_AXIS)
    axb.set_title(f"(b)  full budget: {e2e/60:.1f} min per cosmology",
                  fontsize=FS_AXIS, loc="left", pad=9)
    axb.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="white") for _, _, c in segs],
               labels=[s[0] for s in segs], fontsize=FS_LEGEND - 1.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.75), ncol=3, frameon=False, handlelength=1.4)

    out = out_dir / "production_timing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\n[timing] -> {out}")


if __name__ == "__main__":
    main()
