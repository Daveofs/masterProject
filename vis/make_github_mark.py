#!/usr/bin/env python3
"""Render the GitHub mark to a vector PDF for the code-availability appendix.

The thesis is built on a machine with no fontawesome/academicons TeX packages, so
the logo cannot come from a font. Rather than depend on a package being installed
wherever the document is next compiled, the mark is rendered once, here, from its
published SVG path into a self-contained PDF that \\includegraphics can take.

Only the path grammar the mark actually uses is implemented (M/m, L/l, H/h, V/v,
C/c, S/s, Z/z -- no arcs, no quadratics).

Usage
-----
    /users/damrein/miniforge3/bin/python make_github_mark.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

# GitHub mark, 24x24 viewBox, all cubic segments.
GITHUB = (
    "M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 "
    "0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 "
    "17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 "
    "1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465"
    "-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 "
    "3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 "
    "3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 "
    "1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627"
    "-5.373-12-12-12"
)

TOKEN = re.compile(r"([MmLlHhVvCcSsZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse(d: str):
    """SVG path string -> (vertices, codes) for matplotlib.path.Path."""
    toks = [(c or n) for c, n in TOKEN.findall(d)]
    verts, codes = [], []
    i, cmd = 0, None
    cur = start = (0.0, 0.0)
    prev_ctrl = None
    num = lambda k: float(toks[k])
    while i < len(toks):
        if re.match(r"[A-Za-z]", toks[i]):
            cmd = toks[i]; i += 1
            if cmd in "Zz":
                codes.append(MPath.CLOSEPOLY); verts.append(start)
                cur = start; prev_ctrl = None
                continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = num(i), num(i + 1); i += 2
            cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
            start = cur
            verts.append(cur); codes.append(MPath.MOVETO)
            cmd = "l" if rel else "L"          # implicit lineto for extra pairs
            prev_ctrl = None
        elif c == "L":
            x, y = num(i), num(i + 1); i += 2
            cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
            verts.append(cur); codes.append(MPath.LINETO); prev_ctrl = None
        elif c == "H":
            x = num(i); i += 1
            cur = (cur[0] + x, cur[1]) if rel else (x, cur[1])
            verts.append(cur); codes.append(MPath.LINETO); prev_ctrl = None
        elif c == "V":
            y = num(i); i += 1
            cur = (cur[0], cur[1] + y) if rel else (cur[0], y)
            verts.append(cur); codes.append(MPath.LINETO); prev_ctrl = None
        elif c in ("C", "S"):
            if c == "C":
                p1 = (num(i), num(i + 1)); p2 = (num(i + 2), num(i + 3))
                p3 = (num(i + 4), num(i + 5)); i += 6
            else:
                p2 = (num(i), num(i + 1)); p3 = (num(i + 2), num(i + 3)); i += 4
                p1 = (0.0, 0.0) if rel else (
                    2 * cur[0] - prev_ctrl[0], 2 * cur[1] - prev_ctrl[1]
                ) if prev_ctrl else cur
                if rel and prev_ctrl:
                    p1 = (cur[0] - prev_ctrl[0], cur[1] - prev_ctrl[1])
            if rel:
                p1 = (cur[0] + p1[0], cur[1] + p1[1])
                p2 = (cur[0] + p2[0], cur[1] + p2[1])
                p3 = (cur[0] + p3[0], cur[1] + p3[1])
            verts += [p1, p2, p3]
            codes += [MPath.CURVE4] * 3
            prev_ctrl, cur = p2, p3
        else:
            raise ValueError(f"unsupported SVG command {cmd!r}")
    return verts, codes


def main():
    verts, codes = parse(GITHUB)
    # SVG y grows downward; flip so the mark is upright.
    verts = [(x, 24.0 - y) for x, y in verts]

    fig = plt.figure(figsize=(1.0, 1.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.add_patch(PathPatch(MPath(verts, codes), fc="#161D33", ec="none"))
    ax.set_xlim(-0.4, 24.4); ax.set_ylim(-0.4, 24.4); ax.set_aspect("equal")

    out = Path("/users/damrein/masterProject/report/plots/github_mark.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, transparent=True); plt.close(fig)
    print(f"[github] {len(verts)} vertices -> {out}")


if __name__ == "__main__":
    main()
