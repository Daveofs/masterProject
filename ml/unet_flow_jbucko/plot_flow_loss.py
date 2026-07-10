#!/usr/bin/env python3
"""Plot train/validation loss for a jbucko flow run.

Reads the train_log.jsonl that unet_flow_jbucko/train_flow.py writes (one JSON row
per epoch: {epoch, time_s, lr, train_loss, val_loss}). Train and validation are the
SAME flow-matching MSE loss < ||v_theta(x_t,t) - (x1-x0)||^2 >; validation is computed
on HELD-OUT COSMOLOGIES (split_by_cosmo in dataset.py), so a gap between the two curves
is the generalization gap. No model/training code here -- pure plotting glue.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="train_flow.py --out-dir (has train_log.jsonl)")
    p.add_argument("--out", default=None, help="output png (default: <run-dir>/loss_curve.png)")
    args = p.parse_args()

    log_path = Path(args.run_dir) / "train_log.jsonl"
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {log_path}")
    ep = np.array([r["epoch"] for r in rows])
    tr = np.array([r["train_loss"] for r in rows])
    va = np.array([r["val_loss"] for r in rows])

    best_ep = int(ep[np.argmin(va)])
    best_va = float(va.min())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, tr, "-o", ms=3, color="steelblue", label="train")
    ax.plot(ep, va, "-o", ms=3, color="tomato", label="validation (held-out cosmologies)")
    ax.axvline(best_ep, color="0.6", ls=":", lw=1.0)
    ax.scatter([best_ep], [best_va], color="tomato", zorder=5,
               label=f"best val {best_va:.4f} @ epoch {best_ep}")
    ax.set_xlabel("epoch"); ax.set_ylabel("flow-matching MSE loss")
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Conditional flow matching (low->high): train vs validation\n"
                 r"loss $=\langle\,\|v_\theta(x_t,t)-(x_1-x_0)\|^2\,\rangle$, "
                 r"$x_0$=low, $x_1$=high, $x_t=(1-t)x_0+t x_1$", fontsize=10)
    out = Path(args.out or (Path(args.run_dir) / "loss_curve.png"))
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"[plot] {len(rows)} epochs | best val {best_va:.5f} @ {best_ep} -> {out}", flush=True)


if __name__ == "__main__":
    main()
