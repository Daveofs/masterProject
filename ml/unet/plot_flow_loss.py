#!/usr/bin/env python3
"""Plot train/validation loss for a jbucko flow run.

Reads the train_log.jsonl that unet_flow_jbucko/train_flow.py writes (one JSON row
per epoch: {epoch, time_s, lr, train_loss, val_loss}). Train and validation are the
SAME flow-matching MSE loss < ||v_theta(x_t,t) - (x1-x0)||^2 >; validation is computed
on HELD-OUT COSMOLOGIES (split_by_cosmo in dataset.py), so a gap between the two curves
is the generalization gap. No model/training code here -- pure plotting glue, on the
SAME shared figure (analysis.plot_train_val_loss) transfer_function.py's emulator loss
plot uses, so the two pipelines' training diagnostics are structurally identical.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.plotting import plot_train_val_loss  # noqa: E402


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

    out = Path(args.out or (Path(args.run_dir) / "loss_curve.png"))
    plot_train_val_loss(
        ep, tr, va, out, xlabel="epoch", ylabel="flow-matching MSE loss",
        val_label="validation (held-out cosmologies)",
        formula="Conditional flow matching (low->high): train vs validation\n"
                r"loss $=\langle\,\|v_\theta(x_t,t)-(x_1-x_0)\|^2\,\rangle$, "
                r"$x_0$=low, $x_1$=high, $x_t=(1-t)x_0+t x_1$")


if __name__ == "__main__":
    main()
