#!/usr/bin/env python3
"""Plot train/validation loss for a diffusion run.

Reads the train_log.jsonl that train_diffusion.py writes (one JSON row per epoch:
{epoch, time_s, lr, train_loss, val_loss}). Train and validation are the SAME
sigma-weighted EDM denoising loss; validation is computed on HELD-OUT COSMOLOGIES
(split_by_cosmo in dataset.py), so a gap between the two curves is the
generalization gap. No model/training code here -- pure plotting glue, on the SAME
shared figure (analysis.plot_train_val_loss) every other pipeline's loss plot uses.
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
    p.add_argument("--run-dir", required=True, help="train_diffusion.py --out-dir (has train_log.jsonl)")
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
        ep, tr, va, out, xlabel="epoch", ylabel="EDM weighted denoising loss",
        val_label="validation (held-out cosmologies)", skip_first=1, smooth_window=9,
        formula="EDM conditional diffusion (low->high): train vs validation\n"
                r"loss $=\langle\,\lambda(\sigma)\|D_\theta(x_1+\sigma\epsilon,\sigma,\mathrm{cond})-x_1\|^2\,\rangle$, "
                r"$\ln\sigma\sim\mathcal{N}(P_\mathrm{mean},P_\mathrm{std}^2)$, "
                r"$\mathrm{cond}$=low, $x_1$=high")


if __name__ == "__main__":
    main()
