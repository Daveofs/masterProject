#!/usr/bin/env python3
"""Plot train/validation loss for a sphere-flow run -- the SAME figure, source and
shared plotting code as unet/plot_flow_loss.py.

Reads the train_log.jsonl that train_sphere_flow.py writes (one JSON row per epoch:
{epoch, time_s, lr, train_loss, val_loss}). Train and validation are the SAME
flow-matching MSE loss; validation is computed on HELD-OUT COSMOLOGIES
(dataset.split_by_cosmo), so a gap between the two curves is the generalization
gap. Pure plotting glue on analysis.plot_train_val_loss -- the ONE canonical loss
figure unet (plot_flow_loss.py) and transfer (transfer_function.py train()) also
use, so all three pipelines' training diagnostics are structurally identical.

The only thing that differs from unet's is the printed formula: sphere-flow's
x0 is fresh NOISE (x0~N(0,I)) and x1 is the arcsinh-signal of the high map, whereas
unet interpolates from x0=low -- same flow-matching objective, different endpoints.

  python sphereflow/plot_sphere_loss.py --run-dir <train_sphere_flow --out-dir>
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
    p.add_argument("--run-dir", required=True,
                   help="train_sphere_flow.py --out-dir (has train_log.jsonl)")
    p.add_argument("--out", default=None, help="default: <run-dir>/loss_curve.png")
    args = p.parse_args()

    log_path = Path(args.run_dir) / "train_log.jsonl"
    if not log_path.exists():
        raise SystemExit(
            f"no train_log.jsonl in {args.run_dir} -- this run predates the "
            f"train/val trainer (e.g. the old streaming v3 model only saved a "
            f"training-loss EMA in meta.npz['loss_hist'], with no validation). "
            f"Retrain with train_sphere_flow.py to get a train-vs-val curve.")
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
        formula="DeepSphere conditional flow matching (direct): train vs validation\n"
                r"loss $=\langle\,\|v_\theta(x_t,t,\mathrm{cond})-(x_1-x_0)\|^2\,\rangle$, "
                r"$x_0\sim\mathcal{N}(0,I)$, $x_1=\mathrm{signal}(high)$, "
                r"$\mathrm{cond}=\mathrm{signal}(low)$")


if __name__ == "__main__":
    main()
