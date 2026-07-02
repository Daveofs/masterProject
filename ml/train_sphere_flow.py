#!/usr/bin/env python3
"""Train the DeepSphere-conv flow-matching generator (sphere_flow.SphereFlowNet).

Generative small-scale correction: learns to sample delta_high (per-shell high-res
overdensity) conditioned on delta_low, using spherical Chebyshev graph convs. Unlike
the deterministic L2 correction, this restores stochastic small-scale power.

Data-parallel with PyTorch DDP (torchrun). Shells are streamed one .npz file at a
time (per rank), so RAM is bounded regardless of the 20k-170k shell count. Maps are
split into order-based patches at high nside.

Launch (see run_sphere_flow.sh):
  torchrun --nproc_per_node=4 train_sphere_flow.py --data-root ... --nside 2048 --order 8
"""

from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import healpy as hp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sphere_flow as sf


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}"))
        return local, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


# ---------------------------------------------------------------------------
# Streaming data: one .npz file pair at a time, per-shell overdensity, patches
# ---------------------------------------------------------------------------

class FlowShellStreamer:
    """Yields (x1, cond, cosmo) patch batches for flow matching.

    x1   = delta_high / delta_scale   (target; per-shell overdensity, unit-ish var)
    cond = delta_low  / delta_scale   (conditioning)
    cosmo= per-file conditioning vector (cosmo params + shell-index scalar)

    Streams file pairs assigned to this rank; loads one pair, splits into patches,
    yields shuffled batches, moves on. Endless (the loop controls #steps).
    """

    def __init__(self, file_pairs, cosmo_vecs, nside, order, sig_scale, nest=True,
                 softening=1.0, seed=0):
        self.file_pairs = list(file_pairs)
        self.cosmo_vecs = list(cosmo_vecs)
        self.nside, self.order, self.nest = nside, order, nest
        self.npix = hp.nside2npix(nside)
        self.scale = sig_scale
        self.softening = softening
        self.rng = np.random.RandomState(seed)

    def _read(self, path):
        m = np.asarray(np.load(str(path))["shells"], dtype=np.float32)
        if m.shape[1] != self.npix:
            o = "NESTED" if self.nest else "RING"
            m = np.stack([hp.ud_grade(s, self.nside, order_in=o, order_out=o) for s in m])
        return m.astype(np.float32)

    def load_file(self, fi):
        low, high = self.file_pairs[fi]
        lo, hi = self._read(low), self._read(high)
        dlo, _ = sf.to_overdensity(lo)
        dhi, _ = sf.to_overdensity(hi)
        # arcsinh variance-stabilizing transform (tames shot-noise heavy tail).
        x1 = sf.map_to_patches(sf.signal_forward(dhi, self.scale, self.softening), self.order)
        cond = sf.map_to_patches(sf.signal_forward(dlo, self.scale, self.softening), self.order)
        # Per-shell cosmo vector repeated across that shell's patches.
        c = np.repeat(self.cosmo_vecs[fi][None], x1.shape[0], axis=0).astype(np.float32)
        return x1, cond, c

    def batches(self, batch_size, device):
        order = np.arange(len(self.file_pairs))
        while True:
            self.rng.shuffle(order)
            for fi in order:
                x1, cond, cosmo = self.load_file(int(fi))
                idx = self.rng.permutation(x1.shape[0])
                for b in range(0, len(idx) - batch_size + 1, batch_size):
                    bi = idx[b:b + batch_size]
                    yield (torch.from_numpy(x1[bi]).to(device),
                           torch.from_numpy(cond[bi]).to(device),
                           torch.from_numpy(cosmo[bi]).to(device))


def build_file_pairs(data_root, test_cosmo, low_name, high_name, include_test):
    import yaml
    data_root = Path(data_root)
    cosmos = sorted(d for d in data_root.iterdir()
                    if d.is_dir() and d.name.startswith("cosmo_"))
    pairs, cosmo_vecs = [], []
    for c in cosmos:
        if (not include_test) and c.name == test_cosmo:
            continue
        params_f = None
        runs = [r for r in sorted(c.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        for ld in (runs or [c]):
            low, high = ld / low_name, ld / high_name
            if not (low.exists() and high.exists()):
                continue
            pf = ld / "params.yml"
            if not pf.exists():
                pf = c / "params.yml"
            vec = _cosmo_vector(pf) if pf.exists() else np.zeros(1, np.float32)
            pairs.append((low, high))
            cosmo_vecs.append(vec)
    return pairs, cosmo_vecs


def _cosmo_vector(params_yml):
    import yaml
    p = yaml.safe_load(Path(params_yml).read_text())
    keys = sorted(k for k, v in p.items() if _is_num(v))
    return np.array([float(p[k]) for k in keys], dtype=np.float32)


def _is_num(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False


def estimate_signal_scale(file_pairs, nside, nest, softening=1.0, n=1):
    """Global std of arcsinh(delta_low) over a few files (normalizes the signal)."""
    ds = []
    for low, _ in file_pairs[:max(n, 1)]:
        m = np.asarray(np.load(str(low))["shells"], dtype=np.float32)
        if m.shape[1] != hp.nside2npix(nside):
            o = "NESTED" if nest else "RING"
            m = np.stack([hp.ud_grade(s, nside, order_in=o, order_out=o) for s in m])
        d, _ = sf.to_overdensity(m)
        ds.append(np.arcsinh(d / softening).std())
    return float(np.mean(ds) + 1e-12)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    local, rank, world = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    pairs, cosmo_vecs = build_file_pairs(
        args.data_root, args.test_cosmo, args.low_name, args.high_name, args.include_test)
    if not pairs:
        raise RuntimeError("no training file pairs found")
    # Pad cosmo vectors to a common length, then STANDARDIZE per-dimension. Raw
    # params span ~14 orders of magnitude (e.g. bary_Mc~1e14, As~1e-9); feeding
    # them unnormalized into the conditioning MLP produces NaN immediately.
    clen = max(len(v) for v in cosmo_vecs)
    carr = np.stack([np.pad(v, (0, clen - len(v))) for v in cosmo_vecs]).astype(np.float64)
    cmean = carr.mean(0)
    cstd = carr.std(0)
    cstd = np.where(cstd < 1e-8, 1.0, cstd)          # constant dims -> 0 after norm
    cosmo_vecs = [((carr[i] - cmean) / cstd).astype(np.float32) for i in range(len(carr))]
    cond_dim = clen

    sig_scale = estimate_signal_scale(pairs, args.nside, args.nest, args.softening,
                                      args.stat_files)

    # Shard files across ranks.
    my_pairs = pairs[rank::world] or [pairs[rank % len(pairs)]]
    my_cosmo = cosmo_vecs[rank::world] or [cosmo_vecs[rank % len(cosmo_vecs)]]
    if is_main():
        print(f"[data] {len(pairs)} files | {world} ranks | cond_dim={cond_dim} | "
              f"sig_scale={sig_scale:.4g} softening={args.softening} | "
              f"nside={args.nside} order={args.order}", flush=True)

    L = sf.healpix_laplacian(args.nside, order=args.order, nest=args.nest)
    net = sf.SphereFlowNet(L, cond_dim=cond_dim, hidden=args.hidden,
                           n_layers=args.n_layers, K=args.K).to(dev)
    if world > 1:
        net = DDP(net, device_ids=[local], broadcast_buffers=False)
    raw = net.module if world > 1 else net

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    streamer = FlowShellStreamer(my_pairs, my_cosmo, args.nside, args.order,
                                 sig_scale, nest=args.nest, softening=args.softening,
                                 seed=rank)

    # #patches this rank sees per epoch (approx) -> total steps.
    ppf = (sf.n_patches(args.order) if args.order > 1 else 1)
    shells = sum(np.load(str(p[0]))["shells"].shape[0] for p in my_pairs[:1]) * len(my_pairs)
    steps_per_epoch = max(shells * ppf // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))

    if is_main():
        print(f"[train] ~{steps_per_epoch:,} steps/epoch x {args.epochs} = "
              f"{total_steps:,} steps | batch/gpu={args.batch_size}", flush=True)

    net.train()
    it = streamer.batches(args.batch_size, dev)
    t0 = time.time()
    loss_hist = []
    for step in range(1, total_steps + 1):
        x1, cond, cosmo = next(it)
        opt.zero_grad()
        # fp32: torch.sparse.mm (the Chebyshev conv) has no bf16 CUDA kernel.
        loss = sf.flow_matching_loss(raw, x1, cond, cosmo)
        if not torch.isfinite(loss):   # safety net: skip a bad batch, don't poison weights
            opt.zero_grad(set_to_none=True); sched.step()
            if is_main():
                print(f"  step {step}: non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if is_main() and step % args.log_every == 0:
            r = step / (time.time() - t0)
            print(f"  step {step:,}/{total_steps:,} | loss={loss.item():.4f} | "
                  f"{r:.1f} steps/s | lr={sched.get_last_lr()[0]:.2e}", flush=True)
            loss_hist.append(loss.item())

    if is_main():
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save(raw.state_dict(), out / "sphere_flow.pth")
        np.savez(out / "meta.npz", nside=args.nside, order=args.order, K=args.K,
                 hidden=args.hidden, n_layers=args.n_layers, cond_dim=cond_dim,
                 sig_scale=sig_scale, softening=args.softening,
                 cosmo_mean=cmean, cosmo_std=cstd, loss_hist=np.array(loss_hist))
        print(f"[train] saved model + meta to {out}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--test-cosmo", default="cosmo_000001")
    p.add_argument("--low-name", default="shells_nside=2048.npz")
    p.add_argument("--high-name", default="compressed_shells.npz")
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument("--order", type=int, default=8)
    p.add_argument("--nest", action="store_true", default=True)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--softening", type=float, default=1.0,
                   help="arcsinh softening for the density transform (smaller = more "
                        "compression of the shot-noise tail).")
    p.add_argument("--stat-files", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out-dir", default="./sphere_flow_model")
    args = p.parse_args()
    try:
        train(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
