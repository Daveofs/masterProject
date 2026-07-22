#!/usr/bin/env python3
"""Train the conditional flow-matching model (see flow_model.py) on patches
from make_patch_dataset.py. Same DDP/wandb/checkpoint infra as train.py.

Single GPU:
    python train_flow.py --patch-dir $sdir/sphereflow/patches/nside512_256_100k \
        --out-dir $sdir/sphereflow/runs/flow_v1

Multi-GPU (single node, DDP via torchrun - see train_flow.sbatch):
    python3 -m torch.distributed.run --standalone --nproc_per_node=4 train_flow.py ...
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PatchDataset, split_by_cosmo, transform_pair, cosmo_z_vector
from flow_model import FlowUNet, residual_target


def is_distributed():
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def reduce_mean(value: float, device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              grad_clip: float, is_main: bool, hp_cutoff: float, hp_transition: float,
              space: str, log_every: int = 100, wandb_run=None):
    model.train(train)
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for step, batch in enumerate(loader):
            low_raw = batch["low"].to(device, non_blocking=True)
            high_raw = batch["high"].to(device, non_blocking=True)
            x0, high_f = transform_pair(low_raw, high_raw, space)
            # HIGH-PASS RESIDUAL target (see flow_model.py's module docstring): x1 is
            # IDENTICAL to x0 at large scales, so the target velocity x1-x0 is a pure
            # high-pass field -- the ODE can only ever add small-scale content.
            x1 = x0 + residual_target(x0, high_f, hp_cutoff, hp_transition)

            # built unconditionally -- forward() ignores it when the model was
            # constructed with use_cosmo_cond=False, so no branching needed here.
            cosmo = batch["cosmo"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True).to(x0.dtype)
            cosmo_z = cosmo_z_vector(cosmo, z).to(x0.dtype)

            bs = x0.shape[0]
            t = torch.rand(bs, device=device, dtype=x0.dtype)
            t_bcast = t[:, None, None, None]
            xt = (1 - t_bcast) * x0 + t_bcast * x1
            target_v = x1 - x0

            pred_v = model(xt, t, cosmo_z=cosmo_z)
            # Per-patch variance normalization (2026-07-21): a dense/well-resolved
            # shell's TRUE residual is naturally tiny, so plain MSE gives it far
            # weaker gradient signal than a sparse shell's large residual -- the
            # network learns sparse-shell corrections well but never gets pushed to
            # calibrate the "correct answer is near-zero here" case precisely,
            # showing up as a persistent few-percent over/undershoot in the
            # densest-shell Cl-ratio panel that cutoff/transition tuning and more
            # epochs both failed to fix (see [[deepsphere-shell-correction]] memory).
            #
            # BOUNDED, BATCH-RELATIVE (2026-07-21, fixed same day -- diffusion's own
            # version of this raw-epsilon-floored divisor caused a non-converging,
            # oscillating loss when it compounded with EDM's own per-noise-level
            # weight; unet has no such second weight to compound with, but a raw
            # divisor is still fragile and the loss barely moved epoch-to-epoch,
            # 0.97->0.89 over 200 epochs -- suspiciously flat, consistent with an
            # unbounded reweighting washing out useful gradient signal). Normalizing
            # by the BATCH's own mean variance (a "typical" patch gets weight 1) and
            # clamping to >=0.1 (at most 10x upweighting) keeps the same direction
            # of correction without the unbounded blowup.
            patch_var = target_v.var(dim=(1, 2, 3), keepdim=True)
            rel_var = torch.clamp(patch_var / (patch_var.mean() + 1e-8), min=0.1)
            loss = (((pred_v - target_v) ** 2) / rel_var).mean()

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * bs
            n += bs

            if train and is_main and (step + 1) % log_every == 0:
                dt = time.time() - t0
                rate = (step + 1) / dt
                eta = (n_batches - step - 1) / rate
                print(f"  [step {step + 1}/{n_batches}] loss={loss.item():.5f} "
                      f"grad_norm={grad_norm:.3f} ({rate:.2f} it/s/rank, eta {eta / 60:.1f} min)", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/step_loss": loss.item(),
                                   "train/grad_norm": grad_norm.item(), "epoch": epoch})

    mean_loss = total_loss / n
    if is_distributed():
        mean_loss = reduce_mean(mean_loss, device)
    return mean_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--time-emb-dim", type=int, default=128)
    p.add_argument("--use-cosmo-cond", action=argparse.BooleanOptionalAction, default=True,
                   help="condition FlowUNet on cosmology (H0,Omega_cdm,Ob,Om,ns,s8,w0) + "
                        "shell redshift, injected at the bottleneck latent (see "
                        "flow_model.FlowUNet). Default: on. Pass --no-use-cosmo-cond to "
                        "train the original, unconditioned model for an A/B comparison.")
    # High-pass residual formulation (see flow_model.py's module docstring): the flow
    # target x1 is x0 + highpass(high-x0), so it can only ever ADD small-scale
    # content, never drift x0's already-correct large scales. Same defaults/semantics
    # as diffusion/train_diffusion.py's --hp-cutoff/--hp-transition (fractions of
    # patch Nyquist) -- MUST be saved into the checkpoint (they are, via args.json/
    # ckpt["args"]) so apply_flow.py composes with the exact cutoff the target used.
    p.add_argument("--hp-cutoff", type=float, default=0.10,
                   help="high-pass cutoff as a fraction of patch Nyquist (below this: "
                        "pinned to the low map, not generated)")
    p.add_argument("--hp-transition", type=float, default=0.10,
                   help="width of the raised-cosine hand-over band above --hp-cutoff, "
                        "as a fraction of patch Nyquist")
    # 'delta' (linear overdensity) is the space analysis.full_sky.od_cl actually
    # measures -- see dataset.raw_to_delta_pair for why 'log1p' was a formulation bug
    # (identical finding to diffusion/train_diffusion.py's --space). 'log1p' kept for
    # comparison only.
    p.add_argument("--space", choices=["delta", "log1p"], default="delta",
                   help="field the residual is modelled in (default: delta, the space "
                        "od_cl measures)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-lr-scaling", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="sphereflow")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    distributed = is_distributed()
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)
    is_main = rank == 0

    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)

    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                                config=vars(args), dir=str(out_dir), mode=args.wandb_mode)

    train_idx, val_idx, val_cosmos = split_by_cosmo(args.patch_dir, args.val_frac, args.seed)
    if is_main:
        print(f"[train_flow] {len(train_idx)} train patches, {len(val_idx)} val patches "
              f"({len(val_cosmos)} held-out cosmologies: {val_cosmos})")
        if distributed:
            print(f"[train_flow] distributed: world_size={world_size}, "
                  f"global batch={args.batch_size * world_size}")

    train_ds = PatchDataset(args.patch_dir, train_idx)
    val_ds = PatchDataset(args.patch_dir, val_idx)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                            shuffle=True, seed=args.seed, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                   drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    lr = args.lr if args.no_lr_scaling else args.lr * world_size
    model = FlowUNet(in_channels=1, out_channels=1, base_channels=args.base_channels,
                      time_emb_dim=args.time_emb_dim,
                      use_cosmo_cond=args.use_cosmo_cond).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if distributed else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        if is_main:
            print(f"[train_flow] resumed from {args.resume} at epoch {start_epoch}")

    log_path = out_dir / "train_log.jsonl"
    # A fresh (non-resumed) run must not APPEND onto a stale log left behind by an
    # earlier training run that reused this same --out-dir (e.g. same RUN_NAME
    # resubmitted) -- the per-epoch write below always opens in append mode (correct
    # for --resume, which continues an existing log), so without this a rerun
    # silently duplicates every epoch 0..N-1 on top of the old ones, corrupting
    # plot_flow_loss.py's loss curve with repeated/overlapping segments. Confirmed:
    # flow_nside512_patch256_n100000_ch32_b32_e40/train_log.jsonl had epochs 0-39
    # three times over (120 rows, 40 unique) from three non-resumed reruns.
    if not args.resume and is_main and log_path.exists():
        log_path.unlink()
    n_params = sum(p.numel() for p in (model.module if distributed else model).parameters())
    if is_main:
        print(f"[train_flow] model has {n_params:,} parameters, device={device}, lr={lr:.2e}, "
              f"use_cosmo_cond={args.use_cosmo_cond}, space={args.space}, "
              f"hp_cutoff={args.hp_cutoff}, hp_transition={args.hp_transition}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, True,
                                epoch, args.grad_clip, is_main, args.hp_cutoff,
                                args.hp_transition, args.space, wandb_run=wandb_run)
        val_loss = run_epoch(model, val_loader, optimizer, device, False,
                              epoch, args.grad_clip, is_main, args.hp_cutoff,
                              args.hp_transition, args.space, wandb_run=wandb_run)
        scheduler.step()
        dt = time.time() - t0

        if not is_main:
            continue

        row = {"epoch": epoch, "time_s": dt, "lr": scheduler.get_last_lr()[0],
               "train_loss": train_loss, "val_loss": val_loss}
        print(f"[epoch {epoch}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} ({dt:.1f}s)", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if wandb_run is not None:
            wandb_run.log({f"epoch/{k}": v for k, v in row.items()})

        ckpt = {
            "model": (model.module if distributed else model).state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_val": best_val, "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, out_dir / "best.pt")

    if is_main:
        print(f"[train_flow] done. best val_loss={best_val:.5f}")
        if wandb_run is not None:
            wandb_run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
