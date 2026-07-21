#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=sphereflow-compare
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/compare-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/compare-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# MULTI-NODE EVAL (2026-07-20): the --nodes=1 header above is just the sbatch
# default -- override at submission (`sbatch --nodes=4 sphereflow/run_diagnostics_only.sh`)
# to scale past one node's 4 GPUs. WORLD_SIZE below is derived from however many
# nodes SLURM actually gave this job (SLURM_NNODES x GPUS_PER_NODE), via a genuine
# multi-node torchrun c10d rendezvous (MASTER_ADDR from the node list, srun spans
# every node). This is a DIFFERENT risk profile from sphereflow TRAINING's multi-node
# ban (see run_sphere_flow.sh's header): the crashes that forced training to 1 node
# were from SUSTAINED per-step DDP gradient allreduce over ~13k+ steps hammering the
# Slingshot fabric; this eval job's own distributed calls are a couple of
# dist.all_gather_object per diagnostic stage (zbin-grid, kappa) -- a handful of
# collectives total, not thousands -- so it is a materially lighter cross-node
# workload. Not proven crash-proof, just a different, lower-frequency usage pattern.

# ============================================================================
# Run apply_sphere_flow.py's full comparison suite on the sphereflow "direct"
# checkpoint (run 3826942 -- the best surviving generative candidate of the
# 2026-07-14 model survey, see the script's docstring) through the SAME shared
# analysis/ diagnostics apply_transfer.py uses, so its plots (in OUT_DIR) can
# be compared side-by-side with the transfer function's own eval outputs under
# /capstor/scratch/cscs/damrein/outputs/transfer/ and unet's under
# /capstor/scratch/cscs/damrein/outputs/flowruns/<run>/eval.
#
# Emits the SAME figure set as transfer/run_diagnostic_only.sh and
# unet/run_diagnostics_only.sh (shared analysis/ plotting code, same shells
# 5 10 15 30 50, same kappa nside=1024/lmax=2048), so the three eval dirs can be
# compared plot-by-plot:
#   example_patches.png, patch_power_ratio_pctile_band.png, moments_vs_shell.png,
#   example_histograms.png, cl_ratio_by_zbin_grid.png, kappa_cl_per_cosmology.png,
#   kappa_cl_pctile_band.png, kappa_moments_scatter.png
#
# Single GPU, single cosmology (cosmo_000122 -- the only one this checkpoint was
# actually held out on). --kappa is the cost driver: on top of the ~20-35 unique
# shells the patch/full-sky/zbin-grid diagnostics ODE-sample (8 steps each since
# the 2026-07-21 x0=cond change, cached and reused across plot stages), it samples
# EVERY usable shell in z<=1.05 (dozens).
# Drop --kappa below (and expect a few minutes to ~an hour instead) if you only want
# the cheaper diagnostics.
#
# apply_sphere_flow.py enables --compile + --amp (bf16 autocast) and a bigger
# --patch-batch (512, vs. training's memory-bound 256 sweet spot) by default --
# "free" speedups (same math, faster; see sphere_flow.sample_ode's docstring
# for why precision doesn't degrade). Override via EXTRA_ARGS, e.g.
# EXTRA_ARGS="--steps 25" sbatch sphereflow/run_diagnostics_only.sh for a (lower-
# fidelity) step-count reduction on top.
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow

# GPUS_PER_NODE x SLURM_NNODES = total ranks -- apply_sphere_flow.py splits its two
# dominant-cost stages (plot_cl_zbin_grid, plot_kappa) across ALL of them via
# torch.distributed. SLURM_NNODES reflects whatever --nodes this job was actually
# submitted with (see the multi-node note above); GPUS_PER_NODE=4 is this cluster's
# fixed per-node GPU count, not a tunable.
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-1}
EVAL_GPUS=$(( GPUS_PER_NODE * NNODES ))
# Pin OpenMP-using CPU work (healpy's Cl/anafast, UFalcon's kappa-map construction)
# to the cores SLURM actually allocated PER NODE (--cpus-per-task=128 above), DIVIDED
# across the GPUS_PER_NODE concurrent rank processes THAT NODE RUNS -- the parallel
# zbin-grid/kappa sections run all ranks' CPU-bound work at once, so giving every
# rank the full per-node core count would oversubscribe.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-128} / GPUS_PER_NODE ))

# All env-overridable so run_sphere_flow.sh can submit this as a dependent eval job
# for whatever model/dataset it just trained. --run-dirs is NOT passed ->
# apply_sphere_flow.py reads the held-out set straight from MODEL_DIR/meta.npz's
# test_cosmos, so this always evaluates on exactly what that model held out.
#
# DATA_ROOT must be the root the model TRAINED from (its held-out cosmos' data
# lives there). Job 4221138 failed with 7x FileNotFoundError from exactly this:
# the *_cos200 model -- DESPITE the name -- was trained on cosmogridv1's 44
# cosmos, not 200 grid ones (checked: its patch set nside512_order16_n200000
# holds exactly the 44 cosmogridv1 cosmos; DATA_ROOT/DATA_TAG defaults won at
# its submission), so its 7 held-out cosmos only have data under cosmogridv1.
DATA_ROOT="${DATA_ROOT:-/capstor/scratch/cscs/damrein/cosmogridv1}"
MODEL_DIR="${MODEL_DIR:-/capstor/scratch/cscs/damrein/outputs/sphereflow/x0cond_hpres_ovlp_nside512_o16_n100000_h128_b248_e40}"
OUT_DIR="${EVAL_OUT_DIR:-${MODEL_DIR}/eval}"
# NSIDE deliberately UNSET by default: apply_sphere_flow.py now takes the data
# nside from the model's own meta.npz (a 512 model + this script's old hardcoded
# NSIDE=2048 default was the other half of job 4221138's failure). Set NSIDE only
# to force a mismatch check.
NSIDE_FLAG=""; [ -n "${NSIDE}" ] && NSIDE_FLAG="--nside ${NSIDE}"
LMAX="${LMAX:-3000}"
# 10 -> 8 (2026-07-21): matches apply_sphere_flow.py's own new default now that
# sample_ode starts from x0=cond (informative start, see sphere_flow.py's
# docstring) instead of noise -- far fewer steps needed.
STEPS="${STEPS:-8}"
MAX_COSMOLOGIES="${MAX_COSMOLOGIES:-10}"
KAPPA="${KAPPA:-1}"
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa --kappa-nside 1024 --kappa-lmax 2048"
# OVERLAP checkpoints only (meta['patch_mode']=='overlap', see sphere_flow.py) --
# ignored for pre-2026-07-20 disjoint checkpoints. Both left UNSET by default so
# apply_sphere_flow.py's own auto-scaled/UNTUNED defaults apply; override to
# re-run diagnostics at a different center density / taper sharpness.
NSIDE_CENTERS="${NSIDE_CENTERS:-32}"
NSIDE_CENTERS_FLAG=""; [ -n "${NSIDE_CENTERS}" ] && NSIDE_CENTERS_FLAG="--nside-centers ${NSIDE_CENTERS}"
TAPER_POWER_FLAG=""; [ -n "${TAPER_POWER}" ] && TAPER_POWER_FLAG="--taper-power ${TAPER_POWER}"
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

echo "==== sphereflow diagnostics | job ${SLURM_JOB_ID} | ${NNODES} node(s) x ${GPUS_PER_NODE} GPU = ${EVAL_GPUS} ranks | model ${MODEL_DIR} | data ${DATA_ROOT} ===="

# srun spans EVERY allocated node (one task/node, torchrun fans out to
# GPUS_PER_NODE local ranks) -- NOT --nodes=1 like the old single-node version.
srun --ntasks=${NNODES} --ntasks-per-node=1 --gres=gpu:${GPUS_PER_NODE} uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python -m torch.distributed.run \
    --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    sphereflow/apply_sphere_flow.py \
      --model-dir  '${MODEL_DIR}' \
      --data-root  '${DATA_ROOT}' \
      --max-cosmologies ${MAX_COSMOLOGIES} \
      ${NSIDE_FLAG} \
      --lmax       ${LMAX} \
      --steps      ${STEPS} \
      --patch-shells 5 10 15 30 50 \
      --fullsky-shells 5 10 15 30 50 \
      --n-zbins 3 --n-shells-per-zbin 5 \
      ${KAPPA_FLAG} ${NSIDE_CENTERS_FLAG} ${TAPER_POWER_FLAG} \
      --out-dir    '${OUT_DIR}' \
      $EXTRA_ARGS
"

echo "sphereflow diagnostics job ${SLURM_JOB_ID} finished at $(date) -> ${OUT_DIR}"
