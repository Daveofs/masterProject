#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sphereflow-compare
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/compare-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/compare-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

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
# shells the patch/full-sky/zbin-grid diagnostics ODE-sample (50 steps each, cached
# and reused across plot stages), it samples EVERY usable shell in z<=1.05 (dozens).
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
DATA_ROOT="${DATA_ROOT:-/capstor/scratch/cscs/damrein/grid}"
MODEL_DIR="${MODEL_DIR:-/capstor/scratch/cscs/damrein/outputs/sphereflow/direct_grid_nside512_o16_n200000_h128_b128_e40}"
OUT_DIR="${EVAL_OUT_DIR:-${MODEL_DIR}/eval}"
# NSIDE deliberately UNSET by default: apply_sphere_flow.py now takes the data
# nside from the model's own meta.npz (a 512 model + this script's old hardcoded
# NSIDE=2048 default was the other half of job 4221138's failure). Set NSIDE only
# to force a mismatch check.
NSIDE_FLAG=""; [ -n "${NSIDE}" ] && NSIDE_FLAG="--nside ${NSIDE}"
LMAX="${LMAX:-3000}"
STEPS="${STEPS:-50}"
MAX_COSMOLOGIES="${MAX_COSMOLOGIES:-10}"
KAPPA="${KAPPA:-1}"
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa --kappa-nside 1024 --kappa-lmax 2048"
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

echo "==== sphereflow diagnostics | job ${SLURM_JOB_ID} | model ${MODEL_DIR} | data ${DATA_ROOT} ===="

srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python sphereflow/apply_sphere_flow.py \
      --model-dir  '${MODEL_DIR}' \
      --data-root  '${DATA_ROOT}' \
      --max-cosmologies ${MAX_COSMOLOGIES} \
      ${NSIDE_FLAG} \
      --lmax       ${LMAX} \
      --steps      ${STEPS} \
      --patch-shells 5 10 15 30 50 \
      --fullsky-shells 5 10 15 30 50 \
      --n-zbins 3 --n-shells-per-zbin 5 \
      ${KAPPA_FLAG} \
      --out-dir    '${OUT_DIR}' \
      $EXTRA_ARGS
"

echo "sphereflow diagnostics job ${SLURM_JOB_ID} finished at $(date) -> ${OUT_DIR}"
