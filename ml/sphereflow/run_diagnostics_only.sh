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

DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"
MODEL_DIR="/capstor/scratch/cscs/damrein/outputs/sphereflow/3826942"
OUT_DIR="${MODEL_DIR}/compare"
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

echo "==== sphereflow compare | job ${SLURM_JOB_ID} | model ${MODEL_DIR} ===="

srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python sphereflow/apply_sphere_flow.py \
      --model-dir  '${MODEL_DIR}' \
      --data-root  '${DATA_ROOT}' \
      --nside      2048 \
      --lmax       3000 \
      --steps      50 \
      --patch-shells 5 10 15 30 50 \
      --fullsky-shells 5 10 15 30 50 \
      --n-zbins 3 --n-shells-per-zbin 5 \
      --kappa --kappa-nside 1024 --kappa-lmax 2048 \
      --out-dir    '${OUT_DIR}' \
      $EXTRA_ARGS
"

echo "sphereflow compare job ${SLURM_JOB_ID} finished at $(date) -> ${OUT_DIR}"
