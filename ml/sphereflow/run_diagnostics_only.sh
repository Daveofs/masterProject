#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sphereflow-compare
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/compare-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/compare-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# Run apply_sphere_flow.py's full comparison suite on the sphereflow "direct"
# checkpoint (run 3826942 -- the best surviving generative candidate of the
# 2026-07-14 model survey, see the script's docstring) through the SAME shared
# analysis/ diagnostics apply_transfer.py uses, so its plots (in OUT_DIR) can
# be compared side-by-side with the transfer function's own eval outputs under
# /capstor/scratch/cscs/damrein/outputs/transfer/.
#
# Single GPU, single cosmology (cosmo_000122 -- the only one this checkpoint
# was actually held out on): ~20-35 unique shells get ODE-sampled (50 steps
# each) across the patch/full-sky/zbin-grid diagnostics, cached and reused
# across plot stages -- a few minutes to ~an hour depending on GPU load, well
# within the 4h budget above. Pass --kappa to also add it (expensive: dozens
# more shells) once the cheaper diagnostics look right.
#
# apply_sphere_flow.py enables --compile + --amp (bf16 autocast) and a bigger
# --patch-batch (512, vs. training's memory-bound 256 sweet spot) by default --
# "free" speedups (same math, faster; see sphere_flow.sample_ode's docstring
# for why precision doesn't degrade). Override via EXTRA_ARGS, e.g.
# EXTRA_ARGS="--steps 25" sbatch run_sphere_flow_compare.sh for a (lower-
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
      --out-dir    '${OUT_DIR}' \
      $EXTRA_ARGS
"

echo "sphereflow compare job ${SLURM_JOB_ID} finished at $(date) -> ${OUT_DIR}"
