#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=diffusion-diag
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/diffusion/diag-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/diffusion/diag-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Diagnostic-only run against the ALREADY-TRAINED checkpoint in OUT_DIR below (NO
# retraining): apply_diffusion.py's full eval suite, including the full-sky section
# (--data-root given) and --kappa. Sibling of unet/run_diagnostics_only.sh,
# sphereflow/run_diagnostics_only.sh and transfer/run_diagnostic_only.sh -- same
# shells (5 10 15 30 50), same kappa resolution -- so all four pipelines' eval dirs
# can be diffed plot-by-plot.
#
# WHY THIS EXISTS / WHEN TO USE IT: re-evaluate an already-trained checkpoint after
# an INFERENCE-side change, without paying for retraining. Used 2026-07-18 for the
# sharpened-taper fix (TAPER_POWER below): the ~16x-overlap cosine blend was averaging
# each sky pixel's ~6 independent diffusion samples, shrinking the stochastic part of
# the correction to ~0.41 of its amplitude -- the huge downward Cl percentile band on
# faint shells. taper_power=32 (soft nearest-tile-wins, the Ronneberger overlap-tile
# idea) measured on the delta-512 checkpoint, shell 5 ell800-1535: corrected/high
# 0.597 -> 0.800 (low baseline 0.535); shell 15: 0.904 -> 0.977. NOTE an earlier
# "shared global sphere noise" fix was tried and REVERTED: cropping noise through the
# gnomonic index map made it ~8%% correlated (not white), out-of-distribution for the
# denoiser -- high-k went 1.013 -> 0.799, worse than no model.

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DIFFUSION=/users/damrein/masterProject/ml/diffusion

# Overridable so this can target ANY finished run without editing the file, e.g.
#   sbatch --export=ALL,PATCH_DIR=...,OUT_DIR=...,DATA_ROOT=... diffusion/run_diagnostics_only.sh
# PATCH_DIR/OUT_DIR must be the pair that were actually TRAINED together
# (apply_diffusion.py reads the source nside from PATCH_DIR's metadata, so a
# mismatched pair is a silent resolution error, not a crash), and DATA_ROOT must be
# where PATCH_DIR's own held-out cosmologies actually live -- see run_diffusion.sh's
# DATA_TAG note for why a mismatch silently breaks the full-sky/kappa sections.
# Defaults point at the DELTA-space nside=512 run (job 4237844) -- the current best
# checkpoint. (The older diffusion_cosmogridv1_* runs are LOG1P-space and their
# small-scale numbers are misleading, see dataset.raw_to_delta_pair.)
DATA_ROOT=${DATA_ROOT:-/capstor/scratch/cscs/damrein/grid}
PATCH_DIR=${PATCH_DIR:-/capstor/scratch/cscs/damrein/outputs/flowpatches/grid_nside512_256_100000}
OUT_DIR=${OUT_DIR:-/capstor/scratch/cscs/damrein/outputs/diffusionruns/diffusion_delta_grid_nside512_patch256_n100000_ch32_b32_e40}

# EDM sampler settings -- must match how you want to sample, NOT stored in the
# checkpoint (unlike hp_cutoff/hp_transition/sigma_data, which apply_diffusion.py
# reads back from ckpt["args"] so the composition can't drift from training).
STEPS=${STEPS:-32}
SIGMA_MIN=${SIGMA_MIN:-0.002}
SIGMA_MAX=${SIGMA_MAX:-80.0}
RHO=${RHO:-7.0}
TAPER_POWER=${TAPER_POWER:-32}

export PYTHONUNBUFFERED=1
# EVAL_GPUS defined here (used below by both OMP_NUM_THREADS and the torchrun launch)
# -- apply_diffusion.py splits its zbin-grid/kappa sections (dominant cost) across
# this many GPUs via torch.distributed, single-node intra-NVLink. Drop to 1 to fall
# back to the original single-process path.
EVAL_GPUS=${EVAL_GPUS:-4}
# Pin OpenMP-using CPU work (healpy's Cl/anafast, UFalcon's kappa-map construction)
# to the cores SLURM actually allocated (--cpus-per-task=128 above), DIVIDED across
# the EVAL_GPUS concurrent rank processes -- the parallel zbin-grid/kappa sections
# run all ranks' CPU-bound work at once, so giving every rank the full core count
# would oversubscribe (EVAL_GPUS x 128 threads contending for 128 cores).
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-128} / EVAL_GPUS ))
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa"
# 3 cosmologies: enough for a REAL 16-84th percentile band (1 cosmology = no band at
# all, which is what the first delta-512 eval shipped with) at ~3x the kappa cost.
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-3}
mkdir -p /capstor/scratch/cscs/damrein/outputs/logs/diffusion

# COST: the full-sky sections reconstruct the whole sphere per shell, and each tile
# costs a ~STEPS-step Heun ODE integration (2 network evals per step) -- materially
# more expensive per shell than unet's 8-step flow. The tile GEOMETRY is cached across
# shells/cosmologies (analysis.patch_tiling.gnomonic_index_maps, ~3.2GB RAM at
# nside=2048/patch=256), so the GPU sampling is the limiter and --kappa-max-cosmologies
# / MAX_COSMOLOGIES are the real cost knobs. Drop STEPS if this needs to fit in less
# walltime (quality/speed tradeoff, unlike the tiling which is free after the cache).

srun --nodes=1 --ntasks=1 --gres=gpu:${EVAL_GPUS} uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${DIFFUSION}/plot_diffusion_loss.py --run-dir '${OUT_DIR}'
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=${EVAL_GPUS} \
    ${DIFFUSION}/apply_diffusion.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${OUT_DIR}/eval' \
    --steps ${STEPS} --sigma-min ${SIGMA_MIN} --sigma-max ${SIGMA_MAX} --rho ${RHO} \
    --taper-power ${TAPER_POWER} \
    --data-root '${DATA_ROOT}' \
    --shell-indices --example-shells 5 10 15 30 50 \
    --fullsky-patch-size 256 --max-cosmologies ${MAX_COSMOLOGIES} \
    --kappa-max-cosmologies ${MAX_COSMOLOGIES} ${KAPPA_FLAG}
"
echo "diffusion diagnostics-only job \${SLURM_JOB_ID} finished at \$(date) -> ${OUT_DIR}/eval"
