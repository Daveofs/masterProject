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
#
# MULTI-NODE EVAL (2026-07-21, matching sphereflow/run_diagnostics_only.sh's
# identical pattern): the --nodes=1 header above is just the sbatch default --
# override at submission (`sbatch --nodes=4 diffusion/run_diagnostics_only.sh`) to
# scale past one node's 4 GPUs. WORLD_SIZE below is derived from however many
# nodes SLURM actually gave this job (SLURM_NNODES x GPUS_PER_NODE), via a genuine
# multi-node torchrun c10d rendezvous (MASTER_ADDR from the node list, srun spans
# every node). This eval job's own distributed calls are a couple of
# dist.all_gather_object per diagnostic stage (zbin-grid, kappa) -- a handful of
# collectives total -- so it's a materially lighter cross-node workload than
# sustained DDP training. Not proven crash-proof, just a different, lower-frequency
# usage pattern.

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
# GPUS_PER_NODE x SLURM_NNODES = total ranks -- apply_diffusion.py splits its two
# dominant-cost sections (zbin-grid, kappa) across ALL of them via
# torch.distributed. SLURM_NNODES reflects whatever --nodes this job was actually
# submitted with (see the multi-node note above); GPUS_PER_NODE=4 is this
# cluster's fixed per-node GPU count, not a tunable.
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-1}
EVAL_GPUS=$(( GPUS_PER_NODE * NNODES ))
# Pin OpenMP-using CPU work (healpy's Cl/anafast, UFalcon's kappa-map construction)
# to the cores SLURM actually allocated PER NODE (--cpus-per-task=128 above),
# DIVIDED across the GPUS_PER_NODE concurrent rank processes THAT NODE RUNS -- the
# parallel zbin-grid/kappa sections run all ranks' CPU-bound work at once, so
# giving every rank the full per-node core count would oversubscribe.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-128} / GPUS_PER_NODE ))
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

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

echo "==== diffusion diagnostics | job ${SLURM_JOB_ID} | ${NNODES} node(s) x ${GPUS_PER_NODE} GPU = ${EVAL_GPUS} ranks ===="

# srun spans EVERY allocated node (one task/node, torchrun fans out to
# GPUS_PER_NODE local ranks) -- NOT --nodes=1 like the old single-node version.
srun --ntasks=${NNODES} --ntasks-per-node=1 --gres=gpu:${GPUS_PER_NODE} uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${DIFFUSION}/plot_diffusion_loss.py --run-dir '${OUT_DIR}'
  python -m torch.distributed.run \
    --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
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
