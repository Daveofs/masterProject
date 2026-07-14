#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unet-diag
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/unet/diag-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/unet/diag-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Diagnostic-only run against the ALREADY-TRAINED checkpoint in OUT_DIR below (no
# retraining): apply_flow.py's full eval suite, including the full-sky section
# (--data-root given: full-sky reconstruction + real angular Cl) and --kappa.
# Emits the SAME figure set as transfer/run_diagnostic_only.sh and
# sphereflow/run_diagnostics_only.sh -- same shells (5 10 15 30 50), same kappa
# resolution (nside=1024, lmax=2048, apply_flow.py's defaults) -- so the three
# pipelines' eval dirs can be diffed plot-by-plot.

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"
UNET=/users/damrein/masterProject/ml/unet
PATCH_DIR=/capstor/scratch/cscs/damrein/outputs/flowpatches/nside2048_256_100000
OUT_DIR=/capstor/scratch/cscs/damrein/outputs/flowruns/flow_nside2048_patch256_n100000_ch32_b32_e40_cond

export PYTHONUNBUFFERED=1
# weak-lensing kappa map diagnostic -- off by default, see run_flow.sh's KAPPA note.
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa"

# SPEED: job 4201387 TIMED OUT at 10h having done only 61 shell reconstructions --
# full-sky tiling was re-projecting all 12288 gnomonic tiles from scratch for every
# shell (~8 min/shell of single-threaded CPU, dwarfing the GPU work it was feeding).
# The tile geometry is value-independent, so it is now built once and reused across
# every shell AND cosmology (analysis.patch_tiling.gnomonic_index_maps): ~108x faster
# per shell (~500s -> ~5s), leaving the GPU flow-ODE integration (~1.5 min/shell) as
# the limiter. --kappa-max-cosmologies is therefore the real cost knob now, not the
# tiling. NOTE: the index cache costs ~3.2GB RAM at nside=2048/patch=256.

srun --nodes=1 --ntasks=1 --gres=gpu:1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${UNET}/plot_flow_loss.py --run-dir '${OUT_DIR}'
  python ${UNET}/apply_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${OUT_DIR}/eval' \
    --steps 8 \
    --data-root '${DATA_ROOT}' \
    --shell-indices --example-shells 5 10 15 30 50 \
    --fullsky-patch-size 256 ${KAPPA_FLAG}
"
echo "unet diagnostics-only job \${SLURM_JOB_ID} finished at \$(date) -> ${OUT_DIR}/eval"
