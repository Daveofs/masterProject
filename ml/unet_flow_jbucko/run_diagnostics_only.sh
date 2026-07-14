#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=flow-diag
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/flowjbucko/diag-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/flowjbucko/diag-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# Diagnostic-only run against the ALREADY-TRAINED nside=512/patch=256 checkpoint --
# runs apply_flow.py's full-sky section (--data-root given: full-sky reconstruction +
# real Cl + analysis.plot_example_full_sky_grid) WITHOUT retraining or colliding with
# the separate, currently-running nside=2048/patch=1024 job (4130899).

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"
PIPE=/users/damrein/masterProject/ml/unet_flow_jbucko
PATCH_DIR=/capstor/scratch/cscs/damrein/outputs/flowpatches/nside2048_256_100000
OUT_DIR=/capstor/scratch/cscs/damrein/outputs/flowruns/flow_nside2048_patch256_n100000_ch32_b32_e40_cond

export PYTHONUNBUFFERED=1
# weak-lensing kappa map diagnostic -- off by default, see run_flow.sh's KAPPA note.
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa"

srun --nodes=1 --ntasks=1 --gres=gpu:1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${PIPE}/plot_flow_loss.py --run-dir '${OUT_DIR}'
  python ${PIPE}/apply_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --model     '${OUT_DIR}/best.pt' \
    --out-dir   '${OUT_DIR}/eval' \
    --steps 8 \
    --data-root '${DATA_ROOT}' \
    --shell-indices --example-shells 5 10 15 20 30 50 \
    --fullsky-patch-size 256 ${KAPPA_FLAG}
"
echo "flow diagnostics-only job \${SLURM_JOB_ID} finished at \$(date) -> ${OUT_DIR}/eval"
