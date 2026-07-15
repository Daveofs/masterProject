#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=sphere-flow
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# SINGLE-MODEL sphere-flow (formulation=direct): DeepSphere Chebyshev graph-conv
# flow matching conditioned on the raw DISCO map, generating the high-res signal.
#
#   stage 0  low/high shell stacks at nside            (preprocess/prepare_maps.py)
#   stage 1  (low, high) HEALPix-superpixel patch set  (sphereflow/make_patch_dataset.py)
#   stage 2  DDP train the flow, 1 node / 4 GPUs       (sphereflow/train_sphere_flow.py)
#   stage 3  full eval suite vs DISCO + CosmoGrid      (sphereflow/apply_sphere_flow.py)
#
# REBUILT 2026-07-14 on unet/run_flow.sh's structure, which is stable, after this
# job died the same way twice (jobs 3852435 and 4210107, both SIGABRT/NCCL around
# step ~13.2k with "NET/OFI ... PTLTE_NOT_FOUND" / "Device or resource busy").
#
# NODES: 1, deliberately -- this is THE fix, not a tuning knob.
#   Every crash was a CROSS-NODE Slingshot/libfabric (CXI) error. At 1 node x 4
#   GPUs, NCCL stays on intra-node NVLink and never touches the fabric, so that
#   entire failure class is gone by construction rather than papered over with
#   FI_CXI_* tuning (which was tried repeatedly and kept not working). All the
#   FI_CXI_*/NCCL_* env vars that used to be here are deleted for the same reason:
#   unet/run_flow.sh sets NONE of them and trains reliably on 1 node.
#   Scaling back out to multiple nodes means re-opening that can of worms; do it
#   only if a single node's 4 GPUs genuinely become the bottleneck.
#
# The other half of the fix is in the trainer: a materialized patch dataset +
# DistributedSampler (drop_last) instead of a hand-rolled per-rank producer
# thread, so ranks CANNOT desync -- see train_sphere_flow.py's header.
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow
# DATA_ROOT overridable: cosmogridv1 (44 cosmos, default) or the grid (198 cosmos
# with DISCO input -- 4.5x more cosmological diversity). find_ready_runs / prepare_maps
# both auto-skip the ~2300 grid cosmos that lack a DISCO low map, so pointing at the
# grid dir just picks the 198 runnable ones. PATCH_DIR/RUN_NAME below fold NPATCH etc
# but NOT the data root, so give a distinct RUN_NAME when switching datasets.
DATA_ROOT="${DATA_ROOT:-/capstor/scratch/cscs/damrein/cosmogridv1}"
SPHEREFLOW=/users/damrein/masterProject/ml/sphereflow

NSIDE=${NSIDE:-512}
ORDER=${ORDER:-16}            # 12*ORDER^2 patches/shell; must match the model's graph
NPATCH=${NPATCH:-200000}      # ~26 GB at nside=2048/order=16 (16,384 px/patch, low+high)
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-32}            # per GPU; patches are 16,384 px, so far smaller than unet's
HIDDEN=${HIDDEN:-64}
N_LAYERS=${N_LAYERS:-6}
K=${K:-5}
LR=${LR:-2e-4}
VAL_FRAC=${VAL_FRAC:-0.15}
SEED=${SEED:-0}
STEPS=${STEPS:-50}            # ODE steps at eval
# Held-out cosmologies: drawn by dataset.split_by_cosmo from --val-frac/--seed, and
# saved into meta.npz so stage 3 evaluates on exactly what training excluded. Pin
# them explicitly with TEST_COSMOS="cosmo_000003 cosmo_000006 ..." (e.g. to match
# the transfer pipeline's split).
TEST_COSMOS=${TEST_COSMOS:-}
TEST_COSMOS_FLAG=""
[ -n "$TEST_COSMOS" ] && TEST_COSMOS_FLAG="--test-cosmos ${TEST_COSMOS}"
# weak-lensing kappa diagnostic in stage 3 -- EXPENSIVE for sphereflow (every usable
# shell of every eval cosmology is a full ODE sample). KAPPA=0 skips it.
KAPPA=${KAPPA:-1}
KAPPA_FLAG=""; [ "${KAPPA}" = "1" ] && KAPPA_FLAG="--kappa --kappa-nside 1024 --kappa-lmax 2048"
MAX_COSMOLOGIES=${MAX_COSMOLOGIES:-3}

# DATA_TAG distinguishes patch caches / run dirs built from DIFFERENT datasets at
# the same NPATCH (e.g. "grid" vs the default cosmogridv1) so they never collide.
DATA_TAG=${DATA_TAG:+${DATA_TAG}_}
PATCH_DIR="/capstor/scratch/cscs/damrein/outputs/sphereflow/patches/${DATA_TAG}nside${NSIDE}_order${ORDER}_n${NPATCH}"
RUN_NAME=${RUN_NAME:-direct_${DATA_TAG}nside${NSIDE}_o${ORDER}_n${NPATCH}_h${HIDDEN}_b${BATCH}_e${EPOCHS}}
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/sphereflow/${RUN_NAME}"
mkdir -p "$PATCH_DIR" "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
ulimit -c 0    # never dump multi-GB core files into home on a crash

echo "==== sphere-flow | job ${SLURM_JOB_ID} | nside=${NSIDE} order=${ORDER} n=${NPATCH} ===="

# ---- stage 0: low/high shell stacks (skips runs already prepared) ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside ${NSIDE} --num-workers 5
"

# ---- stage 1: build the (low, high) superpixel patch dataset ----
# metadata.npy is the LAST file make_patch_dataset.py writes (low.npy/high.npy are
# allocated as empty memmaps up front), so checking it -- not low.npy -- is what
# distinguishes a COMPLETE dataset from a killed half-written one.
if [ ! -f "${PATCH_DIR}/metadata.npy" ]; then
  srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
    source ${VENV}/bin/activate
    python ${SPHEREFLOW}/make_patch_dataset.py \
      --data-dir  '${DATA_ROOT}' \
      --out-dir   '${PATCH_DIR}' \
      --nside ${NSIDE} --order ${ORDER} \
      --n-patches ${NPATCH} --seed ${SEED} \
      --num-workers ${SLURM_CPUS_PER_TASK}
  "
else
  echo '[stage1] patch dataset already exists, skipping'
fi

# ---- stage 2: DDP training, 1 node / 4 GPUs (torchrun --standalone) ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  cd ${SPHEREFLOW}
  python -m torch.distributed.run --standalone --nproc_per_node=4 train_sphere_flow.py \
    --patch-dir '${PATCH_DIR}' \
    --out-dir   '${OUT_DIR}' \
    --epochs ${EPOCHS} --batch-size ${BATCH} --lr ${LR} \
    --hidden ${HIDDEN} --n-layers ${N_LAYERS} --K ${K} \
    --val-frac ${VAL_FRAC} --seed ${SEED} ${TEST_COSMOS_FLAG} \
    --num-workers $((SLURM_CPUS_PER_TASK / 4))
"
STAGE2_RC=$?
if [ "$STAGE2_RC" -ne 0 ]; then
    echo "stage 2 FAILED (rc=$STAGE2_RC). Per-epoch checkpoints are in ${OUT_DIR}"
    echo "(last.pt/best.pt) -- resume with --resume ${OUT_DIR}/last.pt after fixing."
    exit 1
fi

# ---- stage 2b: train/validation loss curve -> ${OUT_DIR}/loss_curve.png. SAME
#      figure + shared analysis.plot_train_val_loss as unet/plot_flow_loss.py, read
#      from the train_log.jsonl stage 2 wrote (per-epoch train_loss + held-out
#      val_loss). CPU-only, seconds. ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python ${SPHEREFLOW}/plot_sphere_loss.py --run-dir '${OUT_DIR}'
"

# ---- stage 3: the FULL eval suite, submitted as a SEPARATE dependent GPU job.
#      SAME figures/statistics/shells as transfer/apply_transfer.py and
#      unet/apply_flow.py (shared analysis/ code):
#        example_patches.png, patch_power_ratio_pctile_band.png,
#        moments_vs_shell.png, example_histograms.png, cl_ratio_by_zbin_grid.png,
#        kappa_cl_{per_cosmology,pctile_band}.png, kappa_moments_scatter.png
#
#      WHY a separate job, not inline: sphereflow's eval ODE-samples every shell it
#      touches (zbin grid ~45 + kappa's ~47-per-cosmology full-sky reconstructions),
#      which is HOURS. Run inline, it competes with training for this job's walltime
#      and gets killed mid-plot -- exactly how job 4211337 died (8h limit hit during
#      the zbin grid, so cl_ratio_by_zbin_grid.png AND all kappa plots were never
#      written). run_diagnostics_only.sh gets its OWN 12h single-GPU walltime.
#      Depends on THIS job (afterok) so it only runs if training succeeded, and
#      reads the held-out set from ${OUT_DIR}/meta.npz (no --run-dirs drift).
sbatch --dependency=afterok:${SLURM_JOB_ID} \
       --export=ALL,MODEL_DIR="${OUT_DIR}",DATA_ROOT="${DATA_ROOT}",EVAL_OUT_DIR="${OUT_DIR}/eval",NSIDE="${NSIDE}",STEPS="${STEPS}",MAX_COSMOLOGIES="${MAX_COSMOLOGIES}",KAPPA="${KAPPA}" \
       ${SPHEREFLOW}/run_diagnostics_only.sh \
  && echo "[stage 3] diagnostics job submitted (afterok:${SLURM_JOB_ID}) -> ${OUT_DIR}/eval"

echo "sphere-flow TRAINING job ${SLURM_JOB_ID} finished at $(date) -> ${OUT_DIR} (eval runs as a dependent job)"
