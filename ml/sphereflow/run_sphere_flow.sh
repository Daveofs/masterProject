#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=sphere-flow
#SBATCH --partition=normal
#SBATCH --account=sk037
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/sphereflow/slurm-%j.err
#SBATCH --chdir=/users/damrein/masterProject/ml

# ============================================================================
# SINGLE-MODEL sphere-flow (v3, formulation=direct): DeepSphere-conv flow
# matching conditioned on the raw DISCO map, generating the high-res signal.
#
# v3 fixes vs the failed v2 run (3825944):
#   * direct formulation (no global resid_scale — the per-shell heteroscedastic
#     residual scale was miscalibrating faint vs dense shells)
#   * shell-index conditioning restored (model knows faint vs dense regime)
#   * batch 512 -> 128: best measured throughput AND ~4x more optimizer steps per
#     shell READ (12 vs 3) -> far less Lustre pressure (see below)
#   * multi-threaded prefetch + 30-min NCCL timeout (survive Lustre I/O stragglers)
#
# SCALE NOTE (job 3842721 crashed here): at 10 nodes = 40 ranks, all mmap-reading
# 14 GB shell files, Lustre contention drove throughput DOWN to 0.47 steps/s (vs
# 2.1 at 4 nodes) and one rank stalled >10 min -> NCCL collective timeout -> crash.
# For the CURRENT ~44 runs, 4-6 nodes is FASTER and more stable than 10 (fewer
# ranks contending for the same Lustre bandwidth). Scale nodes up only once you
# have many more DISCO pairs. --nodes below is left at your value; consider 4.
# ============================================================================

export UENV_REPO_PATH=/capstor/scratch/cscs/damrein/.uenv-images
VENV=/capstor/scratch/cscs/damrein/venvs/sphereflow

DATA_ROOT="/capstor/scratch/cscs/damrein/cosmogridv1"
# STABLE run name (NOT the job id): resubmitting after a crash then AUTO-RESUMES
# from OUT_DIR/checkpoint.pt instead of restarting from scratch. Start a brand-new
# training by choosing a new RUN_NAME (or deleting the checkpoint).
RUN_NAME=${RUN_NAME:-v3_direct_44runs}
OUT_DIR="/capstor/scratch/cscs/damrein/outputs/sphereflow/${RUN_NAME}"
TEST_COSMO=cosmo_000122
mkdir -p "$OUT_DIR" /capstor/scratch/cscs/damrein/outputs/logs/sphereflow

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_ATS=0
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=memhooks
# At 10 nodes x 4 GPUs = 40 ranks, NCCL opens enough concurrent rendezvous
# channels to exhaust the CXI provider's HARDWARE tag-matching table
# (Slingshot-11), surfacing as "NET/OFI ... RC: 5. Error: 26 (PTLTE_NOT_FOUND)".
# Switching to software tag matching removes that hardware limit (small latency
# cost, not throughput-limiting for this workload). Standard fix at this scale.
export FI_CXI_RX_MATCH_MODE=software
# PTLTE_NOT_FOUND recurred mid-run (job 3852435, step 13k) even with software
# matching -> also enlarge the CXI completion/transmit queues (CSCS-recommended
# for ML workloads). Note these reduce, not eliminate, transient fabric errors —
# the trainer now also CHECKPOINTS every 500 steps and auto-resumes on restart.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1
export LD_LIBRARY_PATH=/opt/cray/pe/lib64:/opt/cray/libfabric/lib64:$LD_LIBRARY_PATH
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
# Do NOT dump core files on crash: a SIGABRT from 16 ranks wrote ~27 GB of
# multi-GB `core_nid*` files into the (home) working dir and blew the 50 GB home
# quota. We never gdb these; suppress them so crashes don't fill home.
ulimit -c 0
# DIAGNOSTIC per-rank heartbeat window (see train loop). Active in [HB_LO,HB_HI];
# set HB_HI=0 to disable. Covers the deterministic ~13,254 hang.
export HB_LO=${HB_LO:-13240}
export HB_HI=${HB_HI:-13270}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== sphere-flow v3 (single model, direct) | job ${SLURM_JOB_ID} ===="

# ---- stage 0: ensure raw npy dataset exists (decompress DISCO+high npz; CPU) ----
#      map-only, no transfer function / SHT. Skips runs already prepared.
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python preprocess/prepare_maps.py --data-dir '${DATA_ROOT}' --nside 2048 --num-workers 5
"

# ---- stage 1: DDP training (4 nodes x 4 GPUs) ----
srun uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python -m torch.distributed.run \
           --nnodes=${SLURM_NNODES} --nproc_per_node=${GPUS_PER_NODE} \
           --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    sphereflow/train_sphere_flow.py \
      --data-root   '${DATA_ROOT}' \
      --test-cosmo  ${TEST_COSMO} \
      --include-test \
      --formulation direct \
      --nside       2048 \
      --order       16 \
      --hidden      64 \
      --n-layers    6 \
      --K           5 \
      --epochs      8 \
      --batch-size  128 \
      --patch-frac  0.5 \
      --lr          2e-4 \
      --log-every   50 \
      --ckpt-every  200 \
      --no-compile \
      --out-dir     '${OUT_DIR}'
"
STAGE1_RC=$?

# ---- auto-requeue on crash (recurring Slingshot PTLTE_NOT_FOUND fabric errors at
#      this node count -- not fully eliminated by the CXI env vars above). Training
#      exits nonzero on a crashed rank; checkpoint.pt already has the last <=500
#      steps of progress, so just resubmit ourselves (same RUN_NAME -> auto-resume)
#      instead of making the user notice and resubmit by hand every time.
#      CHAIN_DEPTH guards against infinite resubmission if something is genuinely
#      broken (not just a transient fabric error).
CHAIN_DEPTH=${CHAIN_DEPTH:-0}
MAX_CHAIN=${MAX_CHAIN:-15}
if [ "$STAGE1_RC" -ne 0 ]; then
    if [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN" ]; then
        echo "stage 1 crashed (rc=$STAGE1_RC), chain depth $CHAIN_DEPTH/$MAX_CHAIN -> resubmitting (will auto-resume from checkpoint)"
        sbatch --export=ALL,RUN_NAME="${RUN_NAME}",CHAIN_DEPTH=$((CHAIN_DEPTH + 1)),MAX_CHAIN="${MAX_CHAIN}" \
            "${BASH_SOURCE[0]}"
    else
        echo "stage 1 crashed (rc=$STAGE1_RC) and hit MAX_CHAIN=$MAX_CHAIN -- NOT resubmitting further. Check ${OUT_DIR}/checkpoint.pt and the logs; something beyond transient fabric errors may be wrong."
    fi
    echo "sphere-flow v3 job ${SLURM_JOB_ID} exiting early (stage 1 incomplete) at $(date)"
    exit 0
fi

# ---- stage 2: evaluate flow vs DISCO baseline vs CosmoGrid ----
srun --nodes=1 --ntasks=1 uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "
  source ${VENV}/bin/activate
  python sphereflow/apply_sphere_flow.py \
      --model-dir '${OUT_DIR}' \
      --data-root '${DATA_ROOT}' \
      --test-cosmo ${TEST_COSMO} \
      --shell-indices 3 30 50 --steps 50 --lmax 3000
"

echo "sphere-flow v3 job ${SLURM_JOB_ID} finished at $(date)"
