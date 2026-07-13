#!/bin/bash
#SBATCH --job-name=transfer-vs-jbucko
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/compare-plots-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/compare-plots-%j.err
source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=32

RUN=/capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000122/run_0
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/compare_${SLURM_JOB_ID}
mkdir -p "$OUT"

echo "[1/2] example patches (vs jbucko's example_patches.png layout)"
python /users/damrein/masterProject/ml/transfer/plot_example_patches.py \
  --run-dir "$RUN" \
  --counts /capstor/scratch/cscs/damrein/outputs/transfer/poisson_iter_3891261/cosmo_000122_counts.npz \
  --shells 3 10 30 50 --n-per-shell 1 --patch-size 256 --nside 2048 --seed 0 \
  --out "$OUT/example_patches.png"

echo "[2/2] Poisson-calibration convergence (vs jbucko's loss_curve.png layout)"
python /users/damrein/masterProject/ml/transfer/plot_poisson_convergence.py \
  --run-dir "$RUN" \
  --corrected /capstor/scratch/cscs/damrein/outputs/transfer/noclip_3886275/cosmo_000122_corrected_noclip.npz \
  --shells 3 10 30 --n-avg 4 --n-iter 5 --damp 0.4 --seed 0 \
  --out "$OUT/poisson_convergence.png"

echo "[$(date --iso-8601=seconds)] done -> $OUT"
