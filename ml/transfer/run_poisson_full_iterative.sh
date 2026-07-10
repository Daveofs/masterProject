#!/bin/bash
#SBATCH --job-name=pois-full-iter
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/transfer/pois-full-iter-%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/transfer/pois-full-iter-%j.err
# Full 69-shell lognormal+Poisson count map with the VALIDATED iterative recipe
# (smooth ell_c taper + damped/averaged per-ell Cl calibration). Measured on
# shells 3/10/30: Cl_counts/Cl_high within ~1-7% at every band tested, mean exact,
# sparsity matches truth to <1pp, non-negative integer counts. Built from the
# --no-clip (Cl-optimal, continuous, can-go-negative) transfer-corrected field.
source /users/damrein/miniforge3/etc/profile.d/conda.sh
conda activate deepSphere
export OMP_NUM_THREADS=64
OUT=/capstor/scratch/cscs/damrein/outputs/transfer/poisson_iter_${SLURM_JOB_ID}
mkdir -p "$OUT"
python /users/damrein/masterProject/ml/transfer/poisson_resample.py \
  --corrected /capstor/scratch/cscs/damrein/outputs/transfer/noclip_3886275/cosmo_000122_corrected_noclip.npz \
  --run-dir /capstor/scratch/cscs/damrein/cosmogridv1/cosmo_000122/run_0 \
  --lmax 3000 --nside 2048 --ell-c 300 --taper 100 --n-avg 4 --n-iter 5 --damp 0.4 --seed 0 \
  --out "$OUT/cosmo_000122_counts.npz"
echo "[$(date --iso-8601=seconds)] done -> $OUT"
