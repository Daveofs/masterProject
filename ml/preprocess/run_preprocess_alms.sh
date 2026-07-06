#!/bin/bash
#SBATCH --job-name=preprocess_alms
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=0
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/preprocess_alms/preprocess_alms_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/preprocess_alms/preprocess_alms_%j.err

# Activate environment
source /users/damrein/miniforge3/bin/activate

DATA_DIR="/capstor/scratch/cscs/damrein/cosmogridv1"

# Run script
python /users/damrein/masterProject/ml/preprocess/preprocess_alms.py \
    --data-dir "$DATA_DIR" \
    --lmax 1536 \
    --num-workers 30 \
