#!/bin/bash
#SBATCH --job-name=apply_flow
#SBATCH --account=sk037
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/apply_flow/apply_flow_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/apply_flow/apply_flow_%j.err

# Activate environment
source /users/damrein/miniforge3/bin/activate

# Arguments
MODEL="/capstor/scratch/cscs/damrein/outputs/flow_matching/3704630/flow_mlp.pth"
INPUT="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/shells_nside=2048.npz"
PARAMS="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/params.yml"
NSIDE_PATCH=16
SHELL_INDEX=-1  
T=0.0
BATCH_SIZE=256
DEVICE="cpu"
OUT="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/shells_nside=2048_corrected.npz"

# Run script
python /users/damrein/masterProject/ml/apply_flow_correction.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --params "$PARAMS" \
    --nside-patch "$NSIDE_PATCH" \
    --shell-index "$SHELL_INDEX" \
    --t "$T" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --out "$OUT" \
