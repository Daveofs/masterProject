#!/bin/bash
#SBATCH --job-name=apply_flow
#SBATCH --account=sk037
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/capstor/scratch/cscs/damrein/outputs/logs/apply_flow/apply_flow_%j.out
#SBATCH --error=/capstor/scratch/cscs/damrein/outputs/logs/apply_flow/apply_flow_%j.err

source /users/damrein/miniforge3/bin/activate

# Parallelize healpy SHTs
export OMP_NUM_THREADS=32

MODEL="/capstor/scratch/cscs/damrein/outputs/flow_matching/3740493/cosmo_000001/model/flow_mlp.pth"
INPUT="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/shells_nside=2048.npz"
PARAMS="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/params.yml"
SHELL_INDEX=-1
STEPS=25
DEVICE="cuda:0"
OUT="/capstor/scratch/cscs/damrein/cosmogridv1_test2/cosmo_000001/run_0/shells_nside=2048_corrected.npz"

python /users/damrein/masterProject/ml/apply_flow_correction.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --params "$PARAMS" \
    --shell-index "$SHELL_INDEX" \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --out "$OUT"
