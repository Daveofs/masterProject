#!/bin/bash
#SBATCH --job-name=pkdgrav
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --output=outputs/pkdgrav_%j.out
#SBATCH --error=outputs/pkdgrav_%j.err

cd /cluster/home/damrein/pkdgrav/pkdgrav3_dev-master/build
srun ./pkdgrav3 /cluster/home/damrein/project/cosmogridv1/cosmo_000001/param_files/cosmology.par