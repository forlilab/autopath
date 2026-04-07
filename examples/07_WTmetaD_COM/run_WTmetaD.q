#!/bin/bash
#SBATCH -e 6e23_A.err
#SBATCH -o 6e23_A.out
#SBATCH --gres=gpu#:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=alphafold,forli
#SBATCH --job-name="WTmetaD-6e23_A"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath3
python ap_WTmetaD_COM.py --sysname 6e23_A\