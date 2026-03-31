#!/bin/bash
#SBATCH -e sMD-3ptb.err
#SBATCH -o sMD-3ptb.out
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=alphafold,forli
#SBATCH --job-name="sMD-3ptb"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath

python ap_sMD.py --sysname 3ptb\