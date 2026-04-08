#!/bin/bash
#SBATCH -e WTmetaD-3ptb.err
#SBATCH -o WTmetaD-3ptb.out
#SBATCH --gres=gpu#:rtxa6000:1
#SBATCH --time=1-0
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --partition=forli-pro,alphafold,forli
#SBATCH --job-name="WTmetaD-3ptb"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

# module load cuda/12.9

source ~/.bashrc
micromamba activate autopath
python ap_WTmetaD_COM.py --sysname 3ptb\