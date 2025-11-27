#!/bin/bash
#SBATCH -e 3ptb/3ptb_rep1.err
#SBATCH -o 3ptb/3p3ptb_rep1tb.out
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=forli,alphafold
#SBATCH --job-name="3ptb_rep1"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath3

 python ap_vanillaMD.py --sysname 3ptb\
  --ligname UNK --replica rep1