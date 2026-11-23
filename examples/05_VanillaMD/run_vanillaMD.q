#!/bin/bash
# Logs go to the submit directory: <sysname>/ does not exist before the first run
#SBATCH -e 3ptb_rep1.%j.err
#SBATCH -o 3ptb_rep1.%j.out
# --- EDIT FOR YOUR CLUSTER: partition, GPU type, exclude list, env activation ---
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1-0
#SBATCH --partition=forli,alphafold
#SBATCH --job-name="3ptb_rep1"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath
# ------------------------------------------------------------------------------

python ap_vanillaMD.py --sysname 3ptb \
  --resname UNK --replica rep1
