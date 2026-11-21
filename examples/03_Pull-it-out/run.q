#!/bin/bash
#SBATCH -e sMD-3ptb.err
#SBATCH -o sMD-3ptb.out
# --- EDIT FOR YOUR CLUSTER: partition, GPU type, exclude list, env activation ---
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=alphafold,forli
#SBATCH --job-name="sMD-3ptb"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath
# ------------------------------------------------------------------------------

# Pulls only; submit run_analysis.q after this job
python ap_sMD.py --sysname 3ptb --lig ../data/3ptb.sdf --reps 10
