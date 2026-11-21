#!/bin/bash
#SBATCH -e sMD-analysis-3ptb.err
#SBATCH -o sMD-analysis-3ptb.out
# --- EDIT FOR YOUR CLUSTER: partition, memory, env activation ---
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-0
#SBATCH --partition=shared,highmem
#SBATCH --job-name="sMD-analysis-3ptb"

source ~/.bashrc
micromamba activate autopath
# ------------------------------------------------------------------------------

python analyze_sMD.py --sysname 3ptb --lig ../data/3ptb.sdf
