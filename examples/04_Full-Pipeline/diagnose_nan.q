#!/bin/bash
#SBATCH -e 3ptb_nan_%j.err
#SBATCH -o 3ptb_nan_%j.out
#SBATCH --partition=forli,forli-pro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-4
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="3ptb_nan"

source ~/.bashrc
micromamba activate autopath
python diagnose_nan.py --arm "$ARM" --ns 5
