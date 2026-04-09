#!/bin/bash
#SBATCH -e 3ptb/lie/3ptb.err
#SBATCH -o 3ptb/lie/3ptb.out
#SBATCH --time=1-0
#SBATCH --partition=forli,forli-pro,alphafold,shared
#SBATCH --job-name="LIE_3ptb"

source ~/.bashrc
micromamba activate autopath
python run_LIE.py -s 3ptb
