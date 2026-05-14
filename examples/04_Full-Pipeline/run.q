#!/bin/bash
#SBATCH -e 3ptb.err
#SBATCH -o 3ptb.out
##SBATCH --gres=gpu#:rtxa6000:1
#SBATCH --time=1-0
#SBATCH --partition=alphafold,forli,forli-pro
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="3ptp"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath

python run_AutoPath.py --rec ../data/3ptb.pdb --lig ../data/3ptb.sdf\
    --protocol ../../autopath/data/eq_lig-prot_5ns_4fs.json\
