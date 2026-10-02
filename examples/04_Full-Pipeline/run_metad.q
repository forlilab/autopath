#!/bin/bash
#SBATCH -e 3ptb_metad.err
#SBATCH -o 3ptb_metad.out
#SBATCH --partition=forli,forli-pro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0
# exclude the slow GPU nodes
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="3ptb_metad"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

source ~/.bashrc
micromamba activate autopath

python run_milestones_metad.py --rec ../data/3ptb.pdb --lig ../data/3ptb.sdf \
    --protocol ../../autopath/data/eq_lig-prot_5ns_4fs.json
