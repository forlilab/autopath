#!/bin/bash
#SBATCH -e sMD.err
#SBATCH -o sMD.out
#SBATCH --gres=gpu#:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=alphafold
#SBATCH --job-name="WTmetaD-2hu4"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath2
python run_WTmetaD.py --rec ../data/2hu4.pdb --lig ../data/2hu4.sdf