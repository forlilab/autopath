#!/bin/bash
#SBATCH -e WTmetaD-3ptb.err
#SBATCH -o WTmetaD-3ptb.out
#SBATCH --gres=gpu#:rtxa6000:1
#SBATCH --time=1-0
#SBATCH --exclude=nodea0110,nodea0111,nodec0821,nodea0410
#SBATCH --partition=forli-pro,alphafold,forli
#SBATCH --job-name="WTmetaD-3ptb"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

# module load cuda/12.4

echo "Node allocated: $(hostname)"
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA version: $(nvcc --version)"


source ~/.bashrc
micromamba activate autopath
python ap_WTmetaD_COM.py --sysname 3ptb\