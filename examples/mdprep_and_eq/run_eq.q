#!/bin/bash
#SBATCH -e sMD-3ptb.err
#SBATCH -o sMD-3ptb.out
#SBATCH --gres=gpu#:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=alphafold,forli
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="sMD-3ptb"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath3
python ap_equilibration.py --rec ../data/3ptb.pdb --lig ../data/3ptb.sdf --protocol eq_lig-prot_5ns_4fs.json\
# resname and smiles are optional
#  --resname UNK\
#  --smiles 'CCC(CC)O[C@@H]1C=C(C[C@@H]([C@H]1NC(=O)C)N)C(=O)[O-]'\
