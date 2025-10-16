#!/bin/bash
#SBATCH -e sMD.err
#SBATCH -o sMD.out
#SBATCH --gres=gpu#:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=alphafold,forli
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="sMD-2hu4"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath3
python autopath_equilibration.py --rec ../data/2hu4.pdb --lig ../data/2hu4.sdf --protocol equilibration_lig_prot_memb.json\
 # resname and smiles are optional
#  --resname UNK\
#  --smiles 'CCC(CC)O[C@@H]1C=C(C[C@@H]([C@H]1NC(=O)C)N)C(=O)[O-]'\