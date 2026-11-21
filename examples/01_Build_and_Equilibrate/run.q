#!/bin/bash
# --- EDIT FOR YOUR CLUSTER: partition, GPU type, exclude list, env activation ---
#SBATCH -e 3ptb.err
#SBATCH -o 3ptb.out
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1-0
#SBATCH --partition=alphafold,forli-pro,forli
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="3ptb_equilibration"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath
# --- end cluster-specific block ---

# ../data/3ptb.* are the shipped example inputs; replace with your own files.
# The protocol defaults to the packaged eq_lig-prot_5ns_4fs.json (--protocol to override).
python ap_equilibration.py --rec ../data/3ptb.pdb --lig ../data/3ptb.sdf

# optional: --sysname NAME (default: ligand stem), --outdir DIR, --resname UNK,
#  --smiles 'CCC(CC)O[C@@H]1C=C(C[C@@H]([C@H]1NC(=O)C)N)C(=O)[O-]'
