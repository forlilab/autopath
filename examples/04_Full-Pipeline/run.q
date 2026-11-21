#!/bin/bash
# --- EDIT FOR YOUR CLUSTER: partition, GPU type, exclude list, env activation ---
#SBATCH -e 3ptb.err
#SBATCH -o 3ptb.out
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=3-0
#SBATCH --partition=alphafold,forli
#SBATCH --job-name="AutoPath-3ptb"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath
# --- end cluster-specific block ---

# ../data/3ptb.* are the shipped example inputs; replace with your own files.
# The protocol defaults to the packaged eq_lig-prot_5ns_4fs.json (--protocol to override).
# --pocket_resids takes residue numbers of the ORIGINAL receptor PDB (3ptb: trypsin S1 site).
python run_AutoPath.py --rec ../data/3ptb.pdb --lig ../data/3ptb.sdf \
    --pocket_resids 133 134 135 136 137 138 139 140 156 157 158 159 160 161 162 \
                    180 181 182 183 196 197 198 199 200 201 204 205 206 207 208 \
                    209 210 211 212 213 214 215 226 227 228 229 230
# add --max_polar_cwrmsf <A> to skip pulling for unstable poses
