#!/bin/bash
#SBATCH -e toy-Open-HW.err
#SBATCH -o toy-Open-HW.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --partition=shared,highmem
#SBATCH --job-name="toy-Open-HW"

source ~/.bashrc
micromamba activate autopath
export OPENMM_CPU_THREADS=1 OMP_NUM_THREADS=1

python run_toy.py --scenario Open-HW
