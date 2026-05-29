#!/bin/bash
#SBATCH -e 3ptb/mmgbsa_igb8_WAT/3ptb_mmpbsa.err
#SBATCH -o 3ptb/mmgbsa_igb8_WAT/3ptb_mmpbsa.out
#SBATCH --time=0:10:00
#SBATCH --partition=highmem,shared,gpu
#SBATCH --exclude=nodea0111,nodea0110 # EXCLUDE KNOWN PROBLEMATIC NODES
#SBATCH --ntasks=222  # Request 32 separate MPI processes/slots
#SBATCH --cpus-per-task=1 # Each process uses 1 CPU. for MPI runs
#SBATCH --job-name="mmpbsa_3ptb"

# module purge
module load openmpi/3.1.6
# module load gcc

source ~/.bashrc
micromamba activate autopath

module load amber/24
#export OMP_NUM_THREADS=222

echo "Starting mmpbsa calculation for 3ptb at $(date)"
echo "Running on $(hostname)"
echo "Entering output directory 3ptb/mmgbsa_igb8_WAT ..."
cd 3ptb/mmgbsa_igb8_WAT

echo "Running ante-mmpbsa to generate prmtop files..."
ante-MMPBSA.py -p /gpfs/home/mllanos/forlilab/autopath/examples/01_Build_and_Equilibrate/3ptb/system.prmtop -s "((:WAT,HOH)&!(:233))|:POP:NA:CL:K:MG" -n :UNK --radii mbondi3 -c complex.prmtop -r receptor.prmtop -l ligand.prmtop

echo "Finished ante-mmpbsa at $(date)"
echo "Running mmpbsa.py for trajectory /gpfs/home/mllanos/forlilab/autopath/examples/01_Build_and_Equilibrate/3ptb/equilibration/equilibration_3ptb_aligned.dcd ..."

mpirun -np 222 --display-allocation MMPBSA.py.MPI -O -i /gpfs/home/mllanos/forlilab/autopath/examples/02_iAnalysis/3ptb/mmgbsa_igb8_WAT/3ptb_mmgbsa.in -o FINAL_RESULTS_mmpbsa.dat -do FINAL_DECOMP_mmpbsa.dat -sp /gpfs/home/mllanos/forlilab/autopath/examples/01_Build_and_Equilibrate/3ptb/system.prmtop -y /gpfs/home/mllanos/forlilab/autopath/examples/01_Build_and_Equilibrate/3ptb/equilibration/equilibration_3ptb_aligned.dcd -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop
