#!/bin/bash
#SBATCH -e ${out_dir}/${sysname}_mmgbsa.err
#SBATCH -o ${out_dir}/${sysname}_mmgbsa.out
#SBATCH --time=${time}
#SBATCH --partition=${partition}
#SBATCH --exclude=nodea0111,nodea0110 # EXCLUDE KNOWN PROBLEMATIC NODES
#SBATCH --ntasks=${omp_threads}  # Request 32 separate MPI processes/slots
#SBATCH --cpus-per-task=1 # Each process uses 1 CPU. for MPI runs
#SBATCH --job-name="mmgbsa_${sysname}"

# module purge
module load openmpi/3.1.6
# module load gcc

source ~/.bashrc
micromamba activate autopath

module load amber/24
#export OMP_NUM_THREADS=${omp_threads}

echo "Starting mmgbsa calculation for ${sysname} at $$(date)"
echo "Running on $$(hostname)"
echo "Entering output directory ${out_dir} ..."
cd ${out_dir}

echo "Running ante-mmpbsa to generate prmtop files..."
ante-MMPBSA.py -p ${system_prmtop} -s "${strip_selection}" -n ${lig_selection} --radii ${radii} -c complex.prmtop -r receptor.prmtop -l ligand.prmtop

echo "Finished ante-mmpbsa at $$(date)"
echo "Running mmgbsa.py for trajectory ${trajectory} ..."

mpirun -np ${omp_threads} --display-allocation MMPBSA.py.MPI -O -i ${mmpbsa_in} -o FINAL_RESULTS_mmgbsa.dat -do FINAL_DECOMP_mmgbsa.dat -sp ${system_prmtop} -y ${trajectory} -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop
