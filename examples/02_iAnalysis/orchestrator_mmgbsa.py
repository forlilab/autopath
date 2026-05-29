import os
import shutil
from autopath.ap_PLIP import ProteinLigandAnalyzer
import MDAnalysis as mda
from glob import glob

EXPERIMENT = 'mmgbsa_igb8_WAT'

trajectories = glob(f"../01_Build_and_Equilibrate/*/equilibration/equilibration_*_aligned.dcd")

print(f"Found {len(trajectories)} systems to process.")

for trajectory in trajectories:
    sysname = os.path.basename(os.path.dirname(os.path.dirname(trajectory)))
    print(f"Processing system: {sysname}")
    
    output_folder = f"{sysname}/{EXPERIMENT}"
    # if os.path.exists(output_folder):
    #     print(f"Removing existing directory {output_folder} for a fresh start.")
    #     shutil.rmtree(output_folder)
        
    topology = f"../01_Build_and_Equilibrate/{sysname}/system.prmtop"
        
    ProteinLigandAnalyzer.prepare_mmgbsa_batch(sysname, topology, trajectory,
                    output_folder=output_folder,
                    mmpbsa_in="mmgbsa_igb8.in",
                    radii='mbondi3',
                    slurm_template_fname=None, #'mmgbsa_slurm_template.q', #if None will get it from the package
                    persistent_waters_cutoff=0.9,
                    # traj_slice=(0, 221, 1),
                    #  mpi_threads=64
                    )