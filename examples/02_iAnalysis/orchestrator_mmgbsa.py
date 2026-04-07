import os
import shutil
from autopath.ap_PLIP import ProteinLigandAnalyzer
import MDAnalysis as mda

EXPERIMENT = 'mmgbsa_igb8_WAT'
systems = ['3ptb']

for sysname in systems:
    print(f"Processing system: {sysname}")
    output_folder = f"{sysname}/{EXPERIMENT}"
    
    # if os.path.exists(output_folder):
    #     print(f"Removing existing directory {output_folder} for a fresh start.")
    #     shutil.rmtree(output_folder)
        
    prmtop = f"../01_Build_and_Equilibrate/{sysname}/system.prmtop"
    traj = f"../01_Build_and_Equilibrate/{sysname}/equilibration/equilibration_{sysname}_aligned.xtc"
    # ProteinLigandAnalyzer.prepare_mmpbsa_batch(sysname, prmtop, traj,
    #                 output_folder=output_folder,
    #                 mmpbsa_in="mmgbsa_igb8.in",
    #                 radii='mbondi3',
    #                 slurm_template_fname=None, #will get it from the package
    #                 persistent_waters_cutoff=0.9,
    #                 # traj_slice=(0, 221, 1),
    #                 #  mpi_threads=64
    #                 )
    
    aplip = ProteinLigandAnalyzer(top=prmtop, trajs=[traj],
                                ligand_mda_selection='resname UNK',
                                protein_mda_selection='protein',
                                outdir=f"{sysname}/lie")
    
    print("Computing LIE for all residues...")
    lie_df = aplip.compute_LIE(use_residues=None,
                            ligand_amber_selection=':UNK'
                            )

    aplip.plot_lie_components(lie_df)
    lie_df.to_csv(f"{sysname}/lie/LIE_ALL.csv")