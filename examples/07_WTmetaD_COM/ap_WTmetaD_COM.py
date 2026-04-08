import os
import logging
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import MetadynamicsMD
from autopath.cv import com_cv
from autopath.utils import load_system, calculate_com_distance
from autopath.utils import setup_logging

from openmm.app import PDBFile

def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs Well-Tempered Metadynamics MD simulation (WTmetaD) for a protein-ligand complex.",
        epilog="""
        REPORTING BUGS
                Please report bugs to:
                AutoDock mailing list   http://autodock.scripps.edu/mailing_list\n

        COPYRIGHT
                Copyright (C) 2025 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )

    parser.add_argument(
        "-s",
        "--sysname",
        dest="sysname",
        required=True,
        action="store",
        help="name for the system, all output will be stored in a folder with this name",
    )

    return parser.parse_args()


def main():

    args = cmd_lineparser()

    sys_name = args.sysname # results will be written to a folder with this name

    # Run multiple walkers WTmetaD in a sequential mode. I you have more ligands that GPUs, this is the best way.
    # If you only have few ligands it might be faster to spread the walkers across GPUs and run this in parallel.
    N_WALKERS = 3
    ligand_resname = "UNK"  # Change this to your ligand residue name
    pocket_selection = '(resid 145-153 183-190) and name CA'
    
    # Setup logging
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")
    
    checkpoint = f'../01_Build_and_Equilibrate/{sys_name}/equilibration/checkpoint_equil_{sys_name}.chk'
    system_fname = f'../01_Build_and_Equilibrate/{sys_name}/equilibration/system_equil_{sys_name}.xml'
    prmtop_fname = f'../01_Build_and_Equilibrate/{sys_name}/system.prmtop'
    pdb_fname = f'../01_Build_and_Equilibrate/{sys_name}/equilibration/{sys_name}_equilibrated.pdb'
    # topology = AmberPrmtopFile(prmtop_fname).topology
    
    topology = PDBFile(pdb_fname).topology
    system = load_system(system_fname)

    u = mda.Universe(pdb_fname)
    pocket_atoms = u.select_atoms(pocket_selection)
    pocket_atoms_idx = [a.index for a in pocket_atoms]
    ligand_atoms = u.select_atoms(f'resname {ligand_resname} and not name H*')
    ligand_atoms_idx = [a.index for a in ligand_atoms]
    com_dist = calculate_com_distance(u, ligand_atoms, pocket_atoms, wrap=False)[0]/10
    logger.info(f'Initial COM distance is {com_dist:.2f} nm') # We'll use this for set the boundaries later
    MIN_COM = com_dist * 0.75
    MAX_COM = com_dist * 2.0
    logger.info(f'Setting COM CV boundaries to {MIN_COM:.2f} nm and {MAX_COM:.2f} nm')
    
    ########################################################################################
    ############################## Well-tempered MD - COM CV ###############################
    ########################################################################################

    metad =  MetadynamicsMD(
        pocket_atoms=pocket_atoms_idx,
        ligand_atoms=ligand_atoms_idx,
        topology=topology,
        out_dir=f"{sys_name}/WTmetaD_COM",
        timestep=0.004,
        is_membrane=False,
        # platform="CUDA",
    )

    # here is where you define your CV. In this case, we use a simple distance between the COM of the ligand and the COM of the pocket.
    # The com_cv here is an instance of the CVSpec class, which is a wrapper around the CV definition that MetadynamicsMD expects. 
    # check cv.py to see examples of how to define your own CVs, and some pre-packed examples like this one.
    cv = com_cv(
        pocket_atoms=pocket_atoms_idx,
        ligand_atoms=ligand_atoms_idx,
        grid_min=MIN_COM,
        grid_max=MAX_COM,
        hill_width=0.025,   # nm
        grid_points=100,
    )

    for walker in range(1, N_WALKERS+1):
        rep_id = metad.run(
                system=system,
                checkpoint_file=checkpoint,
                cv_specs=[cv],
                run_id=f'{walker}',
                mMD_time=2,       # ns
                bias_factor=12,  # WTmetaD bias factor
                hill_height=1.2,  # kJ/mol approx 0.5 KbT
                biasFrequency=2,  # ps
                saveFrequency=50,
            )

        ####################### Post-processing ######################
        ### Wrap, align and save the clean trajectory
        traj_fname = f"{sys_name}/WTmetaD_COM/WTMetaD_{rep_id}.dcd"
        traj = md.load(traj_fname, top=prmtop_fname)
        traj = traj.center_coordinates()
        traj = traj.image_molecules()
        try: # if there's no protein
            backbone = traj.topology.select("backbone")
            traj = traj.superpose(traj[0], atom_indices=backbone)
        except Exception as e:
            logger.warning(f"Superposition failed: {e}. Proceeding without superposition.")
        traj.save(traj_fname.replace(".dcd", "_aligned.dcd"))
        os.remove(traj_fname) # remove the unaligned trajectory to save space

    return

if __name__ == "__main__":
    main()
