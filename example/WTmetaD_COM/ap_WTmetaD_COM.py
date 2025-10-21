import os
import logging
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import SystemPreparation, Equilibration, MetadynamicsMD
from autopath.analysis import plot_atomic_rmsf
from autopath.utils import fix_pdb, save_pdb, load_system, align_trajectory, calculate_com_distance, compute_rmsd
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
    pocket_residues = [218, 219, 262, 263, 305, 306, 49, 50, 91, 92, 133, 134, 175, 176]  # Change this to your pocket residue IDs

    # Setup logging
    os.makedirs(sys_name, exist_ok=True)
    logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{sys_name}/{sys_name}.log", mode="a"),
        logging.StreamHandler(),
    ],
    )
    
    checkpoint = f'../equilibration/{sys_name}/equilibration/checkpoint_equil_{sys_name}.chk'
    system_fname = f'../equilibration/{sys_name}/equilibration/system_equil_{sys_name}.xml'
    system = load_system(system_fname)
    prmtop_fname = f'../equilibration/{sys_name}/system.prmtop'
    # topology = AmberPrmtopFile(prmtop_fname).topology
    pdb_fname = f'../equilibration/{sys_name}/equilibration/{sys_name}_equilibrated.pdb'
    topology = PDBFile(pdb_fname).topology

    u = mda.Universe(pdb_fname)
    pocket_atoms = u.select_atoms(f'resid {" ".join(map(str, pocket_residues))} and name CA')
    pocket_atoms_idx = [a.index for a in pocket_atoms]
    ligand_atoms = u.select_atoms(f'resname {ligand_resname} and not name H*')
    ligand_atoms_idx = [a.index for a in ligand_atoms]
    com_dist = calculate_com_distance(u, ligand_atoms, pocket_atoms, wrap=False)[0]/10
    print(f'Initial COM distance is {com_dist:.2f} nm') # We'll use this for set the boundaries later
    MIN_COM = com_dist * 0.75
    MAX_COM = com_dist * 3.0

    ########################################################################################
    ############################## Well-tempered MD - COM CV ###############################
    ########################################################################################

    metad =  MetadynamicsMD(
        pocket_atoms=pocket_atoms_idx,
        ligand_atoms=ligand_atoms_idx,
        topology=topology,
        out_dir=f"{sys_name}/WTmetaD_COM",
        timestep=0.004,
        is_membrane=False
    )

    for walker in range(1, N_WALKERS+1):
        rep_id = metad.run(
                system=system,
                checkpoint_file=checkpoint,
                mMD_CV='com',
                mMD_time=1, #ns
                bias_factor=10,
                hill_height=1.2, #kJ/mol approx 0.5 KbT 0.3 Kcal/mol
                hill_width=0.05, #nm
                biasFrequency=2, #ps
                saveFrequency=50,
                grid_dimensions=(MIN_COM, MAX_COM), #nm
                grid_points=125
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
            logging.warning(f"Superposition failed: {e}. Proceeding without superposition.")
        traj.save(traj_fname.replace(".dcd", "_aligned.dcd"))
        # os.remove(smd_traj)

    return

if __name__ == "__main__":
    main()
