import os
import logging
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import SteeredMD
from autopath.utils import load_system
from openmm.app import PDBFile, AmberPrmtopFile

def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs steered molecular dynamics simulation (sMD) for a protein-ligand complex.",
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
    sys_name = args.sysname
    
    N_REPS = 50 # how many pulling replicates to run
    DIRECTION = 'forward' # 'forward' or 'backward'
    SPEED = 0.0005 #nm/ps
    sMD_spring_cte_per_atom = 50 * 4.184  # KJ/mol/nm2, converted from kcal. This affects thermal fluctuations

    ligand_resname = "UNK"  # Change this to your ligand residue name
    # pocket_residues = [218, 219, 262, 263, 305, 306, 49, 50, 91, 92, 133, 134, 175, 176]  # Change this to your pocket residue IDs
    pocket_residues = [134, 135, 136, 137, 138, 139, 140, 157, 158, 159, 160, 161, 162, 180,
                       181, 182, 183, 211, 212, 213, 214, 215, 226, 227, 228, 229]
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
    
    os.makedirs(sys_name, exist_ok=True)

    logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{sys_name}/{sys_name}.log", mode="a"),
        logging.StreamHandler(),
    ],
    )
    
    ########################################################################################
    ###################################### Steered MD ######################################
    ########################################################################################

    sMD_spring_cte = sMD_spring_cte_per_atom * len(ligand_atoms_idx)  # Normalize by ligand size

    # Run steered MD
    sMD = SteeredMD(
        system=system,
        topology=topology,
        groupA_atoms=ligand_atoms_idx,
        groupB_atoms=pocket_atoms_idx,
        restrained_atoms=None, #restrained_atoms_indices,
        restart_velocities=True,
        autostop_freq=50,
        out_dir=f"{sys_name}/sMD",
    )

    for rep_idx in range(1, N_REPS+1):
        rep_id = sMD.run(
                # max_time=1500, #ps
                max_displacement=2.5, #nm 
                # steps_per_move=2500,
                dx_per_move=0.001, #nm
                pulling_speed=SPEED, #nm/ps
                sMD_spring_cte=sMD_spring_cte,
                checkpoint_file=checkpoint,
                pdb_file=None,
                pulling_direction=DIRECTION,
                )
    

        ####################### Post-processing ######################
        ### Wrap, align and save the clean trajectory
        smd_traj = f"{sys_name}/sMD/sMD_{rep_id}.dcd"
        traj = md.load(smd_traj, top=prmtop_fname)
        traj = traj.center_coordinates()
        traj = traj.image_molecules()
        try: # if there's no protein
            backbone = traj.topology.select("backbone")
            traj = traj.superpose(traj[0], atom_indices=backbone)
        except Exception as e:
            logging.warning(f"Superposition failed: {e}. Proceeding without superposition.")
        traj.save(smd_traj.replace(".dcd", "_aligned.dcd"))
        # os.remove(smd_traj)
    
    return

if __name__ == "__main__":
    main()
