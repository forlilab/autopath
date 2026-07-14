import os
import logging
import argparse

import MDAnalysis as mda

from autopath import SteeredMD
from autopath.utils import load_system, setup_logging, wrap_align_save_traj
from openmm.app import PDBFile, AmberPrmtopFile


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs steered molecular dynamics simulation (sMD) for a protein-ligand complex.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
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
        
    DATADIR = f'/gpfs/home/mllanos/forlilab/autopath/examples/01_Build_and_Equilibrate/{sys_name}'
    N_REPS = 10 # how many pulling replicates to run
    DIRECTION = 'forward' # 'forward' or 'backward'
    SPEED = 0.01 #nm/ps
    DX_PER_MOVE = 0.001 #nm, how often to update the reference position for the spring
    SPRING_CTE = 50 * 4.184  # KJ/mol/nm2/atom, converted from kcal. This affects thermal fluctuations
    
    # Setup logging
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")
    logger.info(f"Starting steered MD simulations for {sys_name} with {N_REPS} replicates in {DIRECTION} direction.")
    
    ligand_resname = "UNK"  # Change this to your ligand residue name
    pocket_residues = [134, 135, 136, 137, 138, 139, 140, 157, 158, 159, 160, 161, 162, 180,
                       181, 182, 183, 211, 212, 213, 214, 215, 226, 227, 228, 229]
    
    checkpoint = f'{DATADIR}/equilibration/checkpoint_equil_{sys_name}.chk'
    system_fname = f'{DATADIR}/equilibration/system_equil_{sys_name}.xml'
    system = load_system(system_fname)
    prmtop_fname = f'{DATADIR}/system.prmtop'
    pdb_fname = f'{DATADIR}/equilibration/{sys_name}_equilibrated.pdb'
    topology = PDBFile(pdb_fname).topology
    
    u = mda.Universe(pdb_fname)
    pocket_atoms = u.select_atoms(f'resid {" ".join(map(str, pocket_residues))} and name CA')
    pocket_atoms_idx = [a.index for a in pocket_atoms]
    ligand_atoms = u.select_atoms(f'resname {ligand_resname} and not name H*')
    ligand_atoms_idx = [a.index for a in ligand_atoms]
    
    os.makedirs(sys_name, exist_ok=True)
    
    ########################################################################################
    ###################################### Steered MD ######################################
    ########################################################################################

    sMD_spring_cte = SPRING_CTE * len(ligand_atoms_idx)  # Normalize by ligand size
    logger.info(f"sMD spring constant set to {sMD_spring_cte} KJ/mol/nm2 for ligand of {len(ligand_atoms_idx)} atoms.")
    
    # Run steered MD
    sMD = SteeredMD(
        system=system,
        topology=topology,
        groupA_atoms=ligand_atoms_idx,
        groupB_atoms=pocket_atoms_idx,
        restrained_atoms=None,
        restart_velocities=True,
        out_dir=f"{sys_name}/sMD",
        dx_per_move=DX_PER_MOVE,
        max_displacement=2.5,
        sMD_spring_cte=sMD_spring_cte,
        # Log per-frame geometry (contacts, min-distance, ligand shape) to
        # sMD_<run_id>_geom.dat at the DCD cadence, reusing positions already
        # pulled for autostop. These complement the trace features in
        # clustering (AnalysisSMD.run(..., geom_features=True)) without a
        # post-processing pass over the trajectories. Pass a list to select a
        # subset, e.g. log_geom_features=["nc", "rog", "npr1", "npr2"].
        log_geom_features=True,
    )

    for rep_idx in range(1, N_REPS+1):
        rep_id = sMD.run(
                pulling_speed=SPEED, #nm/ps
                checkpoint_file=checkpoint,
                pulling_direction=DIRECTION,
                )
        
        logger.info(f"Completed sMD replicate {rep_idx}/{N_REPS} with ID: {rep_id}")

        ####################### Post-processing ######################
        ########## Wrap, align and save the clean trajectory #########
        ##############################################################
        
        smd_traj = f"{sys_name}/sMD/sMD_{rep_id}.dcd"
        wrap_align_save_traj(smd_traj, prmtop_fname)
    
    return

if __name__ == "__main__":
    main()
