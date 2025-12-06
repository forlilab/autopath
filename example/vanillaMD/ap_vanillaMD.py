import os
import logging
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import VanillaMD
from autopath.analysis import plot_atomic_rmsf
from autopath.utils import load_system, compute_rmsd
from openmm.app import PDBFile

def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs vanilla MD of a system, that has been previously equilibrated.",
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
        help="Name of the system, used to create output directory and files",
    )
    
    parser.add_argument(
        "-n",
        "--ligname",
        dest="ligname",
        required=False,
        default=None,
        help="Residue name of the ligand, if present.",
    )

    parser.add_argument(
        '-r',
        '--replica',
        dest='replica',
        required=False,
        default=None,
        help='Replica name, used to differentiate multiple runs on the same system.'
    )
    return parser.parse_args()


def main():

    args = cmd_lineparser()
    lig_resname = args.ligname
    sys_name = args.sysname
    replica = args.replica
    if replica is None:
        run_id = sys_name
    else:
        run_id = f'{sys_name}_{replica}'
    
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
    ######################################## Vanilla MD ####################################
    ########################################################################################
    
    MD_TIME = 100  # ns
    TIMESTEP = 0.004  # 4 fs timestep
    TEMPERATURE = 300  # 300 K
    SAVE_FREQ = 25000  # save /0.1ns at 4fs
    RESTART_VELOCITIES = True  # whether to restart velocities from checkpoint
    RESTRAINED_ATOMS = None  # list of atom indices to restrain, or None for no restraints
    
    system_pdb_file = f"{sys_name}/system.pdb"
    system_prmtop = f"{sys_name}/system.prmtop"
    topology = PDBFile(system_pdb_file).topology
    
    system = load_system(f"{sys_name}/equilibration/system_equil_{sys_name}.xml")
    checkpoint = f"{sys_name}/equilibration/checkpoint_equil_{sys_name}.chk"

    vanilla_md = VanillaMD(
        system=system,
        topology=topology,
        restrained_atoms=RESTRAINED_ATOMS,
        timestep=TIMESTEP,
        temperature=TEMPERATURE,
        save_freq=SAVE_FREQ,
        out_dir=f"{sys_name}/MD"
        )
    
    system = vanilla_md.run(
                            checkpoint_file=checkpoint,
                            pdb_file=None, 
                            run_id=run_id,
                            MD_time=MD_TIME,
                            restart_velocities=RESTART_VELOCITIES
                            )

    traj_fname = f"{sys_name}/MD/MD_{run_id}.dcd"
    
    # Wrap, align and save the clean trajectory
    # Adjust based on your needs!
    traj = md.load(traj_fname, top=system_prmtop)
    traj = traj.center_coordinates()
    traj = traj.image_molecules()
    try: # if there's no protein
        backbone = traj.topology.select("backbone")
        traj = traj.superpose(traj[0], atom_indices=backbone)
    except Exception as e:
        logging.warning(f"Superposition failed: {e}. Proceeding without superposition.")
    traj.save(traj_fname.replace(".dcd", "_aligned.dcd"))
    os.remove(traj_fname)
    logging.info(f"Aligned trajectory saved to {traj_fname.replace('.dcd', '_aligned.dcd')}")
    
    # Calculate RMSD and RMSF of the ligand
    u = mda.Universe(system_prmtop, traj_fname.replace(".dcd", "_aligned.dcd"), in_memory=True)
    # u_ref = mda.Universe(system_prmtop, system_pdb_file) # you can use other references here
    
    rmsd_df = compute_rmsd(u, u,
                            alig_select="backbone", 
                            groupselections={
                                            "ligand":f"resname {lig_resname} and not name H*" if lig_resname is not None else None,
                                            "protein":'protein and not name H*'
                                            },
                            plots_outdir=f"{sys_name}/MD",
                            suffix=run_id,
                            )
    rmsd_df.to_csv(f"{sys_name}/MD/{run_id}_rmsd.csv", index=False)
    
    if lig_resname is not None:
        plot_atomic_rmsf(u, outname=f"{sys_name}/MD/{run_id}_RMSF.png", log_rmsf=True)

    return

if __name__ == "__main__":
    main()
