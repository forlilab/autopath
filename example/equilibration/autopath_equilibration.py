import os
import logging
import argparse

import MDAnalysis as mda

from autopath import SystemPreparation, Equilibration, SteeredMD
from autopath.analysis import plot_rmsd, plot_atomic_rmsf
from autopath.utils import fix_pdb, save_pdb, load_system, align_trajectory, calculate_com_distance, compute_rmsd
from openmm.app import PDBFile

def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs equilibration of a protein-ligand system.",
        epilog="""
        REPORTING BUGS
                Please report bugs to:
                AutoDock mailing list   http://autodock.scripps.edu/mailing_list\n

        COPYRIGHT
                Copyright (C) 2025 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )

    parser.add_argument(
        "-r",
        "--rec",
        dest="rec",
        required=True,
        action="store",
        help="path to the receptor PDB file",
    )

    parser.add_argument(
        "-l",
        "--lig",
        dest="lig",
        required=True,
        action="store",
        help="path to the ligand SDF file",
    )
    
    parser.add_argument(
        "-p",
        "--protocol",
        dest="protocol",
        required=True,
        action="store",
        help="path to the JSON file with the equilibration protocol",
    )
    
    parser.add_argument(
        "-n",
        "--resname",
        dest="lig_resname",
        required=False,
        default="UNK",
        help="residue name of the ligand",
    )
    return parser.parse_args()


def main():

    args = cmd_lineparser()
    receptor = args.rec
    ligand = args.lig 
    equilibration_scheme = args.protocol # Make sure to customize the equilibration scheme as needed
    
    sys_name = os.path.splitext(os.path.basename(receptor))[0]
    os.makedirs(sys_name, exist_ok=True)

    logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{sys_name}/{sys_name}.log", mode="a"),
        logging.StreamHandler(),
    ],
    )
    
    lig_resname = "UNK"

    # Fix/prepare the receptor
    protein_pdb = fix_pdb(pdbfile=receptor, keep_heterogens=True, pH=7.4)
    pdb_name = os.path.splitext(os.path.basename(receptor))[0]
    prot_path=f"{sys_name}/{pdb_name}_fixed.pdb"
    save_pdb(protein_pdb.topology, protein_pdb.positions, prot_path)

    ########################################################################################
    #################################### System preparation ################################
    ########################################################################################

    # Prepare the system
    prepare_system = SystemPreparation(
            out_dir=sys_name,
            boxShape="dodecahedron",
            padding=1.0,
            lig_ff="espaloma",
            ionicStrength=0.15,
            is_membrane=False,
            lipid_type="POPC",
            forcefield = [
            'amber14-all.xml',
            "amber14/tip3pfb.xml",
            "amber/tip3pfb_HFE_multivalent.xml"   
            ]
    )

    # Variants is a dictionary which specifies the chain:resid for the variant e.g. {"A:123": "CYX"}
    # If you re-run the script and the system is already prepared comment the following line
    # system, topo = prepare_system.run(protein=prot_path, variants=None, ligands=ligand)

    ########################################################################################
    ###################################### Equilibration ###################################
    ########################################################################################

    system_pdb_file = f"{sys_name}/system.pdb"
    topo = PDBFile(system_pdb_file).topology
    system = load_system(f"{sys_name}/system.xml")

    # Run restrained equilibration
    equilibration = Equilibration(
        system=system,
        topology=topo,
        protocol_fname=equilibration_scheme,
        is_membrane=False,
        restrained_minimization=True,
        out_dir=f"{sys_name}/equilibration",
        )
    
    # If you re-run the script and the system is equilibrated prepared comment the following line
    # system_eq = equilibration.run(pdb_file=system_pdb_file, run_id=sys_name)

    system_prmtop = f"{sys_name}/system.prmtop"
    equilibrated_traj = f"{sys_name}/equilibration/trajectory_equilibration_{sys_name}.dcd"
    # Wrap, align and save the clean trajectory
    align_trajectory(
        system_prmtop,
        equilibrated_traj,
        out_fname=f"{sys_name}/equilibration/{sys_name}_aligned.dcd",
        strip_mask=None #you can dry the traj or remove garbage
    )
    equilibrated_traj = f"{sys_name}/equilibration/{sys_name}_aligned.dcd"
    # Calculate RMSD and RMSF of the ligand
    u_eq = mda.Universe(system_prmtop, equilibrated_traj, in_memory=True)
    
    lig_rmsd_equilibration = compute_rmsd(u_eq, u_eq,
                                          alig_select="backbone", 
                                          groupselections={"ligand":f"resname {lig_resname} and not name H*", 
                                                           "protein":'protein and not name H*'},
                                          out_dir=f"{sys_name}/equilibration"
                                          )
    lig_rmsd_equilibration.to_csv(f"{sys_name}/equilibration/{sys_name}_ligand_rmsd.csv", index=False)
    plot_atomic_rmsf(u_eq, outname=f"{sys_name}/equilibration/{sys_name}_RMSF.png", log_rmsf=False)

    ########################################################################################
    ###################################### Steered MD ######################################
    ########################################################################################

    # equilibrated_pdb = f"{sys_name}/equilibration/{sys_name}_equilibrated.pdb"
    equilibrated_chk = f"{sys_name}/equilibration/checkpoint_equil_{sys_name}.chk"
    system_file = f"{sys_name}/equilibration/system_equil_{sys_name}.xml"
    system_eq = load_system(system_file)

    # load last frame of the equilibrated and WRAPPED trajectory, DO NOT USE THE STRIPPED ONE or the PDB
    u_eq = mda.Universe(system_prmtop, equilibrated_traj)
    u_eq.trajectory[-1]  # set pointer to last frame

    # Get pocket atoms
    pocket_atoms = u_eq.select_atoms("same residue as protein and (around 5 resname UNK) and (name CA C N)")
    ligand_atoms = u_eq.select_atoms(f"resname {lig_resname} and (not name H*)")
    restrained_atoms = u_eq.select_atoms("same residue as (protein and around 6 resname UNK) and name CA")
    
    restrained_atoms_indices = [atom.index for atom in restrained_atoms]
    ligand_atoms_indices = [atom.index for atom in ligand_atoms]
    pocket_atom_indices = [atom.index for atom in pocket_atoms]
    pocket_full_names = [f"{atom.resname}_{atom.resid}_{atom.index}" for atom in pocket_atoms]
    # This is to check that the selection is correct
    # PLEASE debug your own selection 
    # logging.info(f"Pocket atoms are: {', '.join(set(pocket_full_names))}")

    # Calculate COM distance after equilibration
    # You can use this to approximate the pulling distance
    # Additionally, you could use this as a checkpoint and stop here if the ligand has drifted too far
    eq_com = calculate_com_distance(u_eq, ligand_atoms, pocket_atoms, weighByMass=True)
    final_com = eq_com.values[-1][0]
    logging.info(f"COM distance after equilibration is: {final_com:.2f} A")       

    # Run steered MD
    sMD = SteeredMD(
        system=system_eq,
        topology=topo,
        ligand_atoms=ligand_atoms_indices,
        pocket_atoms=pocket_atom_indices,
        restrained_atoms=restrained_atoms_indices,
        restart_velocities=True,
        out_dir=f"{sys_name}/sMD",
    )

    sMD.run(sMD_time=1, #ns
            displacement=1.5, #nm 
            pulling_force=1000, #KJ/mol/nm^2
            replicas=3, 
            checkpoint_file=equilibrated_chk,
            do_backwards=False # this is for the reverse pulling
            )
    
    ########################################################################################
    ###################################### Post-processing #################################
    ########################################################################################
    # Load and align the sMD trajectories
    for i in range(1, 4):
        traj_file = f"{sys_name}/sMD/trajectory_sMD_replica_{i}_forward.dcd"
        align_trajectory(
                        prmtop_file=system_prmtop,
                        traj_file=traj_file,
                        out_fname=traj_file.replace(".dcd", "_aligned.dcd"),
                    )
        os.remove(traj_file) # remove the dcd

    return

if __name__ == "__main__":
    main()
