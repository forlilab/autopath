import os
import logging
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import SystemPreparation, Equilibration
from autopath.analysis import plot_atomic_rmsf
from autopath.utils import fix_pdb, save_pdb, load_system, setup_logging, compute_rmsd
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
        required=False,
        default=None,
        help="path to the ligand SDF/PDB file",
    )
    
    parser.add_argument(
        "-n",
        "--resname",
        dest="resname",
        required=False,
        default="UNK",
        help="residue name of the ligand",
    )
    parser.add_argument(
        "-s",
        "--smiles",
        dest="smiles",
        required=False,
        default=None,
        help="Smiles of the ligand. Will be used to assign bond orders if provided",
    )

    parser.add_argument(
        "--lig_from_xray",
        action='store_true',
        help="include flag if ligand is directly extracted from crystal structure without any processing",
        default=False
    )
    
    parser.add_argument(
        "-f",
        "--fixpdb",
        dest="fixpdb",
        required=False,
        default=True,
        help="whether to fix the input PDB file (default: True)",
    )
    
    parser.add_argument(
        "-p",
        "--protocol",
        dest="protocol",
        required=True,
        action="store",
        help="path to the JSON file with the equilibration protocol",
    )
    return parser.parse_args()


def main():

    args = cmd_lineparser()
    receptor = args.rec
    lig_path = args.lig 
    lig_resname = args.resname
    if lig_path is not None:
        ligands = [(lig_resname,lig_path,args.smiles,args.lig_from_xray)]
    else:
        ligands = None
    equilibration_scheme = args.protocol # Make sure to customize the equilibration scheme as needed
    
    sys_name = os.path.splitext(os.path.basename(receptor))[0]
    os.makedirs(sys_name, exist_ok=True)

    # Setup logging
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")
    logger.info("Starting equilibration process")

    ########################################################################################
    #################################### System preparation ################################
    ########################################################################################

    # Fix/prepare the receptor
    if not args.fixpdb:
        fixed_receptor = receptor
    else:
        fixed_receptor = fix_pdb(pdbfile=receptor, 
                            replace_nonstandard_residues=True,
                            ignore_terminal_missing_residues=True,
                            keep_heterogens=True, 
                            pH=7.4
                            )
        pdb_name = os.path.splitext(os.path.basename(receptor))[0]
        fixed_receptor_path=f"{sys_name}/{pdb_name}_fixed.pdb"
        save_pdb(fixed_receptor.topology, fixed_receptor.positions, fixed_receptor_path)
        
    # Assemble and parameterize the system
    prepare_system = SystemPreparation(
            boxShape="dodecahedron",
            padding=1.2,
            hydrogenMass=1.5,
            lig_ff="openff",
            ionicStrength=0.15,
            ions=("Na+", "Cl-"),
            is_membrane=False,
            lipid_type="POPC",
            out_dir=sys_name,
            forcefield = [
            'amber14-all.xml',
            "amber14/tip3pfb.xml",
            "amber/tip3pfb_HFE_multivalent.xml"   
            ]
    )

    # Variants is a dictionary which specifies the chain:resid for the variant e.g. {"A:123": "CYX"}
    # If you re-run the script and the system is already prepared comment the following line
    system, topology = prepare_system.run(protein=fixed_receptor_path, variants=None, ligands=ligands)

    ########################################################################################
    ###################################### Equilibration ###################################
    ########################################################################################

    system_pdb_file = f"{sys_name}/system.pdb"
    topology = PDBFile(system_pdb_file).topology
    system = load_system(f"{sys_name}/system.xml")

    # Run restrained equilibration
    equilibration = Equilibration(
        system=system,
        topology=topology,
        protocol_fname=equilibration_scheme,
        is_membrane=False,
        restrained_minimization=False,
        out_dir=f"{sys_name}/equilibration",
        )
    
    # If you re-run the script and the system is equilibrated prepared comment the following line
    system = equilibration.run(pdb_file=system_pdb_file, run_id=sys_name)
        
    ########################################################################################
    ###################################### Post-processing #################################
    ########################################################################################

    system_prmtop = f"{sys_name}/system.prmtop"
    equilibrated_traj = f"{sys_name}/equilibration/equilibration_{sys_name}.dcd"
    
    # Wrap, align and save the clean trajectory
    traj = md.load(equilibrated_traj, top=system_pdb_file)
    traj = traj.center_coordinates()
    traj = traj.image_molecules()
    try: # if there's no protein
        backbone = traj.topology.select("backbone")
        traj = traj.superpose(traj[0], atom_indices=backbone)
    except Exception as e:
        logging.warning(f"Superposition failed: {e}. Proceeding without superposition.")
    traj.save(equilibrated_traj.replace(".dcd", "_aligned.xtc"))
    os.remove(equilibrated_traj)
    logger.info(f"Aligned trajectory saved to {equilibrated_traj.replace('.dcd', '_aligned.xtc')}")

    # Calculate RMSD and RMSF of the ligand
    # Make sure to customize/add the selections as needed
    u_eq = mda.Universe(system_pdb_file, equilibrated_traj.replace(".dcd", "_aligned.xtc"), in_memory=True)
    rmsd_equilibration = compute_rmsd(u_eq, u_eq,
                                          alig_select="backbone", 
                                          groupselections={
                                                           "ligand":f"resname {lig_resname} and not name H*" if ligands is not None else None, 
                                                           "protein":'protein and not name H*'
                                                           },
                                          plots_outdir=f"{sys_name}/equilibration"
                                          )
    rmsd_equilibration.to_csv(f"{sys_name}/equilibration/RMSD_{sys_name}.csv", index=False)
    if ligands is not None:
        plot_atomic_rmsf(u_eq, lig_resname,
                         outname=f"{sys_name}/equilibration/RMSF_{sys_name}.png",
                        )
    return

if __name__ == "__main__":
    main()
