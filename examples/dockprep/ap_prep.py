import os
import logging
import argparse

import MDAnalysis as mda
import mdtraj as md

from autopath import SystemPreparation, Equilibration
from autopath.analysis import plot_atomic_rmsf
from autopath.utils import fix_pdb, fetch_pdb, save_receptor_and_ligand_from_pdb, save_receptor_and_ligand_from_openmm, save_pdb, save_receptor_w_colig, load_system, setup_logging, compute_rmsd
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
        "--rec",
        default=None,
        help="path to the receptor PDB file",
    )

    parser.add_argument(
        "--lig",
        default=None,
        help="path to the ligand SDF/PDB file",
    )
    
    parser.add_argument(
        "--lig_smiles",
        default=None,
        type=str,
        help="Smiles of the ligand",
    )

    parser.add_argument(
        "--org_colig",
        default=None,
        help="path to the organic co-ligand SDF/PDB file",
    )

    parser.add_argument(
        "--org_colig_smiles",
        default=None,
        type=str,
        help="Smiles of the organic co-ligand",
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
        "--restrained_minimization_only",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--equil_rec_only", 
        action='store_true',
        help='equilibrate the receptor without any ligands or cofactors',
        default=False
    )

    parser.add_argument(
        "--fetch_pdb", 
        action='store_true', 
        default=False
    )

    parser.add_argument(
        "--pdb_id", 
        type=str, 
        default=None, 
        help='pdb id to fetch. can be of the format XXXX or XXXX_A, where the latter specifies the chain id after the pdb id'
    )

    parser.add_argument(
        "--pdb_chain_ids", 
        type=str, 
        default=None, 
        help='chain ids to fetch. if multiple, assumes chain ids are passed as A-B-C (i.e. dash separated)'
    )

    parser.add_argument(
        "--pdb_lig_name", 
        type=str, 
        default=None, 
        help='if provided, will extract all ligands corresponding to this name from the pdb'
    )

    parser.add_argument(
        "--use_ccd_smiles_for_lig", 
        action='store_true',
        help='use ccd smiles to create rdkit mol. otherwise will default to using molscrub', 
        default=False
    )

    parser.add_argument(
        "--pdb_lig_resid", 
        type=str, 
        default=None, 
        help='if provided alongside pdb_lig_name, will extract the ligand corresponding to this residue id and name <pdb_lig_name> from the pdb. if needing to pass in chain id as well, can specify as <resid>_<chainid>'
    )

    parser.add_argument(
        "--pdb_inorg_cofactor_name", 
        type=str, 
        default=None, 
        help='if provided, will extract receptor with all inorganic cofactors corresponding this name (e.g. MG, ZN, etc.)'
    )

    parser.add_argument(
        "--pdb_inorg_cofactor_resid", 
        type=str, 
        default=None, 
        help='if provided alongside pdb_inorg_cofactor_name, will extract receptor with inorganic cofactor corresponding to this residue id and name <pdb_inorg_cofactor_name> from the pdb. if needing to pass in chan id as well, can specify as <resid>_<chainid>' 
    )

    parser.add_argument(
        "--pdb_org_colig_name", 
        type=str, 
        default=None, 
        help='if provided, will extract all organic co-ligands corresponding to this name (e.g. HEM, NAD, etc.)'
    )

    parser.add_argument(
        "--pdb_org_colig_resid", 
        type=str, 
        default=None, 
        help='if provided alongside pdb_org_colig_name, will extract receptor with organic co-ligand corresponding to this residue id and name <pdb_org_colig_name> from the pdb. if needing to pass in chain id as well, can specify as <resid>_<chainid>'
    )

    parser.add_argument(
        "--use_ccd_smiles_for_colig", 
        action='store_true',
        help='use ccd smiles to create rdkit mol. otherwise will default to using molscrub', 
        default=False
    )

    parser.add_argument(
        "--ignore_colig_simulation", 
        action='store_true',
        help='dont include coligand in simulation. relevant if forcefield has issue parameterizing molecule (e.g. heme)',
        default=False
    )

    parser.add_argument(
        "--ignore_crystallographic_waters",
        action='store_true',
        help='dont use crystallographic waters when minimizing or equilibrating receptor',
        default=False
    )

    parser.add_argument(
        "--water_resids",
        type=str,
        nargs='+',
        help='saves waters with receptor whose residue id is contained in this list. (example: 25 29 32)', 
        default=None 
    )

    parser.add_argument(
        "--water_chainids",
        type=str,
        nargs='+',
        help='restricts waters to be saved with receptor to those whose chain id is contained in this list. must specify if water_resids is included to prevent inclusion of waters added upon solvation by OpenMM. (example: A B)', 
        default=None 
    )

    parser.add_argument(
        "--lig_resname",
        dest="lig_resname",
        required=False,
        default="UNK",  
        help="residue name to assign ligand during system prep",
    )

    parser.add_argument(
        "--save_dir", 
        type=str, 
        default=None, 
        help='directory to save files. will resort to name of pdb file if not specified'
    )


    return parser.parse_args()


def main():

    args = cmd_lineparser()

    if args.ignore_crystallographic_waters:
        include_waters = False
    else:
        include_waters = True

    if args.water_resids and args.water_chainids is None:
        msg = "must specify water_chainids if including water_resids to prevent inclusion of waters added upon solvation by OpenMM"
        raise ValueError(msg) 

    if args.fetch_pdb and args.rec is None:
        sys_name = args.pdb_id 
        print(f"fetching {args.pdb_id}")
        pdb_path = fetch_pdb(args.pdb_id, args.pdb_chain_ids, args.save_dir)
        print('extracting receptor/ligand')
        rec_path, lig_path, lig_smiles, org_colig_path, org_colig_smiles = save_receptor_and_ligand_from_pdb(pdb_path, 
                                                                                                             args.pdb_id, 
                                                                                                             args.save_dir, 
                                                                                                             args.pdb_inorg_cofactor_name,
                                                                                                             args.pdb_inorg_cofactor_resid,
                                                                                                             args.pdb_org_colig_name, 
                                                                                                             args.pdb_org_colig_resid,
                                                                                                             args.ignore_colig_simulation,
                                                                                                             args.pdb_lig_name,                                                                                                                                                                     args.pdb_lig_resid,
                                                                                                             args.use_ccd_smiles_for_lig,
                                                                                                             args.use_ccd_smiles_for_colig,
                                                                                                             include_waters,
                                                                                                             args.water_resids,
                                                                                                             args.water_chainids)
        if args.lig_smiles:
            lig_smiles = args.lig_smiles 
        ligs_from_xray = True 
    elif args.rec is not None:
        sys_name = os.path.splitext(os.path.basename(args.rec))[0]
        rec_path = args.rec
        lig_path = args.lig 
        lig_smiles = args.lig_smiles
        org_colig_path = args.org_colig
        org_colig_smiles = args.org_colig_smiles       
        ligs_from_xray = False 

    lig_resname = args.lig_resname
    if args.pdb_org_colig_name is not None:
        org_colig_resname = args.pdb_org_colig_name
    else:   
        org_colig_resname = None

    ligands = []
    if lig_path is not None and not args.equil_rec_only:
        ligands.append((lig_resname,lig_path,lig_smiles,ligs_from_xray))
    if org_colig_path is not None and not args.ignore_colig_simulation and not args.equil_rec_only:
        ligands.append((org_colig_resname,org_colig_path,org_colig_smiles,ligs_from_xray))
    if len(ligands) == 0:
        ligands = None

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.splitext(os.path.basename(rec_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    equilibration_scheme = args.protocol # Make sure to customize the equilibration scheme as needed
    
    # Setup logging
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")
    logger.info("Starting equilibration process")

    # Fix/prepare the receptor
    protein_pdb = fix_pdb(pdbfile=rec_path, keep_heterogens=True, pH=7.4)
    pdb_name = os.path.splitext(os.path.basename(rec_path))[0]
    #save full receptor 
    prot_path = f"{save_dir}/{pdb_name}_fixed.pdb"
    save_pdb(protein_pdb.topology, protein_pdb.positions, prot_path)
    #add co-ligand to fixed receptor if applicable 
    if org_colig_path is not None:
        prot_path_w_colig=f"{save_dir}/{pdb_name}_fixed_w_colig.pdb"
        save_receptor_w_colig(prot_path, org_colig_path, org_colig_smiles, prot_path_w_colig, args.pdb_org_colig_name)
        fixed_path = prot_path_w_colig
        base_fname = f"{pdb_name}_fixed_w_colig"
    else:
        fixed_path = prot_path
        base_fname = f"{pdb_name}_fixed"
    
    #save fixed receptor wo/solvent
    save_receptor_and_ligand_from_openmm(pdb_path=fixed_path, 
                                         save_dir=save_dir, 
                                         inorg_cofactor_name=args.pdb_inorg_cofactor_name, 
                                         org_colig_name=args.pdb_org_colig_name, 
                                         ligand_name=None, 
                                         water_resids=None,
                                         output_fname_rec=f"{base_fname}_wo_solvent.pdb")
    #save fixed receptor w/solvent with solvent if water_resids are specified  
    if args.water_resids is not None:
        save_receptor_and_ligand_from_openmm(pdb_path=fixed_path, 
                                             save_dir=save_dir, 
                                             inorg_cofactor_name=args.pdb_inorg_cofactor_name, 
                                             org_colig_name=args.pdb_org_colig_name, 
                                             ligand_name=None, 
                                             water_resids=args.water_resids,
                                             water_chainids=args.water_chainids,
                                             output_fname_rec=f"{base_fname}_w_struct_waters.pdb")


    ########################################################################################
    #################################### System preparation ################################
    ########################################################################################

    # Prepare the system
    prepare_system = SystemPreparation(
            out_dir=save_dir,
            boxShape="dodecahedron",
            padding=1.2,
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
    system, topo = prepare_system.run(protein=prot_path, variants=None, ligands=ligands)

    ########################################################################################
    ###################################### Equilibration ###################################
    ########################################################################################

    system_pdb_file = f"{save_dir}/system.pdb"
    topo = PDBFile(system_pdb_file).topology
    system = load_system(f"{save_dir}/system.xml")

    # Run restrained equilibration
    equilibration = Equilibration(
        system=system,
        topology=topo,
        protocol_fname=equilibration_scheme,
        is_membrane=False,
        restrained_minimization=True,
        restrained_minimization_only=args.restrained_minimization_only,
        out_dir=f"{save_dir}/equilibration",
        )
    
    # If you re-run the script and the system is equilibrated prepared comment the following line
    logger.info("Running equilibration...")
    system_eq = equilibration.run(pdb_file=system_pdb_file, run_id=sys_name)
        
    ########################################################################################
    ###################################### Post-processing #################################
    ########################################################################################

    restrained_min_struct = f"{save_dir}/equilibration/{sys_name}_minim.pdb"
    if org_colig_path is not None and args.ignore_colig_simulation:
        #need to add co-ligand back to minimized structure since was not included in simulation  
        save_receptor_w_colig(restrained_min_struct, org_colig_path, org_colig_smiles, restrained_min_struct, org_colig_resname)
 
    #save minimized structure without solvent
    save_receptor_and_ligand_from_openmm(pdb_path=restrained_min_struct, 
                                         save_dir=f"{save_dir}/equilibration",
                                         inorg_cofactor_name=args.pdb_inorg_cofactor_name, 
                                         org_colig_name=org_colig_resname,
                                         remove_H_colig=True,
                                         ligand_name=lig_resname,
                                         water_resids=None,
                                         water_chainids=None,
                                         output_fname_rec=f"{sys_name}_minim_receptor_wo_solvent.pdb",
                                         output_fname_lig=f"{sys_name}_minim_ligand.pdb")
    #save minimized structure with solvent if water_resids are specified 
    if args.water_resids is not None:
        save_receptor_and_ligand_from_openmm(pdb_path=restrained_min_struct, 
                                             save_dir=f"{save_dir}/equilibration",
                                             inorg_cofactor_name=args.pdb_inorg_cofactor_name, 
                                             org_colig_name=org_colig_resname,
                                             remove_H_colig=True,
                                             ligand_name=None,
                                             water_resids=args.water_resids,
                                             water_chainids=args.water_chainids,
                                             output_fname_rec=f"{sys_name}_minim_receptor_w_struct_waters.pdb")


    if args.restrained_minimization_only:
        return 

    system_prmtop = f"{save_dir}/system.prmtop"
    equilibrated_traj = f"{save_dir}/equilibration/equilibration_{sys_name}.dcd"
    
    # Wrap, align and save the clean trajectory
    traj = md.load(equilibrated_traj, top=system_pdb_file)
    traj = traj.center_coordinates()
    traj = traj.image_molecules()
    try: # if there's no protein
        backbone = traj.topology.select("backbone")
        traj = traj.superpose(traj[0], atom_indices=backbone)
    except Exception as e:
        logger.warning(f"Superposition failed: {e}. Proceeding without superposition.")
    traj.save(equilibrated_traj.replace(".dcd", "_aligned.xtc"))
    os.remove(equilibrated_traj)
    logger.info(f"Aligned trajectory saved to {equilibrated_traj.replace('.dcd', '_aligned.xtc')}")

    if ligands is not None:
        # Calculate RMSD and RMSF of the ligand
        u_eq = mda.Universe(system_pdb_file, equilibrated_traj.replace(".dcd", "_aligned.xtc"), in_memory=True)
        lig_rmsd_equilibration = compute_rmsd(u_eq, u_eq,
                                              alig_select="backbone", 
                                              groupselections={"ligand":f"resname {lig_resname} and not name H*", 
                                                               "protein":'protein and not name H*'},
                                              plots_outdir=f"{save_dir}/equilibration"
                                              )
        lig_rmsd_equilibration.to_csv(f"{save_dir}/equilibration/{sys_name}_ligand_rmsd.csv", index=False)
        plot_atomic_rmsf(u_eq, outname=f"{save_dir}/equilibration/{sys_name}_RMSF.png", log_rmsf=True)

    return

if __name__ == "__main__":
    main()
