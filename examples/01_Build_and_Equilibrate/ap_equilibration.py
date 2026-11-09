import os
import argparse

import MDAnalysis as mda
from rdkit import Chem
from openmm.app import PDBFile

from autopath import SystemPreparation, Equilibration
from autopath.membrane_orient import orient_membrane
from autopath.ap_PLIP import plot_atomic_property, cw_rmsf_windows
from autopath.utils import save_pdb, load_system, setup_logging, compute_rmsd, wrap_align_save_traj
from autopath.pdb_preprocessor import PDBPreprocessor


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="This script prepares and equilibrates a protein-ligand system using AutoPath.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )
    parser.add_argument("-r", "--rec", dest="rec", required=True,
                        help="path to the receptor PDB file")
    parser.add_argument("-l", "--lig", dest="lig", default=None,
                        help="path to the ligand SDF/PDB file")
    parser.add_argument("-n", "--resname", dest="resname", default="UNK",
                        help="residue name of the ligand (default: UNK)")
    parser.add_argument("-s", "--smiles", dest="smiles", default=None,
                        help="SMILES of the ligand. Used to assign bond orders if provided")
    parser.add_argument("--lig_from_xray", action="store_true", default=False,
                        help="the ligand is taken directly from the crystal structure without any processing")
    parser.add_argument("--no_fixpdb", dest="fixpdb", action="store_false",
                        help="skip PDBFixer (use the receptor as is)")
    parser.add_argument("-p", "--protocol", dest="protocol", required=True,
                        help="path to the JSON file with the equilibration protocol")
    return parser.parse_args()


def main():

    args = cmd_lineparser()
    receptor = args.rec
    lig_resname = args.resname
    if args.lig is not None:
        ligands = [(lig_resname, args.lig, args.smiles, args.lig_from_xray)]
    else:
        ligands = None
    equilibration_scheme = args.protocol  # customize the equilibration scheme as needed

    sys_name = os.path.splitext(os.path.basename(receptor))[0]
    os.makedirs(sys_name, exist_ok=True)

    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")
    logger.info("Starting equilibration process")

    ########################################################################################
    #################################### System preparation ################################
    ########################################################################################

    # Fix/prepare the receptor: missing atoms/loops, non-standard residues, protonation, caps
    if args.fixpdb:
        fixed_receptor = PDBPreprocessor(receptor).fix(
                            replace_nonstandard_residues=True,
                            ignore_terminal_missing_residues=True,
                            keep_heterogens=True,
                            cap_termini=True,
                            pH=7.4
                            )
        receptor_path = f"{sys_name}/{sys_name}_fixed.pdb"
        save_pdb(fixed_receptor.topology, fixed_receptor.positions, receptor_path)
    else:
        receptor_path = receptor

    # Put the membrane normal on z before the bilayer is built. addMembrane lays the
    # bilayer in xy and never moves the solute, so a tilted receptor silently builds a
    # tilted system. Set this to True whenever is_membrane=True below; the receptor and
    # every ligand pose are moved by the same transform, so docked poses stay in register.
    membrane_orientation = False
    if membrane_orientation:
        receptor_path, ligands = orient_membrane(receptor_path, ligands, sys_name)

    # Assemble and parameterize the system (defaults used in the AutoPath paper)
    prepare_system = SystemPreparation(
            boxShape="dodecahedron",
            padding=1.2,           # nm
            hydrogenMass=1.5,      # amu, HMR for a 4 fs timestep
            lig_ff="openff-2.3.0",
            ionicStrength=0.15,    # M
            ions=("Na+", "Cl-"),
            is_membrane=False,
            out_dir=sys_name,
            forcefield=[
                'amber14-all.xml',
                'amber14/tip3pfb.xml',
                'amber/tip3pfb_HFE_multivalent.xml',
            ]
    )

    # variants: chain:resid -> residue variant, e.g. {"A:123": "CYX"}
    # If the system is already prepared you can comment out this line
    prepare_system.run(protein=receptor_path, variants=None, ligands=ligands)

    ########################################################################################
    ###################################### Equilibration ###################################
    ########################################################################################

    system_pdb_file = f"{sys_name}/system.pdb"
    topology = PDBFile(system_pdb_file).topology
    system = load_system(f"{sys_name}/system.xml")

    # restrained_minimization=False (as in the AutoPath pipeline): minimise with every
    # component restrained at the warm-up force constants of the protocol
    equilibration = Equilibration(
        system=system,
        topology=topology,
        protocol_fname=equilibration_scheme,
        is_membrane=False,
        restrained_minimization=False,
        out_dir=f"{sys_name}/equilibration",
        )

    # If the system is already equilibrated you can comment out this line
    equilibration.run(pdb_file=system_pdb_file, run_id=sys_name)

    ########################################################################################
    ###################################### Post-processing #################################
    ########################################################################################

    eq_dir = f"{sys_name}/equilibration"
    # Wrap, centre and superpose on the protein backbone -> equilibration_<sys>_aligned.dcd
    aligned_traj = wrap_align_save_traj(f"{eq_dir}/equilibration_{sys_name}.dcd", system_pdb_file)[0]
    logger.info(f"Aligned trajectory saved to {aligned_traj}")

    u_eq = mda.Universe(system_pdb_file, aligned_traj, in_memory=True)

    # RMSD of protein and ligand (customize/add selections as needed)
    groups = {"protein": "protein and not name H*"}
    if ligands is not None:
        groups["ligand"] = f"resname {lig_resname} and not name H*"
    rmsd_equilibration = compute_rmsd(u_eq, u_eq, alig_select="backbone", groupselections=groups,
                                      plots_outdir=eq_dir)
    rmsd_equilibration.to_csv(f"{eq_dir}/{sys_name}_rmsd.csv", index=False)

    if ligands is None:
        return

    # Ligand stability: per-atom RMSF, contact frequency and contact-weighted RMSF (cw-RMSF).
    # Whole run, and the "free window" (frames after the last ligand restraint is released).
    lig_sel = f"resname {lig_resname}"
    # polar (N, O, S), non-polar (C, halogens) and all heavy atoms; NaN if a set never contacts the pocket
    cw, per_atom = cw_rmsf_windows(u_eq, lig_sel, f"{eq_dir}/equilibration_{sys_name}.csv", equilibration_scheme)
    cw.to_csv(f"{eq_dir}/{sys_name}_cwRMSF.csv")
    logger.info("cw-RMSF (A):\n" + cw.round(3).to_string())
    for window, pa in per_atom.items():
        pa.to_csv(f"{eq_dir}/{sys_name}_RMSF_{window}.csv", index=False)
    ref_mol = Chem.SDMolSupplier(args.lig)[0] if args.lig.endswith(".sdf") else None
    plot_atomic_property(u_eq, per_atom["whole"].rmsf.values, lig_resname=lig_sel, ref_mol=ref_mol,
                         outname=f"{eq_dir}/{sys_name}_RMSF.svg")
    plot_atomic_property(u_eq, per_atom["whole"].contact_freq.values, lig_resname=lig_sel, ref_mol=ref_mol,
                         outname=f"{eq_dir}/{sys_name}_contact_freq.svg", color='r')
    return


if __name__ == "__main__":
    main()
