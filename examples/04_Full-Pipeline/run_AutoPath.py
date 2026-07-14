import os
import argparse
from glob import glob
import shutil
from autopath.autopath_core import AutoPath
from autopath.utils import setup_logging

def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs the full AutoPath pipeline for a given receptor and ligand.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
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
    
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    receptor = args.rec
    ligand = args.lig
    protocol = args.protocol
    sys_name = os.path.basename(ligand).split('.')[0]
    os.makedirs(sys_name, exist_ok=True)

    # Setup logging
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")
    
    pocket_residues = [180, 181, 182, 183, 226, 227, 228, 229, 230, 204, 205, 206, 
                       207, 208, 209, 210, 211, 212, 213, 214, 215, 196, 197, 198, 
                       199, 200, 201, 133, 134, 135, 136, 137, 138, 139, 140, 156, 
                       157, 158, 159, 160, 161, 162]
    pocket_selection = f'resid {" ".join(map(str, pocket_residues))} and name CA'
    
    ap = AutoPath(pdb_path=receptor,
                    pocket_selection=pocket_selection,

                    do_fix_pdb=False,
                    run_preparation=False,
                    padding=1.2,
                    hydrogenMass=1.5, 
                    ionicStrength=0.15,
                    boxShape="dodecahedron",

                    lig_ff='openff-2.3.0',
                    
                    platform='OpenCL',
                    
                    run_equilibration=False,
                    protocol_fname=protocol,

                    run_sMDpulling=False,
                    sMD_outdir='sMD',
                    sMD_ligand_anchor_mode='murcko',
                    sMD_converge_speeds=True,
                    sMD_max_replicas = 50,
                    sMD_pulling_speeds={
                                        0.020: 5,
                                        0.015: 5,
                                        0.010: 5, # nm/ps equivalent to 10.0 m/s nm/ns
                                        # 0.0050: 25, # nm/ps equivalent to 5.0 m/s nm/ns
                                        # 0.0010: 25, # nm/ps equivalent to 1.0 m/s nm/ns
                                        },

                    sMD_run_analysis=True,

                    extract_milestones=False,
                    n_milestones=3,
                    
                    run_metadynamics=False,
                    mMD_use_funnel_potential=False,
                    mMD_preseed_bias=False,
                    mMD_milestone_seeding=False,
                    
                    mMD_time=10, # ns
                    mMD_bias_frequency=2,
                    mMD_hill_width=0.05,
                    mMD_hill_height=1.2,
                    mMD_bias_factor=15
                )
    
    ap.run(ligand_file=ligand)

if __name__ == "__main__":
    main()