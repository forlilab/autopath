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

    ap = AutoPath(pdb_path=receptor,
                    pocket_selection='(resid 145-153 183-190) and name CA',

                    do_fix_pdb=True,
                    run_preparation=True,
                    padding=1.2,
                    hydrogenMass=1.5, 
                    ionicStrength=0.15,
                    boxShape="dodecahedron",

                    lig_ff='OPENFF',

                    run_equilibration=True,
                    protocol_fname=protocol,

                    run_sMDpulling=True,
                    sMD_outdir='sMD',
                    sMD_ligand_anchor_mode='lig_com',
                    sMD_max_replicas = 50,
                    sMD_pulling_speeds={
                                        0.0050:None, # nm/ps equivalent to 5.0 m/s nm/ns
                                        0.0010:None, # nm/ps equivalent to 1.0 m/s nm/ns
                                        0.010:None, # nm/ps equivalent to 10.0 m/s nm/ns
                                        },

                    sMD_autostop_freq=250, #moves
                    sMD_dx_per_move=0.001, # nm

                    sMD_run_analysis=True,
                    sMD_clust_selection=None,
                    
                    extract_milestones=True,
                    n_milestones=5,

                    run_metadynamics=True,
                    mMD_time=10,
                    mMD_bias_frequency=2,
                    mMD_hill_width=0.05,
                    mMD_hill_height=1.2,
                    mMD_bias_factor=15
                )
    ap.run(ligand_file=ligand)

if __name__ == "__main__":
    main()
