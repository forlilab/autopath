import os
import logging
import argparse
from glob import glob
from autopath.config import Config
from autopath.autopath_core import AutoPath


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs AutoPath simulation.",
        epilog="""
        REPORTING BUGS
                Please report bugs to:
                AutoDock mailing list   http://autodock.scripps.edu/mailing_list\n

        COPYRIGHT
                Copyright (C) 2023 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        required=True,
        action="store",
        help="path to the json config file",
    )

    parser.add_argument(
        "-l",
        "--lig",
        dest="lig",
        required=True,
        action="store",
        help="ligand SDF file or path ligands directory",
    )

    return parser.parse_args()


def main():

    args = cmd_lineparser()
    config_file = args.config
    ligand = args.lig

    if os.path.isfile(ligand):
        ligands = [ligand]
    elif os.path.isdir(ligand):
        ligands = glob(f"{ligands}/*.sdf")

    # Instantiate the AutoPath class with the configuration
    config = Config.from_config(config_file)
    autopath_simulation = AutoPath(
        # General
        VS_mode=config.VS_mode,
        pdb_path=config.pdb_path,
        do_fix_pdb=config.do_fix_pdb,
        pocket_selection=config.pocket_selection,
        temperature=config.temperature,
        random_state=config.random_state,
        # Preparation
        run_preparation=config.run_preparation,
        forcefield=config.forcefield,
        lig_ff=config.lig_ff,
        boxShape=config.boxShape,
        padding=config.padding,
        ionicStrength=config.ionicStrength,
        variants=config.variants,
        is_membrane=config.is_membrane,
        lipid_type=config.lipid_type
        # Equilibration
        run_equilibration=config.run_equilibration,
        equilibration_scheme=config.equilibration_scheme,
        # Steered MD
        run_sMDpulling=config.run_sMDpulling,
        sMD_pulling_dist=config.sMD_pulling_dist,
        sMD_time=config.sMD_time,
        sMD_replicas=config.sMD_replicas,
        sMD_steps_per_move=config.sMD_steps_per_move,
        sMD_pulling_force=config.sMD_pulling_force,
        sMD_autostop=config.sMD_autostop,
        # Milestones
        extract_milestones=config.extract_milestones,
        n_milestones=config.n_milestones,
        run_relax=config.run_relax,
        relax_steps=config.relax_steps,
        # Metadynamics
        run_metadynamics=config.run_metadynamics,
        mMD_walkers=config.mMD_walkers,
        mMD_time=config.mMD_time,
        mMD_bias_factor=config.mMD_bias_factor,
        mMD_hill_height=config.mMD_hill_height,
        mMD_hill_width=config.mMD_hill_width,
    )

    for lig in ligands:

        # sys_name = os.path.splitext(os.path.basename(lig))[0]
        # os.makedirs(sys_name, exist_ok=True)

        # logging.basicConfig(
        #     level="INFO",
        #     format="%(asctime)s [%(levelname)s] %(message)s",
        #     handlers=[
        #         logging.FileHandler(f"{sys_name}/{sys_name}.log", mode="a"),
        #         logging.StreamHandler(),
        #     ],
        # )

        autopath_simulation.run(lig)


if __name__ == "__main__":
    main()
