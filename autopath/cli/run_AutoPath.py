import os
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
        ligands = sorted(glob(os.path.join(ligand, "*.sdf")))
    else:
        raise FileNotFoundError(f"Ligand input not found: {ligand}")

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
        platform=config.platform,
        # Preparation
        run_preparation=config.run_preparation,
        forcefield=config.forcefield,
        hydrogenMass=config.hydrogenMass,
        timestep=config.timestep,
        lig_ff=config.lig_ff,
        boxShape=config.boxShape,
        padding=config.padding,
        ionicStrength=config.ionicStrength,
        variants=config.variants,
        is_membrane=config.is_membrane,
        lipid_type=config.lipid_type,
        # Equilibration
        run_equilibration=config.run_equilibration,
        protocol_fname=config.protocol_fname,
        # Steered MD
        run_sMDpulling=config.run_sMDpulling,
        sMD_outdir=config.sMD_outdir,
        sMD_pulling_dir=config.sMD_pulling_dir,
        sMD_pulling_speeds=config.sMD_pulling_speeds,
        sMD_max_pulling_dist=config.sMD_max_pulling_dist,
        sMD_time=config.sMD_time,
        sMD_steps_per_move=config.sMD_steps_per_move,
        sMD_dx_per_move=config.sMD_dx_per_move,
        sMD_spring_cte=config.sMD_spring_cte,
        sMD_ligand_anchor_mode=config.sMD_ligand_anchor_mode,
        sMD_autostop_freq=config.sMD_autostop_freq,
        sMD_run_analysis=config.sMD_run_analysis,
        sMD_clust_selection=config.sMD_clust_selection,
        # Milestones
        extract_milestones=config.extract_milestones,
        milestone_mode=config.milestone_mode,
        milestone_min_frame_separation=config.milestone_min_frame_separation,
        n_milestones=config.n_milestones,
        relax_steps=config.relax_steps,
        # Metadynamics
        run_metadynamics=config.run_metadynamics,
        mMD_use_funnel_potential=config.mMD_use_funnel_potential,
        mMD_bias_factor=config.mMD_bias_factor,
        mMD_bias_frequency=config.mMD_bias_frequency,
        mMD_hill_height=config.mMD_hill_height,
        mMD_hill_width=config.mMD_hill_width,
        mMD_time=config.mMD_time,
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
