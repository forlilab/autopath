import os
import sys
import logging
import argparse
from glob import glob
from autopath.config import Config
from autopath.autopath_core import AutoPath

logger = logging.getLogger(__name__)


def cmd_lineparser():
    """Build and return the CLI argument parser for AutoPath.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with attributes:

        ``config`` : str
            Path to the JSON configuration file.
        ``lig`` : str
            Path to a single ligand SDF file or a directory of SDF files.
    """
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
        help="ligand SDF file or path to a directory of ligand SDF files",
    )

    return parser.parse_args()


def main():
    """Entry point for the AutoPath CLI.

    Parses command-line arguments, loads the JSON configuration, instantiates
    an :class:`~autopath.autopath_core.AutoPath` simulation object, and runs
    the full pipeline for each ligand in the provided SDF file or directory.

    Arguments (via CLI)
    -------------------
    -c / --config : str
        Path to the JSON configuration file.
    -l / --lig : str
        Path to a single ligand SDF file, or to a directory containing
        ``*.sdf`` files (processed in sorted order).

    Exits
    -----
    Exits with code 1 on ``FileNotFoundError``, ``ValueError``, or any
    unexpected exception, logging the error before exiting.
    """
    try:
        args = cmd_lineparser()
        config_file = args.config
        ligand = args.lig

        if os.path.isfile(ligand):
            ligands = [ligand]
        elif os.path.isdir(ligand):
            ligands = sorted(glob(os.path.join(ligand, "*.sdf")))
            if not ligands:
                raise FileNotFoundError(f"No SDF files found in directory: {ligand}")
        else:
            raise FileNotFoundError(f"Ligand input not found: {ligand}")

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
            sMD_run_analysis=config.sMD_run_analysis,
            sMD_clust_selection=config.sMD_clust_selection,
            sMD_conv_window=config.sMD_conv_window,
            sMD_conv_streak=config.sMD_conv_streak,
            sMD_autostop_estimator=config.sMD_autostop_estimator,
            sMD_alternate_speeds=config.sMD_alternate_speeds,
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
            autopath_simulation.run(lig)

    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
