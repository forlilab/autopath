import os
import argparse

import pandas as pd
import MDAnalysis as mda

from autopath.autopath_core import AutoPath
from autopath.ap_PLIP import cw_rmsf_windows
from autopath.utils import setup_logging


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs the full AutoPath pipeline for a given receptor and ligand.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )
    parser.add_argument("-r", "--rec", dest="rec", required=True, help="path to the receptor PDB file")
    parser.add_argument("-l", "--lig", dest="lig", required=True, help="path to the ligand SDF file")
    parser.add_argument("-p", "--protocol", dest="protocol", required=True,
                        help="path to the JSON file with the equilibration protocol")
    parser.add_argument("--max_polar_cwrmsf", type=float, default=None,
                        help="skip pulling if the free-window polar cw-RMSF (A) exceeds this value "
                             "(default: no filter, the score is only reported)")
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    receptor, ligand, protocol = args.rec, args.lig, args.protocol
    sys_name = os.path.splitext(os.path.basename(ligand))[0]
    os.makedirs(sys_name, exist_ok=True)
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")

    # Settings shared by both phases (defaults of the AutoPath paper)
    settings = dict(
        # Pocket for the pulling CV: CA atoms of the beta-sheet core around the S1 site, as
        # residue numbers of the ORIGINAL receptor PDB (translated to system numbering).
        # None = protein CA atoms within 8 A of the ligand heavy atoms.
        pocket_selection=[133, 134, 135, 136, 137, 138, 139, 140, 156, 157, 158, 159, 160, 161, 162,
                          180, 181, 182, 183, 196, 197, 198, 199, 200, 201, 204, 205, 206, 207, 208,
                          209, 210, 211, 212, 213, 214, 215, 226, 227, 228, 229, 230],

        # System preparation
        padding=1.2,              # nm
        hydrogenMass=1.5,         # amu (HMR, 4 fs timestep)
        ionicStrength=0.15,       # M
        boxShape="dodecahedron",
        lig_ff="openff-2.3.0",
        # platform="OpenCL",

        # Equilibration
        protocol_fname=protocol,

        # Steered MD: forward pulls, Murcko-scaffold anchor, spring 418.4 kJ/mol/nm2 per
        # ligand heavy atom (sMD_spring_cte=None), 0.001 nm restraint steps up to 3.5 nm,
        # autostop at 1% of the initial ligand-pocket contacts
        sMD_outdir="sMD",
        sMD_ligand_anchor_mode="murcko",
        sMD_pulling_speeds={      # nm/ps -> minimum replicas (0.001 nm/ps = 1 m/s)
            0.015: 10,
            0.010: 10,
            0.005: 10,
        },
        sMD_converge_speeds=True,  # add replicas until the PMF converges ...
        sMD_max_replicas=50,       # ... or this many per speed
        sMD_autostop_nc=0.01,
        sMD_max_pulling_dist=3.5,
        sMD_log_geom_features=True,

        # Pathway clustering + cumulant / Jarzynski / force (v->0) estimators
        sMD_path_model="dtw",
        cluster_across_speeds=True,

        # Metadynamics seeded from pulling milestones is experimental and not part
        # of the paper protocol; to try it set extract_milestones/run_metadynamics=True:
        # n_milestones=3, mMD_use_funnel_potential=True, mMD_milestone_seeding=True,
        # mMD_time=10, mMD_bias_frequency=2, mMD_hill_width=0.05, mMD_hill_height=1.2, mMD_bias_factor=12
        extract_milestones=False,
        run_metadynamics=False,
    )

    # Phase 1: prepare and equilibrate (also writes RMSD, per-atom RMSF and contact frequency)
    AutoPath(pdb_path=receptor, do_fix_pdb=True,
             run_preparation=True, run_equilibration=True,
             run_sMDpulling=False, sMD_run_analysis=False, **settings).run(ligand_file=ligand)

    # Early filter: contact-weighted RMSF of the ligand over the equilibration
    eq = f"{sys_name}/equilibration"
    u = mda.Universe(f"{sys_name}/system.pdb", f"{eq}/equilibration_{sys_name}_aligned.dcd", in_memory=True)
    cw, _ = cw_rmsf_windows(u, "resname UNK", f"{eq}/equilibration_{sys_name}.csv", protocol)
    cw.to_csv(f"{eq}/{sys_name}_cwRMSF.csv")
    logger.info("cw-RMSF (A):\n" + cw.round(3).to_string())
    if args.max_polar_cwrmsf is not None:
        window = "free" if "free" in cw.index else "whole"
        polar = cw.loc[window, "cw_rmsf_polar"]
        if pd.isna(polar):
            logger.warning("Polar cw-RMSF undefined (no polar ligand atom contacts the pocket): filter not applied.")
        elif polar > args.max_polar_cwrmsf:
            logger.warning(f"Polar cw-RMSF ({window}) {polar:.2f} A > {args.max_polar_cwrmsf} A: pose unstable, "
                           "skipping steered MD.")
            return

    # Phase 2: steered MD, pathway clustering and free-energy / friction / koff analysis
    AutoPath(pdb_path=receptor, do_fix_pdb=False,
             run_preparation=False, run_equilibration=False,
             run_sMDpulling=True, sMD_run_analysis=True, **settings).run(ligand_file=ligand)


if __name__ == "__main__":
    main()
