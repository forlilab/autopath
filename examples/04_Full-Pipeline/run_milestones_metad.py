"""Milestone extraction + funnel metadynamics on top of an existing sMD run.

Preparation, equilibration, sMD pulling and analysis are skipped; the medoid
trajectories and medoid_info.json under sMD/analysis_traces are reused.

--walkers     shared (one bias, walkers run sequentially) or isolated
--seed-order  order shared walkers are seeded in
--cv          path progress or funnel z coordinate
--no-extract  reuse existing milestones
"""
import os
import argparse
from autopath.autopath_core import AutoPath
from autopath.utils import setup_logging


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Milestone extraction + funnel metadynamics for an existing sMD run.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )
    parser.add_argument("-r", "--rec", dest="rec", required=True, action="store",
                        help="path to the receptor PDB file")
    parser.add_argument("-l", "--lig", dest="lig", required=True, action="store",
                        help="path to the ligand SDF file")
    parser.add_argument("-p", "--protocol", dest="protocol", required=True, action="store",
                        help="path to the JSON file with the equilibration protocol")
    parser.add_argument("--seed-order", dest="seed_order", default="unbound_first",
                        choices=["unbound_first", "bound_first"],
                        help="order sequential shared-bias walkers are seeded in")
    parser.add_argument("--outdir-suffix", dest="suffix", default="")
    parser.add_argument("--walkers", choices=["shared", "isolated"], default="shared",
                        help="shared: one bias, walkers run one after another; "
                             "isolated: every walker its own bias")
    parser.add_argument("--n-milestones", dest="n_milestones", type=int, default=3)
    parser.add_argument("--ns", type=float, default=5.0, help="ns per walker")
    parser.add_argument("--cv", choices=["path", "funnel_z"], default="path")
    parser.add_argument("--no-extract", dest="extract", action="store_false",
                        help="reuse the milestones already in <sys>/milestones")
    parser.add_argument("--boundary-correction", action=argparse.BooleanOptionalAction, default=True,
                        help="divide each hill by its fraction inside the walls (McGovern-de Pablo)")
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    receptor = args.rec
    ligand = args.lig
    protocol = args.protocol
    sys_name = os.path.basename(ligand).split('.')[0]
    os.makedirs(sys_name, exist_ok=True)

    # separate log so it does not interleave with the sMD run's autopath.log
    logger = setup_logging(f"{sys_name}/autopath_metad{args.suffix}.log", log_level="INFO")

    # must match the sMD run; system.pdb sequential resids, not PDB numbering
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


                  # same platform as the sMD run
                  platform='OpenCL',

                  run_equilibration=False,
                  protocol_fname=protocol,

                  # --- reuse the existing sMD run ---
                  run_sMDpulling=False,
                  sMD_outdir='sMD',
                  sMD_ligand_anchor_mode='murcko',
                  sMD_converge_speeds=True,
                  sMD_max_replicas=50,
                  sMD_pulling_speeds={
                      0.020: 10,
                      0.015: 10,
                      0.010: 10,
                  },
                  # medoids read from sMD/analysis_traces/path_analysis/medoid_info.json
                  sMD_run_analysis=False,

                  # --- milestone extraction ---
                  extract_milestones=args.extract,
                  n_milestones=args.n_milestones,
                  milestone_mode='all_medoids',

                  # --- metadynamics ---
                  run_metadynamics=True,
                  mMD_use_funnel_potential=True,
                  mMD_preseed_bias=False,
                  mMD_multiple_walkers=args.walkers == "shared",
                  mMD_cv=args.cv,
                  mMD_seed_order=args.seed_order,
                  mMD_outdir_suffix=args.suffix,
                  mMD_milestone_seeding=True,   # each walker starts at its own milestone

                  mMD_time=args.ns,  # ns per walker
                  mMD_bias_frequency=2,
                  mMD_hill_width=0.05,
                  mMD_hill_height=1.2,
                  mMD_bias_factor=15,
                  mMD_boundary_correction=args.boundary_correction,
                  )

    ap.run(ligand_file=ligand)


if __name__ == "__main__":
    main()
