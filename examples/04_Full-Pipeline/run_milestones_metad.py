"""Milestone extraction + funnel metadynamics from an existing sMD run.

Everything upstream of milestone extraction is switched off and read back from
disk: preparation, equilibration, sMD pulling and sMD analysis all already ran.
The medoid trajectories and medoid_info.json under sMD/analysis_traces are the
only inputs needed here.

3 walkers, each seeded from its own milestone. No bias pre-seeding, no shared
(multiple-walker) bias.
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
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    receptor = args.rec
    ligand = args.lig
    protocol = args.protocol
    sys_name = os.path.basename(ligand).split('.')[0]
    os.makedirs(sys_name, exist_ok=True)

    # Separate log file on purpose: appending to autopath.log would interleave
    # this run with the earlier sMD run's lines and make attribution ambiguous.
    logger = setup_logging(f"{sys_name}/autopath_metad.log", log_level="INFO")

    # Same pocket definition as the sMD run that produced the trajectories being
    # reused; these are system.pdb sequential resids, not original PDB numbering.
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

                  # 2 fs, NOT the 4 fs default. This system was prepared with
                  # hydrogenMass=1.5 amu (1605 solute H at 1.5; water H stay at
                  # 1.008 and are rigid), which is too little repartitioning for
                  # 4 fs -- marginally stable, so a walker blows up at random.
                  # A 5 ns walker died with "Particle position is NaN" at 1.7 ns
                  # at 4 fs while two others survived. run_preparation=False
                  # reuses the existing masses, so the timestep is the only knob.
                  timestep=0.002,

                  # OpenCL, not CUDA: this is what the sMD run used on these same
                  # RTX PRO 6000 (Blackwell) nodes.
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
                      0.020: 5,
                      0.015: 5,
                      0.010: 5,
                  },
                  # Analysis already ran; medoids are read from
                  # sMD/analysis_traces/path_analysis/medoid_info.json.
                  sMD_run_analysis=False,

                  # --- milestone extraction ---
                  extract_milestones=True,
                  n_milestones=3,
                  milestone_mode='all_medoids',

                  # --- metadynamics ---
                  run_metadynamics=True,
                  mMD_use_funnel_potential=True,
                  mMD_preseed_bias=False,       # no FES pre-filling
                  mMD_multiple_walkers=False,   # independent bias per walker
                  mMD_milestone_seeding=True,   # each walker starts at its own milestone

                  mMD_time=5,  # ns per walker
                  mMD_bias_frequency=2,
                  mMD_hill_width=0.05,
                  mMD_hill_height=1.2,
                  mMD_bias_factor=15,
                  )

    ap.run(ligand_file=ligand)


if __name__ == "__main__":
    main()
