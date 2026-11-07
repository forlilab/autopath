import os
import argparse
from glob import glob

import MDAnalysis as mda
from rdkit import Chem
from openmm.app import PDBFile

from autopath import SteeredMD
from autopath.utils import (load_system, setup_logging, wrap_align_save_traj,
                            get_pocket_atoms_idxs, get_ligand_anchor_atoms)


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs steered molecular dynamics (sMD) pulls for an equilibrated protein-ligand complex.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )
    parser.add_argument("-s", "--sysname", dest="sysname", required=True,
                        help="system name; outputs go to <sysname>/sMD")
    parser.add_argument("--eqdir", default=None,
                        help="folder of the equilibrated system (default: ../01_Build_and_Equilibrate/<sysname>)")
    parser.add_argument("-l", "--lig", default=None,
                        help="ligand SDF, only used to draw the anchor atoms (optional)")
    parser.add_argument("--ligand_sel", default="resname UNK", help="MDAnalysis selection of the ligand")
    parser.add_argument("--pocket_sel", default=None,
                        help="MDAnalysis selection of the pocket atoms, in system.pdb numbering "
                             "(default: CA atoms of residues within 8 A of the ligand)")
    parser.add_argument("--anchor", default="murcko",
                        choices=["murcko", "lig_ha", "lig_com", "pocket_com", "contacts", "inertia"],
                        help="ligand atoms pulled by the restraint (default: murcko)")
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.005, 0.010, 0.015],
                        help="pulling speeds in nm/ps (0.001 nm/ps = 1 m/s; default: 5, 10, 15 m/s)")
    parser.add_argument("--reps", type=int, default=10, help="replicas per speed")
    return parser.parse_args()


def main():

    args = cmd_lineparser()
    sys_name = args.sysname
    eq_root = args.eqdir or f"../01_Build_and_Equilibrate/{sys_name}"
    eq_dir = f"{eq_root}/equilibration"

    # Settings used in the AutoPath paper
    TIMESTEP = 0.004        # ps (needs HMR)
    TEMPERATURE = 300       # K
    DX_PER_MOVE = 0.001     # nm, restraint-centre increment
    MAX_DISPLACEMENT = 3.5  # nm, maximum pulling distance
    SPRING_PER_ATOM = 100 * 4.184  # kJ/mol/nm2 per ligand heavy atom (418.4)
    AUTOSTOP_NC = 0.01      # stop once ligand-pocket contacts < 1% of the initial value ...
    AUTOSTOP_WINDOW = 5     # ... for this many consecutive samples

    os.makedirs(sys_name, exist_ok=True)
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")

    system = load_system(f"{eq_dir}/system_equil_{sys_name}.xml")
    checkpoint = f"{eq_dir}/checkpoint_equil_{sys_name}.chk"
    state_xml = f"{eq_dir}/checkpoint_equil_{sys_name}.xml"
    system_pdb = f"{eq_root}/system.pdb"
    eq_pdb = f"{eq_dir}/{sys_name}_equilibrated.pdb"
    topology = PDBFile(system_pdb).topology

    # Pocket and ligand anchor atoms (system.pdb numbering), from the equilibration
    u = mda.Universe(eq_pdb, f"{eq_dir}/equilibration_{sys_name}_aligned.dcd")
    lig_ha = f"({args.ligand_sel}) and not name H*"
    if args.pocket_sel is not None:
        pocket_idx = get_pocket_atoms_idxs(u, pocket_selection=args.pocket_sel)
    else:
        pocket_idx = get_pocket_atoms_idxs(u, ligand_selection=lig_ha)
    ref_mol = Chem.SDMolSupplier(args.lig)[0] if args.lig else None
    anchor_idx = get_ligand_anchor_atoms(u, args.ligand_sel, mode=args.anchor, n_atoms=5,
                                         out_dir=sys_name, ref_mol=ref_mol)
    n_heavy = len(u.select_atoms(lig_ha))
    logger.info(f"{len(pocket_idx)} pocket atoms; {len(anchor_idx)} '{args.anchor}' anchor atoms")

    ########################################################################################
    ###################################### Steered MD ######################################
    ########################################################################################

    # The spring scales with the total number of ligand heavy atoms (not anchor atoms)
    spring = SPRING_PER_ATOM * n_heavy
    logger.info(f"sMD spring constant {spring:.1f} kJ/mol/nm2 for {n_heavy} ligand heavy atoms")

    smd_dir = f"{sys_name}/sMD/trajectories"
    sMD = SteeredMD(
        system=system,
        topology=topology,
        groupA_atoms=anchor_idx,
        groupB_atoms=pocket_idx,
        restart_velocities=True,
        timestep=TIMESTEP,
        temperature=TEMPERATURE,
        out_dir=smd_dir,
        dx_per_move=DX_PER_MOVE,
        max_displacement=MAX_DISPLACEMENT,
        sMD_spring_cte=spring,
        autostop_nc=AUTOSTOP_NC,
        autostop_nc_window=AUTOSTOP_WINDOW,
        autostop_min_displacement=0.5,
        save_freq=5,
        # Per-frame geometry (contacts, min-distance, ligand shape) logged to
        # sMD_<run_id>_geom.dat; used as clustering features by analyze_sMD.py
        log_geom_features=True,
    )

    # Each replica starts from the equilibrated state with new velocities,
    # relaxes 50 ps with the spring at its start position, then pulls forward.
    for speed in args.speeds:
        for rep in range(1, args.reps + 1):
            run_id = sMD.run(pulling_speed=speed, pulling_direction="forward",
                             checkpoint_file=checkpoint, state_xml_file=state_xml)
            logger.info(f"Completed sMD replica {rep}/{args.reps} at {speed} nm/ps: {run_id}")

    # Wrap, centre and superpose the pulling trajectories
    trajs = [f for f in sorted(glob(f"{smd_dir}/sMD_replica-*.dcd")) if "aligned" not in f]
    wrap_align_save_traj(trajs, system_pdb)


if __name__ == "__main__":
    main()
