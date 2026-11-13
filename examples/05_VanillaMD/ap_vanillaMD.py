import os
import argparse

import pandas as pd
import MDAnalysis as mda

from autopath import VanillaMD
from autopath.ap_PLIP import plot_atomic_property, ligand_rmsf_contacts, cw_rmsf_scores
from autopath.utils import load_system, compute_rmsd, setup_logging, wrap_align_save_traj

from openmm.app import PDBFile

def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description="Runs vanilla MD of a system, that has been previously equilibrated.",
        epilog="""
        COPYRIGHT
                Copyright (C) 2026 Forli Lab, Center for Computational Structural Biology,
                             Scripps Research.""",
    )

    parser.add_argument(
        "-s",
        "--sysname",
        dest="sysname",
        required=True,
        action="store",
        help="Name of the system, used to create output directory and files",
    )
    
    parser.add_argument(
        "-n",
        "--ligname",
        dest="ligname",
        required=False,
        default=None,
        help="Residue name of the ligand, if present (e.g. UNK).",
    )

    parser.add_argument(
        "--eqdir",
        dest="eqdir",
        default=None,
        help="Folder of the equilibrated system (default: ../01_Build_and_Equilibrate/<sysname>)",
    )

    parser.add_argument(
        '-r',
        '--replica',
        dest='replica',
        required=False,
        default=None,
        help='Replica name, used to differentiate multiple runs on the same system.'
    )
    return parser.parse_args()


def main():

    args = cmd_lineparser()
    lig_resname = args.ligname
    sys_name = args.sysname
    replica = args.replica
    if replica is None:
        run_id = sys_name
    else:
        run_id = f'{sys_name}_{replica}'
    
    eq_root = args.eqdir or f"../01_Build_and_Equilibrate/{sys_name}"
    os.makedirs(sys_name, exist_ok=True)

    # Setup logging
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")
    
    ########################################################################################
    ######################################## Vanilla MD ####################################
    ########################################################################################
    
    MD_TIME = 100  # ns
    TIMESTEP = 0.004  # 4 fs timestep
    TEMPERATURE = 300  # 300 K
    SAVE_FREQ = 25000  # save /0.1ns at 4fs
    RESTART_VELOCITIES = True  # whether to restart velocities from checkpoint
    RESTRAINED_ATOMS = None  # list of atom indices to restrain, or None for no restraints
    
    system_pdb_file = f"{eq_root}/system.pdb"
    topology = PDBFile(system_pdb_file).topology
    
    system = load_system(f"{eq_root}/equilibration/system_equil_{sys_name}.xml")
    checkpoint = f"{eq_root}/equilibration/checkpoint_equil_{sys_name}.chk"

    vanilla_md = VanillaMD(
        system=system,
        topology=topology,
        restrained_atoms=RESTRAINED_ATOMS,
        timestep=TIMESTEP,
        temperature=TEMPERATURE,
        save_freq=SAVE_FREQ,
        out_dir=f"{sys_name}/MD"
        )
    
    system = vanilla_md.run(
                            checkpoint_file=checkpoint,
                            pdb_file=None, 
                            run_id=run_id,
                            MD_time=MD_TIME,
                            restart_velocities=RESTART_VELOCITIES
                            )

    traj_fname = f"{sys_name}/MD/MD_{run_id}.dcd"

    # Wrap, centre and superpose on the protein backbone -> MD_<run_id>_aligned.dcd
    aligned_traj = wrap_align_save_traj(traj_fname, system_pdb_file)[0]
    logger.info(f"Aligned trajectory saved to {aligned_traj}")

    u = mda.Universe(system_pdb_file, aligned_traj, in_memory=True)

    groups = {"protein": "protein and not name H*"}
    if lig_resname is not None:
        groups["ligand"] = f"resname {lig_resname} and not name H*"
    rmsd_df = compute_rmsd(u, u, alig_select="backbone", groupselections=groups,
                           plots_outdir=f"{sys_name}/MD", suffix=run_id)
    rmsd_df.to_csv(f"{sys_name}/MD/{run_id}_rmsd.csv", index=False)

    if lig_resname is not None:
        # Ligand per-atom RMSF / pocket-contact frequency and contact-weighted RMSF
        # (polar N/O/S, non-polar C/halogens, all heavy atoms; NaN if a set never contacts the pocket)
        lig_sel = f"resname {lig_resname}"
        per_atom = ligand_rmsf_contacts(u, lig_sel)
        per_atom.to_csv(f"{sys_name}/MD/{run_id}_RMSF.csv", index=False)
        cw = pd.DataFrame([cw_rmsf_scores(per_atom, on_undefined="nan")])
        cw.to_csv(f"{sys_name}/MD/{run_id}_cwRMSF.csv", index=False)
        logger.info("cw-RMSF (A):\n" + cw.round(3).to_string(index=False))
        plot_atomic_property(u, per_atom.rmsf.values, lig_resname=lig_sel,
                             outname=f"{sys_name}/MD/{run_id}_RMSF.svg")
        plot_atomic_property(u, per_atom.contact_freq.values, lig_resname=lig_sel,
                             outname=f"{sys_name}/MD/{run_id}_contact_freq.svg", color='r')

    return

if __name__ == "__main__":
    main()
