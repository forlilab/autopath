"""Run a relaxed milestone state with one force at a time to find which one NaNs.

--arm plain    unbiased MD, no funnel, no bias
--arm funnel   funnel restraint only (zero hill height)
--arm bias     metadynamics bias only
"""
import argparse
import logging
import os

from openmm import XmlSerializer

from autopath.utils import setup_logging, load_system
from autopath.vanilla_md import VanillaMD
from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD
from autopath.metadynamics.CV import pathCV_cv
from autopath.metadynamics.Funnel import (create_funnel_force_from_trajectory_analysis,
                                          load_funnel_params)
from openmm.app import PDBFile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["plain", "funnel", "bias"])
    ap.add_argument("--milestone", default="milestone_2_frame_145")
    ap.add_argument("--ns", type=float, default=5.0)
    args = ap.parse_args()

    base = "3ptb"
    out = f"{base}/nan_diagnosis/{args.arm}"
    os.makedirs(out, exist_ok=True)
    setup_logging(f"{out}/diagnose.log", log_level="INFO")
    log = logging.getLogger("autopath")

    ms_dir = f"{base}/milestones"
    state = f"{ms_dir}/{args.milestone}_relax_state.xml"
    system_xml = f"{ms_dir}/{args.milestone}_relax_system.xml"
    topology = PDBFile(f"{base}/system.pdb").topology
    log.info("arm=%s milestone=%s state=%s", args.arm, args.milestone, state)

    if args.arm == "plain":
        md = VanillaMD(system=load_system(system_xml), topology=topology,
                       timestep=0.004, temperature=300,
                       out_dir=out, platform="OpenCL")
        md.run(state_file=state, run_id=f"plain_{args.milestone}", MD_time=args.ns)
        log.info("PLAIN ARM SURVIVED %.1f ns", args.ns)
        return

    # both biased arms need the milestone list for the path CV
    from glob import glob
    milestones = sorted(
        (m for m in glob(f"{ms_dir}/milestone_*_frame_*.xml")
         if "_relax" not in os.path.basename(m)),
        key=lambda x: int(os.path.basename(x).split("_frame_")[0].rsplit("_", 1)[-1]))
    log.info("milestones: %s", [os.path.basename(m) for m in milestones])

    u = PDBFile(f"{base}/system.pdb")
    import MDAnalysis as mda
    uni = mda.Universe(f"{base}/system.pdb")
    lig = uni.select_atoms("resname UNK and not name H*")
    pocket_residues = [180, 181, 182, 183, 226, 227, 228, 229, 230, 204, 205, 206,
                       207, 208, 209, 210, 211, 212, 213, 214, 215, 196, 197, 198,
                       199, 200, 201, 133, 134, 135, 136, 137, 138, 139, 140, 156,
                       157, 158, 159, 160, 161, 162]
    pocket = uni.select_atoms(f'resid {" ".join(map(str, pocket_residues))} and name CA')

    funnel_force = None
    if args.arm == "funnel":
        params = load_funnel_params(f"{base}/metadynamics_funnel/funnel_params.npz")
        funnel_force = create_funnel_force_from_trajectory_analysis(params)
        log.info("funnel arm: funnel force attached, bias hill height set to 0")

    metad = MetadynamicsMD(topology=topology,
                           ligand_atoms=[a.index for a in lig],
                           pocket_atoms=[a.index for a in pocket],
                           restrained_atoms=None, timestep=0.004, temp=300,
                           out_dir=out, platform="OpenCL")
    cv = pathCV_cv(topology=topology, milestones=milestones,
                   pocket_atoms=[a.index for a in pocket],
                   ligand_atoms=[a.index for a in lig],
                   system=load_system(f"{base}/system.xml"),
                   grid_min=0.0, grid_max=1.0, hill_width=0.05, sigma="auto")

    # funnel arm: no hills, so only the funnel can destabilise it
    hill_height = 0.0 if args.arm == "funnel" else 1.2
    metad.run(state_xml_file=state, system=load_system(system_xml),
              run_id=f"{args.arm}_{args.milestone}", cv_specs=[cv],
              mMD_time=args.ns, bias_factor=15, hill_height=hill_height,
              biasFrequency=2, funnel_force=funnel_force,
              funnel_params=None, bias_dir=out)
    log.info("%s ARM SURVIVED %.1f ns", args.arm.upper(), args.ns)


if __name__ == "__main__":
    main()
