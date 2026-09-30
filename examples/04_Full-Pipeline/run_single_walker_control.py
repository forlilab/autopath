"""Single-walker control for the path-seeding comparison.

One walker for 3 x mMD_time, started from the equilibrated bound pose. This is
the reference a path-seeded run has to beat: same GPU, same total sampling, same
path CV and funnel, only the restart points differ.

Everything that could shift the free-energy surface is pinned to the seeded run:
the funnel is loaded from its saved parameters rather than re-derived, and the
path CV is built from the same milestone files in the same bound -> unbound
order, so the [0, 1] progress coordinate means the same thing in both.
"""
import argparse
import logging
import os
from glob import glob

import MDAnalysis as mda
from openmm.app import PDBFile
from rdkit import Chem

from autopath.clustering import order_milestones_bound_to_unbound
from autopath.utils import (setup_logging, load_system, get_pocket_atoms_idxs,
                            get_ligand_anchor_atoms)
from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD
from autopath.metadynamics.CV import pathCV_cv
from autopath.metadynamics.Funnel import (create_funnel_force_from_trajectory_analysis,
                                          load_funnel_params)

POCKET_RESIDUES = [180, 181, 182, 183, 226, 227, 228, 229, 230, 204, 205, 206,
                   207, 208, 209, 210, 211, 212, 213, 214, 215, 196, 197, 198,
                   199, 200, 201, 133, 134, 135, 136, 137, 138, 139, 140, 156,
                   157, 158, 159, 160, 161, 162]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="3ptb")
    ap.add_argument("--ns", type=float, default=15.0)
    ap.add_argument("--funnel-from", default="3ptb/metadynamics_funnel/funnel_params.npz",
                    help="reuse the seeded run's funnel instead of re-deriving it")
    ap.add_argument("--out", default="3ptb/metadynamics_single_walker")
    ap.add_argument("--seed-from", default="equil",
                    help="'equil' for the equilibrated pose, or a milestone name "
                         "like milestone_2_frame_145 to seed along the path")
    ap.add_argument("--bias-dir", default=None,
                    help="shared bias directory; concurrent walkers pointing at "
                         "the same one build a single surface together")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--extend-path-to-nm", dest="extend_path", type=float,
                    default=None)
    ap.add_argument("--lig", default="../data/3ptb.sdf")
    ap.add_argument("--expect-cv-length", type=float, default=11.72,
                    help="Å per unit of progress in the run being compared "
                         "against; the control aborts if its own CV disagrees")
    args = ap.parse_args()

    base = args.base
    os.makedirs(args.out, exist_ok=True)
    setup_logging(f"{args.out}/autopath_{args.run_id or args.seed_from}.log",
                  log_level="INFO")
    log = logging.getLogger("autopath")

    # Selections have to come from the same helpers the pipeline uses. A plain
    # heavy-atom selection is NOT the same anchor: the run being compared against
    # pulls on the Murcko scaffold, and the two differ by ~1 Å of CV length.
    pocket_selection = f'resid {" ".join(map(str, POCKET_RESIDUES))} and name CA'
    eq_pdb = f"{base}/equilibration/3ptb_equilibrated.pdb"
    eq_traj = f"{base}/equilibration/equilibration_3ptb_aligned.dcd"
    u_eq = (mda.Universe(eq_pdb, eq_traj, in_memory=True)
            if os.path.exists(eq_traj) else mda.Universe(eq_pdb))
    pocket = get_pocket_atoms_idxs(u_eq, pocket_selection)
    lig = get_ligand_anchor_atoms(u_eq, "resname UNK", mode="murcko", n_atoms=5,
                                  out_dir=args.out,
                                  ref_mol=Chem.SDMolSupplier(args.lig)[0])
    log.info("pocket atoms: %d; ligand anchor atoms: %s", len(pocket), list(lig))
    topology = PDBFile(f"{base}/system.pdb").topology

    # Same milestone files and same direction as the seeded run: the path CV has
    # to be identical or the two dG(t) curves are not on the same coordinate.
    milestones = sorted(
        (m for m in glob(f"{base}/milestones/milestone_*_frame_*.xml")
         if "_relax" not in os.path.basename(m)),
        key=lambda x: int(os.path.basename(x).split("_frame_")[0].rsplit("_", 1)[-1]))
    milestones = order_milestones_bound_to_unbound(milestones, pocket, lig)
    log.info("path CV milestones: %s", [os.path.basename(m) for m in milestones])

    params = load_funnel_params(args.funnel_from)
    funnel_force = create_funnel_force_from_trajectory_analysis(params)
    log.info("funnel reused from %s", args.funnel_from)

    if args.seed_from == "equil":
        state = f"{base}/equilibration/checkpoint_equil_3ptb.xml"
        system_xml = f"{base}/equilibration/system_equil_3ptb.xml"
        log.info("starting from the equilibrated pose: %s", state)
    else:
        state = f"{base}/milestones/{args.seed_from}_relax_state.xml"
        system_xml = f"{base}/milestones/{args.seed_from}_relax_system.xml"
        log.info("starting from milestone %s: %s", args.seed_from, state)
    for _f in (state, system_xml):
        if not os.path.exists(_f):
            raise SystemExit(f"missing seed file: {_f}")

    cv = pathCV_cv(topology=topology, milestones=milestones,
                   pocket_atoms=pocket, ligand_atoms=lig,
                   system=load_system(f"{base}/system.xml"),
                   grid_min=0.0, grid_max=1.0, hill_width=0.05, sigma="auto",
                   extend_path_to_nm=args.extend_path)

    # Fail loudly rather than produce a curve on a different coordinate: the
    # comparison is only meaningful if one unit of progress means the same
    # distance in both runs.
    got = cv.cv_length_per_unit_ang
    if got is None or abs(got - args.expect_cv_length) > 0.01:
        raise SystemExit(
            f"CV length is {got} Å/unit but the run being compared against has "
            f"{args.expect_cv_length}. The selections or milestones differ, so the "
            "two ΔG(t) curves would not be comparable.")
    log.info("CV length matches the seeded run: %.2f Å per unit of progress", got)

    metad = MetadynamicsMD(topology=topology, ligand_atoms=lig,
                           pocket_atoms=pocket, restrained_atoms=None,
                           timestep=0.004, temp=300,
                           out_dir=args.out, platform="OpenCL")
    run_id = args.run_id or f"walker_{args.seed_from}"
    bias_dir = args.bias_dir or args.out
    os.makedirs(bias_dir, exist_ok=True)
    log.info("run_id=%s bias_dir=%s (%s)", run_id, bias_dir,
             "SHARED with concurrent walkers" if args.bias_dir else "private")
    metad.run(state_xml_file=state, system=load_system(system_xml),
              run_id=run_id, cv_specs=[cv],
              mMD_time=args.ns, bias_factor=15, hill_height=1.2,
              biasFrequency=2, funnel_force=funnel_force,
              funnel_params=params, bias_dir=bias_dir)
    log.info("finished: %.1f ns, seed=%s", args.ns, args.seed_from)


if __name__ == "__main__":
    main()
