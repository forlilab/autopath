"""Prepare MM-GBSA jobs (AMBER MMPBSA.py) for every equilibrated system.

One job per system and GB model, all frames of the aligned equilibration
trajectory, solvent and ions stripped. The four GB models of the AutoPath paper
are available; which one ranks best is target dependent.

Usage
-----
    python orchestrator_mmgbsa.py [--gb OBC2 GBn2] [--persistent_waters 0.9]
    bash run_mmgbsa_batch.sh
"""
import argparse
import os
from glob import glob

from autopath.ap_PLIP import ProteinLigandAnalyzer

# GB model -> (igb, radii set)
GB_MODELS = {
    "HCT":  (1, "mbondi"),
    "OBC2": (5, "mbondi2"),
    "GBn":  (7, "bondi"),
    "GBn2": (8, "mbondi3"),
}
TEMPLATE = "mmgbsa_igb8.in"   # igb is replaced per model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eq_root", default="../01_Build_and_Equilibrate",
                    help="directory holding one <sys>/ folder per equilibrated system")
    ap.add_argument("--gb", nargs="+", default=list(GB_MODELS), choices=list(GB_MODELS),
                    help="GB models to run (default: all four)")
    ap.add_argument("--persistent_waters", type=float, default=None,
                    help="keep interfacial waters present in at least this fraction of frames "
                         "(default: strip all waters, as in the paper)")
    args = ap.parse_args()

    trajectories = sorted(glob(f"{args.eq_root}/*/equilibration/equilibration_*_aligned.dcd"))
    print(f"Found {len(trajectories)} systems to process.")
    template = open(TEMPLATE).read()

    for gb in args.gb:
        igb, radii = GB_MODELS[gb]
        mmpbsa_in = f"mmgbsa_{gb}.in"
        with open(mmpbsa_in, "w") as f:
            f.write(template.replace("igb=8,", f"igb={igb},"))
        experiment = f"mmgbsa_{gb}" + ("_WAT" if args.persistent_waters is not None else "")

        for trajectory in trajectories:
            sysname = os.path.basename(os.path.dirname(os.path.dirname(trajectory)))
            print(f"Processing system: {sysname} ({gb}: igb={igb}, {radii})")
            ProteinLigandAnalyzer.prepare_mmgbsa_batch(
                f"{sysname}_{gb}",  # job/qfile name, unique per model
                f"{args.eq_root}/{sysname}/system.prmtop",
                trajectory,
                output_folder=f"{sysname}/{experiment}",
                mmpbsa_in=mmpbsa_in,
                radii=radii,
                slurm_template_fname=None,  # bundled template; or 'mmgbsa_slurm_template.q'
                persistent_waters_cutoff=args.persistent_waters,
                # traj_slice=(start, end, step),  # 1-based MMPBSA frames; default: all
                # mpi_threads=64,
            )


if __name__ == "__main__":
    main()
