"""Contact-weighted ligand RMSF (cw-RMSF) for every equilibrated system.

Scans ``<eq_root>/*/equilibration/equilibration_*_aligned.dcd`` (the output of
``01_Build_and_Equilibrate``) and, for each system, computes the per-atom RMSF and
pocket-contact frequency of the ligand heavy atoms and the cw-RMSF over polar
(N, O, S), non-polar (C, F, Cl, Br, I) and all heavy atoms, for the whole run and
for the free window (frames after the ligand restraint is released, located from
the run's ``equilibration_protocol.json``).

Writes ``cwrmsf/<sys>_RMSF_{whole,free}.csv`` (per atom) and ``cwrmsf/cwrmsf_summary.csv``.
Lower cw-RMSF = more stable pose.

Usage
-----
    python run_cwrmsf.py [--eq_root ../01_Build_and_Equilibrate] [--ligand "resname UNK"] [-j 8]
"""
import argparse
import os
from glob import glob
from multiprocessing import Pool

import pandas as pd


def score_system(job):
    import MDAnalysis as mda
    from autopath.ap_PLIP import ligand_rmsf_contacts, cw_rmsf_scores
    from autopath.equilibration import free_window_start

    sysname, run_dir, ligand, outdir = job
    eq = f"{run_dir}/equilibration"
    u = mda.Universe(f"{run_dir}/system.pdb", f"{eq}/equilibration_{sysname}_aligned.dcd", in_memory=True)
    free_start = free_window_start(f"{eq}/equilibration_{sysname}.csv", f"{eq}/equilibration_protocol.json",
                                   component="ligand", n_frames=len(u.trajectory))
    rows = []
    for window, start in (("whole", None), ("free", free_start)):
        per_atom = ligand_rmsf_contacts(u, ligand, start=start)
        per_atom.to_csv(f"{outdir}/{sysname}_RMSF_{window}.csv", index=False)
        rows.append({"system": sysname, "window": window, "n_frames": len(u.trajectory) - (start or 0),
                     **cw_rmsf_scores(per_atom, on_undefined="nan")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eq_root", default="../01_Build_and_Equilibrate",
                    help="directory holding one <sys>/ folder per equilibrated system")
    ap.add_argument("--ligand", default="resname UNK", help="MDAnalysis selection of the ligand")
    ap.add_argument("--outdir", default="cwrmsf")
    ap.add_argument("-j", "--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()

    trajs = sorted(glob(f"{args.eq_root}/*/equilibration/equilibration_*_aligned.dcd"))
    print(f"Found {len(trajs)} equilibrated systems.")
    os.makedirs(args.outdir, exist_ok=True)
    jobs = []
    for traj in trajs:
        run_dir = os.path.dirname(os.path.dirname(traj))
        jobs.append((os.path.basename(run_dir), run_dir, args.ligand, args.outdir))

    if args.n_jobs > 1:
        with Pool(args.n_jobs) as pool:
            results = pool.map(score_system, jobs)
    else:
        results = [score_system(j) for j in jobs]

    summary = pd.DataFrame([r for rows in results for r in rows])
    summary.to_csv(f"{args.outdir}/cwrmsf_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
