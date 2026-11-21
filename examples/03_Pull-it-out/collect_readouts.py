"""Collect one ΔG readout per ligand from several analysed sMD systems, for ranking.

Scans ``<root>/*/sMD/analysis_traces/`` (output of ``analyze_sMD.py`` here, or of
``04_Full-Pipeline``) and writes ``readouts.csv`` with, per system, estimator and speed:

- ``dG_profile``: ΔG at the most prominent PMF peak for the cumulant, the profile
  maximum for Jarzynski and force (affinity ranking within a series).
- ``dG_rupture``: ΔG at d*, the median over all systems found of the force-peak
  (rupture) displacement (screening). Rerun whenever the set of systems changes.

Cumulant and Jarzynski are read per pulling speed, force at v->0 only.

Usage
-----
    python collect_readouts.py [--root .] [--out readouts.csv]
"""
import argparse
import os
from glob import glob

import pandas as pd

from autopath.pulling.readouts import profile_readouts, rupture_displacement, rupture_readouts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="folder holding one <sysname>/ folder per analysed system")
    ap.add_argument("--out", default="readouts.csv")
    args = ap.parse_args()

    dirs = sorted(d for d in glob(f"{args.root}/*/sMD/analysis_traces")
                  if os.path.isfile(f"{d}/mixture_pmfs.csv") and os.path.isfile(f"{d}/sMD_processed_data.csv"))
    if not dirs:
        raise SystemExit(f"no <sysname>/sMD/analysis_traces with mixture_pmfs.csv under {args.root}")
    print(f"Found {len(dirs)} analysed systems.")

    pmfs, ruptures, prof, npulls = {}, {}, [], []
    for d in dirs:
        sysname = d.split(os.sep)[-3]
        pmfs[sysname] = pd.read_csv(f"{d}/mixture_pmfs.csv")
        proc = pd.read_csv(f"{d}/sMD_processed_data.csv", usecols=["trajname", "speed", "step", "r_coord", "force"])
        ruptures[sysname] = rupture_displacement(proc)
        prof.append(profile_readouts(pmfs[sysname]).assign(system=sysname))
        npulls.append(proc.groupby("speed").trajname.nunique().rename("n_pulls").reset_index().assign(system=sysname))

    out = rupture_readouts(pmfs, ruptures).merge(pd.concat(prof), on=["system", "estimator", "speed"], how="outer")
    n = pd.concat(npulls)
    out = out.merge(n, on=["system", "speed"], how="left")
    # force at v->0 pools every speed
    v0 = n.groupby("system").n_pulls.sum()
    out.loc[out.speed == 0.0, "n_pulls"] = out.loc[out.speed == 0.0, "system"].map(v0)
    out = out[["system", "estimator", "speed", "n_pulls", "dG_profile", "peak_fell_back", "d_star", "dG_rupture"]]
    out.sort_values(["estimator", "speed", "system"]).to_csv(args.out, index=False)
    print(out.pivot_table(index="system", columns=["estimator", "speed"], values=["dG_profile", "dG_rupture"])
          .round(1).to_string())
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
