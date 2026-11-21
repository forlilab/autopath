"""Persistent-contact and all-residue LIE for one equilibrated system.

Usage
-----
    python run_LIE.py -s SYS [--eq_root ../01_Build_and_Equilibrate] [--resname UNK]
"""
import argparse
import os

import MDAnalysis as mda

from autopath.ap_PLIP import ProteinLigandAnalyzer


def to_netcdf(top, traj, out):
    """Rewrite ``traj`` as AMBER NetCDF; cpptraj misreads the DCD unit cell."""
    u = mda.Universe(top, traj)
    with mda.Writer(out, u.atoms.n_atoms) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--sys", dest="sysname", required=True, help="system name")
    parser.add_argument("--eq_root", default="../01_Build_and_Equilibrate",
                        help="directory holding one <sys>/ folder per equilibrated system")
    parser.add_argument("--resname", default="UNK", help="ligand residue name (as in 01_Build_and_Equilibrate)")
    args = parser.parse_args()
    sysname = args.sysname
    top = f"{args.eq_root}/{sysname}/system.prmtop"
    os.makedirs(f"{sysname}/lie", exist_ok=True)
    traj = to_netcdf(top, f"{args.eq_root}/{sysname}/equilibration/equilibration_{sysname}_aligned.dcd",
                     f"{sysname}/lie/equilibration_{sysname}_aligned.nc")

    aplip = ProteinLigandAnalyzer(
        top=top,
        trajs=[traj],
        ligand_mda_selection=f"resname {args.resname}",
        protein_mda_selection='protein',
        outdir=f"{sysname}/lie")

    important_residues, persistence_byRes, persistence_byRes_byType = aplip.get_persistent_interactions(frequency_cutoff=0.5, n_jobs=1)
    print("Important residues:", important_residues)
    print("Computing LIE for important residues...")
    lie_df = aplip.compute_LIE(use_residues=important_residues,
                               ligand_amber_selection=f":{args.resname}")
    aplip.plot_lie_components(lie_df)
    lie_df.to_csv(f"{sysname}/lie/LIE_importance05.csv")

    print("Computing LIE for all residues...")
    lie_df = aplip.compute_LIE(use_residues=None,
                               ligand_amber_selection=f":{args.resname}")
    aplip.plot_lie_components(lie_df)
    lie_df.to_csv(f"{sysname}/lie/LIE_ALL.csv")


if __name__ == "__main__":
    main()
