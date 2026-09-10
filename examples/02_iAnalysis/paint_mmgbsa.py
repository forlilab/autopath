"""Color a structure by per-residue MMGBSA energy decomposition.

Walks the ``mmgbsa_*`` subdirectories of an equilibration run, parses each
``FINAL_DECOMP_*.dat``, and writes B-factor painted PDBs (plus the per-residue
bar charts) into ``<mmgbsa_dir>/painted/``.

The decomposition is reported in *stripped complex* numbering (protein 1..N,
ligand N+1), which is the numbering carried by ``complex.prmtop`` -- not by the
solvated ``system.prmtop``, where the ligand sits after the waters. So the dry
complex topology is the one used for painting, and matching coordinates are
carved out of the solvated equilibrated PDB.

Usage
-----
    python paint_mmgbsa.py <equilibration_dir> [--pdb PDB] [--normalize]
"""

import argparse
import os
from glob import glob

import MDAnalysis as mda

from autopath.ap_PLIP import ProteinLigandAnalyzer

# resnames dropped when carving the dry complex out of the solvated system
SOLVENT = "HOH WAT SOL NA CL K MG Na+ Cl- K+ Mg+ POP"


def build_dry_pdb(system_prmtop, coord_pdb, complex_prmtop, out_pdb):
    """Write coordinates matching ``complex_prmtop`` from a solvated frame.

    Verifies atom-for-atom correspondence before writing; MMPBSA's stripped
    topology has to line up with the coordinates or the B-factors land on the
    wrong atoms.
    """
    sol = mda.Universe(system_prmtop, coord_pdb)
    dry = sol.select_atoms(f"not resname {SOLVENT}")
    cx = mda.Universe(complex_prmtop)

    if dry.n_atoms != cx.atoms.n_atoms:
        raise ValueError(
            f"stripped system has {dry.n_atoms} atoms but {complex_prmtop} has "
            f"{cx.atoms.n_atoms}; the strip_mask and SOLVENT list disagree."
        )
    if not (dry.names == cx.atoms.names).all():
        raise ValueError(
            "atom order differs between the stripped system and complex.prmtop."
        )

    dry.write(out_pdb)
    return out_pdb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("equil_dir", help="equilibration/ directory of an autopath run")
    ap.add_argument("--pdb", default=None,
                    help="coordinates to paint (default: *_equilibrated.pdb in equil_dir)")
    ap.add_argument("--system-prmtop", default=None,
                    help="solvated topology (default: ../system.prmtop)")
    ap.add_argument("--component", default="all",
                    help="MMGBSA component, or 'all' (default)")
    ap.add_argument("--normalize", action="store_true",
                    help="also rescale B-factors to 0-100")
    ap.add_argument("--top-residues", type=int, default=10,
                    help="residues per tail in the bar charts")
    args = ap.parse_args()

    equil = os.path.abspath(args.equil_dir)
    run = os.path.dirname(equil)

    coord_pdb = args.pdb or next(iter(sorted(glob(f"{equil}/*_equilibrated.pdb"))), None)
    if coord_pdb is None:
        raise SystemExit(f"no *_equilibrated.pdb in {equil}; pass --pdb")
    system_prmtop = args.system_prmtop or f"{run}/system.prmtop"

    decomps = sorted(glob(f"{equil}/mmgbsa*/FINAL_DECOMP_*.dat"))
    if not decomps:
        raise SystemExit(f"no FINAL_DECOMP_*.dat under {equil}/mmgbsa*/")

    print(f"coordinates : {coord_pdb}")
    print(f"solvated top: {system_prmtop}")

    for decomp in decomps:
        mmgbsa_dir = os.path.dirname(decomp)
        complex_prmtop = f"{mmgbsa_dir}/complex.prmtop"
        if not os.path.exists(complex_prmtop):
            print(f"[skip] {mmgbsa_dir}: no complex.prmtop")
            continue

        outdir = f"{mmgbsa_dir}/painted"
        os.makedirs(outdir, exist_ok=True)
        print(f"\n=== {os.path.basename(mmgbsa_dir)} -> {outdir}")

        dry_pdb = build_dry_pdb(system_prmtop, coord_pdb, complex_prmtop,
                                f"{outdir}/complex_dry.pdb")

        df = ProteinLigandAnalyzer.parse_mmpbsa_deltas_all_components(decomp)
        n_r = (df["location"] == "R").sum()
        n_l = (df["location"] == "L").sum()
        print(f"    parsed {len(df)} residues ({n_r} receptor, {n_l} ligand)")

        ProteinLigandAnalyzer.plot_mmpbsa_byresidue(
            df, top_residues=args.top_residues, out_dir=outdir)

        ProteinLigandAnalyzer.paint_mmpbsa_byresidue(
            df, pdb_file=dry_pdb, prmtop_file=complex_prmtop,
            mmpbsa_component=args.component, normalize=False, outdir=outdir)
        if args.normalize:
            ProteinLigandAnalyzer.paint_mmpbsa_byresidue(
                df, pdb_file=dry_pdb, prmtop_file=complex_prmtop,
                mmpbsa_component=args.component, normalize=True, outdir=outdir)

        top = df[df["location"] == "R"].nsmallest(5, "TOTAL_Avg")
        print("    most favorable receptor residues (TOTAL, kcal/mol):")
        for _, r in top.iterrows():
            print(f"      {r['label']:>10}  {r['TOTAL_Avg']:8.2f} +/- {r['TOTAL_StdErr']:.2f}")


if __name__ == "__main__":
    main()
