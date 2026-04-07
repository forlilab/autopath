import argparse
import os
from time import time
from glob import glob
from autopath.ap_PLIP import ProteinLigandAnalyzer

def cmd_lineparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sys",
        dest="sysname",
        required=True,
        action="store",
        help="system name",
    )

    return parser.parse_args()

args = cmd_lineparser()
sysname = args.sysname

aplip = ProteinLigandAnalyzer(
    top=f"{sysname}/system.prmtop",
    trajs=[f"{sysname}/equilibration/equilibration_{sysname}_aligned.dcd"],
    ligand_mda_selection='resname UNK',
    protein_mda_selection='protein',
    outdir=f"{sysname}/equilibration/lie")

# important_residues, persistence_byRes, persistence_byRes_byType = aplip.get_persistent_interactions(frequency_cutoff=0.5, n_jobs=None)
# print("Important residues:", important_residues)
# print("Computing LIE for important residues...")
# lie_df = aplip.compute_LIE(use_residues=important_residues)
# aplip.plot_lie_components(lie_df)
# lie_df.to_csv("LIE_importance05.csv")

print("Computing LIE for all residues...")
lie_df = aplip.compute_LIE(use_residues=None,
                           ligand_amber_selection=':UNK'
                           )

aplip.plot_lie_components(lie_df)
lie_df.to_csv(f"{sysname}/equilibration/lie/LIE_ALL.csv")