"""Nonequilibrium analysis of the pulls written by ap_sMD.py (same settings as the AutoPath pipeline).

Clusters the pulls into unbinding pathways (DTW + K-medoids on logged trace and
geometry features, across speeds), reconstructs the free-energy and friction
profiles with the cumulant, Jarzynski and force (v->0) estimators, recombines the
pathways, and records PMF convergence against the number of replicas.

Usage
-----
    python analyze_sMD.py --sysname 3ptb [--eqdir ../01_Build_and_Equilibrate/3ptb] [--lig ../data/3ptb.sdf]

Outputs go to <sysname>/sMD/analysis_traces/.
"""
import argparse
from glob import glob

from autopath.utils import setup_logging
from autopath.pulling import SMDData, SMDAnalysis
from autopath.pulling.PathModel import DTWPathModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--sysname", required=True)
    ap.add_argument("--eqdir", default=None,
                    help="folder of the equilibrated system (default: ../01_Build_and_Equilibrate/<sysname>)")
    ap.add_argument("-l", "--lig", default=None, help="ligand SDF (for depictions; optional)")
    ap.add_argument("--ligand_sel", default="resname UNK")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys_name = args.sysname
    eq_root = args.eqdir or f"../01_Build_and_Equilibrate/{sys_name}"
    eq_pdb = f"{eq_root}/equilibration/{sys_name}_equilibrated.pdb"
    lig_ha = f"({args.ligand_sel}) and not name H*"
    outdir = f"{sys_name}/sMD/analysis_traces"
    logger = setup_logging(f"{sys_name}/autopath.log", log_level="INFO")

    logs = sorted(glob(f"{sys_name}/sMD/trajectories/sMD_*_*_forward.dat"))
    logger.info(f"Found {len(logs)} sMD logs for analysis.")
    smd_data = SMDData(logs, sys_name, temperature=args.temperature, reference_pdb=eq_pdb)

    analysis = SMDAnalysis(sys_name, DTWPathModel(seed=args.seed, do_plots=True, outdir=outdir),
                           estimators=["cumulant", "jarzynski", "force"],
                           do_plots=True, seed=args.seed, temperature=args.temperature,
                           ligand_select=lig_ha, pocket_select=None, outdir=outdir,
                           filter_low_support=True, min_samples_per_step=5, min_support_ratio=1.0,
                           min_replicas_per_path=5, n_paths=None, min_path_steps_ratio=0.6,
                           max_frac_neg_dG_first_half=0.25, min_speeds_for_extrapolation=2)

    # Pathways clustered on trace + logged geometry features, pooled across speeds
    analysis.run(smd_data, group_A=lig_ha, group_B=None, merge_features=False,
                 cluster_across_speeds=True, features=None, ligand_sdf=args.lig,
                 geom_features=True, plateau_frac=0.4, cluster_to_boundary=True,
                 boundary_buffer_frac=0.1, recompute_distances=False)

    # PMF convergence as replicas accumulate (cumulant estimator, the hardest to converge)
    conv_df, traces_df = analysis.check_convergence(
        logs=logs, estimator_name="cumulant", conv_window=5, group_A=lig_ha, group_B=None,
        features=None, merge_features=False, ligand_sdf=args.lig, geom_features=True,
        plateau_frac=0.4, cluster_to_boundary=True, restrict_rmsd_to_boundary=True,
        boundary_buffer_frac=0.1, recompute_distances=False)
    conv_df.to_csv(f"{outdir}/sMD_conv_vALL_metrics.csv", index=False)
    if not traces_df.empty:
        traces_df.to_csv(f"{outdir}/sMD_conv_vALL_traces.csv", index=False)
    logger.info(f"Analysis written to {outdir}")


if __name__ == "__main__":
    main()
