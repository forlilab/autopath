"""Validate the sMD pipeline on the two-channel toy model, whose free energy, friction and unbinding rate are exact.

Steps (``--steps``, default all, in this order):
  pulls     forward sMD pulls with SteeredMD (benchmark settings, contact autostop off)
  escapes   unbiased escapes from the bound well (exact MFPT and exit-tunnel split)
  analysis  SMDAnalysis twice: per-speed clustering (cumulant/Jarzynski) and cross-speed clustering (force)
  validate  score the analysis against the exact answers -> <out>/<scenario>/validate.csv

Usage
-----
    python run_toy.py --scenario Open-HW                                   # small: 20 pulls x 3 speeds, 50 escapes
    python run_toy.py --scenario Open-HW --n-pulls 150 --n-escapes 200     # paper size
"""
import argparse
import os
from glob import glob

import pandas as pd

from autopath.utils import setup_logging
from autopath.pulling import SMDData, SMDAnalysis, toy
from autopath.pulling.PathModel import DTWPathModel

STEPS = ["pulls", "escapes", "analysis", "validate"]


def analyse(smd_dir, scenario, arm, seed):
    """AutoPath analysis of the pulls (benchmark settings); arm dtw_xs clusters across speeds."""
    out = f"{smd_dir}/analysis_{arm}"
    os.makedirs(out, exist_ok=True)
    logs = sorted(glob(f"{smd_dir}/trajectories/sMD_*_*_forward.dat"))
    data = SMDData(logs, scenario, temperature=toy.T, reference_pdb=f"{smd_dir}/start.pdb")
    ana = SMDAnalysis(scenario, DTWPathModel(seed=seed, do_plots=True, outdir=out),
                      estimators=["cumulant", "jarzynski", "force"],
                      do_plots=True, seed=seed, temperature=toy.T,
                      ligand_select="resname UNK", pocket_select="name CA", outdir=out,
                      filter_low_support=True, min_samples_per_step=5, min_support_ratio=1.0,
                      min_replicas_per_path=5, n_paths=None, min_path_steps_ratio=0.6,
                      max_frac_neg_dG_first_half=0.0, min_speeds_for_extrapolation=2)
    ana.run(data, group_A="resname UNK", group_B=None, merge_features=False,
            cluster_across_speeds=(arm == "dtw_xs"), features=None, ligand_sdf=None,
            geom_features=True, plateau_frac=0.4, cluster_to_boundary=True,
            boundary_buffer_frac=0.1, recompute_distances=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="Open-HW", choices=list(toy.SCENARIOS))
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.0, 5.0, 10.0], help="pulling speeds (m/s)")
    ap.add_argument("--n-pulls", type=int, default=20, help="pulls per speed")
    ap.add_argument("--first", type=int, default=1, help="index of the first pull (split pulls across jobs)")
    ap.add_argument("--n-escapes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1, help="pull seed; escapes use it too unless --escape-seed")
    ap.add_argument("--escape-seed", type=int, default=None, help="one escape batch per seed (split escapes across jobs)")
    ap.add_argument("--save-com", action="store_true", help="save escape COM traces (npz)")
    ap.add_argument("--platform", default="Reference")
    ap.add_argument("--out", default=".")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    args = ap.parse_args()

    root = f"{args.out}/{args.scenario}"
    smd_dir, esc_dir = f"{root}/sMD", f"{root}/escapes"
    os.makedirs(esc_dir, exist_ok=True)
    logger = setup_logging(f"{root}/autopath.log", log_level="INFO")

    if "pulls" in args.steps:
        logger.info(f"toy {args.scenario}: {args.n_pulls} pulls x {args.speeds} m/s")
        toy.run_pulls(args.scenario, args.speeds, args.n_pulls, smd_dir, seed=args.seed,
                      platform=args.platform, first=args.first)
    if "escapes" in args.steps:
        seed = args.seed if args.escape_seed is None else args.escape_seed
        logger.info(f"toy {args.scenario}: {args.n_escapes} escapes, seed {seed}")
        e = toy.run_escapes(args.scenario, args.n_escapes, seed, platform=args.platform,
                            out_dir=esc_dir if args.save_com else None)
        e.insert(0, "seed", seed)
        e.to_csv(f"{esc_dir}/escapes_{seed}.csv", index=False)
    if "analysis" in args.steps:
        for arm in ("dtw_ps", "dtw_xs"):
            analyse(smd_dir, args.scenario, arm, seed=42)
    if "validate" in args.steps:
        escapes = pd.concat([pd.read_csv(f) for f in sorted(glob(f"{esc_dir}/escapes_*.csv"))])
        v = toy.validate(f"{smd_dir}/analysis_dtw_ps", args.scenario, escapes,
                         pulls_dir=f"{smd_dir}/trajectories", force_dir=f"{smd_dir}/analysis_dtw_xs")
        v.to_csv(f"{root}/validate.csv", index=False)
        cols = ["estimator", "speed_m_per_s", "barrier_err", "plateau_err", "gamma_rmse", "tau_ratio",
                "exit_t1", "exit_t1_lo", "exit_t1_hi", "esc_t1", "switch"]
        pd.set_option("display.width", 200)
        print(f"{args.scenario}: exact tau {v.tau_exact_ns.iloc[0]:.1f} ns "
              f"[{v.tau_exact_ns_lo.iloc[0]:.1f}, {v.tau_exact_ns_hi.iloc[0]:.1f}] "
              f"from {int(v.n_escapes.iloc[0])} escapes; exact friction {toy.exact_friction(args.scenario):.0f}")
        print(v[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
