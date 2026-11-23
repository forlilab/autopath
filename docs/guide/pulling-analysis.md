# Analyse pulls

`SMDAnalysis` turns the pulls into pathways, free-energy and friction profiles and Kramers k<sub>off</sub>. It runs on CPU and takes minutes to hours per system.

```python
from glob import glob
from autopath.utils import setup_logging
from autopath.pulling import SMDData, SMDAnalysis
from autopath.pulling.PathModel import DTWPathModel

sys_dir, sys_name, lig = "runs/mysys", "mysys", "resname LIG"
lig_ha, out = f"{lig} and not name H*", f"{sys_dir}/sMD/analysis_traces"
setup_logging(f"{sys_dir}/autopath.log")
logs = sorted(glob(f"{sys_dir}/sMD/trajectories/sMD_*_*_forward.dat"))
data = SMDData(logs, sys_name, temperature=300.0,
               reference_pdb=f"{sys_dir}/equilibration/{sys_name}_equilibrated.pdb")

ana = SMDAnalysis(sys_name, DTWPathModel(seed=42, do_plots=True, outdir=out),
                  estimators=["cumulant", "jarzynski", "force"], temperature=300.0, seed=42,
                  ligand_select=lig_ha, pocket_select=None, outdir=out,
                  filter_low_support=True, min_samples_per_step=5, min_support_ratio=1.0,
                  min_replicas_per_path=5, n_paths=None, min_path_steps_ratio=0.6,
                  max_frac_neg_dG_first_half=0.25, min_speeds_for_extrapolation=2)
ana.run(data, group_A=lig_ha, group_B=None, merge_features=False, cluster_across_speeds=True,
        features=None, ligand_sdf=None, geom_features=True, plateau_frac=0.4,
        cluster_to_boundary=True, boundary_buffer_frac=0.1, recompute_distances=False)

conv, traces = ana.check_convergence(logs=logs, estimator_name="cumulant", conv_window=5,
        group_A=lig_ha, group_B=None, features=None, merge_features=False, ligand_sdf=None,
        geom_features=True, plateau_frac=0.4, cluster_to_boundary=True,
        restrict_rmsd_to_boundary=True, boundary_buffer_frac=0.1, recompute_distances=False)
conv.to_csv(f"{out}/sMD_conv_vALL_metrics.csv", index=False)
```

## Method

- **Pathway clustering:** pulls are grouped into unbinding pathways by DTW + K-medoids on the logged force/work traces and geometry features.
- **Estimators:** free-energy profiles come from the cumulant (second-order) estimator, Jarzynski, and a force-based v→0 regression across speeds.
- **Friction and k<sub>off</sub>:** friction Γ(r) is derived from the dissipated work, and k<sub>off</sub> from Kramers mean-first-passage-time theory.
- **Recombination:** pathways are recombined into one profile per estimator and speed.
- **Per-speed meaning:** cumulant and Jarzynski give a free energy per speed. Force gives ΔG only at v→0; its per-speed rows are mean work.

## Outputs (`analysis_traces/`)

| File | Content |
|---|---|
| `mixture_pmfs.csv` | Pathway-recombined ΔG(r) per estimator and speed (speed 0 = v→0) |
| `sMD_processed_data.csv` | Per-step force, work and r for every pull, with its pathway |
| `friction.csv`, `koff_kramers.csv`, `koff_per_path.csv` | Friction and k<sub>off</sub> |
| `path_quality.csv`, `path_merges.csv`, `path_analysis/` | Pathway clusters and a PyMOL session of the paths |
| `sMD_conv_vALL_metrics.csv` | Profile convergence as replicas accumulate |
| `dG*.svg`, `Wdiss*.svg`, `Gamma_*.svg`, `work_profiles_*.svg` | Plots |

## One ΔG per ligand

With one `<sys>/` folder per ligand under a common root, `examples/03_Pull-it-out/collect_readouts.py --root <root>` writes `readouts.csv`. It is built on `autopath.pulling.readouts` and gives two label-free readouts per system, estimator and speed:

- **`dG_profile`:** ΔG at the most prominent peak for the cumulant (the maximum if there is no prominent peak, flagged in `peak_fell_back`), and the maximum of the profile for Jarzynski and force.
- **`dG_rupture`:** ΔG at d\*, the median force-peak (rupture) displacement over all ligands in the set. Every ligand is read at the same d\*. Rerun the script when ligands are added, because d\* depends on the set.

The AutoPath paper used `dG_profile` for affinity series and `dG_rupture` for a screening set. Which one ranks better can depend on the target.

!!! note
    `cluster_across_speeds=True` pools speeds for clustering. The paper read cumulant and Jarzynski from per-speed clustering (`False`), which can give different numbers. Force v→0 is the same either way.

## Before quoting a ΔG

- **Convergence:** check `sMD_conv_vALL_metrics.csv` and add replicas if the profile has not converged.
- **Pathway populations:** check `path_quality.csv`.
- **Periodic images:** check for `[PBC guard]` stops in `autopath.log` and compare `image_distances.csv` with the cutoff.
- **Pull counts:** report `n_pulls` next to each value.
