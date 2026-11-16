# Overview

End-to-end check of the sMD pipeline on a two-channel toy model whose free-energy profile, friction and unbinding time are known exactly (`autopath.pulling.toy`). It runs the pulls with `SteeredMD`, unbiased escapes for the exact rate, the AutoPath analysis, and scores the analysis against the exact answers.

# Model

A stiff five-particle ligand (resname `UNK`) and a ring of eight restrained pocket particles (name `CA`), no solvent. The ligand centre of mass feels an analytic potential: a bound well and two tunnels at ±45° from the pulling direction, an easy one (tunnel 0, E) and a hard one (tunnel 1, H), joined by a smooth minimum. The Langevin bath is the only friction, so the COM friction is exactly *m*γ and the PMF along *r* follows from 3D quadrature.

| Scenario | Landscape |
|---|---|
| `Closed` | two tunnels, barriers 5 and 12 kJ/mol |
| `Open` | `Closed` + a passage joining the tunnels past their barriers |
| `Open-HR` | `Open` + ripples in H beyond the passage |
| `Open-HW` | `Open` + a secondary well in H beyond the passage |

# Workflow

1. **Pulls** (`toy.run_pulls`): 4 fs, 300 K, friction 50 ps<sup>-1</sup>, 0.001 nm restraint steps, spring 418.4 kJ mol<sup>-1</sup> nm<sup>-2</sup> per ligand atom, contact autostop off, pulled to *r* = 1.65 nm, geometry features logged. Each pull is seeded from `--seed` and its replica name, so reruns reproduce it.
2. **Escapes** (`toy.run_escapes`): unbiased runs from the bound well until the COM is 1 nm down a tunnel axis; exact MFPT and exit-tunnel fractions.
3. **Analysis**: `SMDAnalysis` with the benchmark settings, twice: clustering per speed (`analysis_dtw_ps`, cumulant and Jarzynski rows) and across speeds (`analysis_dtw_xs`, force v→0 row and the friction).
4. **Validation** (`toy.validate`): per estimator and speed, the barrier error (max over *r* = 0.56–1.17 nm, profiles aligned at the first grid point), plateau error (mean over the last 0.1 nm), RMSE of the force-regression friction against *m*γ, τ/τ<sub>exact</sub> (Kramers MFPT against the escape mean, 95% bootstrap interval), the fraction of pulls leaving through each tunnel and of pulls that switch tunnel after committing (95% Wilson intervals), and the exit fractions of the escapes.

# Usage

```bash
# small run (~1.5 h on one CPU core, most of it escapes): 20 pulls at 1, 5, 10 m/s, 50 escapes
python run_toy.py --scenario Open-HW
```

Paper size (150 pulls per speed, 200 escapes): split the escapes across jobs with `--escape-seed`, then analyse and validate once they finish. Use CPU partitions; the Reference platform is fast for 13 particles.

```bash
sbatch -p shared,highmem --wrap "python run_toy.py --scenario Open-HW --n-pulls 150 --steps pulls"
sbatch -p shared,highmem --array=1-20 --wrap 'python run_toy.py --scenario Open-HW --steps escapes --n-escapes 10 --escape-seed $((1000+SLURM_ARRAY_TASK_ID))'
# after both finish
sbatch -p shared,highmem --wrap "python run_toy.py --scenario Open-HW --steps analysis validate"
```

or `sbatch run.q` for the small run.

# Output Files

```
{scenario}/
├── autopath.log
├── validate.csv                 # one row per estimator and speed (force at v = 0)
├── escapes/escapes_<seed>.csv   # run, time_ps, exit_tunnel, escaped (+ esc_<seed>_<run>.npz COM traces with --save-com)
└── sMD/
    ├── start.pdb
    ├── trajectories/            # sMD_replica-<id>_v<speed>_forward.{dat,_geom.dat,dcd,pdb}
    ├── analysis_dtw_ps/         # per-speed pathways, PMFs, friction, koff
    └── analysis_dtw_xs/         # cross-speed pathways, force v->0 PMF and friction
```
