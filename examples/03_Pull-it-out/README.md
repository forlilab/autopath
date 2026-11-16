# Overview

Steered MD (sMD) pulls of a ligand out of its pocket, followed by the nonequilibrium analysis of AutoPath: pathway clustering and free-energy, friction and k<sub>off</sub> reconstruction. Start from a system equilibrated with `01_Build_and_Equilibrate`.

`04_Full-Pipeline` runs the same steps inside the `AutoPath` class, adding replicas until the PMF converges. Use this example when you want to drive the pulls yourself.

# Workflow

## 1. Pulling (`ap_sMD.py`, GPU)

* The collective variable is the centre-of-mass distance between ligand **anchor** atoms and **pocket** atoms.
  * Pocket: by default the Cα atoms of residues within 8 Å of the ligand at the end of the equilibration. Pass `--pocket_sel` (MDAnalysis syntax, `system.pdb` numbering) to set it by hand.
  * Anchor: by default the Murcko scaffold heavy atoms (`--anchor murcko`). Other modes: `lig_ha` (all heavy atoms), `lig_com`, `pocket_com`, `contacts`, `inertia` (five atoms, expanded to whole rings). A 2D depiction of the anchor is written to `<sysname>/`.
* Each replica restarts from the equilibrated state with new velocities, relaxes 50 ps with the spring at its starting position, then pulls forward.
* Settings, as in the AutoPath paper: 4 fs timestep, 300 K, restraint centre advanced in 0.001 nm steps up to 3.5 nm, spring 418.4 kJ mol<sup>-1</sup> nm<sup>-2</sup> per ligand heavy atom (total heavy atoms, not anchor atoms), pull stopped once the ligand–pocket contact count stays below 1% of its initial value for 5 samples.
* Speeds: 5, 10 and 15 m/s by default (`--speeds 0.005 0.010 0.015`, nm/ps; 0.001 nm/ps = 1 m/s). At least two speeds are needed for the force (v→0) estimator.
* After pulling, trajectories are wrapped and superposed (`*_aligned.dcd`).

## 2. Analysis (`analyze_sMD.py`, CPU)

* Pulls are grouped into unbinding pathways by DTW + K-medoids on logged trace and geometry features, pooled across speeds.
* Free-energy profiles from the cumulant, Jarzynski and force (v→0 regression) estimators, friction profile and Kramers k<sub>off</sub>; pathways are recombined into one profile.
* PMF convergence as replicas accumulate (cumulant estimator) is written to `sMD_conv_vALL_metrics.csv`.

# Usage

```bash
# pulls (GPU): 10 replicas at each of 5, 10, 15 m/s
python ap_sMD.py --sysname 3ptb --lig ../data/3ptb.sdf --reps 10

# analysis (CPU)
python analyze_sMD.py --sysname 3ptb --lig ../data/3ptb.sdf
```

`--eqdir` points to the equilibrated system (default `../01_Build_and_Equilibrate/<sysname>`).

# Output Files

```
{sysname}/
├── autopath.log
├── pulling_resname_UNK_murcko.png          # anchor atoms
└── sMD/
    ├── trajectories/
    │   ├── sMD_replica-<id>_v<speed>_forward.dat       # per-move log: restraint position, CV, force, work
    │   ├── sMD_replica-<id>_v<speed>_forward_geom.dat  # per-frame geometry features
    │   └── sMD_replica-<id>_v<speed>_forward_aligned.dcd
    └── analysis_traces/                    # pathways, PMFs, friction, koff, convergence
```
