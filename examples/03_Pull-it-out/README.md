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

# Periodic images

Pulling the ligand out, or a protein drifting and rotating in a small box, can bring atoms within the nonbonded cutoff of a periodic image: the system then interacts with its own copy, and works, PMFs and contacts are contaminated.

* **During the pulls.** `SteeredMD` checks, every saved frame of a forward pull, the closest ligand heavy atom to the protein heavy atoms of the 26 neighbouring images. It stops the pull once this falls below the nonbonded cutoff + `pbc_safety_nm` (default 0.2 nm) and logs `[PBC guard] Stopping at move ...` in `autopath.log`.
* **After the pulls.** `ap_sMD.py` re-checks every raw pull before superposing it and writes `<sysname>/sMD/image_distances.csv`: per frame and replica, the closest approach of the protein to its own images (`protein_self`, which the run-time guard does not cover) and of the ligand to the protein images (`ligand_protein`), in Å. The log warns when any frame is inside the cutoff. `01_Build_and_Equilibrate` and `05_VanillaMD` write the same check for their runs.
* **On any trajectory.**

  ```python
  import MDAnalysis as mda
  from autopath.utils import load_system, min_image_distance, periodic_image_report

  u = mda.Universe("system.pdb", "raw_trajectory.dcd")
  d_prot = min_image_distance(u, "protein and not name H*")                      # protein vs its own images
  d_lig = min_image_distance(u, "resname UNK and not name H*", "protein and not name H*")
  report = periodic_image_report(u, load_system("system.xml"), ligand_sel="resname UNK")
  ```

  Each molecule is first moved to the image nearest the protein, so a ligand wrapped to the far side of the box still counts as bound. Use the trajectory **before** `wrap_align_save_traj`: superposition rotates the coordinates but not the box, so `*_aligned.dcd` files are rejected. The raw trajectories are deleted after alignment by default (`remove_original=True`), so run the check first or keep them.

If frames fall inside the cutoff, rebuild the system with a larger `padding`. For sMD, also check that the pulling distance (3.5 nm) fits in the box.

# Usage

```bash
# pulls (GPU): 10 replicas at each of 5, 10, 15 m/s
python ap_sMD.py --sysname 3ptb --lig ../data/3ptb.sdf --reps 10

# analysis (CPU)
python analyze_sMD.py --sysname 3ptb --lig ../data/3ptb.sdf
```

`--sysname` must equal the `--sysname` used in `01_Build_and_Equilibrate` (it names the equilibration files, e.g. `system_equil_<sysname>.xml`). `--eqdir` points to that run (default `../01_Build_and_Equilibrate/<sysname>`). Both scripts stop with the name of the missing file if the contract is not met. `--resname` (default UNK) must match the ligand residue name used in 01.

## SLURM

Pulls need a GPU, analysis only CPUs, so they are separate jobs. Submit the analysis to run after the pulls succeed:

```bash
jid=$(sbatch --parsable run.q); sbatch --dependency=afterok:$jid run_analysis.q
```

The `#SBATCH` lines (partition, GPU type, memory) and the environment activation in `run.q` and `run_analysis.q` are cluster-specific: edit the block marked `EDIT FOR YOUR CLUSTER`. Request a fast GPU type; the pulls are long.

# Output Files

```
{sysname}/
├── autopath.log
├── pulling_resname_UNK_murcko.png          # anchor atoms
└── sMD/
    ├── image_distances.csv                 # closest approach to periodic images, per frame and replica
    ├── trajectories/
    │   ├── sMD_replica-<id>_v<speed>_forward.dat       # per-move log: restraint position, CV, force, work
    │   ├── sMD_replica-<id>_v<speed>_forward_geom.dat  # per-frame geometry features
    │   └── sMD_replica-<id>_v<speed>_forward_aligned.dcd
    └── analysis_traces/                    # pathways, PMFs, friction, koff, convergence
```
