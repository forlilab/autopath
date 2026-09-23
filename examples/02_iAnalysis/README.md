# Overview

This directory contains scripts for post-equilibration interaction analysis of protein-ligand systems. Two complementary methods are provided:

* **MM-GBSA** (`orchestrator_mmgbsa.py`) — end-point free energy calculations using AMBER's MMPBSA.py

* **LIE** (`orchestrator_LIE.py` + `run_LIE.py`) — Linear Interaction Energy decomposition into van der Waals and electrostatic components

Both workflows take the equilibration trajectories produced by `01_Build_and_Equilibrate/` as input, therefore, successful completion of the equilibration step is required. 
The scripts expect:

* `{sysname}/system.prmtop` — AMBER topology
* `{sysname}/equilibration/equilibration_{sysname}_aligned.dcd` — aligned equilibration trajectory

In practice, you can apply these methods to any unbiased MD trajectory.
---

# MM-GBSA

## Workflow Summary

1. `orchestrator_mmgbsa.py` discovers all equilibrated trajectories and, for each system:
   - Prepares a modified MMPBSA input file
   - Generates a SLURM queue file under `qfiles_mmgbsa/`
   - Appends an `sbatch` call to `run_mmgbsa_batch.sh`
2. Submit all jobs at once by running: `./run_mmgbsa_batch.sh`
3. Each SLURM job runs `ante-MMPBSA.py` to prepare stripped topologies, then `MMPBSA.py.MPI` in parallel across trajectory frames.
4. Explore results with `analysis_mmgbsa.ipynb`

## Persistent Waters

Autopath's can identify water molecules that persistently occupy the protein-ligand interface throughout the trajectory. Waters present in at least `persistent_waters_cutoff` fraction of frames (default: 90%) are retained in the MMPBSA calculation as explicit solvent molecules rather than being stripped. Their positions are saved to `{sysname}_persistentWaters.pdb`.
Turn this option ON by setting `persistent_waters_cutoff` flag in `prepare_mmgbsa_batch` function

## MMGBSA Input Config (`mmgbsa_igb8.in`)

The AMBER MMPBSA input file controls the implicit solvation model and all options related to MMGBSA/PBSA:

* `igb=8` — GBn2 GB model (recommended for protein-ligand systems)
* `saltcon=0.15` — implicit salt concentration (150 mM)
* `idecomp=1` — per-residue energy decomposition
* `use_sander=1` — use sander instead of mmpbsa for energy evaluations

You can customize this file for different GB models, salt concentrations, or to disable decomposition.
For more information on how to set up this calculations you can check the Amber reference manual. 

## Key Parameters (`orchestrator_mmgbsa.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EXPERIMENT` | `'mmgbsa_igb8_WAT'` | Output subfolder name |
| `mmpbsa_in` | `'mmgbsa_igb8.in'` | AMBER MMPBSA input config file |
| `radii` | `'mbondi3'` | Atomic radii set for GB calculations |
| `persistent_waters_cutoff` | `0.9` | Fraction of frames a water must appear in to be retained |
| `mpi_threads` | `222` | MPI parallelism (capped by number of trajectory frames) |
| `slurm_template_fname` | `None` | Custom SLURM template; uses built-in default if `None` |

## Output Files

```
{sysname}/
└── {EXPERIMENT}/
    ├── FINAL_RESULTS_mmpbsa.dat       # Overall MM-GBSA energies (mean ± std)
    ├── FINAL_DECOMP_mmpbsa.dat        # Per-residue energy decomposition
    ├── {sysname}_mmgbsa.in            # Modified MMPBSA input (frame range added)
    └── {sysname}_persistentWaters.pdb # Persistent interfacial waters

qfiles_mmgbsa/
└── {sysname}_mmgbsa.q                 # Generated SLURM queue file

run_mmgbsa_batch.sh                    # Batch submission script
```

## Example Usage

```bash
# Step 1: prepare all queue files
python orchestrator_mmgbsa.py

# Step 2: submit all jobs
bash run_mmgbsa_batch.sh
```

---

# LIE

## Workflow Summary

1. `orchestrator_LIE.py` discovers all equilibrated trajectories and generates:
   - A SLURM queue file per system under `qfiles_LIE/`
   - A `run_LIE_batch.sh` batch submission script
2. Submit all jobs: `bash run_LIE_batch.sh`
3. Each SLURM job runs `run_LIE.py -s {sysname}`, which:
   - Identifies the most persistent protein-ligand contacts via ProLIF (frequency cutoff 0.5) and computes LIE restricted to those residues (`LIE_importance05.csv`)
   - Computes LIE again auto-selecting protein residues within 6 Å of the ligand from frame 0 (`LIE_ALL.csv`)
   - Computes van der Waals and electrostatic interaction energies using pytraj
   - Saves results and plots

## Method

LIE (Linear Interaction Energy) estimates the binding energy by computing the van der Waals (VDW) and electrostatic (EELEC) interaction energies between the ligand and its environment directly from the MD trajectory. It is computationally cheaper than MM-GBSA and considers explicit water molecules. However, it requires calibrated scaling coefficients for quantitative predictions.

## Key Parameters (`run_LIE.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-s / --sys` | *(required)* | System name (e.g. `3ptb`) |
| `ligand_mda_selection` | `'resname UNK'` | MDAnalysis selection for the ligand |
| `ligand_amber_selection` | `':UNK'` | AMBER mask for the ligand |
| `cutoff` | `6.0` Å | Distance cutoff for automatic residue selection |
| `lie_options` | `'nopbc cutvdw 10.0 cutelec 10.0'` | pytraj LIE options |


## Output Files

```
{sysname}/
└── lie/
    ├── LIE_importance05.csv   # LIE restricted to ProLIF-persistent contact residues
    ├── LIE_ALL.csv            # Per-frame VDW, EELEC, and Total energies for all replicas
    └── LIE_components.png     # Time series plot of Total, EELEC, and VDW components

qfiles_LIE/
└── {sysname}_LIE.q                # Generated SLURM queue file

run_LIE_batch.sh                   # Batch submission script
```

`LIE_ALL.csv` contains columns: `EELEC`, `VDW`, `Total`, `run` (replica index).

## Example Usage

```bash
# Step 1: prepare all queue files
python orchestrator_LIE.py

# Step 2: submit all jobs
bash run_LIE_batch.sh

# Or run a single system interactively (no SLURM)
python run_LIE.py -s 3ptb
```
