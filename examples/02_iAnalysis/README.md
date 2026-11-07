# Overview

This directory contains scripts for post-equilibration analysis of protein-ligand systems. These are the inexpensive early filters of the AutoPath paper (pose stability and end-point energies), plus LIE:

* **cw-RMSF** (`run_cwrmsf.py`) — contact-weighted ligand RMSF, a pose-stability score

* **MM-GBSA** (`orchestrator_mmgbsa.py`) — end-point free energy calculations using AMBER's MMPBSA.py

* **LIE** (`orchestrator_LIE.py` + `run_LIE.py`) — Linear Interaction Energy decomposition into van der Waals and electrostatic components

Both workflows take the equilibration trajectories produced by `01_Build_and_Equilibrate/` as input, therefore, successful completion of the equilibration step is required. 
The scripts expect:

* `{sysname}/system.prmtop` — AMBER topology (MM-GBSA, LIE)
* `{sysname}/system.pdb` — system topology (cw-RMSF)
* `{sysname}/equilibration/equilibration_{sysname}_aligned.dcd` — aligned equilibration trajectory
* `{sysname}/equilibration/equilibration_{sysname}.csv` and `equilibration_protocol.json` — frame times and protocol, used to locate the free window (cw-RMSF)

In practice, you can apply these methods to any unbiased MD trajectory. By default every script looks for systems under `../01_Build_and_Equilibrate`; change it with `--eq_root`.

Note that the frames of a single 5 ns equilibration are strongly correlated: the scores below are ranking descriptors, not converged binding free energies.

---

# cw-RMSF

Whole-ligand RMSD mixes motion that disrupts binding with motion that does not (a solvent-exposed substituent can dominate it). cw-RMSF weights each ligand atom's fluctuation by how often it touches the pocket:

cw-RMSF = Σ<sub>i</sub> c<sub>i</sub> · RMSF<sub>i</sub> / Σ<sub>i</sub> c<sub>i</sub>

where RMSF<sub>i</sub> is the root-mean-square fluctuation of ligand heavy atom *i* and *c<sub>i</sub>* the fraction of frames in which any pocket heavy atom lies within 3.5 Å of it (pocket = protein heavy atoms within 5 Å of the ligand in the first analysed frame). It is computed over polar (N, O, S), non-polar (C, F, Cl, Br, I) and all heavy atoms, for the whole run and for the free window (the frames after the ligand restraint is released; the last 2.15 ns of the default 5 ns protocol). Lower is more stable. In the paper the polar score was the most informative, and it did not depend on the window analysed.

```bash
python run_cwrmsf.py -j 8
```

Outputs, in `cwrmsf/`:

* `cwrmsf_summary.csv` — one row per system and window: `cw_rmsf_polar`, `cw_rmsf_nonpolar`, `cw_rmsf_all` (Å). NaN means no atom of that set ever contacts the pocket.
* `{sysname}_RMSF_{whole,free}.csv` — per-atom `rmsf` and `contact_freq`

The same calculation from Python, on any aligned trajectory:

```python
from autopath.ap_PLIP import ligand_rmsf_contacts, cw_rmsf
per_atom = ligand_rmsf_contacts(u, "resname UNK")    # start/stop/step select a window
cw_rmsf(per_atom, "polar")
```

---

# MM-GBSA

## Workflow Summary

1. `orchestrator_mmgbsa.py` discovers all equilibrated trajectories and, for each system and GB model:
   - Prepares a modified MMPBSA input file (all frames; solvent and ions stripped)
   - Generates a SLURM queue file `qfiles_mmgbsa/{sysname}_{GB}_mmgbsa.q`
   - Lists it in `run_mmgbsa_batch.sh`
2. Submit all jobs at once by running: `./run_mmgbsa_batch.sh`
3. Each SLURM job runs `ante-MMPBSA.py` to prepare stripped topologies, then `MMPBSA.py.MPI` in parallel across trajectory frames.
4. Explore results with `analysis_mmgbsa.ipynb`

## Persistent Waters

AutoPath can identify water molecules that persistently occupy the protein-ligand interface throughout the trajectory. Waters present in at least the given fraction of frames are retained in the MMPBSA calculation as explicit solvent molecules rather than being stripped. Their positions are saved to `{sysname}_persistentWaters.pdb`.
This is off by default (all waters stripped, as in the paper). Turn it on with `--persistent_waters 0.9`; results then go to `mmgbsa_{GB}_WAT/`.

## GB models

The paper compared four GB models; the best one was target dependent, so all four run by default (`--gb` selects a subset):

| `--gb` | `igb` | radii |
|--------|-------|-------|
| `HCT`  | 1 | `mbondi`  |
| `OBC2` | 5 | `mbondi2` |
| `GBn`  | 7 | `bondi`   |
| `GBn2` | 8 | `mbondi3` |

## MMGBSA Input Config (`mmgbsa_igb8.in`)

The AMBER MMPBSA input template controls the implicit solvation model and all options related to MMGBSA/PBSA. The orchestrator writes one copy per GB model (`mmgbsa_{GB}.in`) with `igb` replaced:

* `igb=8` — GBn2 GB model
* `saltcon=0.15` — implicit salt concentration (150 mM)
* `idecomp=1` — per-residue energy decomposition
* `use_sander=1` — use sander instead of mmpbsa for energy evaluations

You can customize this file for different GB models, salt concentrations, or to disable decomposition.
For more information on how to set up this calculations you can check the Amber reference manual. 

## Key Parameters (`orchestrator_mmgbsa.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eq_root` | `../01_Build_and_Equilibrate` | Where the equilibrated systems are |
| `--gb` | all four | GB models to run |
| `--persistent_waters` | off | Fraction of frames a water must appear in to be retained |
| `mpi_threads` | `222` | MPI parallelism (capped by number of trajectory frames) |
| `slurm_template_fname` | `None` | Custom SLURM template (e.g. `mmgbsa_slurm_template.q`); uses the bundled one if `None` |

## Output Files

```
{sysname}/
└── mmgbsa_{GB}/
    ├── FINAL_RESULTS_mmpbsa.dat       # Overall MM-GBSA energies (mean ± std)
    ├── FINAL_DECOMP_mmpbsa.dat        # Per-residue energy decomposition
    ├── {sysname}_{GB}_mmgbsa.in       # Modified MMPBSA input (frame range added)
    └── {sysname}_{GB}_persistentWaters.pdb # Persistent interfacial waters (if requested)

qfiles_mmgbsa/
└── {sysname}_{GB}_mmgbsa.q            # Generated SLURM queue file

run_mmgbsa_batch.sh                    # Batch submission script
```

## Example Usage

```bash
# Step 1: prepare all queue files
python orchestrator_mmgbsa.py

# Step 2: submit all jobs
bash run_mmgbsa_batch.sh

# Step 3: explore results (analysis_mmgbsa.ipynb), or paint per-residue energies on the structure
python paint_mmgbsa.py ../01_Build_and_Equilibrate/3ptb/equilibration --mmgbsa-dir 3ptb/mmgbsa_GBn2
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
