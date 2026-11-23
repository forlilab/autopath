# Overview

`ap_equilibration.py` is a wrapper around different Autopath classes and functions that assembles, parameterizes, and equilibrates a system for molecular dynamics (MD) simulations. This is the standard first step prior to running any production MD workflow, including:

* Conventional MD (vanillaMD) 

* Steered MD (sMD)


# Workflow Summary

Below is a high-level description of the whole pipeline. By default Autopath will generate a log file at `{sys_name}/autopath.log` that captures key output information about what's happening and, hopefully, can help you troubleshoot potential problems. It is also a record of what you did, in case you come back later and forgot everything.

## PDB fixing
This step is optional and may include:

* Repair missing atoms/residues

* Replace non-standard amino acids

* Protonation at pH 7.4

Note: Heterogens, such as waters, ligands or cofactors, can be removed or retained at this stage.

## System assembly

Depending on the user input, the script can assemble
Protein+Ligand | Protein systems.

* Solvation (default: dodecahedral box). Other options available.

* Ions added to target ionic strength (default: 0.15 M) Set to None to just neutralize charges.

* If your target is embedded in a membrane, set `is_membrane=True` and `lipid_type` to the desired lipid. Only pure bilayers are supported (OpenMM). For complex lipid mixtures, you can pass the path to a custom membrane patch PDB.

* Force-field assignment (default: AMBER14SB + TIP3P-FB + OpenFF 2.3.0). All options available in OpenMM are supported. For small molecules, OpenFF, Espaloma, and GAFF are supported.

* Hydrogen mass repartitioning (1.5 amu) so the protocol can use a 4 fs timestep.

## Equilibration
The overall equilibration protocol is orchestrated by a JSON config file, which enables a detailed control over the process. The JSON file has the following subsections:

#### components_lookup
The components_lookup section allows you to specify different components of the systems that will be treated separately during the equilibration: For example: protein_sidechains, protein_backbone, ligand, membrane, etc. This uses MDAnalysis selection syntax, so its very flexible.

#### minimization
This section controls how the components will be selectively minimized using harmonic restraints. The list of forces should match the list of components previously defined. Forces are in kcal/mol/A<sup>2</sup>. This section is only used when the Equilibration.py class is instantiated with the flag `restrained_minimization=True`. Otherwise, standard minimization is performed with restraints on all components especified before.

#### warmup
This section controls the thermalization or warming up phase. By default, the system is gradually heated to the target temperature in the NVT ensemble using a small timestep. 

#### equilibration
Once we have reached the target temperature, we want to run MD steps while gradually removing the restraints imposed on the system. Each stage in this section allows the user to specify the restraint forces (the list length should match components_lookup), the ensemble (NVT/NPT), the number of steps, and the timestep.
If `npt_flag=True`, a MonteCarloBarostat will be added to the system. Remember to set `is_membrane=True` if you have a membrane so a membrane barostat is used instead.

The default protocol used in the AutoPath paper is `autopath/data/eq_lig-prot_5ns_4fs.json` (5 ns: 0.1 ns warm-up, then eight stages; the ligand restraint is released after 2.85 ns, so the last 2.15 ns are unrestrained, the "free window"). For membrane systems use `eq_lig-prot-memb_10ns_4fs.json`.

NOTE: you can find the protocols in the `autopath/data/` folder. Equilibration lengths are intentionally conservative. If your system is far from equilibrium, you may want to increase the number of steps in some stages and/or reduce timestep. In the equilibration folder, a `equilibration_protocol.json` file is created with the protocol used during the run, in case you need to reproduce it later or just forgot what you did.

## Post-processing
Some basic post-processing is included in this script. Make sure you customize this to your specific system and needs.
* Periodic-image check on the raw trajectory, before superposition: per frame, the closest approach of the protein to its own periodic images and of the ligand to the protein images, flagged against the nonbonded cutoff (`autopath.utils.periodic_image_report`). Frames inside the cutoff mean the system interacts with its own copy: enlarge `padding` (see "Periodic images" in `03_Pull-it-out/README.md`).
* Wraps molecules into the box, centres and superposes the trajectory on the protein backbone (`autopath.utils.wrap_align_save_traj`).
* RMSD of the protein and ligand.
* Ligand per-atom RMSF and pocket-contact frequency, projected on the 2D structure.
* Contact-weighted RMSF (cw-RMSF), the pose-stability score of the AutoPath paper, over the whole run and over the free window.

### cw-RMSF
For each ligand heavy atom *i*, the contact frequency *c<sub>i</sub>* is the fraction of frames in which any pocket heavy atom lies within 3.5 Å (pocket = protein heavy atoms within 5 Å of the ligand in the first analysed frame), and RMSF<sub>i</sub> is the atom's root-mean-square fluctuation:

cw-RMSF = Σ<sub>i</sub> c<sub>i</sub> · RMSF<sub>i</sub> / Σ<sub>i</sub> c<sub>i</sub>

It is reported over polar (N, O, S), non-polar (C, F, Cl, Br, I) and all heavy atoms. Lower is more stable; in the paper the polar score identified native poses and enriched binders from a single 5 ns run. A set with no atom ever in contact is undefined and reported as NaN. The same functions work on any aligned MD trajectory:

```python
from autopath.ap_PLIP import ligand_rmsf_contacts, cw_rmsf, cw_rmsf_scores
from autopath.equilibration import free_window_start

per_atom = ligand_rmsf_contacts(u, "resname UNK", start=None)   # one row per ligand heavy atom
cw_rmsf(per_atom, "polar")                                      # float, Å
start = free_window_start("equilibration_<sys>.csv", "protocol.json", n_frames=len(u.trajectory))
cw_rmsf_scores(ligand_rmsf_contacts(u, "resname UNK", start=start))
```

## Output Files
Upon successful completion, the following files are generated:

### System files

* `{sys_name}/system.pdb`
Final assembled system (protein + ligand + solvent + ions)

* `{sys_name}/system.xml`
OpenMM-serialized system object

* `{sys_name}/system.prmtop`
AMBER-style topology (for interoperability). 

WARNING: Residue numbering in this PRMTOP might differ from the PDB, so be careful with your selections and which one to use to get topologies downstream.

### Equilibration outputs

* `{sys_name}/equilibration/equilibration_{sys_name}_aligned.dcd`
Wrapped, imaged, and aligned equilibration trajectory

* `{sys_name}/equilibration/equilibration_{sys_name}.csv`
State data (time, energies, temperature, density) for every trajectory frame

* `{sys_name}/equilibration/{sys_name}_equilibrated.pdb`, `checkpoint_equil_{sys_name}.chk/.xml`, `system_equil_{sys_name}.xml`
Final structure, state and restraint-free system, the starting point for vanilla MD (`05_VanillaMD`) and steered MD (`03_Pull-it-out`)

* `{sys_name}/equilibration/equilibration_protocol.json`
Copy of the protocol used

* `{sys_name}/equilibration/{sys_name}_image_distances.csv`
Per-frame closest approach (Å) of the protein to its own images (`protein_self`) and of the ligand to the protein images (`ligand_protein`)

* `{sys_name}/equilibration/{sys_name}_rmsd.csv`
RMSD time series for all components. For each component a plot named `RMSD_{component}` will be generated.

* `{sys_name}/equilibration/{sys_name}_RMSF_{whole,free}.csv`
Per-atom ligand RMSF and contact frequency, whole run and free window

* `{sys_name}/equilibration/{sys_name}_cwRMSF.csv`
cw-RMSF (polar, non-polar, all heavy atoms) for each window

* `{sys_name}/equilibration/{sys_name}_RMSF.svg`, `{sys_name}_contact_freq.svg`
Ligand 2D depiction coloured by atomic RMSF and by contact frequency

Arguments

`--rec`
Path to the receptor PDB file.

Optional

`--protocol`
Path to the JSON file defining the equilibration protocol. Default: the packaged `eq_lig-prot_5ns_4fs.json` (found via `importlib.resources`, so it works from any location). Use `eq_lig-prot-memb_10ns_4fs.json` from the same folder for membrane systems.

`--sysname`
System name, used for the output folder and every file name (`system_equil_<sysname>.xml`, `checkpoint_equil_<sysname>.chk`, ...). Default: the ligand file stem (receptor stem if no ligand), so several ligands on one receptor do not overwrite each other. Steps `03_Pull-it-out` and `05_VanillaMD` rebuild these names from their own `--sysname`, which must be the SAME value.

`--outdir`
Parent folder; outputs go to `<outdir>/<sysname>/` (default: current folder). `03`/`05` look in `../01_Build_and_Equilibrate/<sysname>` unless given `--eqdir`/the equivalent option.

`--lig`
Path to ligand file (SDF or PDB).

`--resname`
Residue name assigned to the ligand (default: UNK). Pass the same `--resname` to 02, 03 and 05. If it is not UNK, the ligand selection of the protocol is rewritten to it (`<sysname>/protocol_<resname>.json`).

`--smiles`
SMILES string for the ligand. Used to assign bond orders if provided.

`--lig_from_xray`
Flag indicating that the ligand is taken directly from the crystal structure without reprocessing.

`--no_fixpdb`
Skip PDBFixer and use the receptor as is (by default the receptor is fixed).

## Example Usage
### Protein–ligand equilibration

```
python ap_equilibration.py \
    --rec receptor.pdb \
    --lig ligand.sdf \
    --resname LIG \
    --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
```

The system is named `ligand` (ligand stem) and the packaged 5 ns protocol is used. Add `--protocol my.json` for a custom one.

### Apo protein equilibration
```
python ap_equilibration.py --rec receptor.pdb
```

### Cluster
`run.q` runs the 3ptb example. Its `#SBATCH` lines (partition, GPU type, exclude list) and the environment activation are specific to our cluster: edit the block at the top for yours. The `../data/3ptb.*` paths are the shipped example inputs.
