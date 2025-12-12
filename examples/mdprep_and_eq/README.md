# Overview

`ap_equilibration.py` is a wrapper around different Autopath classes and functions that assembles, parameterizes, and equilibrates a system for molecular dynamics (MD) simulations. This is the standard first step prior to running any production MD workflow, including:

* Conventional MD (vanillaMD) 

* Steered MD (sMD)

* Metadynamics (WTMetaD)


# Workflow Summary

Below is a high level description of the whole pipeline. By default Autopath will generate a log file at `{sys_name}/autopath.log` that captures key output information about what's happening and, hopefully, can help you troubleshoot potential problems. Also, its a record of what you did, in case you come back later of forgot eveything.

## PDB fixing
This step is optional and may include:

* Repair missing atoms/residues

* Replace non-standard amino acids

* Protonation at pH 7.4

Note: Heterogens, such as waters, ligands or cofactors, can be removed or retained at this stage.

## System assembly

Depending on the user input, the script can assemble
Protein+Ligand | Protein | Ligand

* Solvation (default: dodecahedral box). Other options available.

* Ions added to target ionic strength (default: 0.15 M) Set to None to just neutralize charges.

* If your target is embedded in a membrane, set is_membrane=True and lipid_type to the desired lipid. Only pure bilayers are supported (OpenMM). For complex lipid mixtures, you can pass the path to a custom membrane patch PDB.

* Force-field assignment (default: AMBER14SB + TIP3P-FB + Espaloma). All options availabl in OpenMM are supported. For small molecules, Espaloma, OpenFF, and GAFF are supported.

## Equilibration
The overall equilibration protocol is controled by a JSON config file, which enables a detailed control over the equilbration process. The JSON file has the following subsections:

#### components_lookup
The components_lookup section allows you to specify different components of the systems that will be trated separately during the equilibration: For example: protein_sidechains, protein_backbone, ligand, membrane, etc. This uses MDAnalysis selection syntax, so its very flexible.

#### minimization
This section controls how the components will be selectively minimized using harmonic restrainst. The list of forces should match the list of components previously defined. Forces are in Kcal/mol/A2. This section is only used when the Equilibration.py class is instantiated with the flag restrained_minimization=True. Otherwise, standard minimization if performed with restraints in all components especified before.

#### warmup
This section controls the thermalization or warming up phase. By default, the system is gradually heated to the target temperature in the NVT ensemble using a small timestep. 

#### equilibration
Once we have reached the target temperature, we want to run MD steps while gradually removing the restraints imposed on the system. Each stage in this section allows the user to specify the restraint forces (list lenght should match components_lookup), the ensemble (NVP/NPT), the number of steps, and the timestep.
If npt_flag=True, a MonteCarloBarostat will be added to the system. Remember to set is_membrane=True if you have a membrane so the correct barostat is added to the system.

NOTE: you can find some protocol examples in the `autopath/data/` folder. Equilibration lengths are intentionally conservative. If your system is far from equilibrium, you may want to increase the number of steps in some stages and/or reduce timestep. In the equilibration folder, a `equilibration_protocol.json` file is created with the protocol used during the run, in case you need to reproduce it later or just forgot what you did.

## Post-processing
Some basic post-processing is included in this script. Make sure you customize this to your specific system and needs.
* Wraps molecules into the box and center the trajectory.
* Align the trajectory using backbone atoms.
* Calculate and plot RMSD and RMSF for protein and/or ligands. 

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

* `{sys_name}/equilibration/equilibration_{sys_name}_aligned.xtc`
Wrapped, imaged, and aligned equilibration trajectory

* `{sys_name}/equilibration/RMSD_{sys_name}.csv`
RMSD time series for all components. For each component a plot named `RMSD_{component}` will be generated.

* `{sys_name}/equilibration/RMSF_{sys_name}.png`
Ligand 2D depiction, colored by atomic RMSF (if ligand is present). A csv with raw values is also generated.

Arguments

`--rec`
Path to the receptor PDB file.

`--protocol`
Path to the JSON file defining the equilibration protocol.

Optional

`--lig`
Path to ligand file (SDF or PDB).

`--resname`
Residue name assigned to the ligand (default: UNK).

`--smiles`
SMILES string for the ligand. Used to assign bond orders if provided.

`--lig_from_xray`
Flag indicating that the ligand is taken directly from the crystal structure without reprocessing.

`--fixpdb`
Whether to run PDBFixer on the receptor (default: True).

## Example Usage
### Protein–ligand equilibration

```
python ap_equilibration.py \
    --rec receptor.pdb \
    --lig ligand.sdf \
    --resname LIG \
    --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" \
    --protocol equilibration.json
```

### Apo protein equilibration
```
python ap_equilibration.py \
    --rec receptor.pdb \
    --protocol equilibration_apo.json
```