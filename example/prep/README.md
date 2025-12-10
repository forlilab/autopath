#Overview

`ap_prep.py` can assist in preparing structures for docking tasks. Conceptually, `ap_prep.py` is very similar to `ap_equilibration.py`, except it is more general purpose as it allows one to automatically fetch their receptor of interest from the PDB with the relevant chemical context (i.e. ligand, cofactors, waters, etc.). In addition, it will save the relevant files to conduct a docking and corresponding analyses (i.e. receptor, ligand). 

Briefly, `ap_prep` first calls PDBFixer, which will adding missing residues/atoms and replace any nonstandard amino acids. Then, a restrained minimization is run followed by an optional equilibration. The protocol file determines the corresponding restraints applied. The one provided (`equilibration_redock.json`) specifies a restraint constant of 10 kcal/mol/A<sup>2</sup> applied to all heavy atoms and a restraint constant of 100 kcal/mol/A<sup>2</sup> applied to all inorganic cofactors during the restrained minimization procedure. Different protocols can be defined to allow arbitrary constraints to your system, and the specific protocol to use will depend on your system and end goals. 

Upon completion, the following files be written:

* `{save_dir}/{pdb_id}_fixed_wo_solvent.pdb` - receptor and any relevant cofactors/co-ligands post PDBFixer
* `{save_dir}/equilibration/{pdb_id}_minim_receptor_wo_solvent.pdb` - receptor and any relevant cofactors/co-ligands post restrained minimization
* `{save_dir}/{pdb_id}_ligand_fixed.sdf` - ligand that has been successfully converted to an RDKit molecule and saved as an SDF file

If the user specified structural waters to be retained with the receptor, the following files will also be written:

* `{save_dir}/{pdb_id}_fixed_w_struct_waters.pdb` - receptor and any relevant cofactors/co-ligands and any relevant structural waters post PDBFixer
* `{save_dir}/equilibration/{sys_name}_minim_receptor_w_struct_waters.pdb` - this will contain the receptor and any relevant cofactors/co-ligands and any relevant structural waters 





## Arguments

* `rec` - path to receptor PDB file (not needed if fetching from PDB)
* `lig` - path to the ligand SDF/PDB file (not needed if fetching from PDB)
* `lig_smiles` - smiles of the ligand (not needed if fetching from PDB) (type: str) 
* `org_colig` - path to the organic co-ligand SDF/PDB file (not needed if fetching from PDB)
* `org_colig_smiles` - smiles of the organic co-ligand (not needed if fetching from PDB) (type: str)
* `protocol` - path to the JSON file with the equilibration protocol (Required)
* `restrained_minimization_only` - boolean flag indicating to run restrained minimization only
* `equil_rec_only` - boolean flag indicating to equilibrate the receptor without any ligands or cofactors
* `fetch_pdb` - boolean flag indicating to automatically fetch the PDB
* `pdb_id` - pdb id to fetch. can be of the format XXXX or XXXX_A, where the latter specifies the chain id after the pdb id
* `pdb_chain_ids` - chain ids to fetch. if multiple, assumes chain ids are passed as A-B-C (i.e. dash separated)
* `pdb_lig_name` - if provided, will extract all ligands corresponding to this name from the pdb
* `use_ccd_smiles_for_lig` - boolean flag indicating to use ccd smiles to create rdkit mol. otherwise will default to using molscrub. in general, molscrub should work to generate the appropriate smiles, so no need to include this flag. 
* `pdb_lig_resid` - if provided alongside `pdb_lig_name`, will extract the ligand corresponding to this residue id and name from the pdb. if needing to pass in chain id as well, can specify as `<resid>_<chainid>`
* `pdb_inorg_cofactor_name` - if provided, will extract receptor with all inorganic cofactors corresponding this name (e.g. MG, ZN, etc.)
* `pdb_inorg_cofactor_resid` - if provided alongside `pdb_inorg_cofactor_name`, will extract receptor with inorganic cofactor corresponding to this residue id and name from the pdb. if needing to pass in chain id as well, can specify as `<resid>_<chainid>`
* `pdb_org_colig_name` - if provided, will extract all organic co-ligands corresponding to this name (e.g. HEM, NAD, etc.)
* `pdb_org_colig_resid` - if provided alongside `pdb_org_colig_name`, will extract receptor with organic co-ligand corresponding to this residue id and name from the pdb. if needing to pass in chain id as well, can specify as `<resid>_<chainid>`
* `use_ccd_smiles_for_colig` - boolean flag indicating to use ccd smiles to create rdkit mol. otherwise will default to using molscrub. in general, molscrub should work to generate the appropriate smiles, so no need to include this flag. 
* `ignore_colig_simulation` - boolean flag indicating to not include coligand in simulation. relevant if forcefield has issue parameterizing molecule (e.g. heme)
* `ignore_crystallographic_waters` - boolean flag indicating to not use crystallographic waters when minimizing or equilibrating receptor
* `water_resids` - saves waters with receptor whose residue id is contained in this list
* `water_chainids` - restricts waters to be saved with receptor to those whose chain id is contained in this list. must specify if water_resids is included to prevent inclusion of waters added upon solvation by OpenMM
* `lig_resname` - residue name to assign ligand during system prep (default: "UNK")
* `save_dir` - directory to save files. will resort to name of pdb file if not specified

 


## Examples

* Fetching a receptor with its corresponding ligand specified by name and residue id

```python ap_prep.py --protocol equilibration_redock.json --save_dir ./1G9V_RQ3 --restrained_minimization_only --fetch_pdb --pdb_id 1G9V --pdb_lig_name RQ3 --pdb_lig_resid 801_A```

* Same example except structural waters included

```python ap_prep.py --protocol equilibration_redock.json --save_dir ./1G9V_RQ3 --restrained_minimization_only --fetch_pdb --pdb_id 1G9V --pdb_lig_name RQ3 --pdb_lig_resid 801_A --water_resids 918 948 983 1000 water_chainids A C```

* Fetching a receptor (only chain A+B) with its corresponding ligand

```python ap_prep.py --protocol equilibration_redock.json --save_dir ./1HWI_115 --restrained_minimization_only --fetch_pdb --pdb_id 1HWI --pdb_chain_ids A-B --pdb_lig_name 115 --pdb_lig_resid 2_A```

* Fetching a receptor (only chain A) with its corresponding ligand and inorganic cofactor

```python ap_prep.py --protocol equilibration_redock.json --save_dir ./1GKC_NFH --restrained_minimization_only --fetch_pdb --pdb_id 1GKC --pdb_chain_ids A --pdb_lig_name NFH --pdb_inorg_cofactor_name ZN --pdb_inorg_cofactor_resid 1450```


* Fetching a receptor with its corresponding ligand and organic cofactor

```python ap_prep.py --protocol equilibration_redock.json --save_dir ./1SG0_STL --restrained_minimization_only --fetch_pdb --pdb_id 1SG0 --pdb_lig_name STL --pdb_lig_resid 502_B --pdb_org_colig_name FAD --pdb_org_colig_resid 434_A```

* Fetching a receptor with its corresponding ligand specified by name and residue id. The flag `ignore_colig_simulation` is included to bypass issues with parameterizing the cofactor (HEM). 

```python  ap_prep.py --protocol equilibration_redock.json --save_dir ./1R9O_FLP --restrained_minimization_only --fetch_pdb --pdb_id 1R9O --pdb_lig_name FLP --pdb_org_colig_name HEM --ignore_colig_simulation```
