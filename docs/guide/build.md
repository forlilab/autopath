# Build a system

`SystemPreparation.run(protein, variants, ligands)` assembles, solvates and parameterizes the system, and writes `<out_dir>/system.{pdb,xml,prmtop,rst7}`. The defaults below are those used in the AutoPath paper.

```python
import os
from autopath import SystemPreparation
from autopath.pdb_preprocessor import PDBPreprocessor
from autopath.utils import save_pdb, setup_logging

sys_dir = "runs/mysys"
os.makedirs(sys_dir, exist_ok=True)
setup_logging(f"{sys_dir}/autopath.log")

# 1. Fix the receptor: missing atoms/loops, non-standard residues, protonation, caps
fixed = PDBPreprocessor("receptor.pdb").fix(
    replace_nonstandard_residues=True, ignore_terminal_missing_residues=True,
    keep_heterogens=True,            # keeps crystal waters and metal ions (see cofactors below)
    cap_termini=True, pH=7.4)
receptor = f"{sys_dir}/mysys_fixed.pdb"
save_pdb(fixed.topology, fixed.positions, receptor)

# 2. Build
SystemPreparation(
    boxShape="dodecahedron", padding=1.2,          # nm
    hydrogenMass=1.5,                              # HMR, enables a 4 fs timestep
    lig_ff="openff-2.3.0", ionicStrength=0.15, ions=("Na+", "Cl-"),
    is_membrane=False, out_dir=sys_dir,
    forcefield=["amber14-all.xml", "amber14/tip3pfb.xml", "amber/tip3pfb_HFE_multivalent.xml"],
).run(protein=receptor, variants=None,
      ligands=[("LIG", "ligand.sdf", None, False)])   # (resname, file, smiles, from_xray)
```

## System types

| System | `protein` | `ligands` | Notes |
|---|---|---|---|
| Apo protein | fixed PDB | `None` | |
| Protein + ligand | fixed PDB | `[("LIG", "lig.sdf", None, False)]` | The tuple form sets the residue name. A bare path names the ligand `UNK`. |
| Several ligands or cofactors | fixed PDB | one tuple per molecule, distinct resnames | Each gets its own chain. |
| Protein–protein | one PDB with all chains | `None` or tuples | Check the interface after fixing. |
| Membrane protein | fixed, pre-oriented PDB (e.g. from OPM) | optional | `is_membrane=True, lipid_type="POPC"`. Also POPE, DLPC, DLPE, DMPC, DOPC, DPPC, or a lipid-patch PDB path. |
| Small molecule in water | `None` | tuple(s) | Use `padding >= 1.5` nm. With a membrane (`is_membrane=True`) pass one ligand file path (`ligands="ligand.sdf"`); it is placed at `membrane_ligand_z` nm (default 3.0) above the bilayer centre. |

## Inputs

- **Ligand files:**
    - `.sdf` or `.mol2` with bond orders and hydrogens.
    - A `.pdb` ligand or an X-ray extract needs its SMILES: `("LIG", "lig.pdb", "<smiles>", True)`. The corrected molecule is written to `<sys>/<name>_fixed.sdf`.
- **Cofactors (NAD, FAD, heme, ...):** they have no Amber template. Remove them from the receptor PDB and pass each one as a ligand tuple with its SMILES.
- **Metal ions:** kept by `keep_heterogens=True` and parameterized by `tip3pfb_HFE_multivalent.xml`.
- **Protonation states and disulfides:** `variants={"A:57": "HIE", "A:102": "ASH"}` (`chain:resnum` of the fixed PDB → variant).
- **Ligand force field:** `lig_ff` accepts OpenFF (`"openff-2.3.0"`), Espaloma or GAFF names.

## Check the build

```python
import MDAnalysis as mda
u = mda.Universe(f"{sys_dir}/system.pdb")
print(u.dimensions)
print(sorted(set(u.residues.resnames) - {"HOH", "NA", "CL"}))
print(len(u.select_atoms("resname LIG and not name H*")), "ligand heavy atoms")
```

## Common errors

| Error | Fix |
|---|---|
| `No template found for residue ...` | A heterogen is still in the receptor. Pass it as a ligand tuple, or use `keep_heterogens=False`. |
| Kekulization or valence error on a ligand | Give the SMILES and `from_xray=True` |
| `box is X nm across its narrowest axis, below twice the ... cutoff` | Increase `padding` by the amount in the message |
