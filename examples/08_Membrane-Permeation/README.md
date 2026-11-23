# Membrane permeation: system build and starts

Builds a small solute + lipid bilayer system and the equilibrated starts at several depths that a
membrane-permeation free-energy run (Sun, Dama, Tan, Rose & Voth, JCTC 12, 5157 (2016)) starts from.
Everything here is a thin driver over autopath (`SystemPreparation`, `Equilibration`, `SteeredMD`,
`customForces`).

| script | what it does |
|---|---|
| `prepare_permeation_system.py` | ligand (SMILES or SDF) + membrane patch, membrane-centre `DUM`, cylinder restraint; `--both-sides` also writes the system rotated to -z |
| `make_starts.py` | per side: equilibrate the build, steered-MD the ligand toward the centre, re-equilibrate the inner frame |
| `submit.py` | the two steps above as dependent Slurm jobs, per ligand and replica |
| `bilayer_check.py` | bilayer centre, thickness, area per lipid and water penetration along a run's trajectories |
| `equilibrate_ligand_membrane.json`, `equilibrate_long.json` | equilibration protocols |

```bash
python prepare_permeation_system.py --smiles "CCO" --membrane 50_lip_per_leaf_100_percent_POPC_20A_wat \
    --z 2.7 --both-sides --out ethanol_replica_1/md_input
python make_starts.py ethanol_replica_1/md_input            # equilibration_{+-2.7,+-1.0}/
# or both steps on Slurm:
python submit.py ethanol CCO --dry-run
```

Each `equilibration_*` folder holds `system.xml` (with barostat), `system.pdb` and `state.xml`; all
starts share one topology (checked at the end of `make_starts.py`).

The metadynamics runs and their analysis live in HADES: `workflows/metad/permeability`.
`prepare_permeation_system.py --lmz-restraint` was removed with the metadynamics package.
