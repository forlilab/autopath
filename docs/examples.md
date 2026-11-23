# Examples

The [`examples/`](https://github.com/forlilab/autopath/tree/develop/examples) folder holds runnable scripts, each with its own README. They use the trypsin–benzamidine complex (`examples/data/3ptb.pdb`, `3ptb.sdf`). Every `run.q` has a block marked `EDIT FOR YOUR CLUSTER` (partition, GPU type, environment activation).

| Example | What it shows |
|---|---|
| `01_Build_and_Equilibrate` | Build and equilibrate (`--sysname`, `--resname`, `--protocol`), then periodic-image check, RMSD and cw-RMSF |
| `02_iAnalysis` | cw-RMSF, MM-GBSA (four GB models) and LIE for every equilibrated system, with SLURM orchestrators and notebooks |
| `03_Pull-it-out` | Steered MD pulls, then the pull analysis (`analyze_sMD.py`) and one ΔG per ligand (`collect_readouts.py`) |
| `04_Full-Pipeline` | The `AutoPath` class end to end, with an optional cw-RMSF filter before pulling |
| `05_VanillaMD` | Conventional MD replicas from an equilibrated system, with the standard analysis |
| `08_Toy_validation` | Validates the pull analysis on a two-channel toy model with exact free energy, friction and unbinding time |

## Name contract

- **System name:** 03 and 05 rebuild file names from `--sysname`, so it must match the one used in 01, and `--eqdir` must point to that run. Defaults are `--sysname`: the ligand file stem; `--eqdir`: `../01_Build_and_Equilibrate/<sysname>`.
- **Residue name:** `-n/--resname` is the same option in every example and must match the ligand residue name used in 01.
