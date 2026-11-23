# Full pipeline

The `AutoPath` class chains build → equilibration → steered MD → analysis for one receptor–ligand pair. It is driven either by keyword arguments or by a nested JSON config through the CLI.

## CLI

```bash
python autopath/cli/run_AutoPath.py --config autopath/data/config.json --lig ligand.sdf
```

- **`--config`:** a JSON file with one section per stage (`general`, `preparation`, `equilibration`, ...). Each section's keys map to `Config` arguments, and the receptor is `general.pdb_path`.
- **`--lig`:** an SDF, or a directory of SDFs run in sequence.
- **Shipped config:** `autopath/data/config.json` points at `examples/data/3ptb.pdb`.

## Python

From a config file ([Configuration](configuration.md)):

```python
from autopath import AutoPath, Config

cfg = Config.from_config("config.json")
AutoPath(**vars(cfg)).run(ligand_file="compound_001.sdf")
```

Or with keyword arguments:

```python
from autopath.autopath_core import AutoPath

settings = dict(
    pocket_selection=None,                 # None = protein CA within 8 A of the ligand
    padding=1.2, hydrogenMass=1.5, ionicStrength=0.15, boxShape="dodecahedron", lig_ff="openff-2.3.0",
    protocol_fname="/abs/path/eq_lig-prot_5ns_4fs.json",
    sMD_outdir="sMD", sMD_ligand_anchor_mode="murcko",
    sMD_pulling_speeds={0.015: 10, 0.010: 10, 0.005: 10},    # nm/ps -> minimum replicas
    sMD_converge_speeds=True, sMD_max_replicas=50,           # add replicas until the profile converges
    sMD_autostop_nc=0.01, sMD_max_pulling_dist=3.5, sMD_log_geom_features=True,
    sMD_path_model="dtw", cluster_across_speeds=True,
)
AutoPath(pdb_path="receptor.pdb", do_fix_pdb=True,
         run_preparation=True, run_equilibration=True,
         run_sMDpulling=True, sMD_run_analysis=True, **settings).run(ligand_file="ligand.sdf")
```

Each stage has a `run_*` flag. A stage that is switched off reads the outputs a previous run wrote, so the pipeline can be split into calls, for example to filter on cw-RMSF between equilibration and pulls (see example [`04_Full-Pipeline`](https://github.com/forlilab/autopath/tree/develop/examples/04_Full-Pipeline)).

## Differences from the stage-by-stage route

- **Ligand residue name:** a single ligand is always named `UNK`.
- **System name:** taken from the ligand file stem.
- **Fixed receptor:** written next to the input receptor (`<receptor>_fixed.pdb`), so the input folder must be writable.
- **Pocket residues:** `pocket_selection` as a list of residue numbers uses the **original** PDB numbering, which AutoPath translates to system numbering (`<sys>/residue_mapping.json`). Selections passed to `SteeredMD` directly use `system.pdb` numbering.
- **Raw trajectories:** deleted after alignment, so the post-hoc periodic-image check cannot be run on pipeline output. The pulls' run-time PBC guard still applies.

The pipeline's output folders have the same layout as the stage-by-stage route, so every analysis in this guide applies to them. For several ligands, run the pipeline once per ligand and collect one ΔG each with `collect_readouts.py` ([Analyse pulls](pulling-analysis.md#one-g-per-ligand)).
