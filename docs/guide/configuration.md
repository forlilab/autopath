# Configuration

The [full pipeline](pipeline.md) reads a nested JSON file. Each top-level key is a section, and its keys map directly onto `Config` constructor arguments (and so onto `AutoPath` keyword arguments). `Config.from_config()` also accepts a flat JSON file. Legacy key names are remapped with a deprecation warning.

```json
{
  "general": {
    "pdb_path": "protein.pdb",
    "temperature": 300,
    "platform": "fastest",
    "pocket_selection": "same residue as protein and (around 4 resname UNK) and (not name H*)"
  },
  "preparation": {
    "lig_ff": "openff-2.3.0",
    "forcefield": ["amber14-all.xml", "amber14/tip3pfb.xml"],
    "boxShape": "dodecahedron",
    "padding": 1.2,
    "ionicStrength": 0.15
  },
  "equilibration": {
    "protocol_fname": "autopath/data/eq_lig-prot_5ns_4fs.json"
  },
  "sMD": {
    "sMD_pulling_speeds": {"0.005": 5, "0.0025": 5, "0.001": 5},
    "sMD_max_pulling_dist": 3.5
  }
}
```

`autopath/data/config.json` is a complete example that points at `examples/data/3ptb.pdb`. The `Config` class docstring (`autopath/config.py`) lists every option with its default, and is kept in sync with the `AutoPath` constructor.

## `general`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pdb_path` | str | — | Input receptor PDB |
| `temperature` | float | `300` | Temperature (K) |
| `platform` | str | `"fastest"` | OpenMM platform: `"fastest"`, `"CUDA"`, `"OpenCL"` or `"CPU"` |
| `pocket_selection` | str or list | *(see `config.py`)* | Binding-site atoms for the pulling CV and pocket features. A list of residue numbers uses the **original** PDB numbering. |
| `do_fix_pdb` | bool | `true` | Fix the receptor before building: caps, missing atoms, protonation at pH 7.4 |

## `preparation`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lig_ff` | str | `"openff-2.3.0"` | Ligand force field as `"<family>-<version>"`: `openff-*`, `espaloma-*` or `gaff-*` |
| `forcefield` | list | AMBER14 + TIP3P-FB | OpenMM XML force-field files for protein and solvent |
| `boxShape` | str | `"dodecahedron"` | `"dodecahedron"` or `"cube"` |
| `padding` | float | `1.2` | Minimum solute–box distance (nm) |
| `ionicStrength` | float | `0.15` | Ionic strength (M) |
| `is_membrane` | bool | `false` | Embed in a lipid bilayer (switches the box builder and the equilibration protocol) |
| `lipid_type` | str | `null` | Lipid when `is_membrane` is `true` |

## `equilibration`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_equilibration` | bool | `true` | Run the equilibration protocol |
| `protocol_fname` | str | `null` | Protocol JSON; `null` uses the built-in default (see [Equilibrate](equilibrate.md)) |

## `sMD`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sMD_pulling_speeds` | dict | `{0.005: 5, 0.0025: 5, 0.001: 5}` | Pulling speed (nm/ps) → replicas. With `sMD_converge_speeds=true` the count is a minimum; otherwise it is the total. |
| `sMD_max_pulling_dist` | float | `3.5` | Maximum displacement (nm) |
| `sMD_converge_speeds` | bool | `true` | Add replicas per speed until the profile converges (up to `sMD_max_replicas`) |
| `sMD_max_replicas` | int | `50` | Cap on replicas per speed |
| `sMD_clust_selection` | str | `null` | Extra MDAnalysis selection for clustering features |
| `sMD_path_model` | str | `"dtw"` | Pathway clustering: `"dtw"` (DTW + k-medoids) or `"null"` (a single path) |
| `sMD_autostop_nc` | float | `0.01` | Stop a pull once ligand–pocket contacts fall below this fraction of the initial value; `null` disables it |
| `sMD_autostop_estimator` | str | `"cumulant"` | Estimator used to judge convergence: `"cumulant"`, `"jarzynski"` or `"force"` |
