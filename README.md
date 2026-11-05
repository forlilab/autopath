[![License: L-GPL v2.1](https://img.shields.io/badge/License-LGPLv2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html)
[![API stability](https://img.shields.io/badge/stable%20API-no-orange)](https://shields.io/)
[![Docs Status](https://readthedocs.org/projects/autopath/badge/?version=latest)](https://autopath.readthedocs.io/en/latest/?badge=latest)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)       
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)

# AutoPath: A Flexible Framework for Protein-Ligand Unbinding Simulations

**AutoPath**

AutoPath is a modular and extensible framework designed, primarily, for running and analyzing protein-ligand unbinding simulations. AutoPath aims to bridge the gap between **high-throughput docking** and **high-cost physics-based refinement**. It is built for fast, reproducible triage of compounds after virtual screening, helping distinguish the **good, the bad, and the ugly** before expensive downstream simulations.

It integrates system assembly and parameterization, equilibration, and steered MD in a single interface, enabling efficient exploration of ligand dissociation pathways.

Predicting ligand binding is a dual problem: **thermodynamics** (how strongly a ligand binds) and **kinetics** (how long it stays bound). Docking scales well but is often noisy; rigorous free-energy workflows can be accurate but computationally expensive.
AutoPath is designed for the practical middle ground:

- Prioritize compounds by relative behavior.
- Improve the hit enrichment after large docking campaigns.
- Keep workflows modular, reproducible, and scriptable end-to-end.

---

## ⚙️ Core Features

- **End-to-end pipeline**: fully automated 4-stage workflow from raw PDB to free-energy profiles and k_off, driven by a single JSON config file — no manual handoffs between stages.
- **Force field flexibility**: protein parametrized with AMBER14 by default; ligand force field selectable per run from OpenFF, Espaloma, or GAFF (e.g. `"openff-2.3.0"`, `"espaloma-0.3.2"`, `"gaff-2.11"`) to match your accuracy vs. speed trade-off.
- **Membrane support**: solvation in explicit water or embedding in a lipid bilayer (POPC, POPE, DLPC, DLPE, DMPC, DOPC, DPPC, or a custom lipid patch PDB) for membrane-associated targets, with a dedicated membrane equilibration protocol.
- **Steered MD with path analysis**: multi-speed, multi-replica COM-distance pulling with an automatic per-speed convergence ladder; DTW-based clustering separates distinct unbinding pathways (or disable clustering entirely for a single path); Jarzynski, cumulant, and force/raw-work estimators give thermodynamic profiles from nonequilibrium trajectories.
- **Kinetics from nonequilibrium work**: friction profile Γ(r) built from dissipated (or raw) work along the pulling coordinate; k_off estimated via Kramers/Pontryagin mean-first-passage-time theory.
- **Modular and scriptable**: each stage can be run independently via the CLI or imported as a Python library, so you can drop AutoPath classes into your own workflows.

---

## 🧩 Pipeline Overview

AutoPath orchestrates a complete workflow, from protein–ligand complexes to actionable thermodynamics and kinetics profiling:

### 1. System Preparation (`SystemPreparation`)

Fixes and completes the input PDB using PDBFixer, then assigns force field parameters to the protein (AMBER14 by default) and ligand (OpenFF, Espaloma, or GAFF template generators — selectable at runtime). The prepared system is solvated in explicit water or embedded in a lipid bilayer for membrane targets. Stage outputs are a serialized `system.xml` and a `system.pdb` ready for simulation.

### 2. Equilibration (`Equilibration`)

Runs a JSON-driven multi-stage minimization and MD protocol (NVT and NPT ensembles) that progressively releases positional restraints on the protein and ligand across user-defined stages. The protocol is fully configurable without touching any Python code. On completion, a checkpoint file and an equilibrated PDB are saved and passed automatically to the downstream stages.

### 3. Steered MD (`SteeredMD`)

Applies a COM-distance pulling force along a user-defined direction at one or more pulling speeds, running N independent replicas per speed. Each replica produces a `.dat` log file recording force, displacement, and cumulative work at every step, providing the nonequilibrium work data needed for thermodynamic and kinetic analysis.

### 4. SMD Analysis (`SMDAnalysis`)

Runs the full dcTMD post-processing pipeline on the `.dat` work logs. Trajectories are clustered with DTW + k-medoids to identify distinct unbinding pathways; Jarzynski and cumulant estimators compute ΔG and dissipated work W_diss for each path. A friction profile Γ(r) is derived from the dissipated work, and k_off is estimated via Kramers/Pontryagin MFPT theory. Medoid trajectories are saved as representative structures for each path.

---

## 📦 Installation

> **Note:** AutoPath relies on several compiled scientific libraries that must be installed via conda/micromamba. A `pip install` alone is insufficient; follow all four steps below.

### 1) Create environment

```bash
micromamba create -n autopath python=3.11 -y
micromamba activate autopath
```

### 2) Install heavy dependencies via conda

```bash
micromamba install -c conda-forge \
  openmm openmmtools espaloma pdbfixer parmed mdanalysis ambertools \
  rdkit pandas deeptime kmedoids dtaidistance pymol-open-source prolif \
  "openff-toolkit>=0.17" "openff-forcefields==2026.01.0" \
  molscrub meeko -y

micromamba install -c mdtools cvpack -y
```

### 3) Install molscrub (required for PDB preprocessing)

`molscrub` is a companion toolkit used for fixing and standardising receptor PDB files before system assembly. It must be installed from source:

```bash
git clone https://github.com/forlilab/molscrub.git
cd molscrub
pip install -e .
cd ..
```

### 4) Install AutoPath

```bash
git clone https://github.com/forlilab/autopath.git
cd autopath
pip install -e .
```

---

## ⚡ Quickstart

Run the full integrated workflow with a single receptor–ligand pair. The receptor path is set via `pdb_path` inside the JSON config — there is no `--rec` flag; only `--config` and `--lig` are read from the command line:

```bash
python autopath/cli/run_AutoPath.py \
  --config autopath/data/config.json \
  --lig examples/data/3ptb.sdf
```

`autopath/data/config.json` ships pointing at `examples/data/3ptb.pdb`, so the command above runs out of the box. Required inputs:

- `--config`: path to a JSON configuration file covering all pipeline stages (including `pdb_path`, the receptor PDB)
- `--lig`: ligand SDF file, or a directory of `*.sdf` files to run in sequence

Below is a minimal `config.json` skeleton with the most commonly adjusted parameters. All section keys and their defaults match the `Config` class exactly:

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

A complete list of configuration options with defaults is in the `Config` class docstring (`autopath/config.py`), which is kept 1-to-1 in sync with the `AutoPath` constructor.

---

## 🧾 Configuration Model

AutoPath configuration files use a nested JSON structure: each top-level key is a section name, and the values are flat key–value pairs that map directly to `Config` constructor arguments. Below are the most important parameters per section.

**`general`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_path` | str | — | Path to the input protein–ligand PDB file |
| `temperature` | float | `300` | Simulation temperature in Kelvin |
| `platform` | str | `"fastest"` | OpenMM platform: `"fastest"`, `"CUDA"`, `"OpenCL"`, or `"CPU"` |
| `pocket_selection` | str | *(see config.py)* | MDAnalysis selection defining binding-site atoms for COM distance and pocket features |
| `do_fix_pdb` | bool | `true` | Run PDB preprocessing (capping, missing atoms, pH 7.4 protonation) before system build |

**`preparation`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lig_ff` | str | `"openff-2.3.0"` | Ligand force field as `"<family>-<version>"`: `"openff-*"`, `"espaloma-*"`, or `"gaff-*"` (e.g. `"espaloma-0.3.2"`, `"gaff-2.11"`) |
| `forcefield` | list | AMBER14 + TIP3P-FB | OpenMM XML force-field files for protein and solvent |
| `boxShape` | str | `"dodecahedron"` | Simulation box shape: `"dodecahedron"` or `"cube"` |
| `padding` | float | `1.2` | Minimum distance (nm) between solute and box face |
| `ionicStrength` | float | `0.15` | Target ionic strength in mol/L |
| `is_membrane` | bool | `false` | Enable membrane-protein mode (changes box builder and equilibration protocol) |
| `lipid_type` | str | `null` | Lipid residue name when `is_membrane` is `true` |

**`equilibration`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_equilibration` | bool | `true` | Run the equilibration protocol |
| `protocol_fname` | str | `null` | Path to a custom equilibration protocol JSON file; `null` uses the built-in default |

**`sMD`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sMD_pulling_speeds` | dict | `{0.005: 5, 0.0025: 5, 0.001: 5}` | Mapping of pulling speed (nm/ps) to replica count; the count is a minimum-replicas floor when `sMD_converge_speeds=true`, or the total count otherwise |
| `sMD_max_pulling_dist` | float | `3.5` | Maximum COM displacement from the binding site in nm |
| `sMD_converge_speeds` | bool | `true` | Keep adding replicas per speed (up to `sMD_max_replicas`) until convergence, instead of a fixed replica count |
| `sMD_max_replicas` | int | `50` | Hard cap on replicas per speed to prevent unbounded convergence loops |
| `sMD_clust_selection` | str | `null` | Additional MDAnalysis selection for clustering features beyond ligand COM distance |
| `sMD_path_model` | str | `"dtw"` | Path-clustering model: `"dtw"` (DTW + k-medoids) or `"null"` (no clustering, a single path) |
| `sMD_autostop_nc` | float | `0.01` | Stop a replica once native contacts drop below this fraction of their initial value; `null` disables the check |
| `sMD_autostop_estimator` | str | `"cumulant"` | Free-energy estimator used to judge convergence: `"cumulant"`, `"jarzynski"`, or `"force"` |

All parameters, including equilibration options, are documented in `autopath/config.py`.

---

## 🧠 Python API

All major pipeline components are importable as a Python library for custom workflows.

### Class Reference

| Class | Import | Role |
|-------|--------|------|
| `AutoPath` | `from autopath import AutoPath` | Main orchestrator; runs the full pipeline end-to-end (also `from autopath.autopath_core import AutoPath`) |
| `Config` | `from autopath import Config` | Configuration container; reads a JSON file or accepts keyword args |
| `SystemPreparation` | `from autopath import SystemPreparation` | Parametrizes the ligand and solvates (or embeds in membrane) the system |
| `Equilibration` | `from autopath import Equilibration` | JSON-driven multi-stage NPT/NVT equilibration with progressive restraint release |
| `SteeredMD` | `from autopath.pulling import SteeredMD` | OpenMM COM-distance pulling at configurable speeds and replica counts |
| `SMDAnalysis` | `from autopath.pulling import SMDAnalysis` | Full dcTMD analysis: path clustering, PMF, friction profile, k_off via Kramers MFPT |

### Minimal Example

```python
from autopath import AutoPath, Config

# Load config from JSON
cfg = Config.from_config("config.json")

# Run the full pipeline for one ligand
ap = AutoPath(**vars(cfg))
ap.run(ligand_file="compound_001.sdf")
```

### Class Descriptions

**`AutoPath`**

The top-level orchestrator that drives the complete AutoPath pipeline. It accepts all parameters from `Config` directly and executes each stage in sequence — preparation, equilibration, sMD, and nonequilibrium analysis — skipping any stage whose `run_*` flag is `False` and loading previous outputs in its place. The `run()` method takes a ligand SDF file and collects all results under a subdirectory named after the ligand stem.

**`Config`**

A flat container that maps all pipeline parameters to attributes, providing a single source of truth for an AutoPath run. JSON files with either nested section blocks or flat key-value pairs are supported via `Config.from_config()`; the class can also be instantiated directly with keyword arguments. Legacy key names are accepted and remapped automatically, with deprecation warnings printed to standard output.

**`SystemPreparation`**

Handles the full system-assembly workflow prior to simulation. It runs PDBFixer on the receptor, assigns ligand partial charges and bonded parameters using an OpenFF, Espaloma, or GAFF template generator, combines the components under a chosen protein force field, and either solvates the complex in a cubic or dodecahedral box or embeds it in a lipid bilayer. The stage produces `system.xml` and `system.pdb` files consumed by downstream stages.

**`Equilibration`**

Runs a multi-stage equilibration protocol whose stages are defined in a JSON file. Each stage specifies the thermodynamic ensemble (NPT or NVT), simulation duration, barostat settings, and per-component restraint force constants applied to protein heavy atoms and the ligand. Restraint strengths are reduced progressively across stages, allowing the system to relax gradually before production simulations begin.

**`SteeredMD`**

Runs a single COM-distance pulling replica in OpenMM, applying a moving harmonic restraint along the protein–ligand separation coordinate. The `AutoPath` orchestrator calls it in a loop over pulling speeds and replica indices to build the nonequilibrium work ensemble. Each replica writes a `.dat` log file containing time, applied force, displacement, and cumulative work columns used by `SMDAnalysis`.

**`SMDAnalysis`**

Orchestrates the full dcTMD post-processing pipeline for a set of sMD replicas. It loads and filters `.dat` work-profile files via `SMDData`, clusters the trajectories into distinct unbinding paths using DTW + k-medoids (`DTWPathModel`), and estimates ΔG and dissipated work via Jarzynski and second-cumulant estimators. A friction profile Γ(r) is built from the spatially resolved work dissipation, and the unbinding rate k_off is estimated via the Kramers/Pontryagin mean first-passage time formula.

---

## 📚 Examples

All examples use the trypsin–benzamidine complex (`3ptb`) as a reference system and are located in the `examples/` directory. Shared input files (receptor PDB, ligand SDF) live in `examples/data/`.

- **`examples/01_Build_and_Equilibrate`** — Assembles, parameterizes, and equilibrates a protein–ligand system through PDB fixing, solvation, force-field assignment, and a JSON-driven staged equilibration protocol. This is the standard first step before any downstream MD workflow. A Jupyter notebook is not included, but a detailed `README.md` covers all options and output files.

- **`examples/02_iAnalysis`** — End-point free energy analysis of equilibrated trajectories using two complementary methods: MM-GBSA (via AMBER's `MMPBSA.py`) and Linear Interaction Energy (LIE). Includes orchestrator scripts for SLURM batch submission and two Jupyter notebooks (`analysis_mmgbsa.ipynb`, `analysis_LIE.ipynb`) for interactive exploration of results.

- **`examples/03_Pull-it-out`** — Runs steered MD (sMD) pulling simulations on an equilibrated system. Demonstrates the forward-pulling protocol and the associated SLURM queue file.

- **`examples/04_Full-Pipeline`** — Executes the complete AutoPath pipeline end-to-end (preparation → equilibration → sMD → analysis) from a single driver script.

- **`examples/05_VanillaMD`** — Conventional (unbiased) production MD using AutoPath's `VanillaMD` class. Useful as a baseline or for generating reference trajectories.

- **`examples/06_Docking_PrEP`** — Prepares receptor and ligand structures for docking. Can auto-fetch PDB entries, run PDB fixing, apply restrained minimization, and save docking-ready files. Supports cofactors, structural waters, and multi-chain systems.

---

## 📄 License

AutoPath is distributed under the **GNU Lesser General Public License v2.1**. See `LICENSE`.

---

## 🤝 Citation & Support

This software is developed and maintained by the ForliLab at The Scripps Research Institute.
If you use AutoPath in your research, please cite:

> Llanos et al. *AutoPath: A Flexible Framework for Protein-Ligand Unbinding Simulations*. (manuscript in preparation)

Questions, bug reports, and feature requests are welcome via GitHub Issues.
