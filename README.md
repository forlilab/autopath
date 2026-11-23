[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)       
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)
[![API stability](https://img.shields.io/badge/stable%20API-no-orange)](https://shields.io/)
[![Docs](https://github.com/forlilab/autopath/actions/workflows/docs.yml/badge.svg)](https://forlilab.github.io/autopath/)
[![License: L-GPL v2.1](https://img.shields.io/badge/License-LGPLv2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html)

# AutoPath: A Flexible Framework for Protein-Ligand Unbinding Simulations

AutoPath is a modular framework, built on OpenMM, for preparing and simulating protein–ligand systems and estimating how strongly and how long ligands bind. It sits between **high-throughput docking** and **high-cost physics-based refinement**: fast, reproducible triage of compounds after virtual screening, to tell the good, the bad and the ugly apart before expensive downstream simulations.

Binding is a dual problem: **thermodynamics** (how strongly a ligand binds) and **kinetics** (how long it stays bound). Docking scales but is noisy; rigorous free-energy workflows are accurate but expensive. AutoPath targets the practical middle ground: rank compounds by relative behaviour, improve hit enrichment after docking, and keep every step scriptable and reproducible.

## 📖 Documentation

**[forlilab.github.io/autopath](https://forlilab.github.io/autopath/)**: installation, a user guide for every stage, configuration reference, troubleshooting and the API reference.

## ⚙️ What it does

| Stage | Class | Docs |
|---|---|---|
| Build: PDB fixing, ligand parameters (OpenFF, Espaloma, GAFF), solvation or membrane embedding; protein, ligand(s), complexes or a ligand alone | `SystemPreparation` | [Build](https://forlilab.github.io/autopath/guide/build/) |
| Equilibrate: JSON-driven staged restraint release | `Equilibration` | [Equilibrate](https://forlilab.github.io/autopath/guide/equilibrate/) |
| Conventional MD and its analysis: RMSD, contact-weighted RMSF (cw-RMSF), interactions, LIE, periodic-image checks | `VanillaMD` | [MD](https://forlilab.github.io/autopath/guide/md/), [Analysis](https://forlilab.github.io/autopath/guide/md-analysis/) |
| Steered-MD unbinding: multi-speed pulls, pathway clustering, cumulant / Jarzynski / force free-energy profiles, friction, Kramers k<sub>off</sub>, one ΔG per ligand | `SteeredMD`, `SMDAnalysis` | [Pulls](https://forlilab.github.io/autopath/guide/pulling/), [Analysis](https://forlilab.github.io/autopath/guide/pulling-analysis/) |
| MM-GBSA / MM-PBSA with AMBER `MMPBSA.py` | `ProteinLigandAnalyzer` | [MM-GBSA](https://forlilab.github.io/autopath/guide/mmgbsa/) |
| Everything above from one config file | `AutoPath` | [Pipeline](https://forlilab.github.io/autopath/guide/pipeline/) |

Metadynamics has moved out of AutoPath.

## 📦 Installation

AutoPath needs compiled libraries (OpenMM, AmberTools, RDKit, OpenFF), so build the environment with conda/micromamba:

```bash
git clone https://github.com/forlilab/autopath.git
cd autopath
micromamba env create -f environment.yml
micromamba activate autopath
pip install --no-deps -e .
```

Optional dependencies and notes: [Installation](https://forlilab.github.io/autopath/install/).

## ⚡ Quickstart

Run the full pipeline on the shipped trypsin–benzamidine example (`autopath/data/config.json` points at `examples/data/3ptb.pdb`):

```bash
python autopath/cli/run_AutoPath.py --config autopath/data/config.json --lig examples/data/3ptb.sdf
```

Or from Python:

```python
from autopath import AutoPath, Config

cfg = Config.from_config("autopath/data/config.json")
AutoPath(**vars(cfg)).run(ligand_file="examples/data/3ptb.sdf")
```

Each stage can also be run on its own: see the [user guide](https://forlilab.github.io/autopath/guide/overview/) and the [configuration reference](https://forlilab.github.io/autopath/guide/configuration/).

## 📚 Examples

Runnable scripts in [`examples/`](examples/), each with its own README, using the trypsin–benzamidine complex (`examples/data/`):

| Example | What it shows |
|---|---|
| `01_Build_and_Equilibrate` | Build and equilibrate, then RMSD, cw-RMSF and the periodic-image check |
| `02_iAnalysis` | cw-RMSF, MM-GBSA and LIE for many systems, with SLURM orchestrators |
| `03_Pull-it-out` | Steered-MD pulls, their analysis, and one ΔG per ligand |
| `04_Full-Pipeline` | The `AutoPath` class end to end, with a cw-RMSF filter before pulling |
| `05_VanillaMD` | Conventional MD replicas with the standard analysis |
| `06_Docking_PrEP` | Receptor and ligand preparation for docking |
| `08_Toy_validation` | The pull analysis checked on a toy model with exact answers |

## 📄 License

AutoPath is distributed under the **GNU Lesser General Public License v2.1**. See `LICENSE`.

## 🤝 Citation & Support

Developed and maintained by the Forli Lab at Scripps Research. If you use AutoPath, please cite:

> Llanos et al. *AutoPath: A Flexible Framework for Protein-Ligand Unbinding Simulations*. (manuscript in preparation)

Questions, bug reports and feature requests: [GitHub Issues](https://github.com/forlilab/autopath/issues).
