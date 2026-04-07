[![License: L-GPL v2.1](https://img.shields.io/badge/License-LGPLv2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html)
[![API stability](https://img.shields.io/badge/stable%20API-no-orange)](https://shields.io/)
[![Docs Status](https://readthedocs.org/projects/autopath/badge/?version=latest)](https://autopath.readthedocs.io/en/latest/?badge=latest)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)       
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)
# AutoPath: A Flexible Framework for Protein-Ligand Unbinding Simulations

**AutoPath**

AutoPath is a modular and extensible framework designed, primarely, for running and analyzing protein-ligand unbinding simulations. AutoPath aims to bridge the gap between **high-throughput docking** and **high-cost physics-based refinement**. It is built for fast, reproducible triage of compounds after virtual screening, helping distinguish the **good, the bad, and the ugly** before expensive downstream simulations.

It integrates system assembly and parameterization, equilibration, and enhanced sampling in a single interface, enabling efficient exploration of ligand dissociation pathways

Predicting ligand binding is a dual problem: **thermodynamics** (how strongly a ligand binds) and **kinetics** (how long it stays bound). Docking scales well but is often noisy; rigorous free-energy workflows can be accurate but computationally expensive.
AutoPath is designed for the practical middle ground:

- Prioritize compounds by relative behavior.
- Improve the hit enrichment after large docking campaigns
- Keep workflows modular, reproducible, and scriptable end-to-end

---

⚙️ Core Features

📦 Fully integrated with OpenMM, MDAnalysis, MDTraj, and CVPack

While it was initially designed for **small molecule–protein** systems, the Autopath pipeline can applied to **protein–protein assemblies** or **small molecule–nucleic acid complexes**. Moreover
- Solvation, with optional membrane support using single-lipid bilayers or custom lipid patches.
- Membrane-associated targets 
The framework supports force fields available in OpenMM and multiple ligand parameterization strategies, including Espaloma, OpenFF, and GAFF2.

---

## 🧩 Pipeline Overview

AutoPath orchestrates a complete workflow, from protein–ligand complexes to actionable thermodynamics and kinetics profiling:

1. **🛠️ System Preparation**
	- PDB fixing (optional)
	- System assembly and parameterization
	- Solvation, with optional membrane support.

2. **🌡️ Equilibration**
	- JSON-driven staged minimization, heating, and MD (NVT/NPT)
	- Component-wise restraints using MDAnalysis selections.

	A detailed example of how to assemble and equilibrate a protein-ligand system can be found at `autopath/examples/01_Build_and_Equilibrate`

3. **🎯 Steered MD (sMD)**
	- Forward/backward pulling protocols
	- Multi-speed and multi-replica runs
	- Automated stopping criteria for trajectory- and ensemble-level efficiency

4. **📈 Nonequilibrium Analysis**
	- Work-profile preprocessing and filtering
	- Path separation using DTW + K-medoids strategies
	- Jarzynski/cumulant-style estimators for thermodynamics and kinetics analysis

5. **🌀 Metadynamics Refinement**
	- Well-tempered metadynamics (OpenMM)
	- Milestone-informed initialization for pathway-aware sampling

6. **End-point free energy estimation (MMGBSA)**

	- 
	- 
---

## 📦 Installation

### 1) Create environment

```bash
micromamba create -n autopath python=3.11 -y
micromamba activate autopath
```

### 2) Install dependencies

```bash
micromamba install -c conda-forge \
  openmm openmmtools espaloma pdbfixer parmed mdanalysis ambertools \
  rdkit pandas deeptime kmedoids dtaidistance pymol-open-source \
  "openff-toolkit>=0.17" "openff-forcefields==2026.01.0" \
  molscrub meeko -y

micromamba install -c mdtools cvpack -y
```

### 3) Install companion toolkit

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

Run the integrated workflow with one ligand:

```bash
python autopath/cli/run_AutoPath.py \
  --config autopath/data/config.json \
  --rec examples/data/3ptb.pdb \
  --lig examples/data/3ptb.sdf
```

Required inputs:

- `--config`: JSON configuration for all stages
- `--rep`: receptor PDB file
- `--lig`: ligand SDF file

---

## 🧾 Configuration Model

AutoPath config files are organized into sections:

- `general`
- `preparation`
- `equilibration`
- `sMD`
- `milestones`
- `metadynamics`

An example template can be found @ `autopath/data/config.json`
---

## 📚 Examples

- `examples/01_Build_and_Equilibrate` — setup + equilibration
- `examples/03_Pull-it-out` — steered MD workflow
- `examples/04_Full-Pipeline` — Full stack of autopath pipeline
- `examples/05_VanillaMD` — conventional production MD
- `examples/06_Docking_PrEP` — receptor/ligand preparation for docking
- `examples/07_WTmetaD_COM` — COM-based well-tempered metadynamics

---

## 🧠 Python API

AutoPath is also usable programmatically through key classes, so you can build up your own pipeline:

- `SystemPreparation`
- `Equilibration`
- `SteeredMD`
- `SMDAnalysis`
- `MetadynamicsMD`
- `VanillaMD`
- `ProteinLigandAnalyzer`

---

## 📄 License

AutoPath is distributed under the **GNU Lesser General Public License v2.1**. See `LICENSE`.

---

## 🤝 Citation & Support

This software is developed and maintained by the ForliLab at The Scripps Research Institute.
If you use AutoPath in your research, please cite the corresponding publication:
[preprint]

Questions, bug reports, and feature requests are welcome via GitHub Issues.
