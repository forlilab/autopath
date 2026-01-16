# AutoPath

## Installation
Create a clean environment and install the required dependencies:
```bash
$ micromamba create -n autopath
$ micromamba activate autopath 
$ micromamba install -c conda-forge openmm openmmtools espaloma pdbfixer parmed mdanalysis ambertools rdkit pandas deeptime kmedoids dtaidistance pymol-open-source "openff-toolkit>=0.17" "openff-forcefields==2026.01.0"
$ micromamba install -c mdtools cvpack
$ git clone git@github.com:forlilab/molscrub.git
$ cd molscrub
$ pip install -e .
$ cd ..
$ git clone git@github.com:forlilab/autopath.git
$ cd autopath
$ pip install -e .
```
