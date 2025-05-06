# AutoPath

## Installation
Create a clean environment and install the requiered dependencies:
```bash
$ micromamba create -n autopath
$ micromamba activate autopath 
$ micromamba install -c conda-forge openmm openmmtools openff-toolkit openmmforcefields espaloma pdbfixer parmed mdanalysis ambertools rdkit pandas deeptime pyemma 
$ micromamba install -c conda-forge -c mdtools cvpack
$ git clone git@github.com:forlilab/autopath.git
$ cd autopath
$ pip install -e .
```
