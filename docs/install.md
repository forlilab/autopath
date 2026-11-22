# Installation

AutoPath depends on compiled libraries (OpenMM, AmberTools, RDKit, OpenFF, PyMOL), so the environment is built with conda or micromamba from the repository's `environment.yml`. `pip install` alone is not enough.

```bash
git clone https://github.com/forlilab/autopath.git
cd autopath
micromamba env create -f environment.yml
micromamba activate autopath
pip install --no-deps -e .
```

Check the installation:

```bash
python -c "import autopath; from autopath.utils import periodic_image_report; print(autopath.__file__)"
```

## Notes

- `environment.yml` pins `numpy < 2`. A few related pins (`mdtraj < 1.11.1`, `ml_dtypes < 0.6`) exist only because newer releases declare numpy 2.
- A GPU with CUDA or OpenCL is needed for production MD. The OpenMM platform is chosen automatically (`platform="fastest"`).
- MM-GBSA/PBSA jobs call AMBER's `ante-MMPBSA.py` and `MMPBSA.py.MPI` (from AmberTools) and MPI.
- Use one environment per project and avoid installing extra packages into it with pip: a transitive upgrade (for example to numpy 2) can break `import autopath`.
