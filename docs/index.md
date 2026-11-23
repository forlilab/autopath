# AutoPath

AutoPath is a Python framework, built on OpenMM, for preparing and simulating protein–ligand systems and estimating how strongly and how long ligands bind. It sits between high-throughput docking and expensive free-energy methods: fast, reproducible triage of compounds after virtual screening.

It covers the whole route from a receptor PDB and a ligand file to ranked compounds:

| Stage | What it does | Main class |
|---|---|---|
| [Build](guide/build.md) | Fix the receptor, parameterize ligands, solvate or embed in a membrane | `SystemPreparation` |
| [Equilibrate](guide/equilibrate.md) | JSON-driven minimization, warm-up and staged restraint release | `Equilibration` |
| [Conventional MD](guide/md.md) | Unbiased production runs and replicas | `VanillaMD` |
| [Analyse MD](guide/md-analysis.md) | RMSD, ligand RMSF, contact-weighted RMSF (cw-RMSF), interactions, LIE, periodic-image checks | `autopath.ap_PLIP`, `autopath.utils` |
| [Steered MD](guide/pulling.md) | Multi-speed, multi-replica unbinding pulls | `SteeredMD` |
| [Analyse pulls](guide/pulling-analysis.md) | Pathway clustering, cumulant / Jarzynski / force free-energy profiles, friction, Kramers k<sub>off</sub>, one ΔG per ligand | `SMDAnalysis`, `autopath.pulling.readouts` |
| [MM-GBSA / MM-PBSA](guide/mmgbsa.md) | End-point free energies with AMBER `MMPBSA.py` | `ProteinLigandAnalyzer` |
| [Full pipeline](guide/pipeline.md) | Build → equilibrate → cw-RMSF filter → pulls → analysis in one call | `AutoPath` |

Systems can be a protein, a protein with one or several ligands or cofactors, a protein–protein complex, a membrane protein, or a small molecule alone in water.

## Where to start

- New to AutoPath: [Installation](install.md), then the [workflow overview](guide/overview.md).
- Want runnable scripts: the [examples](examples.md) use the trypsin–benzamidine complex (`3ptb`) shipped in `examples/data/`.
- Something failed: [Troubleshooting](guide/troubleshooting.md) lists errors seen in real runs and their fixes.

!!! note "Enhanced sampling"
    Metadynamics has moved out of AutoPath and is not covered here.
