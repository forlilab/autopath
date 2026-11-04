# Overview

`run_AutoPath.py` runs the whole AutoPath workflow for one receptor–ligand pair with the `AutoPath` class and the settings of the AutoPath paper:

1. **System preparation**: PDB fixing (missing atoms/loops, protonation at pH 7.4, caps), solvation in a dodecahedral box (1.2 nm padding, 0.15 M NaCl), AMBER14SB / TIP3P-FB / OpenFF 2.3.0, HMR (1.5 amu).
2. **Equilibration** with the JSON protocol (default `autopath/data/eq_lig-prot_5ns_4fs.json`), then RMSD, per-atom ligand RMSF and contact frequency.
3. **Early filter**: contact-weighted RMSF (cw-RMSF) of the ligand over the whole run and the free window (after the ligand restraint is released), written to `<sys>/equilibration/<sys>_cwRMSF.csv`. With `--max_polar_cwrmsf X` the pipeline stops before pulling when the free-window polar cw-RMSF exceeds X Å; by default the score is only reported (the paper uses it as a ranking descriptor, with no universal cutoff).
4. **Steered MD**: forward pulls at 5, 10 and 15 m/s, Murcko-scaffold anchor, pocket = Cα atoms within 8 Å of the ligand, spring 418.4 kJ mol<sup>-1</sup> nm<sup>-2</sup> per ligand heavy atom, automatic stop when the ligand leaves the pocket. Replicas are added one at a time per speed until the pathway-recombined PMF converges (or 50 replicas).
5. **Analysis**: DTW pathway clustering across speeds; cumulant, Jarzynski and force (v→0) free-energy profiles, friction and Kramers k<sub>off</sub>.

Steps 1–3 and 4–5 run as two `AutoPath` calls so the filter can sit between them; the second call reuses the files of the first.

Metadynamics seeded from pulling milestones is available in the class but is experimental and not part of the paper protocol; it is off here (see the commented options in the script).

# Usage

```bash
python run_AutoPath.py --rec ../data/3ptb.pdb --lig ../data/3ptb.sdf \
    --protocol ../../autopath/data/eq_lig-prot_5ns_4fs.json [--max_polar_cwrmsf 1.0]
```

or `sbatch run.q`. The system name is the ligand file stem (`3ptb/`). Note that PDB fixing writes `<receptor>_fixed.pdb` next to the input receptor.

To define the pocket by hand, set `pocket_selection` to a list of residue numbers in the **original** receptor PDB; AutoPath translates them to system numbering (`<sys>/residue_mapping.json`).

# Main outputs

```
{sys}/
├── system.pdb / system.xml / system.prmtop
├── equilibration/
│   ├── equilibration_{sys}_aligned.dcd, equilibration_{sys}.csv
│   ├── {sys}_rmsd.csv, {sys}_RMSF.csv (+ .svg), {sys}_contact_freq.svg
│   └── {sys}_cwRMSF.csv                   # cw-RMSF, whole run and free window
├── pulling_resname_UNK_murcko.png         # anchor atoms
├── pocket_view.pse                        # PyMOL session of pocket and ligand
└── sMD/
    ├── trajectories/                      # per-replica logs, geometry sidecars, aligned DCDs
    └── analysis_traces/                   # pathways, PMFs, friction, koff, convergence
```
