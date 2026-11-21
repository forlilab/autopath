# Overview

`run_AutoPath.py` runs the whole AutoPath workflow for one receptor–ligand pair with the `AutoPath` class and the settings of the AutoPath paper:

1. **System preparation**: PDB fixing (missing atoms/loops, protonation at pH 7.4, caps), solvation in a dodecahedral box (1.2 nm padding, 0.15 M NaCl), AMBER14SB / TIP3P-FB / OpenFF 2.3.0, HMR (1.5 amu).
2. **Equilibration** with the JSON protocol (default `autopath/data/eq_lig-prot_5ns_4fs.json`), then RMSD, per-atom ligand RMSF and contact frequency.
3. **Early filter**: contact-weighted RMSF (cw-RMSF) of the ligand over the whole run and the free window (after the ligand restraint is released), written to `<sys>/equilibration/<sys>_cwRMSF.csv`. With `--max_polar_cwrmsf X` the pipeline stops before pulling when the free-window polar cw-RMSF exceeds X Å (whole run if the protocol has no free window; an undefined polar score does not filter); by default the score is only reported (the paper uses it as a ranking descriptor, with no universal cutoff).
4. **Steered MD**: forward pulls at 5, 10 and 15 m/s, Murcko-scaffold anchor, pocket = `--pocket_resids` if given, else Cα atoms within 8 Å of the ligand, spring 418.4 kJ mol<sup>-1</sup> nm<sup>-2</sup> per ligand heavy atom, automatic stop when the ligand leaves the pocket. Replicas are added one at a time per speed until the pathway-recombined PMF converges (or 50 replicas).
5. **Analysis**: DTW pathway clustering across speeds; cumulant, Jarzynski and force (v→0) free-energy profiles, friction and Kramers k<sub>off</sub>.

Steps 1–3 and 4–5 run as two `AutoPath` calls so the filter can sit between them; the second call reuses the files of the first.

Metadynamics seeded from pulling milestones is available in the class but is experimental and not part of the paper protocol; it is off here (see the commented options in the script).

# Usage

```bash
python run_AutoPath.py --rec ../data/3ptb.pdb --lig ../data/3ptb.sdf \
    --pocket_resids 133 134 135 136 137 138 139 140 156 157 158 159 160 161 162 \
                    180 181 182 183 196 197 198 199 200 201 204 205 206 207 208 \
                    209 210 211 212 213 214 215 226 227 228 229 230 \
    [--protocol my.json] [--max_polar_cwrmsf 1.0]
```

or `sbatch run.q` (its `#SBATCH` lines, partition, GPU type, exclude list and environment activation are cluster-specific: edit the block at the top).

* `--protocol` defaults to the packaged `eq_lig-prot_5ns_4fs.json`; use `eq_lig-prot-memb_10ns_4fs.json` from the same folder for membranes.
* `--pocket_resids` (optional): residue numbers of the **original** receptor PDB that define the pulling pocket; AutoPath translates them to system numbering (`<sys>/residue_mapping.json`). The list above is the trypsin S1 site of 3ptb and is specific to that target. Omitted, the pocket is the protein Cα atoms within 8 Å of the ligand.
* The system name is the ligand file stem (`3ptb/`), fixed by the package. The ligand residue is always named `UNK`.
* PDB fixing writes `<receptor>_fixed.pdb` next to the input receptor, so the input folder must be writable.

# Analysing the outputs

The pipeline writes the same files as the step-by-step examples, so their analyses apply to its output folders (one `<ligand stem>/` per ligand; run `run_AutoPath.py` once per ligand):

* **Equilibration** (`<sys>/equilibration/`): RMSD, per-atom ligand RMSF, contact frequency and cw-RMSF. What they mean and how they are computed: "Post-processing" and "cw-RMSF" in `01_Build_and_Equilibrate/README.md`.
* **Early filters on the equilibration** (`02_iAnalysis`), with `--eq_root` pointing to this folder:

  ```bash
  cd ../02_iAnalysis
  python run_cwrmsf.py --eq_root ../04_Full-Pipeline          # cw-RMSF of every ligand -> cwrmsf/cwrmsf_summary.csv
  python orchestrator_mmgbsa.py --eq_root ../04_Full-Pipeline # MM-GBSA jobs
  python orchestrator_LIE.py --eq_root ../04_Full-Pipeline    # LIE jobs
  ```

  The ligand is always `UNK` here, the default `--resname`. See `02_iAnalysis/README.md`.
* **Steered MD** (`<sys>/sMD/analysis_traces/`): pathways, free-energy and friction profiles, k<sub>off</sub>, convergence. The files are described in `03_Pull-it-out/README.md` ("Analysis" and "Output Files").
* **Ranking several ligands**: `03_Pull-it-out/collect_readouts.py` reduces every ligand's profiles to the profile and median-rupture readouts ("Ranking several ligands" in `03_Pull-it-out/README.md`):

  ```bash
  python ../03_Pull-it-out/collect_readouts.py --root .
  ```

* **Periodic images**: the pulls stop if the ligand comes within the cutoff of a protein image (`[PBC guard]` in `autopath.log`). The pipeline deletes the raw trajectories after alignment, so the post-hoc check (`autopath.utils.periodic_image_report`, "Periodic images" in `03_Pull-it-out/README.md`) cannot be run on its output; the step-by-step examples 01, 03 and 05 run it before alignment.

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
