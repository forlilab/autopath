Allison's Pippeline: this handles system setup > equlibration ?metadynanmics


# Scripts

All scripts are in this folder. `submit.py` calls the others.

## submit.py

Submits the whole pipeline for one ligand on Slurm: build → starts → metaD walkers, each stage waiting on the one before.
Stages that are already done are skipped.

```
python submit.py NAME SMILES_or_SDF [options]
python submit.py ligands.csv [options]          # CSV rows: name,smiles_or_sdf
```

| option           | default                                      |                                                    |
| ---------------- | -------------------------------------------- | -------------------------------------------------- |
| `--membrane`   | `50_lip_per_leaf_100_percent_POPC_20A_wat` | membrane to build (e.g. `POPC`)                  |
| `--methods`    | `wt tt`                                    | metaD methods (one arm each)                       |
| `--cv2`        | `lmz e2e`                                  | second CV (one arm each)                           |
| `--walkers`    | 4                                            | walkers per arm                                    |
| `--ns`         | 100                                          | ns per walker                                      |
| `--hill-kj`    | 0.02                                         | hill height (kJ/mol)                               |
| `--protocol`   | make_starts default                          | equilibration JSON, e.g. `equilibrate_long.json` |
| `--replicas`   | 1                                            | independent builds                                 |
| `--md-time`    | `4-00:00:00`                               | Slurm time limit per walker                        |
| `--stop-after` | –                                           | `build` or `starts`                            |
| `--dry-run`    | –                                           | print the jobs, submit nothing                     |

Output: `NAME_replica_N/` with `md_input/`, `equilibration_±2.7/`, `smd_output_±1.0/`, `equilibration_±1.0/`, `metadynamics/<arm>/`.

## prepare_permeation_system.py

Builds the ligand + membrane system: ligand parameters, membrane, DUM atom at the membrane centre, and a cylinder restraint.

```
python prepare_permeation_system.py --smiles CCO --membrane POPC --z 2.7 --both-sides --out md_input
```

| option                            | default               |                                    |
| --------------------------------- | --------------------- | ---------------------------------- |
| `--smiles` / `--sdf`          | –                    | ligand                             |
| `--membrane`                    | required              | membrane to build (e.g. `POPC`)  |
| `--z`                           | required              | starting depth of the ligand (nm)  |
| `--out`                         | required              | output folder                      |
| `--both-sides`                  | off                   | also writes the mirrored −z start |
| `--ligand-ff`                   | espaloma              | ligand force field                 |
| `--hmass`                       | 3.0                   | hydrogen mass (amu)                |
| `--ionic-strength`              | 0                     | M, on top of neutralising          |
| `--cyl-k`, `--cyl-r`          | 5 kcal/mol/Å², 5 Å | cylinder restraint                 |
| `--z-wall`, `--lmz-restraint` | off                   | optional extra restraints          |

Output: `system.xml`, `system.pdb`, `system_minus.pdb`, `system.prmtop`, `UNK.sdf`.

## make_starts.py

Makes the walker starts for one side. It equilibrates at the build depth (±2.7 nm), pulls the ligand in with steered MD (farthest atom, 1 nm/ns), and re-equilibrates at ±1.0 nm.

```
python make_starts.py md_input --side plus [--protocol equilibrate_long.json]
```

| option                    | default                              |                                               |
| ------------------------- | ------------------------------------ | --------------------------------------------- |
| `build`                 | required                             | output folder of prepare_permeation_system.py |
| `--side`                | both                                 | `plus`, `minus` or `both`               |
| `--outer` / `--inner` | 2.7 / 1.0                            | start depth / SMD target depth (nm)           |
| `--protocol`            | `equilibrate_ligand_membrane.json` | equilibration protocol                        |
| `--out`                 | next to `build`                    |                                               |

Output: `equilibration_<z>/` (`system.xml`, `system.pdb`, `state.xml`) and `smd_output_<z>/`.

## run_permeation.py

Runs one metaD walker. Walkers in the same `--run-dir` share one bias.

```
python run_permeation.py --start-dirs eq_2.7 eq_1.0 eq_-1.0 eq_-2.7 --ligand-sdf md_input/UNK.sdf --lmz \
    --method wt --layout shared --ns 250 --hill-kj 1.2 --run-dir metadynamics/wt_lmz --walker 1
```

| option                                          | default                            |                                                                                                 |
| ----------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| `--start-dirs`                                | –                                 | equilibration folders; walker k uses folder k                                                   |
| `--system`, `--pdb`, `--state`            | –                                 | instead of `--start-dirs`                                                                     |
| `--lmz` / `--end-to-end` / `--axis-atoms` | –                                 | second CV; omit all three for z only                                                            |
| `--ligand-sdf`                                | –                                 | needed for `--lmz` / `--end-to-end`                                                         |
| `--method`                                    | required                           | `wt`, `tt`, `ttwt`                                                                        |
| `--gamma` / `--tt-gamma` / `--basins`     | 11 (WT), 5 / 5 / symmetric         | bias factors; TT basins                                                                         |
| `--hill-kj`, `--hill-ps`                    | 0.02, 1                            | hill height (kJ/mol), pace (ps)                                                                 |
| `--z-sigma`, `--cos-sigma`                  | 0.2 nm, 0.1                        | hill widths                                                                                     |
| `--half-width`                                | half the smallest start box height | z grid half-width (nm)                                                                          |
| `--relative-to-origin`                        | off                                | measure z and the cylinder from the fixed origin instead of the DUM atom (frees the box height) |
| `--layout`                                    | shared                             | `shared` or `isolated` bias                                                                 |
| `--ns`, `--dcd-ps`                          | 100, 100                           | ns per walker; frame interval (ps)                                                              |
| `--temperature`, `--timestep-fs`            | 310, 4                             |                                                                                                 |
| `--headgroup-atom`                            | off                                | logs the phosphate centroid z (slow; also bonds the lipids to DUM)                              |
| `--run-dir`, `--walker`                     | required                           |                                                                                                 |

Output in `--run-dir`: `HILLS_w<k>.npz`, `COLVAR_w<k>.npy`, `REWEIGHT_w<k>.npz`, `WTMetaD_w<k>.dcd/.pdb`, `walker_<k>.json`, `bias/`.

## analyse_permeation.py

PMFs and convergence for one or more arms.

```
python analyse_permeation.py wt_lmz=metadynamics/wt_lmz wt_e2e=metadynamics/wt_e2e --out analysis --ligand NAME --reports
```

| option                                     | default       |                                                                          |
| ------------------------------------------ | ------------- | ------------------------------------------------------------------------ |
| `NAME=ARM_DIR ...`                       | required      | arms to compare                                                          |
| `--out`                                  | required      | output folder                                                            |
| `--bulk`                                 | 2.6 3.3       | \|z\| window of bulk water (PMF zero), nm                                |
| `--membrane`, `--inner`, `--bilayer` | 1.8, 1.0, 2.0 | nm boundaries for well, crossings, bilayer                               |
| `--ligand`                               | –            | name for plot titles                                                     |
| `--reports`                              | off           | also the autopath report per arm (CV traces, bias, 4 estimators, blocks) |

Output: `pmf.png`, `convergence.png`, `z_traces.png`, `fes-2d.png`, `fes_marginal_<arm>.png`, `e2e_atoms.png`, `summary.csv`, `reports/`.

## bilayer_check.py

Membrane health for one arm, from the walker trajectories.

```
python bilayer_check.py metadynamics/wt_lmz --out analysis/bilayer/wt_lmz --stride 10
```

| option                                                             | default               |                                                                |
| ------------------------------------------------------------------ | --------------------- | -------------------------------------------------------------- |
| `arm_dir`                                                        | required              | folder with `WTMetaD_w<k>.dcd/.pdb`                          |
| `--out`                                                          | required              |                                                                |
| `--stride`                                                       | 10                    | every n-th frame (10 = 1 ns)                                   |
| `--core`, `--shell`                                            | 1.0, 0.5 nm           | core band for counting waters; water–ligand cutoff            |
| `--bin`, `--p-near`                                            | 1.5, 1.2 nm           | height-map cell size; \|z\| counted as a headgroup in the core |
| `--lipid`, `--p-atom`, `--water`, `--ligand`, `--center` | POP, P, HOH, UNK, DUM | residue / atom names                                           |

Output: `bilayer_health.png` (bilayer centre, thickness, area per lipid, box height, mid-plane roughness, phosphates near the centre, per walker), `water_penetration.png`, `bilayer_check.npz`.

## Protocol files

| file                                 | used by                       |                                                                                       |
| ------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------- |
| `equilibrate_ligand_membrane.json` | make_starts.py                | default: minimise, warm up, 1.4 ns NVT/NPT with ligand restraint 5 → 1 kcal/mol/Å² |
| `equilibrate_long.json`            | make_starts.py `--protocol` | same plus 20 ns NPT at 0.1 kcal/mol/Å²                                              |
