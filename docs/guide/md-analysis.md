# Analyse MD

These analyses apply to equilibration and conventional MD trajectories.

**Standard set**, per replica:

1. periodic-image check;
2. RMSD;
3. ligand RMSF, contact frequency and cw-RMSF;
4. persistent protein–ligand interactions.

LIE and [MM-GBSA](mmgbsa.md) are on request.

```python
import MDAnalysis as mda
import pandas as pd
from autopath.utils import load_system, periodic_image_report, wrap_align_save_traj, compute_rmsd
from autopath.ap_PLIP import ligand_rmsf_contacts, cw_rmsf_scores, plot_atomic_property

sys_dir, run_id, lig = "runs/mysys", "mysys_rep1", "resname LIG"
pdb, raw, out = f"{sys_dir}/system.pdb", f"{sys_dir}/MD/MD_{run_id}.dcd", f"{sys_dir}/MD"

# 1. periodic images: on the RAW trajectory
periodic_image_report(mda.Universe(pdb, raw), load_system(f"{sys_dir}/system.xml"),
                      ligand_sel=lig).to_csv(f"{out}/{run_id}_image_distances.csv", index=False)

# 2. wrap, centre, superpose (deletes the raw DCD unless remove_original=False)
u = mda.Universe(pdb, wrap_align_save_traj(raw, pdb)[0], in_memory=True)

# 3. RMSD
compute_rmsd(u, u, alig_select="backbone", plots_outdir=out, suffix=run_id,
             groupselections={"protein": "protein and not name H*", "ligand": f"{lig} and not name H*"}
             ).to_csv(f"{out}/{run_id}_rmsd.csv", index=False)

# 4. ligand stability (needs >= 10 frames)
per_atom = ligand_rmsf_contacts(u, lig)
pd.DataFrame([cw_rmsf_scores(per_atom, on_undefined="nan")]).to_csv(f"{out}/{run_id}_cwRMSF.csv", index=False)
plot_atomic_property(u, per_atom.rmsf.values, lig_resname=lig, outname=f"{out}/{run_id}_RMSF.svg")
```

## Periodic images

`periodic_image_report(u, system, ligand_sel=None)` reports, per frame:

- `protein_self`: how close the protein comes to its own periodic images;
- `ligand_protein`: how close the ligand comes to the protein images.

Both are compared with the system's nonbonded cutoff, and a warning is logged when any frame comes inside it.

- **Co-imaging:** each molecule is first moved to the image nearest the protein, so a ligand wrapped to the far side of the box still counts as bound.
- **Lower-level call:** `min_image_distance(u, sel_a, sel_b=None)` returns the per-frame distance for any two selections. With `sel_b=None` it compares a selection with its own images.
- **Raw trajectories only:** superposed (`*_aligned`) trajectories are rejected, because superposition rotates the coordinates but not the box.
- **Fix:** if frames come inside the cutoff, rebuild with a larger `padding`.

## cw-RMSF

For each ligand heavy atom *i*:

- *c<sub>i</sub>* is the fraction of frames in which any pocket heavy atom lies within 3.5 Å. The pocket is the protein heavy atoms within 5 Å of the ligand in the first analysed frame.
- RMSF<sub>i</sub> is the atom's root-mean-square fluctuation.

$$\text{cw-RMSF} = \frac{\sum_i c_i\,\text{RMSF}_i}{\sum_i c_i}$$

It is reported over polar (N, O, S), non-polar (C, halogens) and all heavy atoms (`cw_rmsf_polar`, `cw_rmsf_nonpolar`, `cw_rmsf_all`, in Å). Lower is more stable. NaN means no atom of that set ever touches the pocket.

On an equilibration, `cw_rmsf_windows(u, lig, state_csv, protocol)` reports it for the whole run and for the free window, after the ligand restraint is released.

## Interactions and LIE

```python
from autopath.ap_PLIP import ProteinLigandAnalyzer

pla = ProteinLigandAnalyzer(top=f"{sys_dir}/system.prmtop", trajs=[traj],
                            ligand_mda_selection=lig, protein_mda_selection="protein",
                            outdir=f"{sys_dir}/lie")
residues, per_res, per_res_type = pla.get_persistent_interactions(frequency_cutoff=0.5)
lie = pla.compute_LIE(use_residues=None, ligand_amber_selection=":LIG")
```

- **ProLIF:** `get_persistent_interactions` uses ProLIF and writes an interaction barcode. To test whether a ligand makes a given interaction, report occupancy for every candidate ligand atom, not just the one reference ligands use.
- **LIE:** pytraj misreads the unit cell of DCD files, so convert to NetCDF first:

```python
def to_netcdf(top, traj, out):
    u = mda.Universe(top, traj)
    with mda.Writer(out, u.atoms.n_atoms) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return out
```

## Reporting

- **Report each replica separately before pooling.** A standard deviation close to the mean usually means a mixture of states.
- **Seed ensemble averages** from time-spaced equilibrium frames, not cluster centroids.
- **Under 10 frames:** RMSF reads as about zero, so skip it.
- **Scripts for many systems:** `examples/02_iAnalysis` (`run_cwrmsf.py`, `orchestrator_LIE.py`).
