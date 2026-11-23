# Equilibrate

`Equilibration` runs a JSON-defined protocol: restrained minimization, a warm-up, then stages that release positional restraints component by component.

## Protocols

AutoPath ships two protocols in `autopath/data/`:

| System | Protocol |
|---|---|
| Soluble protein ± ligands, or a small molecule in water | `eq_lig-prot_5ns_4fs.json`: 5 ns. The ligand restraint is released at 2.85 ns, so the last 2.15 ns are the *free window*. |
| Membrane | `eq_lig-prot-memb_10ns_4fs.json`, with `is_membrane=True` |

A protocol has these parts:

- `components_lookup`: name → MDAnalysis selection, or `{"selection": ..., "type": ...}`.
- `minimization` and `warmup`.
- `equilibration`: a list of stages. Each stage has one force constant per component (kcal mol⁻¹ Å⁻²), plus `npt_flag`, `nsteps` and `stepsize`.

Components that match no atoms (for example the protein components in a ligand-only box) are skipped with a warning.

!!! warning "Ligand residue name"
    The shipped protocols select the ligand as `resname UNK`. If your ligand has another name, rewrite the selection in a copy of the protocol, as below. Otherwise the ligand gets no restraints, and the only sign is a skipped-component warning in the log.

## Run

```python
import json, os
from importlib.resources import files
from openmm.app import PDBFile
from autopath import Equilibration
from autopath.utils import load_system, setup_logging

sys_dir, sys_name, resnames = "runs/mysys", "mysys", ["LIG"]
setup_logging(f"{sys_dir}/autopath.log")

protocol = json.load(open(files("autopath") / "data" / "eq_lig-prot_5ns_4fs.json"))
for k, v in protocol["components_lookup"].items():
    sel = (v["selection"] if isinstance(v, dict) else v).replace("resname UNK", "resname " + " ".join(resnames))
    if isinstance(v, dict):
        v["selection"] = sel
    else:
        protocol["components_lookup"][k] = sel
protocol_path = os.path.abspath(f"{sys_dir}/protocol.json")
json.dump(protocol, open(protocol_path, "w"), indent=2)

Equilibration(
    system=load_system(f"{sys_dir}/system.xml"),
    topology=PDBFile(f"{sys_dir}/system.pdb").topology,
    protocol_fname=protocol_path,          # pass an absolute path
    is_membrane=False,
    restrained_minimization=False,         # as in the AutoPath pipeline
    out_dir=f"{sys_dir}/equilibration",
).run(pdb_file=f"{sys_dir}/system.pdb", run_id=sys_name)
```

In `autopath.log`, `Adding N harmonic restraints to ligand` should report N equal to the total number of ligand heavy atoms.

For a quick test, copy the protocol and divide every `nsteps`: `warmup` by 10 and each `equilibration` stage by 20 gives about 0.25 ns.

## Post-processing

Order matters: periodic-image checks need the raw trajectory, which alignment deletes.

```python
import MDAnalysis as mda
from autopath.utils import periodic_image_report, wrap_align_save_traj, compute_rmsd
from autopath.ap_PLIP import cw_rmsf_windows

eq = f"{sys_dir}/equilibration"
pdb, raw, lig = f"{sys_dir}/system.pdb", f"{eq}/equilibration_{sys_name}.dcd", "resname LIG"

periodic_image_report(mda.Universe(pdb, raw), load_system(f"{sys_dir}/system.xml"),
                      ligand_sel=lig).to_csv(f"{eq}/{sys_name}_image_distances.csv", index=False)

aligned = wrap_align_save_traj(raw, pdb)[0]                 # superposes on the backbone
u = mda.Universe(pdb, aligned, in_memory=True)
compute_rmsd(u, u, alig_select="backbone", plots_outdir=eq,
             groupselections={"protein": "protein and not name H*", "ligand": f"{lig} and not name H*"}
             ).to_csv(f"{eq}/{sys_name}_rmsd.csv", index=False)

cw, per_atom = cw_rmsf_windows(u, lig, f"{eq}/equilibration_{sys_name}.csv", protocol_path)
cw.to_csv(f"{eq}/{sys_name}_cwRMSF.csv")                    # rows whole / free
```

| System | Image check | Align / RMSD | cw-RMSF |
|---|---|---|---|
| Protein + ligand(s) | as above | `backbone` | Per ligand resname |
| Apo or protein–protein | `ligand_sel=None` | `backbone`; one RMSD group per chain | Skip it |
| Small molecule in water | `min_image_distance(u_raw, "resname LIG and not name H*")` | `align_select="resname LIG and not element H"` (mdtraj syntax) | Skip it |
| Membrane | as above | `wrap_align_save_traj(..., is_membrane=True)` | as above |

The cw-RMSF itself is described in [Analyse MD](md-analysis.md#cw-rmsf).
