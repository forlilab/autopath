# Steered MD pulls

`SteeredMD` pulls the ligand out of its pocket along the centre-of-mass distance between ligand **anchor** atoms (group A) and **pocket** atoms (group B). The restraint centre moves in small steps. Each replica restarts from the equilibrated state with new velocities, relaxes 50 ps at its start position, then pulls forward.

```python
import os
from glob import glob
import pandas as pd
import MDAnalysis as mda
from openmm.app import PDBFile
from autopath import SteeredMD
from autopath.utils import (load_system, setup_logging, wrap_align_save_traj, periodic_image_report,
                            get_pocket_atoms_idxs, get_ligand_anchor_atoms)

sys_dir, sys_name, lig = "runs/mysys", "mysys", "resname LIG"
eq, pdb = f"{sys_dir}/equilibration", f"{sys_dir}/system.pdb"
setup_logging(f"{sys_dir}/autopath.log")
system = load_system(f"{eq}/system_equil_{sys_name}.xml")

u = mda.Universe(f"{eq}/{sys_name}_equilibrated.pdb", f"{eq}/equilibration_{sys_name}_aligned.dcd")
lig_ha = f"{lig} and not name H*"
pocket_idx = get_pocket_atoms_idxs(u, ligand_selection=lig_ha)             # CA within 8 A of the ligand
anchor_idx = get_ligand_anchor_atoms(u, lig, mode="murcko", n_atoms=5, out_dir=sys_dir)
n_heavy = len(u.select_atoms(lig_ha))

smd = SteeredMD(
    system=system, topology=PDBFile(pdb).topology,
    groupA_atoms=anchor_idx, groupB_atoms=pocket_idx,
    restart_velocities=True, timestep=0.004, temperature=300,
    out_dir=f"{sys_dir}/sMD/trajectories",
    dx_per_move=0.001, max_displacement=3.5,            # nm
    sMD_spring_cte=418.4 * n_heavy,                     # kJ mol-1 nm-2 per ligand heavy atom
    autostop_nc=0.01, autostop_nc_window=5, autostop_min_displacement=0.5,
    save_freq=5, log_geom_features=True,
)
for speed in (0.005, 0.010, 0.015):                     # nm/ps (0.001 nm/ps = 1 m/s)
    for rep in range(10):
        smd.run(pulling_speed=speed, pulling_direction="forward",
                state_xml_file=f"{eq}/checkpoint_equil_{sys_name}.xml")

trajs = [t for t in sorted(glob(f"{sys_dir}/sMD/trajectories/sMD_replica-*.dcd")) if "aligned" not in t]
pd.concat([periodic_image_report(mda.Universe(pdb, t), system, ligand_sel=lig).assign(replica=os.path.basename(t))
           for t in trajs]).to_csv(f"{sys_dir}/sMD/image_distances.csv", index=False)
wrap_align_save_traj(trajs, pdb)
```

## Settings

These are the AutoPath paper defaults.

| Setting | Default | Notes |
|---|---|---|
| Pocket (group B) | Protein Cα within 8 Å of the ligand | A hand-picked pocket uses `pocket_selection` in `system.pdb` numbering, which is sequential and renumbered, not the original PDB numbers |
| Anchor (group A) | `murcko`: Murcko scaffold heavy atoms | Also `lig_ha`, `lig_com`, `pocket_com`, `contacts`, `inertia`, `ca`, `weighted_com` |
| Spring | 418.4 kJ mol⁻¹ nm⁻² × ligand heavy atoms (all of them, not only the anchors) | The log reports `sigma/dx`; keep it ≥ 10 |
| Speeds | 5, 10, 15 m/s | The force (v→0) estimator needs at least two speeds |
| Replicas | 10 per speed to start | Add more if the profile has not converged |
| Autostop | Ligand–pocket contacts < 1 % of the initial value for 5 samples, after ≥ 0.5 nm | |
| PBC guard | Stops a pull when the ligand comes within cutoff + `pbc_safety_nm` (0.2 nm) of a protein image | Logged as `[PBC guard] Stopping at move ...`. Frequent stops mean the box is too small for the pull length. |

## Output and parallel jobs

- **Output files:** each `run()` writes `sMD_replica-<id>_v<speed>_forward.dat` (restraint position, CV, force and work per move), `_geom.dat` (geometry features per frame) and a DCD.
- **Default ids:** the default `<id>` is the time of day (`HHMMSS`), so parallel jobs on one system should be split **by speed**.
- **Explicit ids:** to split one speed across jobs, pass a unique `run_id`. It must parse as an integer and be a valid `HHMMSS` value.
- **Not covered:** proteins or peptides as the pulled partner. The Murcko anchor and per-heavy-atom spring assume a small molecule.
