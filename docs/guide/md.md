# Conventional MD

`VanillaMD` runs unbiased production MD from an equilibrated system.

```python
from openmm.app import PDBFile
from autopath import VanillaMD
from autopath.utils import load_system, setup_logging

sys_dir, sys_name, run_id = "runs/mysys", "mysys", "mysys_rep1"
eq = f"{sys_dir}/equilibration"
setup_logging(f"{sys_dir}/autopath.log")

md = VanillaMD(
    system=load_system(f"{eq}/system_equil_{sys_name}.xml"),   # restraint-free system
    topology=PDBFile(f"{sys_dir}/system.pdb").topology,
    restrained_atoms=None,
    timestep=0.004,              # requires HMR (hydrogenMass=1.5 at build time)
    temperature=300,
    save_freq=25000,             # steps; 25000 x 4 fs = one frame per 0.1 ns
    out_dir=f"{sys_dir}/MD",
)
md.run(state_file=f"{eq}/checkpoint_equil_{sys_name}.xml",    # portable state XML
       run_id=run_id, MD_time=100,                              # ns
       restart_velocities=True)
```

**Outputs:**

- `<sys>/MD/MD_<run_id>.dcd` and `.csv` (state data)
- `MD_<run_id>_checkpoint.chk/.xml`
- `MD_<run_id>_system.xml`
- `MD_<run_id>.pdb`

`run()` returns `None`.

| Need | Do |
|---|---|
| Start state | `state_file=checkpoint_equil_<sys>.xml`. A `.chk` checkpoint only loads on the same GPU model and platform. When both are given, the state XML wins. |
| Independent replicas | Same start state, `restart_velocities=True`, a distinct `run_id` per replica |
| Extend a run | `state_file=<sys>/MD/MD_<run_id>_checkpoint.xml`, `restart_velocities=False`, and a **new** `run_id`. Reusing the run_id overwrites the trajectory. |
| Positional restraints | `restrained_atoms=` atom indices |
| Enough frames for RMSF | Choose `save_freq` so the run gives at least 100 frames |

The example [`05_VanillaMD`](https://github.com/forlilab/autopath/tree/develop/examples/05_VanillaMD) wraps this with the standard analysis.
