# MM-GBSA / MM-PBSA

`ProteinLigandAnalyzer.prepare_mmgbsa_batch` prepares an AMBER `MMPBSA.py` calculation on an AutoPath trajectory: equilibration or MD, aligned or not. Solvent is stripped and the box is not used. Each call writes:

- `<output_folder>/<name>_mmgbsa.in`;
- a SLURM job `qfiles_mmgbsa/<name>_mmgbsa.q`, which runs `ante-MMPBSA.py` and then `MMPBSA.py.MPI`;
- `run_mmgbsa_batch.sh`, which submits every job in `qfiles_mmgbsa/`.

```python
import os
from autopath.ap_PLIP import ProteinLigandAnalyzer as PLA

sys_dir, sys_name, R = os.path.abspath("runs/mysys"), "mysys", "LIG"
GB = {"HCT": (1, "mbondi"), "OBC2": (5, "mbondi2"), "GBn": (7, "bondi"), "GBn2": (8, "mbondi3")}

igb, radii = GB["GBn2"]
open("mmgbsa_GBn2.in", "w").write(open("mmgbsa_template.in").read().replace("igb=8,", f"igb={igb},"))
PLA.prepare_mmgbsa_batch(
    f"{sys_name}_GBn2",
    f"{sys_dir}/system.prmtop",
    f"{sys_dir}/equilibration/equilibration_{sys_name}_aligned.dcd",
    ligand_amber_selection=f":{R}", ligand_mda_selection=f"resname {R}",
    output_folder=os.path.abspath(f"{sys_name}/mmgbsa_GBn2"),   # the job changes into it
    mmpbsa_in="mmgbsa_GBn2.in", radii=radii,
    traj_slice=None,                  # (start, end, step), 1-based frames; None = all
    mpi_threads=32,                   # capped at the number of frames
    persistent_waters_cutoff=None,    # e.g. 0.9 keeps interfacial waters present in >= 90 % of frames
    partition="your-cpu-partition", time="0-12",
    slurm_template_fname=None,        # bundled template; copy and edit it for your cluster
)
# bash run_mmgbsa_batch.sh
```

## Input template

The template is copied through. Only the lines starting with `#startframe`, `#endframe` and `#interval` are rewritten, plus `strip_mask` when persistent waters are kept, so those lines must be present.

```text
&general
use_sander=1,
#startframe = 1,
#endframe = 100,
#interval = 1,
keep_files=0,
debug_printlevel=2,
netcdf=1,
strip_mask= ":K+:Cl-:CL:CIO:CS:IB:K:LI:MG:NA:RB:PC:PA:OL:POP:WAT:Na+:Cl-:Mg+:K+:HOH:NA:CL:K:MG"
/
&gb
igb=8,
saltcon=0.15,
/
&decomp
csv_format=1,
idecomp=1,
dec_verbose=3,
/
```

For **MM-PBSA**, replace the `&gb` block:

```text
&pb
istrng=0.15,
fillratio=4.0,
radiopt=0,
/
```

PB is much slower than GB. Drop `&decomp` unless you need per-residue terms.

## Choices

| Choice | Guidance |
|---|---|
| GB model | Run `HCT`, `OBC2`, `GBn` and `GBn2` as separate jobs; which one ranks best depends on the target |
| Frames | All frames, or a window through `traj_slice` (for example the equilibration free window). Use time-spaced equilibrium frames, not cluster centroids. |
| Waters | Stripped by default. `persistent_waters_cutoff` keeps bridging waters and writes their PDB next to the results. |
| Entropy | Not computed. Add `entropy=1` to `&general` for quasi-harmonic entropy. |
| SLURM | The bundled template (`autopath/data/mmpbsa_slurm_template.q`) loads `openmpi` and `amber` modules and excludes specific nodes. Copy it, adapt it to your cluster, and pass it as `slurm_template_fname`. |

## Results

Each output folder contains `FINAL_RESULTS_mmpbsa.dat`, `FINAL_DECOMP_mmpbsa.dat` and `complex.prmtop` / `receptor.prmtop` / `ligand.prmtop`.

```python
df = PLA.parse_mmpbsa_differences_table(f"{out}/FINAL_RESULTS_mmpbsa.dat")   # Component, Average, Std_Dev, Std_Err_Mean
dG = df.loc[df.Component == "DELTA TOTAL", "Average"].item()                 # kcal/mol
dec = PLA.parse_mmpbsa_deltas_all_components(f"{out}/FINAL_DECOMP_mmpbsa.dat")
PLA.plot_mmpbsa_byresidue(dec, top_residues=10, out_dir=out)
```

`examples/02_iAnalysis/paint_mmgbsa.py` paints the per-residue energies onto the structure. `orchestrator_mmgbsa.py` prepares jobs for many systems at once.

Frames from one short run are strongly correlated, so treat MM-GBSA as a ranking descriptor and report the pose stability of the same trajectory next to it.
