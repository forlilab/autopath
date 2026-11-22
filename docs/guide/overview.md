# Workflow overview

AutoPath stages are independent Python classes. Each stage reads the files the previous one wrote, so you can run them in one script, in separate jobs, or skip stages whose outputs already exist.

```
receptor.pdb + ligand.sdf
        │  SystemPreparation          (guide/build)
        ▼
<sys>/system.{pdb,xml,prmtop}
        │  Equilibration              (guide/equilibrate)
        ▼
<sys>/equilibration/ ── checkpoint_equil_<sys>.xml, system_equil_<sys>.xml
        ├─ VanillaMD  → <sys>/MD/              → MD analysis
        ├─ SteeredMD  → <sys>/sMD/trajectories → SMDAnalysis → readouts
        └─ MM-GBSA / LIE on the equilibration or MD trajectory
```

## The `<sys>` name

Every file name downstream embeds the system name `<sys>`: the `run_id` passed to `Equilibration.run`. Later stages rebuild paths from it, so they must reuse exactly the same name.

```
<sys>/system.pdb  system.xml  system.prmtop  system.rst7
<sys>/equilibration/
    equilibration_<sys>.dcd / .csv         raw trajectory, state data (Step, Time (ps), ..., Speed (ns/day))
    <sys>_equilibrated.pdb
    checkpoint_equil_<sys>.chk / .xml      start state for MD and pulls (use the .xml)
    system_equil_<sys>.xml                 restraint-free system for MD and pulls
    equilibration_protocol.json            protocol actually used
<sys>/MD/MD_<run_id>.dcd / .csv, MD_<run_id>_checkpoint.xml, MD_<run_id>.pdb
<sys>/sMD/trajectories/sMD_replica-<id>_v<speed>_forward.{dat,_geom.dat,dcd}
<sys>/sMD/analysis_traces/                 mixture_pmfs.csv, sMD_processed_data.csv, friction, koff
```

## Rules that apply to every stage

- **Logging:** call `setup_logging(f"{sys_dir}/autopath.log")` first. Every stage writes what it did there (restraint counts, box size, PBC guard stops).
- **Portable start states:** start MD and pulls from `checkpoint_equil_<sys>.xml`. A `.chk` checkpoint only loads on the same GPU model and platform.
- **PBC-dependent analysis on raw trajectories:** `wrap_align_save_traj` superposes frames (which no longer match the box) and deletes the raw DCD by default. Run periodic-image checks first.
- **Selection languages:** MDAnalysis everywhere except `wrap_align_save_traj(align_select=...)`, which uses mdtraj syntax (`not element H` instead of `not name H*`).
- **Validate small first:** run one system with a shortened protocol before a campaign.
