# Overview

`ap_vanillaMD.py` runs conventional (unbiased) production MD from a system equilibrated with `01_Build_and_Equilibrate`, using AutoPath's `VanillaMD` class, and analyses the trajectory:

* wraps, centres and superposes it on the protein backbone (`MD_<run_id>_aligned.dcd`);
* RMSD of protein and ligand;
* per-atom ligand RMSF and pocket-contact frequency, and the contact-weighted RMSF (cw-RMSF) over polar (N, O, S), non-polar (C, halogens) and all heavy atoms (see `01_Build_and_Equilibrate/README.md` for the definition).

Settings are at the top of `main()`: `MD_TIME` (ns, default 100), `TIMESTEP` (4 fs, needs HMR), `TEMPERATURE` (300 K), `SAVE_FREQ` (steps; 25000 = 0.1 ns), `RESTART_VELOCITIES`, `RESTRAINED_ATOMS`.

# Usage

```bash
python ap_vanillaMD.py --sysname 3ptb --ligname UNK --replica rep1
```

`--eqdir` points to the equilibrated system (default `../01_Build_and_Equilibrate/<sysname>`). Run several `--replica` values for independent replicas from the same equilibrated state (new velocities each time).

# Output Files

```
{sysname}/
├── autopath.log
└── MD/
    ├── MD_{run_id}_aligned.dcd, MD_{run_id}.csv
    ├── {run_id}_rmsd.csv (+ RMSD plots)
    ├── {run_id}_RMSF.csv                  # per-atom rmsf, contact_freq
    ├── {run_id}_cwRMSF.csv                # cw-RMSF (polar, non-polar, all)
    └── {run_id}_RMSF.svg, {run_id}_contact_freq.svg
```

with `run_id = <sysname>_<replica>`.
