# SLC15A4 minibinder sMD campaign — design

Date: 2026-09-24
Status: approved design, not yet planned or implemented

Target project: `/gpfs/group/forli/mllanos/peter/minibinders/` (not a git
repository; this spec is versioned here because it exercises autopath's
untested `force` estimator path). Library code lives in this repo; campaign
scripts land in the target project.

## 1. Purpose

Rank 7 de novo SLC15A4 minibinders by unbinding barrier, computed from steered
MD launched off the 25 equilibrated end points each already has, and compare
that ranking to BLI and to the project's existing MM-PBSA and LIE rankings.

The project's question is whether a computational pipeline separates true
binders from decoys. All 7 systems here are binders, so the operational test is
whether the sMD barrier *orders* them better than MM-PBSA did (which split by
size, not affinity).

### Deliverable

Per design: a v→0 extrapolated barrier height and r_TS, with between-seed
bootstrap error bars; Spearman against BLI Kd; a head-to-head table against the
existing MM-PBSA and LIE rankings. Jarzynski and cumulant estimates are
reported alongside as a sensitivity check — they come free from the same data.

### Non-goals

- Absolute ΔG of binding. sMD gives an unbinding barrier, not a reversible
  binding free energy.
- koff. Kramers friction was previously measured to be floored for 33–43% of
  points and to contribute a near-constant offset, making koff ranking
  rank-identical to barrier ranking. Not worth the extra machinery.
- Any change to autopath's public API. The campaign is driver + analysis
  scripts using autopath as a library.

## 2. Starting material (verified present)

| item | path |
|---|---|
| topology | `systems/<sys>/system.pdb` (~176k atoms, 8,327 protein; slc_m2 = 176,388) |
| System XML | `systems/<sys>/equilibration/system_equil_<sys>.xml` (carries the MonteCarloMembraneBarostat) |
| 25 end points | `systems/<sys>/replicas/rep_NN/<sys>_repNN_final.xml`, NN = 01..25 |

175 end-point State XMLs exist (25 × 7, verified). They are `saveState` output
written by the project's own `run_replica.py` against the same stripped System,
so they do **not** carry the stale `k_*` restraint globals that make
`equilibration/checkpoint_equil_<sys>.xml` unloadable.

Systems: `slc_m1` … `slc_m7`. No sMD has ever been run in this project and no
AutoPath driver exists there; everything is hand-driven library usage.

## 3. Reaction coordinate

COM–COM distance between two Cα groups, pulled by a moving harmonic restraint
(`cvpack.CentroidFunction`, `pbc=True`), `pulling_direction="forward"`.

### groupA — binder Cα

`sMD_ligand_anchor_mode="ca"`, which is already implemented in
`autopath/utils.py::get_ligand_anchor_atoms` (the `mode == "ca"` branch selects
`(<ligand>) and name CA`). No new pulling mode is needed.

| sys | binder resid | n res | n heavy | n Cα |
|---|---|---|---|---|
| slc_m1 | 451–529 | 79 | 666 | 79 |
| slc_m2 | 451–530 | 80 | 667 | 80 |
| slc_m3 | 451–533 | 83 | 639 | 83 |
| slc_m4 | 451–533 | 83 | 648 | 83 |
| slc_m5 | 451–533 | 83 | 634 | 83 |
| slc_m6 | 451–530 | 80 | 669 | 80 |
| slc_m7 | 451–509 | 59 | 466 | 59 |

Binder residues are numbered from 451 in the prmtop's global numbering.
`inputs/residue_map.csv` numbers residues *per chain* and must not be used as a
selection source.

### groupB — 98 shared helical receptor Cα

The receptor is the identical 450-residue SLC15A4 in all 7 systems, so groupB is
shared. This makes the reaction coordinate identical across designs, which is
required for the barriers to be comparable.

Derivation (one-off, already run): over 40 frames of each system's 25 ns
continuation trajectory, take receptor residues with ≥50% frequency of
heavy-atom contact within 6 Å of a **helical** binder residue, intersect with
receptor residues that are themselves helical (mdtraj DSSP H/G/I, majority over
the same frames), then union across all 7 systems.

```
12,15-16,19-20,53-54,57-58,60-61,65,70,109,113-119,121-122,125,128-130,
132-134,136-137,139-141,143-144,147-148,151,196,210-211,214,217,249-258,
260-261,263-266,268-272,283,337-338,340-342,344-345,348-349,351-357,
360-361,363-369,371-372,375,437,440,444
```

23 of these are contacted by all 7 designs; the union is 98.

### Why shared, and why the union

Measured r₀ and pull direction for the three candidate definitions:

| definition | n Cα | r₀ range (nm) | r₀ spread | dir_z | usable pull (nm) |
|---|---|---|---|---|---|
| **shared union** | 98 | 1.37–1.69 | **0.32** | −0.94…−1.00 | 2.82–3.24 |
| per-system helix–helix | 52–62 | 1.21–1.73 | 0.52 | −0.94…−0.98 | 3.05–3.35 |
| shared core | 23 | 1.24–1.60 | 0.36 | −0.87…−0.95 | 3.19–3.71 |

The union has the tightest r₀ spread (so remaining differences are physical, not
definitional), the largest and therefore least noisy centroid, and the most
consistently −z direction. It costs pull length: slc_m2 binds at 2.82 nm.

## 4. Geometry and the pull budget

Equilibrated box 12.67 × 13.30 × 9.84 nm (slc_m1 rep_01; the others differ by
<1% and `SteeredMD` re-evaluates the cap from each seed's own box at runtime).
**z is the thin axis and the pull is
essentially pure −z in all 7 systems** (dir_z −0.94 to −1.00 with the union
selection), so the PBC cap is set by z.

`SteeredMD` caps moves at `min(L_i/2 · |v|/|v_i|) − pbc_safety_nm`, giving
4.45–4.74 nm absolute with the union selection; with r₀ = 1.37–1.69 nm that
leaves 2.82–3.24 nm of travel.

**`max_displacement = 2.7 nm`** — below the 2.82 nm binding case, with margin
for NPT box breathing. This also gives every system and every speed an identical
2700-move protocol grid, which the multi-speed analysis requires.

## 5. Simulation parameters

| knob | value | reasoning |
|---|---|---|
| `max_displacement` | 2.7 nm | §4 |
| `sMD_max_r_offset` | 2.7 nm | matches; code's PBC cap is the backstop |
| `pbc_safety_nm` | 0.5 (default) | z is tight |
| `dx_per_move` | 0.001 nm | 2700 moves; held constant across speeds (required: it is the shared analysis grid) |
| pulling speeds | 0.001 / 0.005 / 0.010 nm/ps | = 1 / 5 / 10 m/s |
| `steps_per_move` | 250 / 50 / 25 (derived) | all ≥ 1.5, no clipping |
| `sMD_spring_cte` | 418.4 × n_binder_heavy → 1.95–2.80·10⁵ kJ/mol/nm² | autopath's per-atom convention |
| σ_thermal | 0.0030–0.0036 nm (derived) | dx = σ/3; see §9 |
| `timestep` | 0.004 ps | matches the project's HMR 4 amu setup |
| `temperature` | 300 K | |
| `restart_velocities` | True | fresh Maxwell draw + 50 ps free relaxation per replica |
| `autostop_nc` | 0.01, window 5, min_disp 0.5 nm | stop at 1% of initial contacts |
| `log_geom_features` | True | Cα-trace shape descriptors → DTW clustering |
| `save_freq` | 5 | single knob: sets DCD, NC-autostop and geom cadence together |
| `restrained_atoms` | None | pilot decides; see §8 |
| `use_NVT` | False | keep the membrane barostat |
| `use_GReweighting` | False | |

Note `log_geom_features` descriptors are computed over `groupA_atoms`, so with
`ca` they describe the binder's **Cα trace**, not its all-atom shape. That is
acceptable as a clustering feature; it is not an all-atom shape descriptor.

### Cost

Per seed across all three speeds: 2.7 + 0.54 + 0.27 ns of pulling plus 3 × 50 ps
relaxation = **3.66 ns**. Per system 91.5 ns. **640 ns for all 7.**

## 6. Seed allocation

**25 seeds × 3 speeds × 1 replica = 75 replicas per system, 525 total.**

Rationale: the project's own MM-PBSA analysis found 73–93% of variance sits
*between* productions, i.e. the starting structure dominates. At fixed cost,
maximising distinct starting structures beats repeating seeds. Because
`restart_velocities=True` already gives every replica a fresh Maxwell velocity
draw plus 50 ps of free relaxation, a repeat from the same seed is strictly
weaker independence than a new seed.

Seed is the blocking factor for error bars. DTW path clustering (unbinding
routes) is an orthogonal grouping, and 75 trajectories per system finally
clears `min_replicas_per_path = 5`.

### Run-id encoding

`run_id = f"00{seed:02d}01"` → filename `sMD_replica-00NN01_v<speed>_forward.dat`.

This is load-bearing and has been verified against the parsers:

- `SMDData._replica_idx_from_log` does `int(id)`, so the id must be numeric —
  a name like `seed07` would raise.
- `SMDData._replica_start_datetime` decodes the id as HHMMSS; `00{seed}{rep}`
  is hh=00, mm=seed, ss=rep, which is always a valid time for seed ≤ 59.
- The ladder sorts replicas by that datetime, so this encoding makes ordering
  **seed-major**: ladder rung *k* means exactly "seeds 1…k".

It also removes the collision risk of the default bare-HHMMSS id when 75 array
tasks start in the same wall-clock second.

## 7. Components

### 7.1 `prep/pull_selections.json` (new, in the target project)

Frozen groupA ranges per system and the shared groupB resid list from §3.
Re-verified against the prmtop at runtime; a mismatch is a hard error, not a
fallback.

### 7.2 `prep/run_smd.py` (new)

One replica per invocation, mirroring the existing `run_replica.py` idiom:

```
python prep/run_smd.py -n <sys> -k <seed 1..25> -v <speed>
```

- loads `system.pdb` (topology) and `equilibration/system_equil_<sys>.xml`
- resolves groupA/groupB atom indices from `pull_selections.json`
- constructs `SteeredMD(...)` with §5
- `sMD.run(pulling_speed=v, pulling_direction="forward", run_id=f"00{k:02d}01",
  state_xml_file=<.../rep_NN/<sys>_repNN_final.xml>)`

`state_xml_file` is the correct entry point: it deserializes the State and sets
box, positions and velocities individually rather than calling `loadState`, so
it is immune to the stale-global-parameter failure.

Output: `systems/<sys>/sMD/trajectories/`.

### 7.3 `qfiles/smd_ARRAY.q` (new)

`--array=1-75` per system; seed = `((t-1)//3)+1`, speed = `SPEEDS[(t-1)%3]`.

The slow-GPU exclude list and a `MIN_NS_PER_DAY` throughput guard go in the
template, never the submit command. Slow cards are ~8× slower on these systems
and a plain partition list does not protect against landing on one.

**Disk.** Full DCDs would total 260 GB (176,388 atoms × 702 frames per seed).
The same array task strips each DCD to protein-only (8,327 atoms, 21× smaller)
and deletes the original → ~12 GB total. The `.dat` and `_geom.dat` sidecars
carry everything the estimators and the clustering need; the stripped DCD is
for QC and visualisation only.

### 7.4 `analysis/smd_barriers.py` (new)

Per system:

1. `SMDData(logs, sysname, temperature=300, reference_pdb=system.pdb)`
2. `SMDAnalysis(path_model='dtw', estimators=['force','jarzynski','cumulant'])`
   `.run(smd, geom_features=True, merge_features=False, cluster_to_boundary=True)`
3. `extrapolate_to_v0(..., min_speeds=2)` → v→0 PMF → barrier + r_TS
4. `force_convergence_ladder(sa, logs, speeds=[0.001,0.005,0.01], conv_window=5)`

The ladder runs **post hoc as a diagnostic**, not as an autostop. The fixed
25-seed grid runs to completion because the seeds are already paid for; the
ladder only answers "had the barrier stopped moving by rung 25?".

Error bars are a **bootstrap over the 25 seeds** — resample seeds with
replacement, redo the v→0 extrapolation, take the percentile interval. This is
the direct lesson from the MM-PBSA campaign, where reported SEs understated the
truth 2–6× by ignoring between-production variance. Within-replica error is not
the relevant scale.

### 7.5 `analysis/smd_figs.py` (new)

Barrier ± seed-bootstrap CI per design, ranked; Spearman vs BLI; head-to-head
against the MM-PBSA and LIE rankings; the same panel repeated for jarzynski and
cumulant as sensitivity.

## 8. Pilot — slc_m2, seeds 01–05, 3 speeds

15 replicas, 18.3 ns, ~2–4 h on one fast GPU. slc_m2 because it is the
PBC-tightest system (2.82 nm usable): if 2.7 nm works there it works everywhere.

Force mode has never been run, and the receptor is unrestrained during the pull
(`autopath_core` hardwires `restrained_atoms=None`), so nothing but the membrane
opposes the −z pull on the binder.

### Go / no-go criteria

1. `autostop_nc` fires before the 2700-move cap in ≥4/5 slow replicas.
   Otherwise the pull is too short and we are measuring a partial barrier.
2. Receptor COM drift < 0.5 nm and no large tilt over the pull → no restraint
   needed. Otherwise add weak Cα restraints on the TM core and re-pilot.
3. No `Capping sMD_moves` warning in any log (would mean PBC clipped us).
4. `extrapolate_to_v0` R² ≥ 0.9 through the barrier region — linear-in-v holds
   over a full decade. If not: add a slower speed, or drop 10 m/s.
5. Realized vs requested speed within 5% (`SteeredMD.run` warns past that).
6. Measured wall-time per replica, to firm the array budget.

Only after all six pass does the full campaign launch.

## 9. Risks

**Linearity over a decade of speed** is the main scientific risk. The force
estimator's ΔG is the v→0 intercept of a linear fit; prior work on this codebase
found the Wdiss extrapolation carries a large bias floor and that the intercept
must be left free, never forced through the origin. Criterion 8.4 tests this
directly and the remedy (a 4th, slower speed) is known.

**dx = σ_thermal/3**, coarser than the σ/10 rule of thumb in `SteeredMD`'s own
docstring, because the per-atom spring convention makes k ≈ 2.5·10⁵ kJ/mol/nm²
for a 600-heavy-atom binder. This is the stiff-spring regime, which is the
favourable one for work and PMF (minimal lag), but the pilot will check the
logged `r_before`/`r_target` columns. autopath only warns past dx > 0.5σ, which
we are under.

**DTW may find no genuine unbinding routes**, in which case the silhouette will
say so. `path_model='null'` (one path, no clustering) is the documented
fallback and does not invalidate the barrier.

**Receptor drift** — see 8.2. Adding restraints biases the PMF, so it is a
measured decision, not a default.

## 10. Open item

The BLI Kd values exist only as `updated_BLI.png`. The final ranking comparison
(§7.5) needs them machine-readable. This does not block the campaign, only the
last figure.
