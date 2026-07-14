# Inline geometry logging for sMD trajectory clustering

**Date:** 2026-07-13
**Status:** Approved (design axes confirmed with user)

> **Update (post-benchmark pivot):** the inline shape backend is **RDKit, built
> once**, not numpy. The benchmark showed numpy ~5× *slower* per frame (LAPACK
> dispatch on tiny matrices), and RDKit's 3D descriptors compute from a
> bond-less, unsanitized, elements-only mol + conformer (no SDF, no bond orders,
> masses from elements) — verified byte-identical to a fully bonded mol. So the
> RDKit path is faster, needs no SDF/element-order coupling (elements come from
> the OpenMM topology), and supports **all** descriptors including
> `asphericity`. The numpy kernel and the asphericity exclusion below are
> **superseded** — the implementation builds one
> `ShapeDescriptorCalculator` (elements → RWMol + conformer) and mutates
> coordinates per frame.

## Motivation

Trace-feature clustering (`DTWPathModel` over `lag`/`work`/`r_before`) works well.
Geometric features (pocket–ligand `dist_*`, ligand 3D shape descriptors) can
improve it, but today they are obtained by **post-processing** the aligned DCDs
with MDAnalysis (`SMDData.calculate_pocket_distances`) and RDKit
(`LigandTrajectoryFeatures`). That re-reads every trajectory from disk.

During the pull, `SteeredMD` **already** pulls atomic positions to the CPU at the
`save_freq` cadence (for NC autostop / verbose monitoring / DCD writing) and
**already** computes the soft contact count `NC`. We can compute a small set of
scalar geometric descriptors from those same positions and log them to a sidecar
file — an almost-free, pre-computed substitute for the post-processing step.

## Goals

1. Speed up / clean up the NC kernel (drop the wasted `sqrt`).
2. Log per-frame scalar geometry during the pull, opt-in, decoupled from autostop.
3. Provide a fast, RDKit-validated **numpy** descriptor kernel for the hot loop;
   keep RDKit as an opt-in and in `LigandFeatures` post-processing.
4. Consume the logged geometry in clustering via the existing feature-merge path,
   with a user-selectable alignment mode.

## Non-goals / YAGNI

- No raw-position or full pairwise-distance-vector logging (duplicates the DCD /
  unstable columns). Only stable scalar columns.
- No change to physics, protocol grid, work/estimator pipeline.
- No removal of the existing post-processing feature builders — they remain the
  fallback and the validation reference.

## Design

### 1. NC / distance kernel (`steered_md.py`)

The switching function is
`NC = Σ 1/(1+(d/r0)^6) = Σ 1/(1+(d²/r0²)³)`, so the `sqrt` inside
`np.linalg.norm` is pure waste (the result is immediately raised to the 6th power).

- `_compute_nc`: compute squared distances via `einsum('ijk,ijk->ij', diff, diff)`
  and evaluate `1/(1+(d²/threshold²)³)`. Mathematically identical to the current
  implementation.
- `_get_pocket_atoms`: compare squared distances against `cutoff²` (skip `sqrt`).
  One-time per replica; kept for clarity/consistency.

Correctness-neutral. A unit test asserts the new NC equals the old norm-based
value on a fixed position array within floating tolerance.

### 2. Inline geometry logging (opt-in, autostop-independent)

New `SteeredMD.__init__` argument:

```
log_geom_features: bool | list[str] = False
```

- `False` (default): no sidecar written, no new code path exercised → **zero
  behavior change; no error when autostop is unused.**
- `True`: log the default set (below).
- `list[str]`: log the named subset.

Default feature set (all scalars):

| column                | needs pocket atoms | source |
|-----------------------|--------------------|--------|
| `geom_nc`             | yes                | existing `_compute_nc` |
| `geom_mindist`        | yes                | min ligand–pocket distance (nm) |
| `geom_rog`            | no                 | mass-weighted radius of gyration (nm) |
| `geom_eccentricity`   | no                 | inertia-tensor kernel |
| `geom_inertial_shape` | no                 | inertia-tensor kernel |
| `geom_npr1`           | no                 | inertia-tensor kernel |
| `geom_npr2`           | no                 | inertia-tensor kernel |
| `geom_spherocity`     | no                 | covariance kernel |
| `geom_pbf`            | no                 | SVD plane-of-best-fit |

`asphericity` is deliberately **excluded** from the numpy backend: its exact RDKit
formula did not reverse-engineer to <1e-4 tolerance. Per project preference we
exclude rather than fall back — requesting `asphericity` under the numpy backend
raises a clear "not supported by numpy backend" error. It remains available via
the opt-in RDKit inline backend and unchanged in post-processing `LigandFeatures`.

Pocket-atom detection (`subset_protein_HA`) is currently gated on
`verbose>0 or autostop_nc is not None`. Extend the gate: also detect when
`log_geom_features` requests a pocket-dependent feature (`geom_nc`/`geom_mindist`).
If pocket atoms are unavailable, those two columns are skipped with a warning;
ligand-only features still logged.

Cadence: same `save_freq` sampling as NC / DCD frames → sidecar rows map 1:1 to
DCD frames. Positions are reused from the `getState(getPositions=True)` already
performed for NC; when autostop is off but geom logging is on, one `getState`
per `save_freq` moves is added (same cost as DCD writing).

Output: `{out_dir}/sMD_{run_id}_geom.dat`, CSV, header
`step,time,<geom_* columns>`. Written incrementally alongside the trace `.dat`.
If no features could be computed for a run, no sidecar is written.

### 3. Numpy descriptor kernel + benchmark

New module `autopath/pulling/geom_kernel.py` with a pure-numpy function taking
heavy-atom positions + masses and returning all requested descriptors from a
**single** mass-weighted inertia-tensor eigendecomposition (+ one covariance
eigendecomposition and one SVD).

Validated formulas (all confirmed to match RDKit to ~1e-16 during design spikes):

- COM (mass-weighted) `c`; centered `X = pos − c`; `r2 = Σ X²` per atom.
- Inertia tensor `I = Σ(m·r2)·E₃ − Σ m·XXᵀ`; eigenvalues ascending
  `I1 ≤ I2 ≤ I3` (== RDKit PMI1/2/3, verified identical).
- `npr1 = I1/I3`, `npr2 = I2/I3`.
- `eccentricity = sqrt(I3² − I1²)/I3`.
- `inertial_shape = I2/(I1·I3)`.
- `rog = sqrt((I1+I2+I3)/(2·M))` in Å, `/10` → nm (matches
  `LigandFeatures._compute_rog` and RDKit `RadiusOfGyration/10`).
- `spherocity`: eigenvalues (ascending `e1≤e2≤e3`) of the **unweighted**
  covariance about the **geometric centroid** `Xg = pos − pos.mean(0)`,
  `cov = Xgᵀ·Xg / N`; `spherocity = 3·e1/Σe`.
- `pbf`: SVD of `Xg`; plane normal = last right-singular vector `n`;
  `pbf = mean(|Xg·n|)`.

**Validation (TDD):** a test asserts each numpy descriptor matches RDKit
(`Descriptors3D.*`, `rdMolDescriptors.CalcPBF`) within 1e-4 on several molecules
and conformers, and that requesting `asphericity` under the numpy backend raises.

**Benchmark:** a script (under `scripts/` or `tests/`) reports µs/frame for the
numpy kernel vs a slimmed RDKit path (reuse one conformer, `SetAtomPosition`, no
re-sanitization / no per-call extra checks). Expected outcome: numpy is much
faster for per-frame use. Default inline backend = **numpy**; RDKit remains an
opt-in (`geom_backend='rdkit'`) and unchanged in `LigandFeatures` post-processing.

### 4. Analysis-side consumption

- `SMDData.load_geom_features()`: read all `sMD_*_geom.dat` sidecars for the loaded
  logs (mirrors `load_logs`), returning a DataFrame with
  `trajname/speed/step/time` + `geom_*` columns. Missing sidecars → return what
  exists (or empty), never error.
- `AnalysisSMD._build_cluster_feature_df`: when geom features are requested and
  sidecars exist, add the geom DataFrame as another source to
  `merge_feature_sets`. If requested but sidecars are missing, warn and skip
  (same pattern as the missing-reference-PDB ligand-feature skip).
- Alignment control `geom_merge`:
  - `'impute'` (**default**): `merge_asof` nearest-fill — every trace row gets the
    closest geom sample. No NaN, PCA-safe, full trace resolution; geom repeats
    between DCD frames.
  - `'aligned'`: keep only rows with an exact geom time match (subsample trace to
    the DCD cadence). No imputation/repeats; coarser grid.
- `geom_*` columns are a small (~10) set, used as ordinary scalar features
  (no forced PCA). The existing `geom_feature_prefix` PCA path in `DTWPathModel`
  is unchanged and still applies to `dist_*` columns.

## Backward compatibility / safety

- All new behavior is behind `log_geom_features` (default `False`) on the pulling
  side and behind explicit geom requests on the analysis side.
- No sidecar / no geom request → identical behavior to today.
- Analysis tolerates absent sidecars without error, satisfying the requirement
  that not using autostop (or geom logging) never breaks pulling or analysis.

## Testing

1. `test_compute_nc_squared_matches_norm` — new NC == old norm-based NC.
2. `test_geom_kernel_matches_rdkit` — numpy descriptors == RDKit within 1e-4 (TDD).
3. `test_pull_geom_logging_off` — default run writes no `_geom.dat`, no error
   (autostop off).
4. `test_pull_geom_logging_on` — sidecar written with expected columns/cadence.
5. `test_load_geom_features_missing` — analysis skips absent sidecars gracefully.
6. `test_geom_merge_modes` — `impute` (no NaN, full rows) vs `aligned` (exact-time
   subset) both produce PCA-usable matrices.
7. Benchmark script (reported, not asserted): numpy vs slim RDKit µs/frame.
