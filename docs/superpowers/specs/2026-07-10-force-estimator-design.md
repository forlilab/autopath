# Force estimator — design

**Date:** 2026-07-10
**Author:** Manuel Llanos (with Claude)
**Status:** Approved, ready for implementation plan

## Motivation

The paper-figures notebook
`scratch/paper_figures/dG_extrapolated_HSP90_6ELO_BAW.ipynb` prototypes a third
free-energy estimator — the **force estimator** — alongside the existing
`cumulant` and `jarzynski` estimators, plus a companion **force-based friction**
route. This spec brings both into `autopath` proper and wires the estimator on
by default whenever multiple pulling speeds are available.

## Core insight

Both the PMF and the friction the notebook computes fall out of the **same
across-speed OLS** that `autopath.pulling.Estimators.extrapolate_to_v0` already
performs, applied to two different per-speed quantities measured at each protocol
step:

| Per-speed quantity regressed vs pulling speed `v` | Intercept (v→0) | Slope |
|---|---|---|
| mean cumulative work `⟨W⟩(r; v)` | **ΔG** (reversible work) → PMF | integrated friction |
| mean restraint force `⟨F⟩(r; v)` | **Feq(r)** (equilibrium force) | **Γ(r) = dF/dv** → local friction |

Because `W = ∫F dr`, the work-slope equals the integrated force-slope — the two
routes are physically consistent. The linear-response relations
`⟨W⟩(v) = ΔG + γ_W·v` and `⟨F⟩(v) = Feq + Γ·v` mean the estimator is **inherently
multi-speed**: there is no v→0 intercept without ≥2 distinct speeds. That is
precisely why it should default on only when multiple speeds exist.

The force estimator differs from the existing two only in the *per-speed input
value* fed to the shared v→0 machinery:

- cumulant:  per-speed `dG = Wmean − β·Var/2`
- jarzynski: per-speed `dG = −(1/β)·ln⟨e^{−βW}⟩`
- **force:   per-speed `dG = Wmean` (raw, uncorrected mean work)**

All three are then extrapolated to v→0 by the same `extrapolate_to_v0` OLS,
yielding the same `R²` / intercept-SE bands.

## Scope

In scope (both confirmed by the user):

1. **ΔG / PMF** via v→0 extrapolation of mean work (notebook's `dG_extrapolated`
   force panel).
2. **Force-based friction** — `Γ = dF/dv`, `Feq` intercept (notebook's
   `friction_four_ways` panels c/d).

Out of scope:

- The Hummer–Szabo PMF on the deconvolved coordinate `z = r_target − F/k`
  (a distinct bin-based estimator, not a per-speed extrapolation).

## Component design

### 1. `ForceEstimator(BaseEstimator)` — PMF part

Location: `autopath/pulling/Estimators.py`, mirroring `CumulantEstimator` /
`JarzynskiEstimator`.

- `name = 'force'`.
- `estimate_dG(raw_W, beta, **kwargs)` returns
  `{'Wmean': Wmean, 'dG': Wmean, 'Wdiss': np.nan}`. The per-speed "PMF" is the
  raw uncorrected mean work; its v→0 intercept is the true ΔG. `Wdiss` is NaN
  because the force estimator's friction is derived from the force column, not
  from a per-speed dissipation split.
- `fit_transform(smd_data)` mirrors the other two estimators (group by
  `['step', 'speed', 'path']`, look up `r_coord` from `protocol_grids`, call
  `estimate_dG`, push via `add_estimator_results('force', ...)`), and
  **additionally** records a `Fmean` column = mean of `group['force']` per
  (step, speed, path). `Fmean` is the input to the force friction.

The per-speed force rows flow unchanged through `calculate_weighted_pmf`
(→ `mixture_pmfs`) and `extrapolate_to_v0(param='dG')`, which produces the force
ΔG at v→0. This means the per-speed "force PMF" (raw work) appears in
`mixture_pmfs` as `estimator='force'` rows — consistent with how cumulant /
jarzynski per-speed PMFs appear, and matching what the notebook plots as the raw
per-speed work.

`Fmean` is **not** added to `calculate_weighted_pmf`'s `weight_cols` (that list
is global; cumulant/jarzynski rows carry no `Fmean` and would raise). It stays in
`sMDDdata.results` on the force rows and is consumed directly by the force
friction methods (§2).

### 2. Force-based friction — `FrictionEstimator`

Location: `autopath/pulling/Estimators.py`. Two new methods mirroring the
existing `gamma_from_wdiss_regression` / `gamma_from_wdiss_derivative` pair, so
`KramersEstimator`'s existing `method='regression'` / `method='derivative'`
selection logic needs no change. Both take the **force-estimator `results`**
(rows with `estimator='force'`, carrying `Fmean`) plus the reused p_eq weights,
and first collapse to a path-mixed `⟨F⟩(step, speed)` (weighted mean of `Fmean`
over paths; for single-path systems this is just the mean) before regressing
across speeds:

- `gamma_from_force_regression(df)`: calls `extrapolate_to_v0(param='Fmean')` on
  the path-mixed `⟨F⟩(step, speed)` frame.
  - `Gamma = Fmean_slope` (local friction Γ = dF/dv — note: for the force route
    the *slope itself* is the local friction, unlike the Wdiss route where the
    slope is `Gamma_integrated` and the local Γ is its r-derivative).
  - `Gamma_integrated = ∫Γ dr` (cumulative trapezoid via the existing
    `_cumulative_integral`).
  - `Feq` = `Fmean` intercept (retained as a diagnostic column).
  - `method='regression'`, `estimator='force'`.
- `gamma_from_force_derivative(df)`: per speed, `Γ_sp(r) = (F(r; v) − Feq(r)) / v`
  using the `Feq(r)` from the regression intercept. `method='derivative'`,
  `estimator='force'`.

Output columns match the existing `friction.csv` schema:
`r_coord, speed, Gamma, Gamma_integrated, method, estimator` (+ `step` when
available).

### 3. `run()` wiring — `AnalysisSMD.py`

- **Speed guard (the "default only when multiple speeds" behavior):** early in
  `run()`, if fewer than 2 distinct speeds are present, drop `'force'` from the
  active estimators and log a warning (`force` needs ≥2 speeds for a v→0
  intercept). This is enforced inside `run()` so it holds regardless of caller,
  and it **warns-and-drops silently** rather than raising.
- **Friction loop branch:** in the friction section (currently ~lines 453–461),
  for `estimator.name == 'force'` call `gamma_from_force_regression` /
  `gamma_from_force_derivative`; for all other estimators keep the existing
  `gamma_from_wdiss_*` calls.
- **Path weights:** set
  `weights_by_estimator['force'] = weights_by_estimator.get('cumulant')` (falling
  back to `jarzynski`, then any available) instead of running `compute_p_eq` on
  the force estimator. Path population is physical / estimator-agnostic, and
  raw-work Boltzmann weights would be dissipation-biased. No effect on
  single-path systems (p_eq = 1).

### 4. Registration & defaults

- Add `'force': ForceEstimator` to `ESTIMATOR_REGISTRY` (`Estimators.py`).
- Add `'force'` to the `_setup_estimators` `estimator_map` (`AnalysisSMD.py`).
- Add `ForceEstimator` to package exports (`autopath/pulling/__init__.py`).
- Add `'force'` to the full-analysis default estimator list
  (`autopath/autopath_core.py` full-analysis caller, and the
  `SMDAnalysis.__init__` default list). The run() speed guard makes this safe
  when only one speed is present.
- **Do not** add `'force'` to `check_convergence`'s `allowed_estimators` —
  convergence is inherently per-speed and the force estimator has no meaningful
  per-speed ΔG.

### 5. Tests — `tests/test_force_estimator.py` (new)

1. `ForceEstimator.estimate_dG` returns `dG == Wmean` and `Wdiss` NaN for a
   sample work array; returns `None` for an empty array.
2. Synthetic `⟨W⟩ = ΔG + γ·v` across ≥3 speeds → `extrapolate_to_v0(param='dG')`
   on force-labelled rows recovers ΔG (intercept) and γ (slope) within
   tolerance.
3. Synthetic `⟨F⟩ = Feq + Γ·v` → `gamma_from_force_regression` recovers `Feq`
   (intercept) and `Γ` (slope).
4. `run()`-level guard: with a single speed and `'force'` requested, force is
   dropped with a warning and the pipeline completes on the remaining
   estimators.

## Data-flow summary

```
raw_data[force, work] ──ForceEstimator.fit_transform──▶ results[estimator='force', dG=Wmean, Fmean]
                                    │                                     │
              calculate_weighted_pmf (reuse cumulant p_eq)   gamma_from_force_regression/derivative
                                    │                          (path-mix Fmean → ⟨F⟩(step,v),
                                    ▼                           reuse cumulant p_eq)
                     mixture_pmfs[estimator='force']                     │
                                    │                                     ▼
                     extrapolate_to_v0(param='dG')            friction.csv[estimator='force']
                                    │                                     │
                            v→0 ΔG (force PMF)                            │
                                    └────────── Kramers (v0 dG + regression Γ) ┘
```

## Files touched

- `autopath/pulling/Estimators.py` — `ForceEstimator`, two friction methods,
  registry entry.
- `autopath/pulling/AnalysisSMD.py` — `_setup_estimators` map, run() speed guard,
  friction-loop branch, force path-weight aliasing, default list.
- `autopath/pulling/__init__.py` — export `ForceEstimator`.
- `autopath/autopath_core.py` — add `'force'` to the full-analysis default list.
- `tests/test_force_estimator.py` — new.
