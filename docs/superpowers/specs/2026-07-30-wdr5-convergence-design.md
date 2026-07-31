# WDR5 sMD convergence — centralized multi-estimator analysis

**Date:** 2026-07-30
**Status:** approved, ready for implementation planning
**Scope:** WDR5 only. HSP90 (koff) is a deliberate follow-up, not part of this spec.

## Problem

Convergence analysis for WDR5 is duplicated across two notebooks and only ever
covers the cumulant estimator:

- `scratch/paper_figures/sMD_WDR5_convergence.ipynb` — single-system deep dive
  (`6dy7_A`, `analysis_features`). Plots per-rung metric traces and draws the
  tolerance lines, but never computes a pass/fail flag.
- `scratch/paper_figures/sMD_WDR5.ipynb` — cross-system. Cell 11 recomputes the
  `converged` flag inline; cell 13 derives wall-clock minutes-to-converge.

Both hardcode the tolerances `4.0 / 3.0 / 0.1` with a comment pointing at
`autopath_core.CONVERGENCE_TOLERANCES`, and neither calls the autopath API.
The deployment pipeline itself only tracks convergence for cumulant.

Two questions we cannot currently answer:

1. Is cumulant genuinely harder to converge than jarzynski or force?
2. If jarzynski/force converge sooner *and* predict pKd as well, is cumulant
   needed at all for pKd work?

A third, cross-cutting question: fast pulling speeds may need more replicas but
each replica is cheaper. That trade-off has never been quantified.

## Goals

- One notebook that owns WDR5 convergence, for **cumulant, jarzynski and force**.
- Reproduce the deployment convergence check **verbatim** wherever the API allows.
- Quantify convergence **cost**, not just replica count, on a hardware-independent
  axis plus a real wall-clock axis.
- Report the result honestly, including censoring.

## Non-goals

- No changes to the `autopath` package.
- No new pulling simulations — this is re-analysis of existing trajectories.
- No re-derivation of pKd values; reuse what exists.
- HSP90.

## Key findings from exploration

- `SMDAnalysis.check_convergence` already accepts `estimator_name`, with
  `allowed_estimators = {'cumulant', 'jarzynski'}`. Jarzynski therefore needs no
  new code and is verbatim.
- `force` is rejected by that check **and is structurally incompatible**: the
  routine loops over speeds independently, whereas `ForceEstimator` is
  "inherently multi-speed: a single speed gives no v→0 intercept".
- `extrapolate_to_v0(results, param='dG_weighted', min_speeds=2)` joins on `step`
  across speeds and is exactly the primitive a force ladder needs. The ladder
  reuses it rather than reimplementing the extrapolation.
- `CONVERGENCE_TOLERANCES` is a **local variable inside `AutoPath.run()`**
  (`autopath_core.py:521`), not importable. Its comments state the values must
  match `check_convergence`'s defaults, so those defaults are the usable
  single source of truth.
- All 36 (system, mode) pairs were enumerated and every one has all 3 speeds
  (0.005 / 0.01 / 0.015) with 15–50 replicas each, so a balanced force ladder is
  feasible everywhere. Ladder depth K = min replicas across speeds ranges from
  15 (`6e22_A/lig_com`) to 50.
- Cost is exactly derivable and hardware-independent: simulated time per replica
  is 344 / 173.5 / 118 ps at v = 0.005 / 0.01 / 0.015 — precisely inverse to speed.
- Real wall-clock is recoverable: replica filenames encode a start timestamp
  (`replica-141506` = 14:15:06) and consecutive replicas run back-to-back
  (that replica's mtime is 14:15:34; the next replica is `replica-141534`).
  Observed ≈22–28 s/replica at v=0.015.
- `autopath.log` is **not** a usable timing source: it interleaves duplicate
  lines from concurrent runs spanning May–July and has no per-replica duration.
- Measured cost of one deployment-faithful `check_convergence`: **56 s** for 50
  replicas at one speed (~1.2 s/rung). Full grid ≈ 4 h single-threaded.

## Architecture

Three new artifacts under `scratch/paper_figures/`, plus edits to two notebooks.

| Artifact | Role |
|---|---|
| `wdr5_conv_lib.py` | shared logic: faithful `SMDAnalysis` construction, force ladder, rung comparison, cost extraction |
| `conv_grid_wdr5.py` | SLURM array driver, one task per (system, mode) = 36 tasks |
| `sMD_WDR5_convergence_all.ipynb` | thin notebook: loads cached CSVs, produces all figures |

"Notebook-only" (the approved choice) means **no changes to the `autopath`
package**. The helper module lives beside the notebook because a SLURM array
cannot import from a `.ipynb`.

### Tolerance handling

Introspected, never restated:

```python
sig = inspect.signature(SMDAnalysis.check_convergence)
TOL = {"dG_weighted-rmsd": sig.parameters["tol_rmsd"].default,     # 4.0
       "barrier_delta":    sig.parameters["tol_barrier"].default,  # 3.0
       "r_ts_delta":       sig.parameters["tol_r_ts"].default}     # 0.1
```

### Deployment-faithful analysis construction

Mirrors `autopath_core.run()` and is already validated by the timing probe:

```python
POCKET = ('(resid 218 219 262 263 305 306 49 50 91 92 133 134 175 176) '
          'and backbone')
LIG_HA = "(resname UNK) and not name H*"

cluster_model = DTWPathModel(seed=42, do_plots=False, outdir=outdir)
sa = SMDAnalysis(system, cluster_model,
                 estimators=['cumulant', 'jarzynski', 'force'],
                 do_plots=False, seed=42, temperature=300,
                 ligand_select=LIG_HA, pocket_select=POCKET, outdir=outdir,
                 filter_low_support=True, min_samples_per_step=5,
                 min_support_ratio=1.0, min_replicas_per_path=5,
                 min_path_steps_ratio=0.6, max_frac_neg_dG_first_half=0.25,
                 min_speeds_for_extrapolation=2,
                 reference_pdb=f"{root}/equilibration/{system}_equilibrated.pdb")
```

Route is fixed to **`analysis_traces`** (`geom_features=True,
merge_features=False`). The `dist_*` pocket-distance cache exists only for
v=0.015 and cannot support multi-speed clustering.

### Estimator paths

- **cumulant, jarzynski** — `check_convergence(estimator_name=...)`, per speed. Verbatim.
- **force** — multi-speed ladder:

```
speeds = [0.005, 0.01, 0.015]
K      = min(n_replicas per speed)          # ladder depth

for k in 2..K:
    per speed: first k replicas, sorted by SMDData._replica_idx_from_log
               (the exact ordering check_convergence uses, so rung k contains
                the same replicas the API would have used)
               -> calculate_weighted_pmf
    concat -> extrapolate_to_v0(param='dG_weighted', min_speeds=2) -> PMF_k
    TS/barrier via force_plateau, plateau_frac=0.4
    compare PMF_k vs PMF_{k-1}:
        dG_weighted-rmsd (restricted to boundary + 0.1 buffer)
        barrier_delta, r_ts_delta
    converged_k = all three < TOL
```

Rung *k* costs `k × Σ_speeds cost_per_replica(speed)`: the force estimator is
charged for all three speeds, which is the honest comparison against
single-speed cumulant/jarzynski.

### Convergence rule and sensitivity

`n_to_converge` = first `k` with `k_consec` consecutive passing rungs.
Deployment uses `k_consec = 2`.

The probe showed the rule is unstable at v=0.015 for `6dy7_A/murcko`: rungs
46–47 pass, 48–50 fail again. A single rule would therefore report a
luck-dependent number. **Every result is computed for `k_consec ∈ {2, 3}`**,
with `k_consec = 2` as the headline (matching deployment) and `k_consec = 3`
reported alongside as a robustness check. Where the two disagree materially,
that disagreement is the finding and must be shown, not smoothed away.

## Validation gate

The rung-comparison logic is the only reimplemented piece, so it is tested
against ground truth before any figure is trusted:

> Run the ladder's comparison functions on **single-speed cumulant** and require
> they reproduce `check_convergence`'s cumulant output exactly — per-rung
> `dG_weighted-rmsd`, `barrier_delta`, `r_ts_delta`, and `n_to_converge`.

Reference output already exists from the probe (`6dy7_A`, `murcko`, v=0.015,
cumulant): 47 rungs, first 2-consecutive pass at k=47, with
`(k=46, rmsd 2.078, barrier 2.678, r_ts 0.00062, True)` and
`(k=47, rmsd 1.880, barrier 1.224, r_ts 0.00124, True)`.

If the reimplementation does not match, force results are reported as untrusted
rather than plotted.

## Data flow

Each array task writes its own files (no write races); the notebook concatenates.

- `conv_grid/<system>_<mode>.csv` — `system, mode, estimator, speed` (`ALL` for
  force), `n_replicas, dG_weighted-rmsd, barrier_delta, r_ts_delta,
  barrier_height, r_ts, converged`
- `cost/<system>_<mode>.csv` — `system, mode, speed, md_ps_per_replica,
  wall_sec_median, wall_sec_mad, n_timed, n_dropped`

Derived in the notebook: `n_to_converge` and `censored` per
(system, mode, estimator, speed, k_consec).

## Figures

1. `n_to_converge` heatmap, estimator × speed, median over systems — the hypothesis test.
2. Cost-to-converge on the same grid, in MD-ns and wall-clock minutes — the speed trade-off.
3. Pareto: cost-to-converge vs pKd predictive quality (Spearman ρ of ΔG vs pKd)
   per estimator — answers whether cumulant is needed for pKd.
4. Per-rung metric traces, one representative system, all three estimators.
5. Censoring report: fraction of (system, mode, speed) never converging within
   available replicas.

Figures 1–3 are produced twice, for `k_consec` 2 and 3.

## Honesty rules

- Never-converged → `n_to_converge = NaN` plus a `censored` flag. Medians are
  reported as "≥K" and never silently drop censored cells. The AmpC convergence
  experiment was 94% censored; a median that ignores censoring is misleading.
- A force rung with fewer than 2 valid speeds → NaN and excluded. No fallback.
- Wall-clock outliers removed by MAD fence with `n_dropped` reported. MD-ns and
  wall-clock are cross-checked; disagreement is flagged as GPU heterogeneity
  rather than silently averaged.
- Timing provenance recorded in the CSV: derived from filename start-time and
  mtime, cross-validated by consecutive-start differencing.

## Changes to existing notebooks

- `sMD_WDR5.ipynb` — remove convergence cells 11 and 13; it keeps pKd/Spearman
  correlation and the cross-system ΔG work. Cell 16's combined heatmap currently
  consumes cell 13's `min_pivot`, so it must be repointed at the new cached CSV
  or have that panel dropped.
- `sMD_WDR5_convergence.ipynb` — remove the convergence-metric cells; it keeps
  the DTW/PCA clustering deep dive and the inline work-profile recompute.

## Risks

- **Reimplementation drift** — mitigated by the validation gate; this is the
  main technical risk and blocks the force results if it fails.
- **Ladder depth varies by pair** (K = 15 for `6e22_A/lig_com`, 25 for
  `6dy7_A/murcko`). Cross-pair medians must account for differing depth, and
  shallow pairs will censor more often.
- **Wall-clock is a proxy.** Start-time/mtime differencing assumes replicas run
  back-to-back on one GPU; it is validated on a sample but not guaranteed for
  every pair. MD-ns is the primary axis precisely because of this.
- **Runtime** ≈4 h single-threaded, ~10 min as a 36-task array plus queue wait.
