# sMD autostop — readout-stability convergence criterion

**Date:** 2026-07-31
**Status:** approved, ready for implementation planning
**Scope:** the `autopath` package (pulling loop + convergence check) plus an offline
calibration driver under `scratch/`. WDR5 is the calibration set.

## Problem

The deployed autostop stops adding replicas when the last two convergence
comparisons both pass:

```python
# autopath_core.py:637
CONVERGED = conv_df['converged'].iloc[-2] and conv_df['converged'].iloc[-1]
```

`converged` is computed per rung in `AnalysisSMD.check_convergence:1089` by
comparing PMF(k) against **PMF(k−1)** on three criteria — window RMSD,
barrier-height change, TS-position change — against tolerances 4.0 kJ/mol,
3.0 kJ/mol, 0.1 nm (`autopath_core.py:521`).

This is an *increment* test, not a *plateau* test. Two consecutive small
increments are sufficient, and on a noisy series they occur by chance.

Evidence from the 108 WDR5 (system, mode, speed) runs, 2026-07-31:

- The decision quantity is the cumulant PMF: `autopath_core.py:595` constructs
  `SMDAnalysis(..., estimators=['cumulant'])` and `check_convergence` defaults to
  `estimator_name='cumulant'`, `quantities=['dG_weighted']`.
- 81 runs stopped on the criterion; 25 hit the 50-replica cap, concentrated at
  the fastest speed (17 of 34 runs at 15 m/s).
- The per-rung `converged` flag flickers rather than latching: the fraction of
  passing rungs per run is 0.00–0.32, median ≈0.13. Over the 20–50 rungs a run
  accumulates, an adjacent pair of passes is close to inevitable.
- Replaying the two-consecutive rule over the logged traces, at the firing point
  the PMF is still a median **12.9 kJ/mol** RMSD from its final shape (90th pct
  51, max 102) and the barrier still moves a median **9.6 kJ/mol** afterwards
  (max 149) — 3× the tolerances that were supposed to certify it. 41 of 61 runs
  exceed the RMSD tolerance after stopping; 40 of 61 exceed the barrier tolerance.
  Worst case 6dy7/inertia at 5 m/s fires at k=7 with a 249 kJ/mol barrier that
  ends at 100 kJ/mol.

The cumulant makes this worse by construction. ΔG_cum = ⟨W⟩ − βσ²_W/2, and with
βσ_W ≈ 16–19 on these systems the variance term dominates; a single new replica
carrying an extreme work value moves σ² sharply, so the rung-to-rung increment
series is heavy-tailed rather than decaying.

Two secondary defects, both real but out of scope to fix beyond what falls out
of this design:

- On restart the loop re-runs `check_convergence` over existing replicas and can
  declare convergence and exit without running anything, so a run that stopped on
  the cap can later be relabelled converged (187 `CONVERGED` log messages for
  6dy7 across 18 real runs).
- The live per-speed decision files `sMD_conv_v{speed}_metrics.csv` no longer
  exist on disk (0 found). Only the pooled `vALL` recomputation survives, and it
  does not reproduce the live flags — `check_convergence` rebuilds the DTW path
  model and force-plateau boundary from whatever data it is handed, so pooled
  and per-speed runs are different statistics. Stopping decisions are auditable
  only from `autopath.log`.

## Goals

- Replace the increment test with a plateau test on the quantities actually read
  off the PMF: **TS position and barrier height**, over a window covering the TS
  plus the kinetics margin beyond it.
- Make the estimator behind the decision configurable, and make multi-speed
  (force) autostop possible.
- Reduce to the current behaviour exactly at `conv_window=1, k_consec=2`, so the
  change is auditable.
- Set defaults only from offline calibration on existing trajectories.

## Non-goals

- No new pulling simulations. Calibration is re-analysis of the 108 existing runs.
- No change to how PMFs, path models or boundaries are computed.
- Not fixing the restart re-confirmation behaviour or restoring the deleted
  per-speed CSVs.
- HSP90 and AmpC recalibration.

## Design

### Rule

At rung *k*, over the window **[r_start, r_TS × (1 + boundary_buffer_frac)]** —
the transition state plus the margin where the Kramers absorbing boundary sits,
which is the existing buffer made explicit rather than incidental:

| quantity | reference | tolerance |
|---|---|---|
| PMF RMSD inside the window | running **mean** PMF over the last `w` rungs | `tol_rmsd` (4.0 kJ/mol) |
| barrier height | running **median** over the last `w` rungs | `tol_barrier` (3.0 kJ/mol) |
| r_TS | running **median** over the last `w` rungs | `tol_r_ts` (0.1 nm) |

All three must pass for `k_consec` consecutive rungs.

Median for the scalar readouts because one extreme replica displaces a mean;
mean for the PMF because it is already an average over replicas.

Starting points for calibration: `w = 5`, `k_consec = 3`.

### Insertion points

- `AnalysisSMD.check_convergence` gains `conv_window: int = 5`. The reference for
  rung *k* becomes the running window rather than rung *k−1*
  (`AnalysisSMD.py:1084-1093`). `conv_window=1` reproduces the current arithmetic
  exactly. The emitted row already carries `barrier_height` and `r_ts`
  (`AnalysisSMD.py:1102-1103`), so no new columns are required for the scalars.
- New module `autopath/pulling/Convergence.py` holding the streak test as a pure
  function over `conv_df`:

  ```python
  def first_streak(df: pd.DataFrame, k_consec: int = 3) -> float:
      """First n_replicas where `converged` holds for k_consec consecutive rungs, else NaN."""
  ```

  `autopath_core.py:637` calls it instead of indexing `iloc[-2]`/`iloc[-1]`. The
  same function backs the offline calibration, so pipeline and analysis cannot
  drift apart.

### Configuration

Four new knobs in `config.py`, plumbed through `AutoPath`. The first two are the
behavioural switches; the last two exist so a calibration outcome can be deployed
without editing library defaults:

- `sMD_autostop_estimator: str = "cumulant"` — one of `{cumulant, jarzynski, force}`.
  Stays `cumulant` for backward compatibility; the default changes only if
  calibration justifies it, in a separate decision.
- `sMD_alternate_speeds: bool = False` — round-robin one replica per speed
  instead of finishing each speed before starting the next.
- `sMD_conv_window: int = 5` — `w`, forwarded to `check_convergence(conv_window=...)`.
- `sMD_conv_streak: int = 3` — `k_consec`, forwarded to `first_streak(k_consec=...)`.

Compatibility is validated at config construction and **raises**; it does not
silently repair an illegal combination:

| estimator | `alternate_speeds=False` | `alternate_speeds=True` |
|---|---|---|
| cumulant | legal (per speed) | legal (per speed) |
| jarzynski | legal (per speed) | legal (per speed) |
| force | **ValueError** | legal (joint v→0 ladder) |

`force` additionally requires ≥2 entries in `sMD_pulling_speeds`, because
`ForceEstimator` has no single-speed v→0 intercept.

`check_convergence` currently rejects anything outside `{cumulant, jarzynski}`
(`AnalysisSMD.py:796-799`). The force path is the multi-speed ladder already
written and validated in `scratch/paper_figures/wdr5_conv_lib.py:conv_force_ladder`
— that code is promoted into the package rather than reimplemented, keeping the
deviation it documents (RMSD r-cap from the slowest speed; barrier/r_TS via
`pmf_peak` because the extrapolated v→0 PMF has no force profile).

### Run loop

`autopath_core.py:574-667`.

- `alternate_speeds=False` — unchanged: `for speed: while not CONVERGED`.
- `alternate_speeds=True` — inverted: `while any speed live: for each live speed:
  run 1 replica; check`. Per-speed estimators retire each speed independently;
  `force` retires all speeds together once the v→0 ladder passes.

The `sMD_max_replicas` cap and the `MAX_CONSECUTIVE_FAILURES` guard apply per
speed in both modes.

## Calibration

No new simulations. A driver replays candidate settings over the 108 existing
WDR5 runs, reusing `wdr5_conv_lib`, and reports per setting:

- **gate** — fraction of runs where, after the stop point, neither readout moves
  beyond its tolerance again relative to its value at the final available rung:
  `|barrier(k_stop) − barrier(k_final)| < tol_barrier` and
  `|r_TS(k_stop) − r_TS(k_final)| < tol_r_ts`, plus window RMSD between PMF(k_stop)
  and PMF(k_final) below `tol_rmsd`. A setting is acceptable at ≥90%. Runs that
  never converge under a setting are censored, counted, and reported separately —
  never dropped from the denominator.
- **cost** — replicas and MD-ns implied, against today's stop points.
- **check** — ΔG-vs-pKd Spearman ρ at the stop point versus using all replicas.
  Reported as a sanity check, **not** tuned on: the ranking is known to be
  insensitive and would license stopping far too early.

Defaults ship only from a setting that passes the gate. If none passes at
acceptable cost, that is the finding and is reported rather than smoothed away.

## Tests

- **Backward-compatibility gate.** `conv_window=1, k_consec=2` reproduces the
  existing fixture exactly (6dy7_A/murcko/v0.015/cumulant): `k=46 → rmsd 2.077676,
  barrier_delta 2.677670, r_ts_delta 0.000619, True`; `k=47 → rmsd 1.880280,
  barrier_delta 1.223554, r_ts_delta 0.001237, True`; `k=48 → False`.
- **`first_streak` units.** A flicker series (alternating pass/fail) must not
  fire at `k_consec=3`; a genuine plateau must fire at the first rung completing
  the streak; a series shorter than `k_consec` returns NaN.
- **Window units.** `conv_window > 1` on a synthetic drifting series must fail
  where the pairwise test passes.
- **Config units.** `force` + `alternate_speeds=False` raises; `force` with one
  speed raises; both legal per-speed combinations construct.

## Risks

- **Behaviour change is the point, so regressions are hard to distinguish from
  the fix.** Mitigated by the `conv_window=1, k_consec=2` equivalence gate: any
  difference from today must be attributable to a deliberate setting change.
- **Cost.** A plateau test necessarily stops later than an increment test. If the
  gate can only be met at a cost the queue cannot absorb, the honest outcome is a
  reported trade-off curve, not a quietly loosened tolerance.
- **Interleaved speeds change the failure surface.** A crash now leaves all
  speeds partially filled rather than some complete. The per-speed cap and
  failure guard limit the blast radius, but restart behaviour under
  `alternate_speeds=True` needs an explicit test.
- **Promoting the force ladder** moves code that currently has no package-level
  test coverage. It arrives with its existing validation gate against
  single-speed cumulant.
- **Calibration set is WDR5 only.** Defaults derived here may not transfer to
  HSP90/AmpC; that transfer is a follow-up, not an assumption.
