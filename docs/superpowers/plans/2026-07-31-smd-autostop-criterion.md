# sMD Autostop — Readout-Stability Criterion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-consecutive-increment sMD autostop with a plateau test on the readouts actually used downstream (TS position and barrier height), make the decision estimator configurable, and set the defaults from an offline replay over existing trajectories.

**Architecture:** A new `autopath/pulling/Convergence.py` holds the pure decision logic (streak test, window references, option validation) so it is unit-testable without simulations. `AnalysisSMD.check_convergence` gains `conv_window`, changing the per-rung reference from "the previous rung" to "the mean/median over the last `w` rungs"; `conv_window=1` reproduces today's arithmetic bit-for-bit and is the audit hinge. The pulling loop in `autopath_core.py` calls `first_streak` instead of indexing the last two rows, and gains a round-robin mode so the multi-speed force estimator can drive the decision.

**Tech Stack:** Python 3.12, `autopath` package, pandas/numpy, pytest 9.1.1.

## Global Constraints

- **Env:** all Python runs use `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python`. The `cosolvkit` env lacks `autopath`.
- **Package tests live in `tests/`** at the repo root (`pytest.ini` is there). Scratch tests stay in `scratch/paper_figures/`.
- **`scratch/*` is gitignored** (`.gitignore:1`). Commit steps apply to `autopath/`, `tests/` and `docs/` only. Never `git add -f` scratch files.
- **Commit messages are short and imperative, with no `Co-Authored-By` trailer.**
- **Backward-compatibility invariant:** `conv_window=1` + `k_consec=2` must reproduce current behaviour exactly. Any task that breaks this fixture is wrong:
  6dy7_A / murcko / v=0.015 / cumulant → 47 rows; `k=46 → rmsd 2.077676, barrier_delta 2.677670, r_ts_delta 0.000619, converged True`; `k=47 → rmsd 1.880280, barrier_delta 1.223554, r_ts_delta 0.001237, converged True`; `k=48 → converged False`.
- **Calibration data root:** `/gpfs/group/forli/mllanos/autopath_benchmarking/WDR5/openFF`; 6 systems × 6 modes × 3 speeds = 108 runs; cached per-rung grid already exists at `scratch/paper_figures/results_conv/conv_grid/` (36 CSVs, 7569 rows, estimators cumulant/jarzynski/force).
- **Validation gate status:** already run green on 2026-07-31 — `max_abs_rmsd_diff = 0.0`, `converged_mismatch_rows = 0`, `n_to_converge_match = True`. The scratch suite (`scratch/paper_figures/test_wdr5_conv_lib.py`, 12 tests, ~286 s) must stay green after Task 6.
- **Defaults that ship in this plan:** `sMD_autostop_estimator="cumulant"`, `sMD_alternate_speeds=False`, `sMD_conv_window=5`, `sMD_conv_streak=3`. Only Task 8 may change them, and only on calibration evidence.
- **Censoring is never hidden.** A setting that never fires is reported as censored with a count, never dropped from a denominator.

## File Structure

| File | Responsibility |
|---|---|
| `autopath/pulling/Convergence.py` | **new** — pure decision logic: `first_streak`, `window_mean_pmf`, `window_reference_scalar`, `validate_autostop_options` |
| `autopath/pulling/ConvergenceLadder.py` | **new** (Task 6) — generic multi-speed force ladder, promoted from `scratch/paper_figures/wdr5_conv_lib.py` |
| `autopath/pulling/AnalysisSMD.py` | modify — `check_convergence` gains `conv_window`; window reference replaces the previous-rung reference; `force` accepted |
| `autopath/autopath_core.py` | modify — call `first_streak`; new knobs; round-robin speed mode |
| `autopath/config.py` | modify — accept the four new keys so JSON configs validate |
| `tests/test_convergence_rule.py` | **new** — unit tests for `Convergence.py` |
| `tests/test_conv_window.py` | **new** — window behaviour + the backward-compatibility fixture |
| `tests/test_autostop_options.py` | **new** — option compatibility matrix |
| `scratch/paper_figures/calibrate_autostop.py` | **new** — offline replay driver over the 108 runs |

---

### Task 1: Pure decision logic module

**Files:**
- Create: `autopath/pulling/Convergence.py`
- Test: `tests/test_convergence_rule.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `first_streak(df: pd.DataFrame, k_consec: int = 3) -> float` — smallest `n_replicas` ending a run of `k_consec` consecutive `True` rungs, else `float("nan")`.
  - `window_mean_pmf(pmfs: Sequence[pd.Series]) -> pd.Series` — elementwise mean over the shared index.
  - `window_reference_scalar(values: Sequence[float]) -> float` — median of finite values, `nan` if none are finite.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_convergence_rule.py`:

```python
import numpy as np
import pandas as pd
import pytest

from autopath.pulling.Convergence import (
    first_streak,
    window_mean_pmf,
    window_reference_scalar,
)


def test_first_streak_needs_consecutive_passes():
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6, 7],
                       "converged": [False, True, False, True, True]})
    assert first_streak(df, k_consec=2) == 7.0


def test_first_streak_k3_rejects_a_flickering_series():
    """The failure mode this whole change exists to fix."""
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6, 7, 8],
                       "converged": [True, True, False, True, True, False]})
    assert first_streak(df, k_consec=2) == 4.0
    assert np.isnan(first_streak(df, k_consec=3))


def test_first_streak_fires_on_a_genuine_plateau():
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6, 7],
                       "converged": [False, False, True, True, True]})
    assert first_streak(df, k_consec=3) == 7.0


def test_first_streak_is_nan_when_shorter_than_the_streak():
    df = pd.DataFrame({"n_replicas": [3, 4], "converged": [True, True]})
    assert np.isnan(first_streak(df, k_consec=3))


def test_first_streak_sorts_by_replica_count():
    df = pd.DataFrame({"n_replicas": [7, 3, 4], "converged": [True, True, True]})
    assert first_streak(df, k_consec=3) == 7.0


def test_window_mean_pmf_of_one_series_is_that_series():
    s = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    pd.testing.assert_series_equal(window_mean_pmf([s]), s)


def test_window_mean_pmf_averages_on_the_shared_index():
    a = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    b = pd.Series([3.0, 4.0], index=[11, 12])
    out = window_mean_pmf([a, b])
    assert list(out.index) == [11, 12]
    assert out.tolist() == [3.0, 3.5]


def test_window_reference_scalar_single_value_is_itself():
    assert window_reference_scalar([2.5]) == 2.5


def test_window_reference_scalar_single_nan_is_nan():
    assert np.isnan(window_reference_scalar([float("nan")]))


def test_window_reference_scalar_ignores_nans_among_finites():
    assert window_reference_scalar([1.0, float("nan"), 3.0]) == 2.0


def test_window_reference_scalar_all_nan_is_nan():
    assert np.isnan(window_reference_scalar([float("nan"), float("nan")]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_convergence_rule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autopath.pulling.Convergence'`

- [ ] **Step 3: Write the implementation**

Create `autopath/pulling/Convergence.py`:

```python
"""Pure decision logic for sMD convergence.

Separated from AnalysisSMD so the stopping rule can be unit-tested without
building PMFs, and so the deployment loop and offline calibration share one
implementation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def first_streak(df: pd.DataFrame, k_consec: int = 3) -> float:
    """Smallest ``n_replicas`` ending a run of ``k_consec`` consecutive passes.

    Returns NaN when the criterion is never met. That is a *censored*
    observation: callers must report it, never silently drop it.
    """
    d = df.sort_values("n_replicas")
    flags = d["converged"].astype(bool).to_numpy()
    ks = d["n_replicas"].to_numpy()
    run = 0
    for flag, k in zip(flags, ks):
        run = run + 1 if flag else 0
        if run >= k_consec:
            return float(k)
    return float("nan")


def window_mean_pmf(pmfs: Sequence[pd.Series]) -> pd.Series:
    """Elementwise mean of PMF series over the index they share.

    A single-element window returns that series unchanged, which is what makes
    ``conv_window=1`` identical to comparing against the previous rung.
    """
    pmfs = list(pmfs)
    if len(pmfs) == 1:
        return pmfs[0]
    frame = pd.concat(pmfs, axis=1, join="inner")
    return frame.mean(axis=1)


def window_reference_scalar(values: Sequence[float]) -> float:
    """Median of the finite values in the window; NaN when none are finite.

    Median rather than mean because one replica carrying an extreme work value
    displaces a mean. NaN entries mean "no peak detected at that rung" and are
    ignored rather than poisoning the reference; a window of only NaNs yields
    NaN, which the caller treats as "criterion waived".
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_convergence_rule.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add autopath/pulling/Convergence.py tests/test_convergence_rule.py
git commit -m "add pure convergence decision logic"
```

---

### Task 2: Window reference in check_convergence

**Files:**
- Modify: `autopath/pulling/AnalysisSMD.py` — signature near line 737-764; loop body lines 985-1112
- Test: `tests/test_conv_window.py`

**Interfaces:**
- Consumes: `Convergence.window_mean_pmf`, `Convergence.window_reference_scalar` from Task 1.
- Produces: `SMDAnalysis.check_convergence(..., conv_window: int = 5)`. Emitted columns are unchanged: `speed, path, n_replicas, {quantity}-rmsd, barrier_delta, r_ts_delta, barrier_height, r_ts, converged, decision_quantity, n_common_points`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_conv_window.py`:

```python
"""The window reference, and the fixture proving conv_window=1 is a no-op."""
import numpy as np
import pytest

from autopath.pulling.Convergence import window_mean_pmf, window_reference_scalar


def test_window_lags_a_drifting_series_more_than_a_pairwise_reference():
    """Why the window fixes the bug: on a steady drift the pairwise step looks
    small while the distance to the window average is w/2 times larger."""
    import pandas as pd
    drift = 1.0
    rungs = [pd.Series([k * drift, k * drift], index=[10, 11]) for k in range(6)]
    pairwise_gap = abs(rungs[5].iloc[0] - rungs[4].iloc[0])
    window_gap = abs(rungs[5].iloc[0] - window_mean_pmf(rungs[0:5]).iloc[0])
    assert pairwise_gap == pytest.approx(1.0)
    assert window_gap == pytest.approx(3.0)
    assert window_gap > pairwise_gap


def test_window_of_one_reproduces_the_pairwise_reference():
    import pandas as pd
    prev = pd.Series([5.0, 6.0], index=[10, 11])
    assert window_mean_pmf([prev]).equals(prev)
    assert window_reference_scalar([7.0]) == 7.0


@pytest.mark.slow
def test_conv_window_1_reproduces_the_deployment_fixture(tmp_path):
    """Backward-compatibility gate. Real data; ~60 s."""
    import sys
    sys.path.insert(0, "scratch/paper_figures")
    import wdr5_conv_lib as lib

    df = lib.conv_api("6dy7_A", "murcko", "cumulant",
                      outdir=str(tmp_path), speeds=[0.015], conv_window=1)
    assert len(df) == 47
    r46 = df[df["n_replicas"] == 46].iloc[0]
    r47 = df[df["n_replicas"] == 47].iloc[0]
    r48 = df[df["n_replicas"] == 48].iloc[0]
    assert abs(r46["dG_weighted-rmsd"] - 2.077676) < 1e-4
    assert abs(r46["barrier_delta"] - 2.677670) < 1e-4
    assert abs(r46["r_ts_delta"] - 0.000619) < 1e-4
    assert bool(r46["converged"]) is True
    assert abs(r47["dG_weighted-rmsd"] - 1.880280) < 1e-4
    assert bool(r47["converged"]) is True
    assert bool(r48["converged"]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_conv_window.py -v -m "not slow"`
Expected: PASS for the two pure tests (they only use Task 1 code).
Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_conv_window.py -v -m slow`
Expected: FAIL — `conv_api() got an unexpected keyword argument 'conv_window'`

- [ ] **Step 3: Add `conv_window` to the signature**

In `autopath/pulling/AnalysisSMD.py`, add to the `check_convergence` keyword list (next to `min_common_points`, around line 755):

```python
        conv_window: int = 5,      # rungs averaged to form the comparison reference; 1 == previous-rung
```

And add the import at the top of the file, next to the existing `from .Estimators import ...`:

```python
from .Convergence import window_mean_pmf, window_reference_scalar
```

- [ ] **Step 4: Replace the previous-rung state with a window**

At the top of the per-speed loop, where `prev_pmf`, `prev_barrier_height` and `prev_r_ts` are initialised, replace them with bounded histories:

```python
        from collections import deque
        _w = max(1, int(conv_window))
        hist_pmf = deque(maxlen=_w)
        hist_barrier = deque(maxlen=_w)
        hist_r_ts = deque(maxlen=_w)
```

Replace the Stage-3 initialisation block (`AnalysisSMD.py:986-993`) with:

```python
                if not hist_pmf:
                    b0, r0 = self._compute_barrier_rts(
                        pmf_k, 1.0/smd.beta, protocol_grid,
                        force_df=force_k, speed=speed,
                        boundary_method=boundary_method, plateau_frac=plateau_frac,
                    )
                    hist_pmf.append(pmf_k)
                    hist_barrier.append(b0)
                    hist_r_ts.append(r0)
                    continue
```

Replace the reference used for the comparison (`AnalysisSMD.py:1006`, and the
`prev_barrier_height` / `prev_r_ts` uses at 1018-1030):

```python
                ref_pmf = window_mean_pmf(hist_pmf)
                ref_barrier = window_reference_scalar(hist_barrier)
                ref_r_ts = window_reference_scalar(hist_r_ts)

                common_r = pmf_k.index.intersection(ref_pmf.index)
```

```python
                if np.isnan(barrier_height) and np.isnan(ref_barrier):
                    barrier_delta = np.nan
                elif np.isnan(barrier_height) or np.isnan(ref_barrier):
                    barrier_delta = np.inf
                else:
                    barrier_delta = abs(barrier_height - ref_barrier)

                if np.isnan(r_ts) and np.isnan(ref_r_ts):
                    r_ts_delta = np.nan
                elif np.isnan(r_ts) or np.isnan(ref_r_ts):
                    r_ts_delta = np.inf
                else:
                    r_ts_delta = abs(r_ts - ref_r_ts)
```

Replace `yNm1 = prev_pmf.loc[common_r].values` (line 1085) with:

```python
                yNm1 = ref_pmf.loc[common_r].values
```

Every one of the three `prev_pmf = pmf_k; prev_barrier_height = ...; prev_r_ts = ...`
state-update blocks (the two early-`continue` branches at lines 1046-1048 and
1079-1081, and the tail at 1110-1112) becomes:

```python
                hist_pmf.append(pmf_k)
                hist_barrier.append(barrier_height)
                hist_r_ts.append(r_ts)
```

- [ ] **Step 5: Thread `conv_window` through the scratch wrapper**

In `scratch/paper_figures/wdr5_conv_lib.py`, add `conv_window: int = 1` to `conv_api`'s signature and forward it to `check_convergence(..., conv_window=conv_window)`. Default 1 there keeps every existing cached result and the 12-test scratch suite reproducible.

- [ ] **Step 6: Run tests to verify they pass**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_conv_window.py -v`
Expected: 3 passed (the slow one takes ~60 s)

- [ ] **Step 7: Run the existing package suite for regressions**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/ -v -m "not slow"`
Expected: no new failures versus the pre-change baseline. Record the baseline first if unknown.

- [ ] **Step 8: Commit**

```bash
git add autopath/pulling/AnalysisSMD.py tests/test_conv_window.py
git commit -m "add conv_window reference to check_convergence"
```

---

### Task 3: Wire the streak test into the pulling loop

**Files:**
- Modify: `autopath/autopath_core.py:635-642` (decision), `:185-260` (knobs)
- Modify: `autopath/config.py:46-60` (accept the keys)

**Interfaces:**
- Consumes: `Convergence.first_streak` from Task 1; `conv_window` from Task 2.
- Produces: `AutoPath(..., sMD_conv_window: int = 5, sMD_conv_streak: int = 3)`.

- [ ] **Step 1: Add the knobs**

In `autopath/autopath_core.py`, add to `AutoPath.__init__`'s keyword list next to `sMD_converge_speeds` (line 191):

```python
        sMD_conv_window: int = 5,
        sMD_conv_streak: int = 3,
```

and next to `self.sMD_converge_speeds = sMD_converge_speeds` (line 255):

```python
        self.sMD_conv_window = sMD_conv_window
        self.sMD_conv_streak = sMD_conv_streak
```

In `autopath/config.py`, add the same two keys with the same defaults to `Config.__init__`'s signature (near line 59) and to the `self.` assignments (near line 298). `Config.from_config` rejects unknown keys (`config.py:417-423`), so a JSON config naming them fails without this.

- [ ] **Step 2: Use the window and the streak**

Add the import at the top of `autopath/autopath_core.py`:

```python
from autopath.pulling.Convergence import first_streak
```

Forward the window in the `check_convergence` call (`autopath_core.py:603-611`), adding one argument:

```python
                                conv_window=self.sMD_conv_window,
```

Replace the decision block (`autopath_core.py:635-642`):

```python
                                # Convergence = the readouts hold a plateau for
                                # sMD_conv_streak consecutive rungs. Two adjacent
                                # passes are not evidence: on WDR5 the per-rung flag
                                # flickers with a pass rate around 0.13, so a pair
                                # arises by chance.
                                if len(conv_df) >= self.sMD_conv_streak:
                                    n_conv = first_streak(conv_df, k_consec=self.sMD_conv_streak)
                                    CONVERGED = bool(np.isfinite(n_conv))
                                else:
                                    logger.info(
                                        f"Only {len(conv_df)} convergence comparisons available for "
                                        f"speed {speed} nm/ps; need {self.sMD_conv_streak}."
                                    )
                                if CONVERGED:
                                    logger.warning(
                                        f"sMD pulling for speed {speed} nm/ps CONVERGED after "
                                        f"{current_replica} replicas (streak of {self.sMD_conv_streak} "
                                        f"at rung {n_conv:.0f}, window {self.sMD_conv_window})."
                                    )
                                    continue
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_autostop_options.py` with the first case:

```python
import numpy as np
import pandas as pd

from autopath.pulling.Convergence import first_streak


def test_deployment_decision_is_a_streak_not_a_pair():
    """The exact series that stopped 6dy7/contacts at v=0.01 must not stop at k=3."""
    df = pd.DataFrame({"n_replicas": [12, 13, 14, 15, 16],
                       "converged": [True, True, False, False, True]})
    assert first_streak(df, k_consec=2) == 13.0
    assert np.isnan(first_streak(df, k_consec=3))
```

- [ ] **Step 4: Run it**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_autostop_options.py -v`
Expected: PASS

- [ ] **Step 5: Verify the module still imports and the knobs exist**

Run:
```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -c "
import inspect
from autopath.autopath_core import AutoPath
from autopath.config import Config
for cls in (AutoPath, Config):
    p = inspect.signature(cls.__init__).parameters
    assert p['sMD_conv_window'].default == 5, cls
    assert p['sMD_conv_streak'].default == 3, cls
print('knobs present with correct defaults')
"
```
Expected: `knobs present with correct defaults`

- [ ] **Step 6: Commit**

```bash
git add autopath/autopath_core.py autopath/config.py tests/test_autostop_options.py
git commit -m "use streak test for sMD autostop decision"
```

---

### Task 4: Estimator and speed-alternation options

**Files:**
- Modify: `autopath/pulling/Convergence.py` (add validator)
- Modify: `autopath/autopath_core.py` (knobs + call the validator), `autopath/config.py` (accept keys)
- Test: `tests/test_autostop_options.py`

**Interfaces:**
- Produces: `validate_autostop_options(estimator: str, alternate_speeds: bool, speeds: Sequence[float]) -> None` — raises `ValueError` on an illegal combination, returns `None` otherwise. `AutoPath(..., sMD_autostop_estimator: str = "cumulant", sMD_alternate_speeds: bool = False)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_autostop_options.py`:

```python
import pytest

from autopath.pulling.Convergence import validate_autostop_options


def test_per_speed_estimators_are_legal_either_way():
    for est in ("cumulant", "jarzynski"):
        for alt in (True, False):
            validate_autostop_options(est, alt, [0.005, 0.01, 0.015])


def test_force_requires_alternating_speeds():
    with pytest.raises(ValueError, match="alternate_speeds"):
        validate_autostop_options("force", False, [0.005, 0.01, 0.015])


def test_force_requires_at_least_two_speeds():
    with pytest.raises(ValueError, match="two speeds"):
        validate_autostop_options("force", True, [0.005])


def test_force_is_legal_with_alternation_and_two_speeds():
    validate_autostop_options("force", True, [0.005, 0.01])


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="Unknown autostop estimator"):
        validate_autostop_options("mbar", False, [0.005, 0.01])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_autostop_options.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate_autostop_options'`

- [ ] **Step 3: Implement the validator**

Append to `autopath/pulling/Convergence.py`:

```python
AUTOSTOP_ESTIMATORS = ("cumulant", "jarzynski", "force")


def validate_autostop_options(estimator: str, alternate_speeds: bool,
                              speeds: Sequence[float]) -> None:
    """Reject illegal autostop option combinations. Raises, never repairs.

    ``force`` has no single-speed v->0 intercept, so it needs at least two
    speeds and needs them advancing together — which only happens when speeds
    are alternated rather than run to completion one at a time.
    """
    if estimator not in AUTOSTOP_ESTIMATORS:
        raise ValueError(
            f"Unknown autostop estimator '{estimator}'. "
            f"Allowed: {sorted(AUTOSTOP_ESTIMATORS)}"
        )
    if estimator == "force":
        if not alternate_speeds:
            raise ValueError(
                "sMD_autostop_estimator='force' requires sMD_alternate_speeds=True: "
                "the force estimator needs >=2 speeds advancing together to form a "
                "v->0 intercept, which the one-speed-at-a-time loop cannot provide."
            )
        if len(speeds) < 2:
            raise ValueError(
                "sMD_autostop_estimator='force' requires at least two speeds in "
                f"sMD_pulling_speeds; got {len(speeds)}."
            )
```

- [ ] **Step 4: Add the knobs and call the validator**

In `autopath/autopath_core.py`, add to `AutoPath.__init__`'s keyword list:

```python
        sMD_autostop_estimator: str = "cumulant",
        sMD_alternate_speeds: bool = False,
```

and in the body, after `self.sMD_pulling_speeds` is assigned:

```python
        self.sMD_autostop_estimator = sMD_autostop_estimator
        self.sMD_alternate_speeds = sMD_alternate_speeds
        validate_autostop_options(sMD_autostop_estimator, sMD_alternate_speeds,
                                  list(sMD_pulling_speeds.keys()))
```

Extend the import from Task 3:

```python
from autopath.pulling.Convergence import first_streak, validate_autostop_options
```

Pass the estimator through to the check (`autopath_core.py:603-611`):

```python
                                estimator_name=self.sMD_autostop_estimator,
```

and to the analysis construction two lines above (`autopath_core.py:595`):

```python
                            smdanalysis = SMDAnalysis(sysname=sys_name, path_model='dtw',
                                                    estimators=[self.sMD_autostop_estimator],
```

Mirror both keys with the same defaults in `autopath/config.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_autostop_options.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add autopath/pulling/Convergence.py autopath/autopath_core.py autopath/config.py tests/test_autostop_options.py
git commit -m "add autostop estimator and speed alternation options"
```

---

### Task 5: Offline calibration driver

**Files:**
- Create: `scratch/paper_figures/calibrate_autostop.py`
- Reads: `scratch/paper_figures/results_conv/conv_grid/*.csv` (36 files, already cached)

**Interfaces:**
- Consumes: `Convergence.first_streak`; the cached per-rung grid columns `system, mode, estimator, speed, n_replicas, dG_weighted-rmsd, barrier_delta, r_ts_delta, barrier_height, r_ts, converged`.
- Produces: `scratch/paper_figures/results_conv/calibration.csv` with one row per `(estimator, k_consec)` and columns `n_cells, n_censored, median_n, p90_n, gate_pass_frac, median_extra_replicas`.

This task calibrates the **streak** dimension only, from the cached `conv_window=1` grid. The window dimension needs the grid recomputed at each `w`, which is a 4 h job; Task 8 does that once the streak sweep has narrowed the candidates.

- [ ] **Step 1: Write the driver**

Create `scratch/paper_figures/calibrate_autostop.py`:

```python
"""Replay candidate autostop settings over the cached WDR5 convergence grid.

Gate: after the stop point, neither readout may move beyond tolerance again
relative to the final available rung. Censored cells are counted, never dropped.
"""
import glob
import os

import numpy as np
import pandas as pd

from autopath.pulling.Convergence import first_streak

GRID = "results_conv/conv_grid/*.csv"
OUT = "results_conv/calibration.csv"
TOL_BARRIER, TOL_RTS = 3.0, 0.1


def load_grid() -> pd.DataFrame:
    files = sorted(glob.glob(GRID))
    if not files:
        raise SystemExit(f"no cached grid at {GRID}; run conv_grid_wdr5.py first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def gate(run: pd.DataFrame, k_stop: float) -> bool:
    """True when the readouts at k_stop already match the final rung."""
    run = run.sort_values("n_replicas")
    at = run[run["n_replicas"] == k_stop]
    if at.empty:
        return False
    final = run.iloc[-1]
    at = at.iloc[0]
    for col, tol in (("barrier_height", TOL_BARRIER), ("r_ts", TOL_RTS)):
        a, b = at[col], final[col]
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) or np.isnan(b) or abs(a - b) >= tol:
            return False
    return True


def main() -> None:
    d = load_grid()
    rows = []
    for k_consec in (2, 3, 4, 5):
        for est, g in d.groupby("estimator"):
            stops, gates, extra = [], [], []
            for _, run in g.groupby(["system", "mode", "speed"]):
                run = run.sort_values("n_replicas")
                k = first_streak(run, k_consec=k_consec)
                stops.append(k)
                if np.isfinite(k):
                    gates.append(gate(run, k))
                    extra.append(k - first_streak(run, k_consec=2))
            stops = np.asarray(stops, float)
            rows.append({
                "estimator": est,
                "k_consec": k_consec,
                "n_cells": len(stops),
                "n_censored": int(np.isnan(stops).sum()),
                "median_n": float(np.nanmedian(stops)) if np.isfinite(stops).any() else np.nan,
                "p90_n": float(np.nanpercentile(stops[np.isfinite(stops)], 90)) if np.isfinite(stops).any() else np.nan,
                "gate_pass_frac": float(np.mean(gates)) if gates else np.nan,
                "median_extra_replicas": float(np.nanmedian(extra)) if extra else np.nan,
            })
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nwritten: {OUT}")
    print("gate target: >=0.90; censored cells are excluded from gate_pass_frac "
          "and reported separately in n_censored")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd scratch/paper_figures && /gpfs/home/mllanos/micromamba/envs/autopath/bin/python calibrate_autostop.py
```
Expected: a table with 12 rows (3 estimators × 4 streak lengths) and `results_conv/calibration.csv` written.

- [ ] **Step 3: Add the pKd ranking check**

The spec requires the downstream ranking to be *reported* alongside the gate, and
never tuned on. The cached grid already carries the readout, so no PMF rebuild is
needed: `barrier_height` at a given rung is TS(ΔG) for that run.

Append to `calibrate_autostop.py`:

```python
from scipy.stats import spearmanr

SYS_PKD = {"6e1z": 3.82, "6dy7": 4.18, "6dya": 6.17,
           "6e1y": 7.66, "6e22": 8.89, "6e23": 10.00}


def ranking_check(d: pd.DataFrame, k_consec: int) -> pd.DataFrame:
    """Spearman rho of TS(dG) vs pKd, at the stop point vs at the final rung.

    Reported only. Tuning on this would license stopping far too early, because
    the ranking is known to be insensitive to PMF drift.
    """
    rows = []
    for (est, mode, speed), g in d.groupby(["estimator", "mode", "speed"]):
        at_stop, at_final, pkd = [], [], []
        for system, run in g.groupby("system"):
            run = run.sort_values("n_replicas")
            k = first_streak(run, k_consec=k_consec)
            if not np.isfinite(k):
                continue
            at_stop.append(float(run[run["n_replicas"] == k]["barrier_height"].iloc[0]))
            at_final.append(float(run["barrier_height"].iloc[-1]))
            pkd.append(SYS_PKD[system.removesuffix("_A")])
        if len(pkd) < 3:
            continue
        rows.append({"estimator": est, "mode": mode, "speed": speed, "k_consec": k_consec,
                     "n_systems": len(pkd),
                     "rho_at_stop": spearmanr(at_stop, pkd)[0],
                     "rho_at_final": spearmanr(at_final, pkd)[0]})
    return pd.DataFrame(rows)
```

and in `main()`, after writing `calibration.csv`:

```python
    rank = pd.concat([ranking_check(d, kc) for kc in (2, 3, 4, 5)], ignore_index=True)
    rank["rho_loss"] = rank["rho_at_final"] - rank["rho_at_stop"]
    rank.to_csv("results_conv/calibration_ranking.csv", index=False)
    print("\npKd ranking check (reported, not tuned on):")
    print(rank.groupby(["estimator", "k_consec"])[["rho_at_stop", "rho_at_final", "rho_loss"]]
              .median().round(3).to_string())
```

- [ ] **Step 4: Record the result in the plan**

Paste both printed tables into this file under Task 8 as "Calibration evidence — streak sweep". This is the input to the defaults decision and must be written down, not remembered.

- [ ] **Step 5: Commit**

Nothing to commit — `scratch/*` is gitignored. Confirm with `git status --short scratch/` that nothing is staged.

---

### Task 6: Promote the force ladder into the package

**Files:**
- Create: `autopath/pulling/ConvergenceLadder.py`
- Modify: `autopath/pulling/AnalysisSMD.py:796-799` (accept `force`)
- Modify: `scratch/paper_figures/wdr5_conv_lib.py` (delegate)

**Interfaces:**
- Consumes: `Estimators.extrapolate_to_v0`, `KramersEstimator.force_plateau_boundary`, `SupportPolicy` — all already imported by `wdr5_conv_lib`.
- Produces: `force_convergence_ladder(sa, logs, speeds, trace_min_replicas=3, trim_fraction=0.1, min_common_points=5) -> pd.DataFrame` with the same columns `check_convergence` emits, and `speed` set to the string `"ALL"`.

- [ ] **Step 1: Establish the regression baseline**

Run the scratch suite and save the output — it is the ground truth the promotion must not disturb:

```bash
cd scratch/paper_figures && /gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -v -p no:cacheprovider 2>&1 | tail -20
```
Expected: 12 passed (~286 s). If it is not green, stop — do not promote code whose gate is failing.

- [ ] **Step 2: Move the generic core**

Create `autopath/pulling/ConvergenceLadder.py` containing the bodies of
`wdr5_conv_lib._speed_state`, `_support_trim_common`, `_compare_rung` and
`conv_force_ladder`, with every WDR5-specific default removed: the new
`force_convergence_ladder` takes an already-built `SMDAnalysis` (`sa`), an
explicit `logs` list and an explicit `speeds` list, and calls
`get_tolerances()`-equivalent introspection on `SMDAnalysis.check_convergence`
rather than importing the scratch helper.

Keep verbatim the two documented deviations, and keep their comments:
- RMSD r-cap from the **slowest** speed's force-plateau boundary, because the extrapolated v→0 PMF has no force profile;
- barrier/r_TS via `boundary_method="pmf_peak"`: when `_find_pmf_peak` finds no peak above the prominence threshold, `_compute_barrier_rts` falls back to `argmax(ΔG)` — not NaN, and not an end-of-range `r_max` fallback — returning that position as `r_ts` and `dG[argmax] - dG[0]` as the barrier; for a monotonic profile with no interior peak this makes `r_ts` the profile endpoint and the barrier the total rise.

- [ ] **Step 3: Make the scratch lib delegate**

Replace `wdr5_conv_lib.conv_force_ladder`'s body with a call into the package,
keeping its WDR5 defaults as the only thing it still owns:

```python
def conv_force_ladder(system: str, mode: str, outdir: str,
                      speeds: list | None = None,
                      logs: list | None = None,
                      trace_min_replicas: int = 3,
                      trim_fraction: float = 0.1,
                      min_common_points: int = 5) -> pd.DataFrame:
    """WDR5-flavoured wrapper around the packaged force ladder."""
    from autopath.pulling.ConvergenceLadder import force_convergence_ladder
    return force_convergence_ladder(
        build_analysis(system, mode, outdir),
        logs if logs is not None else logs_for(system, mode),
        speeds or SPEEDS,
        trace_min_replicas=trace_min_replicas,
        trim_fraction=trim_fraction,
        min_common_points=min_common_points,
    )
```

- [ ] **Step 4: Accept `force` in check_convergence**

In `autopath/pulling/AnalysisSMD.py`, extend `allowed_estimators` (line 796-799)
to include `'force'`, and dispatch to the ladder before the per-speed loop:

```python
        if estimator_name == 'force':
            from .ConvergenceLadder import force_convergence_ladder
            conv_df = force_convergence_ladder(
                self, logs, speeds,
                trace_min_replicas=trace_min_replicas,
                trim_fraction=trim_fraction,
                min_common_points=min_common_points,
            )
            return conv_df, pd.DataFrame()
```

The empty traces frame is deliberate: the ladder produces no per-rung PMF traces, and callers that plot traces must handle an empty frame rather than receive fabricated rows.

- [ ] **Step 5: Run the regression**

```bash
cd scratch/paper_figures && /gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -v -p no:cacheprovider 2>&1 | tail -20
```
Expected: 12 passed, unchanged. In particular `test_validation_gate_matches_api_on_cumulant` must still report `max_abs_rmsd_diff = 0.0`.

- [ ] **Step 6: Run the package suite**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/ -v -m "not slow"`
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
git add autopath/pulling/ConvergenceLadder.py autopath/pulling/AnalysisSMD.py
git commit -m "promote force convergence ladder into package"
```

---

### Task 7: Round-robin speed mode

**Files:**
- Modify: `autopath/autopath_core.py:574-667`

**Interfaces:**
- Consumes: `self.sMD_alternate_speeds`, `self.sMD_autostop_estimator` from Task 4.
- Produces: no new public names; changes the order in which replicas are produced.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_autostop_options.py`:

```python
def test_round_robin_visits_every_live_speed_before_repeating():
    """Ordering contract for alternate_speeds=True, tested on the pure helper."""
    from autopath.pulling.Convergence import round_robin_order

    live = {0.005: True, 0.01: True, 0.015: True}
    assert round_robin_order(live) == [0.005, 0.01, 0.015]

    live[0.01] = False          # retired
    assert round_robin_order(live) == [0.005, 0.015]

    assert round_robin_order({0.005: False}) == []
```

- [ ] **Step 2: Run it**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_autostop_options.py::test_round_robin_visits_every_live_speed_before_repeating -v`
Expected: FAIL — `cannot import name 'round_robin_order'`

- [ ] **Step 3: Implement the helper**

Append to `autopath/pulling/Convergence.py`:

```python
def round_robin_order(live: dict) -> list:
    """Speeds still accepting replicas, slowest first, retired ones dropped."""
    return [s for s in sorted(live) if live[s]]
```

- [ ] **Step 4: Restructure the loop**

In `autopath/autopath_core.py`, wrap the existing `for speed, reps in self.sMD_pulling_speeds.items():` body. When `self.sMD_alternate_speeds` is False the current code path runs unchanged. When True, replace the outer loop with:

```python
            if self.sMD_alternate_speeds and self.sMD_converge_speeds:
                live = {s: True for s in self.sMD_pulling_speeds}
                failures = {s: 0 for s in live}
                while any(live.values()):
                    for speed in round_robin_order(live):
                        reps = self.sMD_pulling_speeds[speed]
                        log_files = glob(f"{sMD_traj_outdir}/sMD_*_v{speed}_{self.sMD_pulling_dir}.dat")
                        if len(log_files) >= self.sMD_max_replicas:
                            logger.warning(
                                f"Reached maximum number of replicas ({self.sMD_max_replicas}) "
                                f"for speed {speed} nm/ps without convergence. Stopping."
                            )
                            live[speed] = False
                            continue
                        try:
                            sMD.run(
                                checkpoint_file=equilibrated_chk,
                                state_xml_file=equilibrated_state_xml,
                                pulling_speed=speed,
                                pulling_direction=self.sMD_pulling_dir,
                            )
                            failures[speed] = 0
                        except Exception as e:
                            failures[speed] += 1
                            logger.error(
                                f"Error during sMD pulling for speed {speed} nm/ps: {e} "
                                f"(consecutive failure {failures[speed]}/5)"
                            )
                            if failures[speed] >= 5:
                                logger.error(
                                    f"Aborting speed {speed} nm/ps after 5 consecutive failures."
                                )
                                live[speed] = False
                            continue
                        if len(log_files) + 1 < reps:
                            continue
                        # decision: per-speed estimators check that speed; force
                        # checks the joint ladder and retires every speed at once
                        conv_df = self._check_speed(
                            sMD_analysis_outdir, sys_name, _lig_sel_ha, reps,
                            speed=None if self.sMD_autostop_estimator == "force" else speed)
                        if conv_df is None or conv_df.empty:
                            continue
                        if len(conv_df) >= self.sMD_conv_streak and np.isfinite(
                                first_streak(conv_df, k_consec=self.sMD_conv_streak)):
                            if self.sMD_autostop_estimator == "force":
                                logger.warning("sMD pulling CONVERGED (force ladder, all speeds).")
                                live = {s: False for s in live}
                            else:
                                logger.warning(
                                    f"sMD pulling for speed {speed} nm/ps CONVERGED after "
                                    f"{len(log_files) + 1} replicas."
                                )
                                live[speed] = False
            else:
                # existing one-speed-at-a-time path, unchanged
```

- [ ] **Step 5: Extract the shared check into `_check_speed`**

The `SMDAnalysis` construction and `check_convergence` call currently inlined at
`autopath_core.py:595-611` are now needed by both branches. Move them verbatim
into a private method on `AutoPath`:

```python
    def _check_speed(self, outdir, sys_name, lig_sel, min_replicas, speed=None):
        """Build the analysis and run the convergence check for one speed.

        speed=None means "all speeds jointly", which is what the force ladder needs.
        `min_replicas` is passed explicitly rather than derived from
        sMD_pulling_speeds, whose values are None in the Config default
        (`config.py:46`) and would raise on min().
        Returns the convergence DataFrame, or None when it cannot be computed yet.
        """
        speeds = None if speed is None else [speed]
        logs = glob(f"{self.sMD_outdir}/trajectories/sMD_*_{self.sMD_pulling_dir}.dat") if speed is None \
            else glob(f"{self.sMD_outdir}/trajectories/sMD_*_v{speed}_{self.sMD_pulling_dir}.dat")
        smdanalysis = SMDAnalysis(sysname=sys_name, path_model='dtw',
                                  estimators=[self.sMD_autostop_estimator],
                                  do_plots=False, seed=self.random_state,
                                  temperature=self.temperature, outdir=outdir,
                                  ligand_select=lig_sel)
        conv_df, _ = smdanalysis.check_convergence(
            logs=logs, speeds=speeds,
            min_replicas=min_replicas,
            estimator_name=self.sMD_autostop_estimator,
            conv_window=self.sMD_conv_window,
            geom_features=bool(self.sMD_log_geom_features),
            plateau_frac=self.sMD_plateau_frac,
            cluster_to_boundary=self.sMD_cluster_to_boundary,
            restrict_rmsd_to_boundary=self.sMD_cluster_to_boundary,
            boundary_buffer_frac=self.sMD_boundary_buffer_frac,
        )
        return conv_df
```

Then have the unchanged sequential branch call `self._check_speed(...)` too, so
there is one construction site rather than two that can drift.

- [ ] **Step 6: Run tests**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/ -v -m "not slow"`
Expected: all pass, including the new round-robin test.

- [ ] **Step 7: Commit**

```bash
git add autopath/autopath_core.py autopath/pulling/Convergence.py tests/test_autostop_options.py
git commit -m "add round-robin speed mode for multi-speed autostop"
```

---

### Task 8: Calibrate the window and set defaults

**Files:**
- Modify: `scratch/paper_figures/conv_grid_wdr5.py` (accept `--conv-window`)
- Modify: `autopath/autopath_core.py`, `autopath/config.py` (defaults, only if evidence supports a change)
- Modify: `docs/superpowers/specs/2026-07-31-smd-autostop-criterion-design.md` (record the outcome)

**Interfaces:**
- Consumes: `calibrate_autostop.py` from Task 5; the packaged `conv_window` from Task 2.
- Produces: final default values, and a recorded evidence table.

- [ ] **Step 1: Add the window sweep to the grid driver**

In `scratch/paper_figures/conv_grid_wdr5.py`, add an argparse flag
`--conv-window` (default 1) forwarded to `lib.conv_api(..., conv_window=...)`,
and include the window value in the output filename so sweeps do not overwrite
each other: `conv_grid/<system>_<mode>_w<window>.csv`.

- [ ] **Step 2: Run the sweep**

Run the 36-task grid for `w ∈ {1, 3, 5, 8}`. Single-threaded this is ~4 h per
window; submit as a SLURM array instead, mirroring the existing
`qfiles_conv/wdr5_conv_array.q` pattern. Do not proceed until all four
windows have written their CSVs.

- [ ] **Step 3: Extend the calibration driver to the window axis**

In `calibrate_autostop.py`, change `GRID` to `results_conv/conv_grid/*_w*.csv`,
parse `w` from the filename into a column, and add `w` to the `groupby` so the
output table has one row per `(estimator, w, k_consec)`.

- [ ] **Step 4: Run it and record the evidence**

```bash
cd scratch/paper_figures && /gpfs/home/mllanos/micromamba/envs/autopath/bin/python calibrate_autostop.py
```

Paste the table into this section under a heading **"Calibration evidence"**,
with the date. Then apply the decision rule from the spec: the shipped default
is the cheapest `(w, k_consec)` whose `gate_pass_frac >= 0.90` for the default
estimator. If no setting reaches 0.90, record that and ship the best available,
saying so explicitly in the spec — do not loosen the tolerances to manufacture a
pass.

- [ ] **Step 5: Apply the defaults**

Edit `sMD_conv_window` and `sMD_conv_streak` defaults in **both**
`autopath/autopath_core.py` and `autopath/config.py` to the chosen values. If
the evidence shows the default estimator should change, raise that as a separate
decision rather than changing it silently here.

- [ ] **Step 6: Record the outcome in the spec**

Add a "Calibration outcome" section to
`docs/superpowers/specs/2026-07-31-smd-autostop-criterion-design.md` stating the
chosen values, the gate pass fraction achieved, the censoring count, and the
median extra replicas versus today's rule.

- [ ] **Step 7: Run the full suite**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/ -v`
Expected: all pass, including the slow backward-compatibility fixture.

- [ ] **Step 8: Commit**

```bash
git add autopath/autopath_core.py autopath/config.py docs/superpowers/specs/2026-07-31-smd-autostop-criterion-design.md
git commit -m "set autostop defaults from WDR5 calibration"
```

---

## Known evidence at plan time (2026-07-31)

From the already-cached `conv_window=1` grid, using the **current** pairwise rule:

| estimator | k_consec | cells | censored | median n | p90 n |
|---|---|---|---|---|---|
| cumulant | 2 | 108 | 47 | 25 | 42 |
| cumulant | 3 | 108 | 86 | 33 | 43 |
| jarzynski | 2 | 108 | 1 | 12 | 17 |
| jarzynski | 3 | 108 | 3 | 16 | 26 |
| force | 2 | 36 | 2 | 12.5 | 20 |
| force | 3 | 36 | 6 | 17 | 29 |

**This is a live risk to the shipped defaults.** With `cumulant` and
`k_consec=3`, 86 of 108 runs never converge even under the *lenient* pairwise
rule — the autostop degenerates into "run to the 50-replica cap". The window
reference is stricter still, so Task 8 is expected to show cumulant censoring at
or above that level. Jarzynski and force converge roughly twice as fast and
almost never censor. If Task 8 confirms this, the honest recommendation is to
change the default estimator, which is a decision for the user, not this plan.
