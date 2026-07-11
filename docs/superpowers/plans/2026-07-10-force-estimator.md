# Force Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `force` free-energy estimator (v→0 extrapolation of mean work for the PMF, plus dF/dv force friction) to the sMD analysis pipeline, defaulted on when ≥2 pulling speeds exist, and switch path weighting to a single robustness-selected reference p_eq shared by all estimators.

**Architecture:** The force estimator's per-speed `dG` is the raw mean work `Wmean`; its v→0 intercept (via the existing `extrapolate_to_v0` OLS) is the true ΔG. It also carries `Fmean`, from which a new pair of `FrictionEstimator` methods derive `Γ = dF/dv` (slope) and `Feq` (intercept) using the same OLS. Path weighting stops being per-estimator: one reference p_eq (fewest negative-dG bins among genuine free-energy estimators, cumulant on ties) is computed once and applied to every estimator.

**Tech Stack:** Python, numpy, pandas, scipy (`scipy.stats.linregress`, `scipy.special.logsumexp`), pytest.

## Global Constraints

- Estimator `estimate_dG(raw_W, beta, **kwargs)` must return a dict with at least `{'Wmean','dG','Wdiss'}` (downstream `calculate_weighted_pmf`, `FrictionEstimator`, Kramers assume these keys).
- `friction.csv` schema per row: `r_coord, speed, Gamma, Gamma_integrated, method, estimator` (+ `step` when available). Force friction must match it.
- The force estimator is inherently multi-speed: with <2 distinct speeds it is dropped from the run (warn, do not raise).
- `force` is NEVER a candidate for the p_eq weighting reference (its `dG=Wmean` is non-negative by construction).
- Temperature for manual checks: `T=310.15`, `kT=0.0083145*T`, `beta=1/kT` (SMDData supplies `beta` at runtime).
- Run the test suite with the autopath env interpreter: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest`.
- Spec: `docs/superpowers/specs/2026-07-10-force-estimator-design.md`. Regime evidence: memory `project_estimator_regime_hsp90.md`.

---

## File Structure

- `autopath/pulling/Estimators.py` — add `ForceEstimator`, two `FrictionEstimator` force methods + a private mixing helper, add to `ESTIMATOR_REGISTRY`.
- `autopath/pulling/SMDData.py` — redefine `_choose_estimator_for_weights(results,'auto')` (robustness) and add `choose_reference_estimator(results)`.
- `autopath/pulling/AnalysisSMD.py` — `_setup_estimators` map, `run()` speed guard, shared reference weights, friction-loop branch, `__init__` default list.
- `autopath/pulling/__init__.py` — export `ForceEstimator`.
- `autopath/autopath_core.py` — add `'force'` to the full-analysis estimator list (line 664).
- `tests/test_force_estimator.py` — new unit tests.
- `tests/regression_run_mixture_pmfs.md` — re-baseline note (weighting change).

---

## Task 1: ForceEstimator class + registry + export

**Files:**
- Modify: `autopath/pulling/Estimators.py` (after `CumulantEstimator`, ~line 127; and `ESTIMATOR_REGISTRY` ~line 1225)
- Modify: `autopath/pulling/__init__.py:31-44`
- Test: `tests/test_force_estimator.py`

**Interfaces:**
- Produces: `ForceEstimator` (BaseEstimator subclass), `name='force'`; `ForceEstimator.estimate_dG(raw_W, beta) -> {'Wmean','dG','Wdiss'}` with `dG==Wmean`, `Wdiss=np.nan`; `fit_transform(smd_data)` appends results rows carrying an extra `Fmean` column; registry key `'force'`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_force_estimator.py
import numpy as np
import pandas as pd
import pytest
from autopath.pulling.Estimators import ForceEstimator, ESTIMATOR_REGISTRY


def test_estimate_dG_is_raw_mean_work():
    W = np.array([10.0, 12.0, 8.0, 14.0])
    out = ForceEstimator.estimate_dG(W, beta=0.4)
    assert out['Wmean'] == pytest.approx(W.mean())
    assert out['dG'] == pytest.approx(W.mean())          # raw work, no correction
    assert np.isnan(out['Wdiss'])                         # friction is force-based, not Wdiss
    assert ForceEstimator.estimate_dG(np.array([]), beta=0.4) is None


def test_force_in_registry():
    assert ESTIMATOR_REGISTRY['force'] is ForceEstimator
    assert ForceEstimator().name == 'force'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -v`
Expected: FAIL with `ImportError: cannot import name 'ForceEstimator'`.

- [ ] **Step 3: Add the ForceEstimator class** (insert after `CumulantEstimator`, before `class FrictionEstimator`, ~line 127 of `Estimators.py`)

```python
class ForceEstimator(BaseEstimator):
    """Force / raw-work estimator.

    Per-speed ``dG`` is the raw mean cumulative work ``Wmean`` (uncorrected);
    its v→0 intercept via :func:`extrapolate_to_v0` is the reversible work = ΔG.
    Also records ``Fmean`` (mean restraint force per step/speed/path) which the
    force-based friction (``FrictionEstimator.gamma_from_force_*``) turns into
    Γ = dF/dv.  Inherently multi-speed: a single speed gives no v→0 intercept.
    """

    @property
    def name(self):
        return 'force'

    @staticmethod
    def estimate_dG(raw_W: np.ndarray, beta: float, **kwargs) -> dict | None:
        raw_W = np.asarray(raw_W, dtype=float)
        if raw_W.size == 0:
            return None
        Wmean = float(raw_W.mean())
        # Per-speed "PMF" is the raw work; dissipation is removed by the v→0
        # extrapolation, not per-speed.  Wdiss is NaN (force friction uses force).
        return {'Wmean': Wmean, 'dG': Wmean, 'Wdiss': np.nan}

    def fit_transform(self, smd_data: SMDData) -> SMDData:
        data = smd_data.raw_data.copy()
        group_keys = ['step', 'speed', 'path']
        results = []
        for (step, speed, path), group in data.groupby(group_keys):
            r_coord = smd_data.protocol_grids[speed].loc[
                smd_data.protocol_grids[speed]['step'] == step,
                'r_target_protocol'
            ].values[0]

            raw_W = group['work'].astype(float).values
            result = self.estimate_dG(raw_W, smd_data.beta)
            if result is None:
                continue

            results.append({
                'r_coord': r_coord,
                'step': step,
                'speed': speed,
                'path': path,
                'n_samples': raw_W.size,
                'Fmean': float(group['force'].astype(float).mean()),
                **result,
            })
        results_df = pd.DataFrame(results)
        smd_data.add_estimator_results(self.name, results_df)
        return smd_data
```

- [ ] **Step 4: Register it** — edit `ESTIMATOR_REGISTRY` (~line 1225):

```python
ESTIMATOR_REGISTRY: dict[str, type[BaseEstimator]] = {
    'jarzynski': JarzynskiEstimator,
    'cumulant': CumulantEstimator,
    'force': ForceEstimator,
}
```

- [ ] **Step 5: Export it** — in `autopath/pulling/__init__.py`, add to the import block (line 31-34) and `__all__` (line 42-43):

```python
from .Estimators import (
    JarzynskiEstimator,
    CumulantEstimator,
    ForceEstimator,
)
```
and add `"ForceEstimator",` to `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add autopath/pulling/Estimators.py autopath/pulling/__init__.py tests/test_force_estimator.py
git commit -m "feat: ForceEstimator (raw-work dG=Wmean, carries Fmean) + registry"
```

---

## Task 2: Robustness-based reference estimator selection

**Files:**
- Modify: `autopath/pulling/SMDData.py:613-640` (`_choose_estimator_for_weights`)
- Test: `tests/test_force_estimator.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on a `results` DataFrame with `estimator`, `dG` columns).
- Produces: `SMDData._choose_estimator_for_weights(results, estimator='auto') -> str` where `'auto'` = fewest negative-dG bins among candidates, candidates = available estimators **excluding `'force'`**, ties broken by order `['cumulant','jarzynski']` then first available; explicit-name path unchanged (raises if the named estimator absent). `SMDData.choose_reference_estimator(results) -> str` = `_choose_estimator_for_weights(results, 'auto')`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_force_estimator.py
from autopath.pulling.SMDData import SMDData


def _results(**dg_by_est):
    frames = []
    for est, dg in dg_by_est.items():
        frames.append(pd.DataFrame({'estimator': est, 'dG': dg,
                                     'step': range(len(dg))}))
    return pd.concat(frames, ignore_index=True)


def test_reference_prefers_fewer_negative_bins():
    # cumulant has negatives, jarzynski none -> jarzynski (the betasigma~10 regime)
    r = _results(cumulant=[-1.0, 5.0, 10.0], jarzynski=[1.0, 6.0, 11.0])
    assert SMDData.choose_reference_estimator(r) == 'jarzynski'


def test_reference_ties_go_to_cumulant():
    r = _results(cumulant=[1.0, 5.0, 10.0], jarzynski=[1.0, 6.0, 11.0])
    assert SMDData.choose_reference_estimator(r) == 'cumulant'


def test_reference_never_selects_force():
    # force dG is all non-negative -> 0 negatives, but must be excluded
    r = _results(cumulant=[-1.0, 5.0], jarzynski=[-1.0, 6.0], force=[9.0, 10.0])
    assert SMDData.choose_reference_estimator(r) in ('cumulant', 'jarzynski')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -k reference -v`
Expected: FAIL with `AttributeError: ... choose_reference_estimator`.

- [ ] **Step 3: Rewrite `_choose_estimator_for_weights` and add helper** — replace the body of `_choose_estimator_for_weights` (`SMDData.py:613-640`):

```python
    @staticmethod
    def _choose_estimator_for_weights(results: pd.DataFrame, estimator: str = 'auto') -> str:
        """Resolve which estimator's dG to use for p_eq computation.

        ``estimator='auto'`` selects by ROBUSTNESS: among genuine free-energy
        candidates (all available estimators EXCLUDING ``'force'``, whose
        ``dG=Wmean`` is non-negative by construction), pick the one with the
        smallest fraction of negative-dG bins — those are exactly the bins
        masked to 0 in ``_compute_p_eq`` and what destabilises the p_eq integral.
        Ties resolve to the order ``['cumulant','jarzynski']`` then first
        available.  An explicitly named estimator is returned as-is (raises if
        not present).
        """
        if 'estimator' not in results.columns:
            raise ValueError("results must include an 'estimator' column.")

        available_estimators = results['estimator'].dropna().unique().tolist()
        if len(available_estimators) == 0:
            raise ValueError("No estimator results available to compute p_eq.")

        if estimator != 'auto':
            if estimator not in available_estimators:
                raise ValueError(
                    f"Estimator '{estimator}' not available in results. "
                    f"Available: {sorted(available_estimators)}"
                )
            return estimator

        candidates = [e for e in available_estimators if e != 'force']
        if not candidates:
            raise ValueError("No non-force estimator available to weight paths.")

        def neg_frac(est: str) -> float:
            dG = results.loc[results['estimator'] == est, 'dG'].to_numpy(dtype=float)
            dG = dG[np.isfinite(dG)]
            return float((dG < 0).mean()) if dG.size else 1.0

        preference = {'cumulant': 0, 'jarzynski': 1}
        # sort by (neg fraction asc, preference asc, name) -> deterministic
        best = min(candidates, key=lambda e: (neg_frac(e),
                                              preference.get(e, 2),
                                              e))
        return best

    @staticmethod
    def choose_reference_estimator(results: pd.DataFrame) -> str:
        """Robustness-selected reference estimator for shared p_eq weighting."""
        return SMDData._choose_estimator_for_weights(results, 'auto')
```

Confirm `import numpy as np` is present at the top of `SMDData.py` (it is).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -k reference -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add autopath/pulling/SMDData.py tests/test_force_estimator.py
git commit -m "feat: robustness-based reference estimator for shared p_eq"
```

---

## Task 3: Force-based friction methods

**Files:**
- Modify: `autopath/pulling/Estimators.py` (add methods to `FrictionEstimator`, after `gamma_from_wdiss_regression`, ~line 400)
- Test: `tests/test_force_estimator.py`

**Interfaces:**
- Consumes: `extrapolate_to_v0(results, param, ...)` (existing, returns `param` intercept + `f"{param}_slope"`), `FrictionEstimator._cumulative_integral`, `FrictionEstimator._smooth_profile`.
- Produces:
  - `FrictionEstimator._mix_force(force_results, weights) -> DataFrame[estimator,step,r_coord,speed,Fbar]` (path-mixed ⟨F⟩).
  - `FrictionEstimator.gamma_from_force_regression(force_results, weights) -> DataFrame` with `r_coord, step, speed(=0.0), Gamma(=Fbar_slope), Gamma_integrated, Feq, method='regression', estimator='force'`.
  - `FrictionEstimator.gamma_from_force_derivative(force_results, weights) -> DataFrame` with per-speed rows `r_coord, step, speed, Gamma(=(Fbar-Feq)/speed), Gamma_integrated, method='derivative', estimator='force'`.
  - `weights` is `{speed: {path: p_eq}}` (single dict; same shape as one estimator's entry in `weights_by_estimator`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_force_estimator.py
from autopath.pulling.Estimators import FrictionEstimator


def _force_results_linear(feq_of_step, gamma_of_step, speeds, paths=('p0',)):
    """Build force results where Fmean(step,speed) = Feq(step) + Gamma(step)*speed."""
    rows = []
    for step, (feq, gam) in enumerate(zip(feq_of_step, gamma_of_step)):
        for sp in speeds:
            for p in paths:
                rows.append(dict(estimator='force', step=step, path=p,
                                 speed=sp, r_coord=float(step) * 0.1,
                                 Fmean=feq + gam * sp))
    return pd.DataFrame(rows)


def test_force_regression_recovers_feq_and_gamma():
    speeds = [0.001, 0.005, 0.01]
    feq = [0.0, 100.0, 250.0, 300.0, 260.0]
    gam = [500.0, 800.0, 1200.0, 900.0, 400.0]
    fr = _force_results_linear(feq, gam, speeds)
    weights = {sp: {'p0': 1.0} for sp in speeds}
    out = FrictionEstimator().gamma_from_force_regression(fr, weights)
    out = out.sort_values('step')
    np.testing.assert_allclose(out['Feq'].to_numpy(), feq, atol=1e-6)
    np.testing.assert_allclose(out['Gamma'].to_numpy(), gam, atol=1e-6)
    assert (out['method'] == 'regression').all()
    assert (out['estimator'] == 'force').all()


def test_force_derivative_matches_slope():
    speeds = [0.001, 0.005, 0.01]
    feq = [0.0, 100.0, 250.0]
    gam = [500.0, 800.0, 1200.0]
    fr = _force_results_linear(feq, gam, speeds)
    weights = {sp: {'p0': 1.0} for sp in speeds}
    out = FrictionEstimator().gamma_from_force_derivative(fr, weights)
    # For exactly-linear data, per-speed (Fbar-Feq)/v == gamma at every speed.
    for sp in speeds:
        g = out[out['speed'] == sp].sort_values('step')
        np.testing.assert_allclose(g['Gamma'].to_numpy(), gam, atol=1e-6)
    assert (out['method'] == 'derivative').all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -k force_regression -v`
Expected: FAIL with `AttributeError: ... gamma_from_force_regression`.

- [ ] **Step 3: Add the methods** (inside `class FrictionEstimator`, after `gamma_from_wdiss_regression`, ~line 400):

```python
    @staticmethod
    def _mix_force(force_results: pd.DataFrame, weights: dict) -> pd.DataFrame:
        """Path-mix Fmean into <F>(step, speed) using p_eq weights.

        ``weights`` is ``{speed: {path: p_eq}}``.  For single-path systems this
        reduces to the plain mean.  Returns columns
        ``estimator, step, r_coord, speed, Fbar``.
        """
        rows = []
        for (speed, step), g in force_results.groupby(['speed', 'step']):
            sw = weights.get(speed, {}) if weights else {}
            num = den = 0.0
            rcoords = []
            for _, row in g.iterrows():
                w = sw.get(row['path'], 0.0) if sw else 1.0
                if w <= 0:
                    continue
                num += w * float(row['Fmean'])
                den += w
                rcoords.append(float(row['r_coord']))
            if den <= 0:
                continue
            rows.append(dict(estimator='force', step=step, speed=speed,
                             r_coord=float(np.mean(rcoords)), Fbar=num / den))
        return pd.DataFrame(rows)

    def gamma_from_force_regression(self, force_results: pd.DataFrame,
                                    weights: dict) -> pd.DataFrame:
        """Γ(r) = dF/dv (slope) and Feq(r) (intercept) via across-speed OLS of <F>."""
        if force_results is None or force_results.empty:
            return pd.DataFrame()
        fbar = self._mix_force(force_results, weights)
        if fbar.empty or fbar['speed'].nunique() < 2:
            return pd.DataFrame()

        v0 = extrapolate_to_v0(results=fbar, param='Fbar', speeds=self.speeds)
        if v0 is None or v0.empty:
            return pd.DataFrame()

        out = v0.copy()
        out['Feq'] = out['Fbar'].astype(float)          # intercept
        out['Gamma'] = out['Fbar_slope'].astype(float)  # local friction dF/dv
        out = out.sort_values('r_coord').reset_index(drop=True)
        r = out['r_coord'].to_numpy(dtype=float)
        out['Gamma_integrated'] = self._cumulative_integral(r, out['Gamma'].to_numpy(dtype=float))
        out['method'] = 'regression'
        out['estimator'] = 'force'
        keep = ['r_coord', 'step', 'speed', 'Gamma', 'Gamma_integrated',
                'Feq', 'method', 'estimator']
        return out[[c for c in keep if c in out.columns]]

    def gamma_from_force_derivative(self, force_results: pd.DataFrame,
                                    weights: dict) -> pd.DataFrame:
        """Per-speed friction Γ_sp(r) = (<F>(r;v) − Feq(r)) / v."""
        if force_results is None or force_results.empty:
            return pd.DataFrame()
        fbar = self._mix_force(force_results, weights)
        if fbar.empty or fbar['speed'].nunique() < 2:
            return pd.DataFrame()

        reg = self.gamma_from_force_regression(force_results, weights)
        if reg.empty:
            return pd.DataFrame()
        feq_by_step = reg.set_index('step')['Feq'].to_dict()

        rows = []
        for speed, g in fbar.groupby('speed'):
            if speed <= 0:
                continue
            g = g.sort_values('r_coord')
            r = g['r_coord'].to_numpy(dtype=float)
            gamma = np.array([(fb - feq_by_step.get(st, np.nan)) / speed
                              for fb, st in zip(g['Fbar'], g['step'])], dtype=float)
            gamma_int = self._cumulative_integral(r, np.nan_to_num(gamma))
            rows.append(pd.DataFrame({
                'r_coord': r, 'step': g['step'].to_numpy(), 'speed': speed,
                'Gamma': gamma, 'Gamma_integrated': gamma_int,
                'method': 'derivative', 'estimator': 'force',
            }))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -k "force_regression or force_derivative" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add autopath/pulling/Estimators.py tests/test_force_estimator.py
git commit -m "feat: force-based friction (dF/dv slope, Feq intercept) in FrictionEstimator"
```

---

## Task 4: run() wiring — speed guard, shared weights, friction branch, defaults

**Files:**
- Modify: `autopath/pulling/AnalysisSMD.py` — `_setup_estimators` (169-172), `__init__` default (106), `run()` speed guard (before 377), weights (407-410), friction loop (455-459)
- Modify: `autopath/autopath_core.py:664`
- Test: `tests/test_force_estimator.py`

**Interfaces:**
- Consumes: `ForceEstimator` (Task 1), `SMDData.choose_reference_estimator` (Task 2), `FrictionEstimator.gamma_from_force_regression/derivative` (Task 3).
- Produces: `SMDAnalysis._drop_force_if_single_speed(estimators, n_speeds) -> list` static helper (warns + drops force when `n_speeds < 2`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_force_estimator.py
from autopath.pulling.AnalysisSMD import SMDAnalysis
from autopath.pulling.Estimators import CumulantEstimator, JarzynskiEstimator


def test_setup_estimators_accepts_force():
    a = SMDAnalysis(estimators=['cumulant', 'force'])
    assert sorted(e.name for e in a.estimators) == ['cumulant', 'force']


def test_speed_guard_drops_force_single_speed(caplog):
    ests = [CumulantEstimator(), JarzynskiEstimator(), ForceEstimator()]
    kept = SMDAnalysis._drop_force_if_single_speed(ests, n_speeds=1)
    assert [e.name for e in kept] == ['cumulant', 'jarzynski']
    kept2 = SMDAnalysis._drop_force_if_single_speed(ests, n_speeds=3)
    assert [e.name for e in kept2] == ['cumulant', 'jarzynski', 'force']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -k "setup_estimators or speed_guard" -v`
Expected: FAIL — `ValueError: Unknown estimator: force` (map) and `AttributeError: _drop_force_if_single_speed`.

- [ ] **Step 3a: Add force to `_setup_estimators` map** (`AnalysisSMD.py:169-172`):

```python
        estimator_map = {
            'jarzynski': JarzynskiEstimator(),
            'cumulant': CumulantEstimator(),
            'force': ForceEstimator(),
        }
```
Ensure `ForceEstimator` is imported in `AnalysisSMD.py` (add to the existing `from autopath.pulling.Estimators import (...)` block).

- [ ] **Step 3b: Add the speed-guard static helper** (add as a `@staticmethod` on `SMDAnalysis`, near `_setup_estimators`):

```python
    @staticmethod
    def _drop_force_if_single_speed(estimators: list, n_speeds: int) -> list:
        """Force needs >=2 speeds for its v->0 intercept; drop it (warn) otherwise."""
        if n_speeds >= 2:
            return list(estimators)
        kept = [e for e in estimators if e.name != 'force']
        if len(kept) != len(estimators):
            logger.warning(
                "Force estimator requires >=2 pulling speeds for v->0 "
                f"extrapolation; only {n_speeds} present — dropping 'force'."
            )
        return kept
```

- [ ] **Step 3c: Apply the guard in `run()`** — immediately before the estimator fit loop (`AnalysisSMD.py:377`, `for estimator in self.estimators:`), insert:

```python
        n_speeds = int(sMDDdata.raw_data['speed'].nunique())
        self.estimators = self._drop_force_if_single_speed(self.estimators, n_speeds)
```

- [ ] **Step 3d: Shared reference weights** — replace the weights dict comprehension (`AnalysisSMD.py:407-410`):

```python
        ref_estimator = SMDData.choose_reference_estimator(sMDDdata.results)
        logger.info(f"[p_eq] shared reference estimator (robustness): {ref_estimator}")
        ref_weights = self.compute_p_eq(sMDDdata, estimator=ref_estimator)
        weights_by_estimator = {est.name: ref_weights for est in active_estimators}
```

- [ ] **Step 3e: Friction-loop branch** — replace the loop body (`AnalysisSMD.py:455-459`):

```python
        for estimator in active_estimators:
            if estimator.name == 'force':
                force_rows = sMDDdata.results[sMDDdata.results['estimator'] == 'force']
                f_deriv = friction_est.gamma_from_force_derivative(force_rows, ref_weights)
                f_regress = friction_est.gamma_from_force_regression(force_rows, ref_weights)
            else:
                f_deriv = friction_est.gamma_from_wdiss_derivative(df, estimator=estimator.name)
                f_regress = friction_est.gamma_from_wdiss_regression(df, estimator=estimator.name)
            friction_deriv_results.append(f_deriv)
            friction_regress_results.append(f_regress)
```

- [ ] **Step 3f: Default list** — `AnalysisSMD.py:106`:

```python
        estimators:Union[list[BaseEstimator], list[str]] = ['jarzynski', 'cumulant', 'force'],
```
and `autopath/autopath_core.py:664`:

```python
                                    estimators=['cumulant', 'jarzynski', 'force'],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/test_force_estimator.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add autopath/pulling/AnalysisSMD.py autopath/autopath_core.py tests/test_force_estimator.py
git commit -m "feat: wire force estimator + shared reference p_eq into run(); default on with >=2 speeds"
```

---

## Task 5: Full-pipeline integration check on real data + regression re-baseline

**Files:**
- Modify: `tests/regression_run_mixture_pmfs.md` (re-baseline note)
- No new source; this task verifies the wired pipeline end-to-end and records the behavior change.

**Interfaces:**
- Consumes: everything from Tasks 1-4.

- [ ] **Step 1: Run the existing suite to confirm no regressions**

Run: `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest tests/ -v`
Expected: existing `test_check_convergence_support.py` and `test_support_policy.py` still PASS; `test_force_estimator.py` PASS. If `test_check_convergence_support.py` fails on the `'auto'` semantics change, inspect — it should not, since it uses the `ESTIMATOR_REGISTRY` cumulant path and does not call `_choose_estimator_for_weights('auto')`.

- [ ] **Step 2: Drive the real pipeline** (verify force appears everywhere and weights are shared). Write `/tmp` scratch script that loads the 6ELO_BAW processed data through `SMDData` + `SMDAnalysis.run` OR, more cheaply, reuse the committed CSVs to confirm the estimator plumbing. Minimal end-to-end assertion script:

```python
# scratch verification (run manually, not committed)
import pandas as pd
from autopath.pulling.SMDData import SMDData
# Rebuild results for the 3 speeds from raw processed data and confirm:
#  - choose_reference_estimator returns 'jarzynski' for 6ELO_BAW (cumulant has neg bins)
#  - a run() over the real logs writes mixture_pmfs with estimator=='force' rows,
#    friction.csv with estimator=='force' method in {regression,derivative},
#    and koff_kramers.csv includes a 'force' row.
```
Expected observations to record: `force` present in `mixture_pmfs.csv` (per-speed + speed=0 v→0 rows), `friction.csv` (both methods), `koff_kramers.csv`; reference estimator logged as `jarzynski` for this dataset.

- [ ] **Step 3: Confirm the shared-weights invariant** — in the same run, assert every entry of `weights_by_estimator` is the identical object/dict (they all point to `ref_weights`). Record the reference chosen.

- [ ] **Step 4: Re-baseline the regression doc** — update `tests/regression_run_mixture_pmfs.md` to note: (a) `force` estimator now present; (b) cumulant/jarzynski `mixture_pmfs` values change for multi-path systems because weighting is now a single shared reference p_eq (robustness-selected) rather than per-estimator self-weighting; (c) single-path systems are unchanged (p_eq=1).

- [ ] **Step 5: Commit**

```bash
git add tests/regression_run_mixture_pmfs.md
git commit -m "test: re-baseline mixture_pmfs regression for shared reference p_eq + force estimator"
```

---

## Self-Review Notes (author)

- **Spec coverage:** ForceEstimator ΔG (Task 1) ✓; force friction dF/dv + Feq (Task 3) ✓; speed guard / default-on ≥2 speeds (Task 4) ✓; shared robustness reference p_eq excluding force (Tasks 2+4) ✓; registry/exports/defaults (Tasks 1,4) ✓; Kramers works unchanged via existing method='regression'/'derivative' rows (verified in Task 5) ✓; check_convergence NOT touched (allowed_estimators left as cumulant/jarzynski) ✓; regression re-baseline (Task 5) ✓.
- **Not in scope:** Hummer–Szabo PMF (explicitly excluded in spec).
- **Type consistency:** `gamma_from_force_regression/derivative(force_results, weights)` signatures match the Task 4 call sites; `choose_reference_estimator(results)` matches Task 4 usage; `_drop_force_if_single_speed(estimators, n_speeds)` matches its test and call site.
- **Watch item during Task 4:** confirm plots (`plot_profile` for `Wdiss`, `plot_extrapolated_param`) tolerate force's all-NaN `Wdiss` (extrapolate_to_v0 drops NaN per estimator, so force yields no Wdiss v→0 rows — no crash expected; verify in Task 5 Step 1).
