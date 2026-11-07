# Hummer–Szabo Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Hummer–Szabo estimator in `autopath.pulling` that gives G0(r) along the pulled COM distance and per-row / per-frame equilibrium weights, with three F_t sources run side by side.

**Architecture:** Pure-array core in `autopath/pulling/hummer_szabo.py` (slice table, F_t sources, log-weights, binned WHAM, G0 profile, bootstrap), an `HummerSzaboEstimator` that runs it on a conditioned `SMDData`, a standalone `hs_frame_weights` for DCD frames, and an opt-in stage at the end of `SMDAnalysis.run()`. Tested against overdamped-Langevin toys with known G0(r) (`tests/hs_toys.py`).

**Tech Stack:** numpy, pandas, scipy (`logsumexp`), MDAnalysis (`DCDFile` for frame counts). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-hummer-szabo-estimator-design.md`

## Global Constraints

- Branch `hs-estimator` (worktree `~/forlilab/autopath/.claude/worktrees/hs-estimator`).
- Python: `~/micromamba/envs/autopath/bin/python`; run tests as `~/micromamba/envs/autopath/bin/python -m pytest -q <file>` from the worktree root. Never install packages into `autopath`.
- Unsupported cases raise `ValueError` whose message starts with `"not supported:"`. No hidden fallbacks.
- ESS is reported, never used to drop data. Rows outside full support get no weight and are counted.
- Units: kJ/mol, nm, ps. β = 1 / (0.008314462618 · T) (same R as `SMDData`).
- Code style: terse comments (1–3 lines, the non-obvious why), numpy docstrings without rationale prose, no system names / measured numbers in repo comments.
- Tests run the code; never assert on source text or signatures. Tests do not import anything outside `autopath/` and `tests/`.
- Commits: one-line conventional subject ≤ 50 chars, no body, no Co-Authored-By trailer.

**Design changes since the approved spec (flagged to the user with this plan):**
- **Primary estimate is all pulls jointly**, not a p_eq density mixture of per-path results. The all-pull ⟨e^{−βW}⟩ already equals the p_neq-weighted sum over paths, so the joint estimate is exact and needs no p_eq (which `compute_p_eq` only gives per speed). Per-path results are reported as diagnostics (`scope="path"`).
- **The `cumulant` and `force` F_t sources are computed from the conditioned rows** with `CumulantEstimator.estimate_dG` / `ForceEstimator.estimate_dG` + `extrapolate_to_v0`, not read from `smd_data.results` (those are per path; the joint estimate needs all-pull values). HS therefore needs no prior estimator run and does not touch `smd_data.results`.

## Review Focus

1. Pulls of one speed with different step size dx (different realized speed) → `"not supported:"`. Test in Task 4.
2. Backward pulls (λ decreasing; spec non-goal) → `"not supported:"`. Test in Task 4.
3. Cumulant source at a slice with one pull (variance undefined) → `"not supported:"`. Test in Task 2.
4. Very large works (βW ~ 1e3) must not overflow: adding a constant to every W leaves the weights unchanged. Test in Task 1.
5. A row whose r lies beyond every λ ± 5σ still gets a finite log-weight (the denominator grid spans all row r). Test in Task 1.

---

### Task 1: Toy pulls and core HS weights

**Files:**
- Create: `tests/hs_toys.py`
- Create: `autopath/pulling/hummer_szabo.py`
- Test: `tests/test_hs_core.py`

**Interfaces:**
- Produces (toys, `tests/hs_toys.py`):
  - `simulate_pulls(G0, dG0, n_pulls, speed, k=3765.6, T=300.0, r_start=1.0, length_nm=1.0, dx=0.001, D=0.01, dt=0.002, lam0="common", r0_jitter_nm=0.0, stop_steps=None, seed=0) -> pd.DataFrame` — overdamped Langevin on `G0` (callable, kJ/mol) with force `-dG0(r)`, harmonic spring k (kJ/mol/nm²) moved by `dx` per move, `dx/speed` ps per move. Columns exactly as `SMDData.raw_data` after conditioning: `trajname, speed, realized_speed, step, time, r_target, r_before, r_after, force, dW_protocol, work, path` (`path=1`). Row i is written after move i, `r_target` for row i is λ0 + (i+1)·dx, `dW_protocol = u(r, λ_new) − u(r, λ_old)` at the instant of the move, `work = cumsum(dW_protocol)`. Initial r sampled from ρ0(r)·e^{−βu(r, λ0)} by inverse CDF on a 1e-4 nm grid. `lam0="common"`: λ0 = r_start for all pulls; `lam0="own"`: λ0 = each pull's initial r (as `steered_md` does), with initial r drawn from ρ0 near r_start and `r0_jitter_nm` added. `stop_steps`: optional per-pull last step (autostop).
  - `protocol_grids(rows) -> dict[float, pd.DataFrame]` — same contract as `SMDData.build_protocol_grids` (`step, r_target_protocol`).
  - `write_dat(rows, out_dir, header_k=True) -> list[str]` — one `sMD_replica-{id}_v{speed}_forward.dat` per pull with the logged columns `step,time,r_target,r_before,r_after,force,U_cvpack,dW_protocol` and, if `header_k`, a `# spring_constant_kJ_mol_nm2=` line.
  - Potentials: `flat()`, `harmonic(r0=1.0, kappa=500.0)`, `double_well(r_a=1.0, r_b=1.6, barrier=10.0)` each returning `(G0, dG0)`.
- Produces (core, `autopath/pulling/hummer_szabo.py`):
  - `check_full_support(rows: pd.DataFrame) -> None` — raises `"not supported: ..."` if any (speed, step) present for a speed lacks a row from every pull of that speed.
  - `slice_table(rows: pd.DataFrame, grids: dict, beta: float) -> pd.DataFrame` — one row per (speed, step): `speed, step, lam, n_pulls, lnZ` (ln mean e^{−βW}), `ess` (Kish ESS of e^{−βW} over the slice's pulls), `Wmean, Wvar` (ddof=1, NaN for one pull). Calls `check_full_support` first.
  - `f_jarzynski(slices: pd.DataFrame, beta: float) -> np.ndarray` — `−lnZ/β`.
  - `slice_weights(slices: pd.DataFrame, mode: str) -> np.ndarray` — `"ess"` → `slices.ess`, `"uniform"` → ones; other → `"not supported:"`.
  - `log_denominator(r: np.ndarray, slices, F, c, k, beta, dr=None) -> np.ndarray` — ln Σ_τ c_τ e^{−β[u(r, λ_τ) − F_τ]} at each r, evaluated on a uniform grid spanning [min r, max r] with spacing `dr` (default √(1/(βk))/10) and linearly interpolated; chunk over slices with `logsumexp`.
  - `hs_log_weights(rows, slices, F, c, k, beta) -> np.ndarray` — per row ln c_s − βW − lnZ_s − log_denominator(r). Rows map to slices by (speed, step). Uses column `r_after` as r.
  - `g0_profile(r, logw, pull, beta, bin_nm=0.02) -> pd.DataFrame` — `r_bin` (bin centre), `G0` (−kT ln(weight/bin_nm), min set to 0), `n_rows`, `ess_pulls` (Kish ESS of per-pull summed weights in the bin).

- [ ] **Step 1: Write the failing tests** in `tests/test_hs_core.py`

```python
def test_flat_landscape_is_recovered():
    rows = simulate_pulls(*flat(), n_pulls=40, speed=0.01, length_nm=0.4, seed=1)
    out = _g0(rows, F="jarzynski")              # helper: slice_table -> F -> c="uniform" -> weights -> g0_profile
    m = out.ess_pulls >= 5
    assert np.ptp(out.G0[m]) < 1.0              # flat within 1 kJ/mol

def test_harmonic_well_is_recovered():
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=60, speed=0.005, r_start=0.9, length_nm=0.2, seed=2)
    out = _g0(rows, F="jarzynski")
    m = out.ess_pulls >= 5
    ref = G0(out.r_bin[m]); ref -= ref.min()
    assert np.max(np.abs(out.G0[m] - ref)) < 1.0

def test_incomplete_support_is_not_supported():
    rows = simulate_pulls(*flat(), n_pulls=4, speed=0.01, length_nm=0.1, stop_steps=[50, 99, 99, 99])
    with pytest.raises(ValueError, match="not supported"):
        slice_table(rows, protocol_grids(rows), BETA)

def test_constant_work_offset_leaves_weights_unchanged():      # Review Focus 4
    rows = simulate_pulls(*flat(), n_pulls=8, speed=0.01, length_nm=0.1, seed=3)
    a = _logw(rows); rows2 = rows.assign(work=rows.work + 1e4)
    b = _logw(rows2)
    np.testing.assert_allclose(a - a.max(), b - b.max(), atol=1e-9)

def test_row_far_from_every_lambda_gets_finite_weight():      # Review Focus 5
    rows = simulate_pulls(*flat(), n_pulls=4, speed=0.01, length_nm=0.1, seed=4)
    rows.loc[rows.index[0], "r_after"] = rows.r_target.max() + 1.0
    assert np.isfinite(_logw(rows)).all()
```

Helpers `_logw(rows)` and `_g0(rows, F)` live in the test file and chain the core functions with `c = slice_weights(slices, "uniform")`.

- [ ] **Step 2: Run to verify they fail**

Run: `~/micromamba/envs/autopath/bin/python -m pytest -q tests/test_hs_core.py`
Expected: FAIL with `ModuleNotFoundError: autopath.pulling.hummer_szabo`

- [ ] **Step 3: Implement `tests/hs_toys.py`** (vectorised over pulls; Euler–Maruyama r += −βD(dG0 + k(r−λ))dt + √(2Ddt)·N(0,1); `dt` must satisfy βDk·dt < 0.1, assert it).

- [ ] **Step 4: Implement the core functions listed under Interfaces in `autopath/pulling/hummer_szabo.py`.**

- [ ] **Step 5: Run to verify they pass**

Run: `~/micromamba/envs/autopath/bin/python -m pytest -q tests/test_hs_core.py`
Expected: 5 passed. If the toy tolerance fails, slow the toy pull (smaller `speed`) — do not loosen the 1 kJ/mol bound.

- [ ] **Step 6: Commit**

```bash
git add tests/hs_toys.py tests/test_hs_core.py autopath/pulling/hummer_szabo.py
git commit -m "feat(pulling): Hummer-Szabo core weights"
```

---

### Task 2: F_t sources and binned WHAM

**Files:**
- Modify: `autopath/pulling/hummer_szabo.py`
- Test: `tests/test_hs_sources.py`

**Interfaces:**
- Consumes: Task 1 core functions and toys.
- Produces:
  - `f_cumulant(slices, beta) -> np.ndarray` — `CumulantEstimator.estimate_dG` formula per slice (Wmean − βWvar/2); any slice with `n_pulls < 2` → `"not supported:"`.
  - `f_force_v0(slices, realized_speed_map: dict) -> np.ndarray` — builds a table `step, r_coord=lam, speed, estimator="force", dG=Wmean` and calls `extrapolate_to_v0(..., param="dG", min_speeds=2, realized_speed_map=...)`; maps the v→0 `dG` back to every slice by `step`; slices with no intercept get NaN. Fewer than 2 speeds → `"not supported:"`.
  - `f_self_consistent(rows, slices, c, k, beta, bin_nm=0.002, stride=1, tol=1e-6, max_iter=10000) -> np.ndarray` — binned WHAM: per slice t, p_t(b) = work-weighted histogram of r_after over the slice's rows (weights e^{−βW}/Σ e^{−βW}, sums to 1); iterate ρ(b) = Σ_t c_t p_t(b) / Σ_t c_t e^{−β[u_t(b) − F_t]}, then F_t = −kT ln Σ_b ρ(b) e^{−βu_t(b)}, shift so F of the first slice = 0, start from `f_jarzynski`, stop when max |ΔF| < tol (kJ/mol); `stride` uses every stride-th step per speed and interpolates F in λ for the skipped slices. Not converged → raise `"not supported: WHAM did not converge ..."`.
  - `F_SOURCES = ("jarzynski", "cumulant", "force", "self_consistent")`.

- [ ] **Step 1: Write the failing tests** in `tests/test_hs_sources.py`

```python
@pytest.fixture(scope="module")
def dw():   # double well, three speeds, common lambda0
    G0, dG0 = double_well(r_a=1.0, r_b=1.6, barrier=10.0)
    rows = pd.concat([simulate_pulls(G0, dG0, n_pulls=60, speed=v, r_start=0.95, length_nm=0.8, seed=i)
                      for i, v in enumerate((0.0025, 0.005, 0.01))])
    return G0, rows

@pytest.mark.parametrize("src", ["jarzynski", "force", "self_consistent"])
def test_double_well_recovered_by(src, dw):
    G0, rows = dw
    out = _g0(rows, F=src, c="ess")
    m = out.ess_pulls >= 5
    ref = G0(out.r_bin[m]); ref -= ref.min()
    assert np.max(np.abs(out.G0[m] - ref)) < 1.5

def test_cumulant_exact_on_harmonic():
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=60, speed=0.005, r_start=0.9, length_nm=0.2, seed=5)
    out = _g0(rows, F="cumulant", c="ess")
    m = out.ess_pulls >= 5
    ref = G0(out.r_bin[m]); ref -= ref.min()
    assert np.max(np.abs(out.G0[m] - ref)) < 1.0

def test_wham_with_jarzynski_F_held_is_classic_hs(dw):
    _, rows = dw                    # one rho step at F = f_jarzynski, c = uniform == hs_log_weights histogram
    ...                             # compare binned rho from both routes, atol 1e-8 after normalisation

def test_force_needs_two_speeds():
    rows = simulate_pulls(*flat(), n_pulls=5, speed=0.01, length_nm=0.1)
    with pytest.raises(ValueError, match="not supported"):
        f_force_v0(slice_table(rows, protocol_grids(rows), BETA), {0.01: 0.01})

def test_cumulant_needs_two_pulls_per_slice():            # Review Focus 3
    rows = simulate_pulls(*flat(), n_pulls=1, speed=0.01, length_nm=0.1)
    with pytest.raises(ValueError, match="not supported"):
        f_cumulant(slice_table(rows, protocol_grids(rows), BETA), BETA)
```

For `test_wham_with_jarzynski_F_held_is_classic_hs`, expose the single ρ step as `wham_density(rows, slices, F, c, k, beta, bin_nm) -> (edges, rho)` and use it from `f_self_consistent`; the test compares it with `np.histogram(r_after, edges, weights=exp(hs_log_weights(...)))`, both normalised to sum 1.

- [ ] **Step 2: Run to verify they fail** — `... -m pytest -q tests/test_hs_sources.py` → ImportError on `f_cumulant`.

- [ ] **Step 3: Implement the sources and `wham_density`.**

- [ ] **Step 4: Run to verify they pass** — expected 7 passed (3 parametrised + 4). Also rerun `tests/test_hs_core.py`.

- [ ] **Step 5: Report, not assert, cumulant on the double well**: add `test_cumulant_double_well_deviation_is_reported` that computes the max deviation and only asserts it is finite; print it with `-s` once and note the number in the Task 7 report.

- [ ] **Step 6: Commit** — `git commit -m "feat(pulling): HS F sources and binned WHAM"`

---

### Task 3: Bootstrap and slice diagnostics

**Files:**
- Modify: `autopath/pulling/hummer_szabo.py`
- Test: `tests/test_hs_bootstrap.py`

**Interfaces:**
- Produces:
  - `estimate(rows, grids, k, beta, f_source, c_mode="ess", bin_nm=0.02, realized_speed_map=None, wham_stride=1) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]` — returns `(g0, logw_per_row, F_per_slice)` for one source on the given rows (full-support rows only; caller trims). Rows whose slice has F = NaN (force source outside the v→0 range) get logw = −inf.
  - `bootstrap_g0(rows, grids, k, beta, f_source, n_boot=200, seed=0, **estimate_kw) -> pd.DataFrame` — resample whole pulls with replacement within each speed (relabel duplicates as distinct pulls), rerun `estimate`, return the point-estimate g0 with `G0_sd` (sd over resamples, per r_bin, after aligning each resample's G0 to the point estimate by the mean over shared bins).
  - `slice_diagnostics(rows, slices, F_by_source: dict[str, np.ndarray], k, beta) -> pd.DataFrame` — per slice: `speed, step, lam, n_pulls, ess`, `F_<source>` for each source, `F_spread` (max − min over sources, after shifting each to 0 at its first common slice), `lag_mean` (mean of λ − r_after), `lag_eq` (dF_jarzynski/dλ / k, `np.gradient` over the slice's speed), `lag_sd`, `lag_sd_eq` (= √(1/(βk))).

- [ ] **Step 1: Write the failing tests**

```python
def test_bootstrap_sd_shrinks_with_more_pulls():
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    small = simulate_pulls(G0, dG0, n_pulls=15, speed=0.005, r_start=0.9, length_nm=0.2, seed=6)
    big = simulate_pulls(G0, dG0, n_pulls=120, speed=0.005, r_start=0.9, length_nm=0.2, seed=7)
    s = _boot(small, n_boot=50); b = _boot(big, n_boot=50)
    assert np.nanmedian(b.G0_sd) < np.nanmedian(s.G0_sd)

def test_harmonic_within_three_sigma():
    ...  # same toy as Task 1 harmonic, n_boot=100; over bins with ess_pulls >= 5:
         # |G0 - ref| <= 3*G0_sd + 0.3 for all bins

def test_lag_diagnostics_match_flat_equilibrium():
    rows = simulate_pulls(*flat(), n_pulls=40, speed=0.001, length_nm=0.1, seed=8)
    d = _diag(rows)
    assert abs(d.lag_mean.mean()) < 0.005               # no mean force on a flat landscape at slow speed
    assert d.lag_sd.mean() == pytest.approx(d.lag_sd_eq.iloc[0], rel=0.15)
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `estimate`, `bootstrap_g0`, `slice_diagnostics`.**
- [ ] **Step 4: Run to verify they pass** (3 passed; earlier HS test files still pass).
- [ ] **Step 5: Commit** — `git commit -m "feat(pulling): HS bootstrap and diagnostics"`

---

### Task 4: `HummerSzaboEstimator` on `SMDData`

**Files:**
- Modify: `autopath/pulling/hummer_szabo.py`
- Modify: `autopath/pulling/__init__.py` (export `HummerSzaboEstimator`, `hs_frame_weights` once Task 5 lands)
- Test: `tests/test_hs_estimator.py`

**Interfaces:**
- Consumes: `SMDData` (`raw_data`, `protocol_grids`, `beta`, `spring_constant`, `realized_speed_map`, `pulling_direction`), `recover_spring_constant`, `trim_results_by_n_samples_support`.
- Produces:
  - `@dataclass HSResult`: `g0: pd.DataFrame` (columns `f_source, scope, path, speed, r_bin, G0, G0_sd, n_rows, ess_pulls`; `scope` ∈ `"all"` — all pulls, all speeds jointly; `"speed"` — one speed, all paths; `"path"` — one path, all speeds; `path`/`speed` NaN where not applicable), `row_weights: pd.DataFrame` (`trajname, speed, step, r, path, f_source, logw, logw_path`; `logw` from scope `"all"`, normalised so Σ exp = 1 per f_source; `logw_path` from that row's path, normalised per (path, f_source)), `slices: pd.DataFrame` (Task 3 diagnostics, scope `"all"`), `excluded: dict` (`n_rows_unsupported`, `n_rows_no_F` per source, `n_pulls_dropped`).
  - `class HummerSzaboEstimator` with `name = "hummer_szabo"`, `__init__(self, f_sources=F_SOURCES, slice_weights="ess", bin_nm=0.02, n_boot=200, seed=0, wham_stride=1)`, and `fit(self, smd_data: SMDData) -> HSResult`.
    `fit` steps: refuse backward pulls and pulls whose per-move dx differs from the speed's median by > 1e-6 nm (`"not supported:"`); k = `smd_data.spring_constant` or `recover_spring_constant(raw_data)`; trim a copy of `raw_data` with `trim_results_by_n_samples_support(min_samples=2, min_support_ratio=1.0, add_support_columns=False)` with `path` set to one value (joint scope), count the dropped rows; run every source (skip `force` with a logged warning when there is one speed — it is still listed in `excluded` as `"single speed"`); then per speed and per path (each re-trimmed within its own group). Never writes to `smd_data.results`.

- [ ] **Step 1: Write the failing tests** (toy rows → `write_dat` → real `SMDData`)

```python
def _smd(tmp_path, rows, header_k=True):
    files = write_dat(rows, tmp_path, header_k=header_k)
    return SMDData(files, temperature=300.0, completion_threshold_nm=0.0)

def test_autostop_pulls_are_trimmed_and_recover_g0(tmp_path):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=40, speed=0.005, r_start=0.9, length_nm=0.2,
                          stop_steps=[150] * 5 + [None] * 35, seed=9)
    res = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=20).fit(_smd(tmp_path, rows))
    assert res.excluded["n_rows_unsupported"] > 0
    assert res.row_weights.step.max() <= 150
    g = res.g0.query("scope == 'all'"); m = g.ess_pulls >= 5
    ref = G0(g.r_bin[m]); ref -= ref.min()
    assert np.max(np.abs(g.G0[m] - ref)) < 1.0

def test_missing_k_header_is_recovered(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=10, speed=0.01, length_nm=0.1, seed=10)
    res = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=5).fit(_smd(tmp_path, rows, header_k=False))
    assert np.isfinite(res.row_weights.logw).all()

def test_own_lambda0_jitter_stays_within_error(tmp_path):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=60, speed=0.005, r_start=0.9, length_nm=0.2,
                          lam0="own", r0_jitter_nm=0.02, seed=11)
    g = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=50).fit(_smd(tmp_path, rows)).g0.query("scope == 'all'")
    m = g.ess_pulls >= 5; ref = G0(g.r_bin[m]); ref -= ref.min()
    assert np.all(np.abs(g.G0[m] - ref) <= 3 * g.G0_sd[m] + 0.3)

def test_two_paths_joint_matches_truth_and_reports_per_path(tmp_path):
    ...  # two toy groups on the same harmonic G0 tagged path=1/2 via a path map applied after SMDData load;
         # scope 'all' recovers G0 (< 1.0); g0 has scope 'path' rows for both paths

def test_results_table_untouched(tmp_path):
    smd = _smd(tmp_path, simulate_pulls(*flat(), n_pulls=6, speed=0.01, length_nm=0.1))
    before = None if smd.results is None else smd.results.copy()
    HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=5).fit(smd)
    assert (smd.results is None and before is None) or smd.results.equals(before)

def test_mixed_step_size_within_speed_is_not_supported(tmp_path):     # Review Focus 1
    a = simulate_pulls(*flat(), n_pulls=3, speed=0.01, dx=0.001, length_nm=0.1, seed=12)
    b = simulate_pulls(*flat(), n_pulls=3, speed=0.01, dx=0.0012, length_nm=0.1, seed=13)
    with pytest.raises(ValueError, match="not supported"):
        HummerSzaboEstimator(f_sources=("jarzynski",)).fit(_smd(tmp_path, _relabel(a, b)))

def test_backward_pulls_are_not_supported(tmp_path):                   # Review Focus 2
    ...  # write_dat with a "_backward.dat" suffix (add `direction="forward"` param to write_dat) -> raises
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `HSResult` and `HummerSzaboEstimator.fit`.**
- [ ] **Step 4: Run to verify they pass** (7 passed) and rerun all `tests/test_hs_*.py`.
- [ ] **Step 5: Commit** — `git commit -m "feat(pulling): HummerSzaboEstimator on SMDData"`

---

### Task 5: Per-frame weights for DCD frames

**Files:**
- Modify: `autopath/pulling/hummer_szabo.py`
- Test: `tests/test_hs_frames.py`

**Interfaces:**
- Consumes: `HummerSzaboEstimator`, `SMDData.traj_files` (the `_aligned.dcd` next to each `.dat`).
- Produces:
  - `frame_steps(n_rows: int, n_frames: int, save_freq: int) -> np.ndarray` — `(j+1)*save_freq - 1` for j in range(n_frames); raises `"not supported: ..."` unless `n_frames == n_rows // save_freq`.
  - `hs_frame_weights(dat_files: list[str], temperature: float = 300.0, f_source: str = "jarzynski", save_freq: int = 5, slice_weights: str = "ess", bin_nm: float = 0.02) -> pd.DataFrame` — builds `SMDData(dat_files, temperature=temperature)`, sets one path, runs `HummerSzaboEstimator(f_sources=(f_source,), n_boot=0, ...)`, reads each DCD's frame count with `MDAnalysis.lib.formats.libdcd.DCDFile(path).n_frames`, and returns `trajname, frame, step, logw` (logw from `row_weights.logw`, renormalised over the returned frames). Frames whose step was trimmed are omitted; `df.attrs["n_frames_excluded"]` holds their count. `n_boot=0` must skip the bootstrap (`G0_sd` NaN) — add that to `bootstrap_g0` if Task 3 didn't.

- [ ] **Step 1: Write the failing tests**

```python
def test_frame_steps_mapping():
    np.testing.assert_array_equal(frame_steps(1201, 240, 5)[:3], [4, 9, 14])
    assert frame_steps(1496, 299, 5)[-1] == 1494

def test_frame_count_mismatch_is_not_supported():
    with pytest.raises(ValueError, match="not supported"):
        frame_steps(1201, 241, 5)

def test_frame_weights_match_row_weights(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=6, speed=0.01, length_nm=0.1, stop_steps=[60] + [None] * 5, seed=14)
    files = write_dat(rows, tmp_path)
    _write_dcds(files, save_freq=5)           # test helper: DCDFile write mode, n_rows // 5 frames of 1 atom
    fw = hs_frame_weights(files, save_freq=5)
    assert fw.attrs["n_frames_excluded"] > 0
    assert set(fw.step) <= set(range(4, 61, 5))
    np.testing.assert_allclose(np.exp(fw.logw).sum(), 1.0)
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `frame_steps` and `hs_frame_weights`; export both from `autopath/pulling/__init__.py`.**
- [ ] **Step 4: Run to verify they pass** (3 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(pulling): HS per-frame DCD weights"`

---

### Task 6: Opt-in stage in `SMDAnalysis.run()`

**Files:**
- Modify: `autopath/pulling/AnalysisSMD.py` (`__init__` signature at line ~95; `run()` right after `self.mixture_pmfs.to_csv(f'{self.outdir}/mixture_pmfs.csv', index=False)`)
- Test: `tests/test_hs_analysis_stage.py`

**Interfaces:**
- Consumes: `HummerSzaboEstimator`, `HSResult`.
- Produces: `SMDAnalysis(..., hummer_szabo: dict | None = None)` — `None` (default) skips HS; a dict is passed as kwargs to `HummerSzaboEstimator`. New method `_run_hummer_szabo(self, sMDDdata: SMDData) -> HSResult | None` sets `self.hs_result` and writes `{outdir}/hs_g0.csv`, `{outdir}/hs_slices.csv`, `{outdir}/hs_row_weights.csv.gz`, `{outdir}/hs_excluded.json`. `run()` calls it at the point above.

- [ ] **Step 1: Write the failing tests** (call `_run_hummer_szabo` on a toy `SMDData`; `SMDAnalysis` built with `do_plots=False`, `outdir=tmp_path`)

```python
def test_stage_writes_outputs_and_leaves_results(tmp_path):
    ...  # files exist; hs_g0.csv has columns f_source, scope, r_bin, G0, G0_sd; smd.results unchanged

def test_stage_off_by_default(tmp_path):
    ...  # SMDAnalysis(...)._run_hummer_szabo(smd) is None and writes nothing
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full HS suite and the existing sMD tests**

Run: `~/micromamba/envs/autopath/bin/python -m pytest -q tests/test_hs_*.py tests/test_force_estimator.py tests/test_summary_estimators.py tests/test_smd_param_logging.py`
Expected: all pass.

- [ ] **Step 5: Commit** — `git commit -m "feat(pulling): opt-in HS stage in SMDAnalysis"`

---

### Task 7: trypsin/3ptb comparison (analysis only, outside the repo)

**Files:**
- Create: `/gpfs/group/forli/mllanos/autopath_benchmarking/trypsin/3ptb/hs_validation/run_hs_validation.py` (scratch script, not committed; writes only inside `hs_validation/`)

**Interfaces:**
- Consumes: `HummerSzaboEstimator`, `CumulantEstimator.estimate_dG`, `JarzynskiEstimator.estimate_dG`, `ForceEstimator.estimate_dG`, `extrapolate_to_v0`.

- [ ] **Step 1: Write the script.** Load `SMDData(glob('.../3ptb/sMD/trajectories/sMD_*_forward.dat'), temperature=300.0)` (default conditioning), set one path. Run `HummerSzaboEstimator(n_boot=200)` (all four sources). On the same trimmed rows compute G(λ) per speed from Jarzynski, cumulant and force-mean-work, and the force v→0 G(λ). Write `hs_g0.csv`, `hs_slices.csv`, `reference_profiles.csv`, `excluded.json`, and one figure `hs_vs_estimators.png` (panel 1: HS G0(r), four sources with ±sd bands, scope "all"; panel 2: G(λ) Jarzynski/cumulant/force-v0 per speed; panel 3: per-slice ESS and F_spread vs λ; panel 4: lag_mean vs lag_eq and lag_sd vs lag_sd_eq). Log wall time.

- [ ] **Step 2: Run it** (CPU only; if it exceeds ~10 min interactively, submit with `sbatch --partition=<cpu partition> --cpus-per-task=4`, no GPU).

Run: `cd /gpfs/group/forli/mllanos/autopath_benchmarking/trypsin/3ptb/hs_validation && PYTHONPATH=~/forlilab/autopath/.claude/worktrees/hs-estimator ~/micromamba/envs/autopath/bin/python run_hs_validation.py`
Expected: the four CSVs, `excluded.json` and the figure; no exception.

- [ ] **Step 3: Write the verdict for the user** (chat, numbers first): supported r range after trimming; per-source G0 barrier and bound→plateau ΔG with sd; pairwise F_spread max; ESS where it drops below 5; lag excess (lag_mean − lag_eq) vs speed; HS G0(r) vs Jarzynski/cumulant/force-v0 G(λ) differences; the cumulant double-well deviation from Task 2. Stop there — nothing feeds the learned CV until the user has seen this.

---

## Follow-up (separate change, not in this plan)

`pocket-cv2` / `poseidon`: delete `autopath/metadynamics/learned/hs.py` and call `autopath.pulling.hs_frame_weights` per speed.
