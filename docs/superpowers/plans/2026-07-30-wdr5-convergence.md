# WDR5 Multi-Estimator Convergence Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one notebook that reports WDR5 sMD convergence for cumulant, jarzynski and force estimators, with replica-count and cost axes, replacing duplicated convergence code in two existing notebooks.

**Architecture:** A scratch-side library (`wdr5_conv_lib.py`) wraps the deployed `SMDAnalysis.check_convergence` verbatim for cumulant/jarzynski and adds a multi-speed force ladder that reuses the same private rung helpers plus `extrapolate_to_v0`. A CLI driver runs one (system, mode) pair per SLURM array task and writes per-pair CSVs. A thin notebook loads those CSVs and produces all figures.

**Tech Stack:** Python 3.12, `autopath` package (micromamba env `autopath`), pandas/numpy, matplotlib/seaborn, pytest 9.1.1, SLURM.

## Global Constraints

- **Env:** all Python runs use `/gpfs/home/mllanos/micromamba/envs/autopath/bin/python`. The `cosolvkit` env lacks `autopath` and must not be used here.
- **No changes to the `autopath` package.** Reading and calling its private helpers is allowed; editing files under `autopath/` is not.
- **`scratch/*` is gitignored** (`.gitignore:1`). Files created under `scratch/paper_figures/` are therefore untracked. Commit steps in this plan apply to `docs/` only. Do **not** `git add -f` scratch files unless the user asks.
- **Data root:** `/gpfs/group/forli/mllanos/autopath_benchmarking/WDR5/openFF`
- **Systems:** `6dy7_A 6dya_A 6e1y_A 6e1z_A 6e22_A 6e23_A`
- **Modes:** `murcko lig_ha contacts lig_com inertia pocket_com` (directory prefix `sMD-`)
- **Speeds:** `0.005 0.01 0.015` — all 36 (system, mode) pairs have all three.
- **Deployment selections (verbatim):**
  - `POCKET = '(resid 218 219 262 263 305 306 49 50 91 92 133 134 175 176) and backbone'`
  - `LIG_HA = "(resname UNK) and not name H*"`
  - `seed=42`, `temperature=300`
- **Analysis route:** `geom_features=True, merge_features=False` (`analysis_traces`). The `dist_*` cache is v0.015-only and cannot support multi-speed clustering.
- **Tolerances are introspected, never hardcoded** — from `SMDAnalysis.check_convergence` signature defaults.
- **Censoring is never hidden.** Non-converging cells are `NaN` + `censored=True`, and medians are reported as "≥K".
- **Reference fixture** (from a verified probe run, `6dy7_A` / `murcko` / v=0.015 / cumulant, all 50 replicas): 47 rows; `k=46 → rmsd 2.077676, barrier_delta 2.677670, r_ts_delta 0.000619, converged True`; `k=47 → rmsd 1.880280, barrier_delta 1.223554, r_ts_delta 0.001237, converged True`; `k=48 → converged False`.

## Design decision made during planning (not in the spec)

The extrapolated v→0 PMF has **no force profile at v=0**, so deployment's
`boundary_method="force_plateau"` cannot be applied to it directly. Resolution:

- **RMSD r-cap** for the force ladder uses the force-plateau boundary computed
  from the **slowest speed (0.005)**, as the closest available proxy to v→0.
- **barrier/r_ts** for the extrapolated PMF use `boundary_method="pmf_peak"`
  (`_compute_barrier_rts` with `force_df=None`), which needs no force profile.
- When `detect_ts` returns `None`, the value is `NaN`; the existing NaN-aware
  delta logic then waives the criterion (both NaN) or marks non-convergence
  (one NaN → `inf`). No `r_max` fallback is introduced.

This is the one unavoidable deviation from deployment and must be stated in the
notebook's own text so no reader mistakes it for verbatim.

## File Structure

| File | Responsibility |
|---|---|
| `scratch/paper_figures/wdr5_conv_lib.py` | Constants, tolerance introspection, faithful `SMDAnalysis` builder, API wrapper, force ladder, `n_to_converge`, cost extraction |
| `scratch/paper_figures/test_wdr5_conv_lib.py` | pytest tests, including the validation gate |
| `scratch/paper_figures/conv_grid_wdr5.py` | CLI: run one (system, mode), write 2 CSVs |
| `scratch/paper_figures/qfiles_conv/wdr5_conv_array.q` | SLURM array, 36 tasks |
| `scratch/paper_figures/sMD_WDR5_convergence_all.ipynb` | Thin notebook: load CSVs → figures |
| `scratch/paper_figures/sMD_WDR5.ipynb` | Modify: strip convergence cells 11, 13; repoint cell 16 |
| `scratch/paper_figures/sMD_WDR5_convergence.ipynb` | Modify: strip convergence-metric cells |

---

### Task 1: Library constants and tolerance introspection

**Files:**
- Create: `scratch/paper_figures/wdr5_conv_lib.py`
- Test: `scratch/paper_figures/test_wdr5_conv_lib.py`

**Interfaces:**
- Consumes: nothing
- Produces: `DATAFOLDER: str`, `SYSTEMS: list[str]`, `MODES: list[str]`, `SPEEDS: list[float]`, `POCKET: str`, `LIG_HA: str`, `SEED: int`, `TEMPERATURE: float`, `get_tolerances() -> dict[str, float]`

- [ ] **Step 1: Write the failing test**

```python
# scratch/paper_figures/test_wdr5_conv_lib.py
import os
import wdr5_conv_lib as lib


def test_constants_match_deployment():
    assert lib.SPEEDS == [0.005, 0.01, 0.015]
    assert len(lib.SYSTEMS) == 6
    assert len(lib.MODES) == 6
    assert lib.LIG_HA == "(resname UNK) and not name H*"
    assert "backbone" in lib.POCKET
    assert lib.SEED == 42
    assert os.path.isdir(lib.DATAFOLDER)


def test_tolerances_are_introspected_from_the_api():
    tol = lib.get_tolerances()
    assert tol == {
        "dG_weighted-rmsd": 4.0,
        "barrier_delta": 3.0,
        "r_ts_delta": 0.1,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'wdr5_conv_lib'`

- [ ] **Step 3: Write minimal implementation**

```python
# scratch/paper_figures/wdr5_conv_lib.py
"""Shared logic for the centralized WDR5 convergence analysis.

Wraps the deployed autopath convergence check verbatim for cumulant and
jarzynski, and adds a multi-speed ladder for the force estimator.
See docs/superpowers/specs/2026-07-30-wdr5-convergence-design.md.
"""
import inspect

from autopath.pulling import SMDAnalysis

DATAFOLDER = "/gpfs/group/forli/mllanos/autopath_benchmarking/WDR5/openFF"
SYSTEMS = ["6dy7_A", "6dya_A", "6e1y_A", "6e1z_A", "6e22_A", "6e23_A"]
MODES = ["murcko", "lig_ha", "contacts", "lig_com", "inertia", "pocket_com"]
SPEEDS = [0.005, 0.01, 0.015]

POCKET = ("(resid 218 219 262 263 305 306 49 50 91 92 133 134 175 176) "
          "and backbone")
LIG_HA = "(resname UNK) and not name H*"
SEED = 42
TEMPERATURE = 300.0

# Deployment's plateau/boundary settings, mirrored from autopath_core.run().
PLATEAU_FRAC = 0.4
BOUNDARY_BUFFER_FRAC = 0.1


def get_tolerances() -> dict:
    """Read convergence tolerances from check_convergence's signature.

    autopath_core.CONVERGENCE_TOLERANCES is a local variable inside run() and
    therefore not importable; its own comments state the values must match
    these defaults, so the signature is the usable single source of truth.
    """
    sig = inspect.signature(SMDAnalysis.check_convergence)
    return {
        "dG_weighted-rmsd": float(sig.parameters["tol_rmsd"].default),
        "barrier_delta": float(sig.parameters["tol_barrier"].default),
        "r_ts_delta": float(sig.parameters["tol_r_ts"].default),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -v
```
Expected: 2 passed

- [ ] **Step 5: No commit (scratch is gitignored)**

Confirm the files are untracked and leave them so:
```bash
cd /gpfs/home/mllanos/forlilab/autopath
git status --short scratch/ | head
```
Expected: no output (ignored).

---

### Task 2: Deployment-faithful analysis builder and API wrapper

**Files:**
- Modify: `scratch/paper_figures/wdr5_conv_lib.py`
- Test: `scratch/paper_figures/test_wdr5_conv_lib.py`

**Interfaces:**
- Consumes: Task 1 constants, `get_tolerances()`
- Produces:
  - `logs_for(system: str, mode: str) -> list[str]`
  - `build_analysis(system: str, mode: str, outdir: str) -> SMDAnalysis`
  - `conv_api(system: str, mode: str, estimator: str, outdir: str, speeds: list[float] | None = None, logs: list[str] | None = None) -> pd.DataFrame`
    returns rows with `speed, n_replicas, dG_weighted-rmsd, barrier_delta, r_ts_delta, barrier_height, r_ts, converged`

- [ ] **Step 1: Write the failing test**

```python
def test_logs_for_excludes_geom_sidecars():
    logs = lib.logs_for("6dy7_A", "murcko")
    assert len(logs) == 112
    assert all(not f.endswith("_geom.dat") for f in logs)
    assert all(f.endswith("_forward.dat") for f in logs)


def test_conv_api_reproduces_probe_reference(tmp_path):
    """Ground truth from a verified probe run (see plan Global Constraints)."""
    df = lib.conv_api("6dy7_A", "murcko", "cumulant",
                      outdir=str(tmp_path), speeds=[0.015])
    assert len(df) == 47
    row46 = df[df["n_replicas"] == 46].iloc[0]
    assert abs(row46["dG_weighted-rmsd"] - 2.077676) < 1e-4
    assert abs(row46["barrier_delta"] - 2.677670) < 1e-4
    assert bool(row46["converged"]) is True
    row48 = df[df["n_replicas"] == 48].iloc[0]
    assert bool(row48["converged"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k "logs_for or probe_reference" -v
```
Expected: FAIL with `AttributeError: module 'wdr5_conv_lib' has no attribute 'logs_for'`

- [ ] **Step 3: Write minimal implementation**

Append to `wdr5_conv_lib.py`:

```python
import os
from glob import glob

import pandas as pd

from autopath.pulling import SMDData  # noqa: F401  (used by callers/tests)
from autopath.pulling.PathModel import DTWPathModel


def logs_for(system: str, mode: str) -> list[str]:
    """Forward sMD logs for one (system, mode), excluding _geom sidecars."""
    pattern = os.path.join(
        DATAFOLDER, system, f"sMD-{mode}", "trajectories", "sMD_*_forward.dat")
    return sorted(f for f in glob(pattern) if not f.endswith("_geom.dat"))


def equilibrated_pdb(system: str) -> str:
    return os.path.join(
        DATAFOLDER, system, "equilibration", f"{system}_equilibrated.pdb")


def build_analysis(system: str, mode: str, outdir: str) -> SMDAnalysis:
    """Construct SMDAnalysis exactly as autopath_core.run() does for WDR5."""
    os.makedirs(outdir, exist_ok=True)
    cluster_model = DTWPathModel(seed=SEED, do_plots=False, outdir=outdir)
    return SMDAnalysis(
        system, cluster_model,
        estimators=["cumulant", "jarzynski", "force"],
        do_plots=False, seed=SEED, temperature=TEMPERATURE,
        ligand_select=LIG_HA, pocket_select=POCKET, outdir=outdir,
        filter_low_support=True,
        min_samples_per_step=5,
        min_support_ratio=1.0,
        min_replicas_per_path=5,
        min_path_steps_ratio=0.6,
        max_frac_neg_dG_first_half=0.25,
        min_speeds_for_extrapolation=2,
        reference_pdb=equilibrated_pdb(system),
    )


def conv_api(system: str, mode: str, estimator: str, outdir: str,
             speeds: list | None = None,
             logs: list | None = None) -> pd.DataFrame:
    """Verbatim deployment convergence check for cumulant / jarzynski."""
    if estimator not in ("cumulant", "jarzynski"):
        raise ValueError(
            f"conv_api handles cumulant/jarzynski only; got {estimator!r}. "
            "Use conv_force_ladder for the force estimator.")
    sa = build_analysis(system, mode, outdir)
    conv_df, _traces = sa.check_convergence(
        logs=logs if logs is not None else logs_for(system, mode),
        speeds=speeds,
        estimator_name=estimator,
        group_A=LIG_HA,
        group_B=None,
        geom_features=True,
        merge_features=False,
        plateau_frac=PLATEAU_FRAC,
        cluster_to_boundary=True,
        restrict_rmsd_to_boundary=True,
        boundary_buffer_frac=BOUNDARY_BUFFER_FRAC,
        recompute_distances=False,
    )
    conv_df = conv_df.copy()
    conv_df["system"] = system
    conv_df["mode"] = mode
    conv_df["estimator"] = estimator
    return conv_df
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k "logs_for or probe_reference" -v
```
Expected: 2 passed (the reference test takes ~60 s)

- [ ] **Step 5: Record the jarzynski smoke result**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -c "
import tempfile, wdr5_conv_lib as lib
d = lib.conv_api('6dy7_A','murcko','jarzynski',outdir=tempfile.mkdtemp(),speeds=[0.015])
print('rows', len(d), 'first converged k',
      d.loc[d.converged,'n_replicas'].min() if d.converged.any() else None)
"
```
Expected: prints a row count near 47 and a first-converged k (or `None`). Note the value in the notebook later; no assertion here.

---

### Task 3: Cost extraction (MD-ns and robust wall-clock)

**Files:**
- Modify: `scratch/paper_figures/wdr5_conv_lib.py`
- Test: `scratch/paper_figures/test_wdr5_conv_lib.py`

**Interfaces:**
- Consumes: Task 2 `logs_for`
- Produces: `cost_per_replica(system: str, mode: str) -> pd.DataFrame` with columns
  `system, mode, speed, md_ps_per_replica, wall_sec_median, wall_sec_mad, n_timed, n_dropped`

- [ ] **Step 1: Write the failing test**

```python
def test_cost_per_replica_shape_and_md_time():
    cost = lib.cost_per_replica("6dy7_A", "murcko")
    assert set(cost["speed"]) == {0.005, 0.01, 0.015}
    assert list(cost.columns) == [
        "system", "mode", "speed", "md_ps_per_replica",
        "wall_sec_median", "wall_sec_mad", "n_timed", "n_dropped",
    ]
    md = cost.set_index("speed")["md_ps_per_replica"]
    # simulated time is inversely proportional to pulling speed
    assert abs(md[0.005] - 344.0) < 5.0
    assert abs(md[0.015] - 118.0) < 5.0
    assert md[0.005] > md[0.01] > md[0.015]


def test_cost_wall_clock_is_positive_and_robust():
    cost = lib.cost_per_replica("6dy7_A", "murcko")
    fast = cost[cost["speed"] == 0.015].iloc[0]
    assert 5.0 < fast["wall_sec_median"] < 300.0
    assert fast["n_timed"] > 0
    assert fast["n_dropped"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k cost -v
```
Expected: FAIL with `AttributeError: ... has no attribute 'cost_per_replica'`

- [ ] **Step 3: Write minimal implementation**

Append to `wdr5_conv_lib.py`:

```python
import datetime as dt
import re

import numpy as np

_REPLICA_RE = re.compile(r"replica-(\d{6})_v([\d.]+)_forward\.dat$")


def _md_ps_from_log(path: str) -> float:
    """Total simulated time (ps) of one replica: last value of the time column."""
    last = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("step"):
                continue
            if line.strip():
                last = line
    if last is None:
        return float("nan")
    return float(last.split(",")[1])


def _wall_seconds(path: str) -> float:
    """Wall-clock seconds for one replica.

    The filename encodes the start time as HHMMSS and the file's mtime is the
    end time (consecutive replicas run back-to-back on one GPU, so the next
    replica's start equals this one's mtime). Returns NaN if unparseable.
    """
    m = _REPLICA_RE.search(os.path.basename(path))
    if not m:
        return float("nan")
    hhmmss = m.group(1)
    end = dt.datetime.fromtimestamp(os.path.getmtime(path))
    start = end.replace(hour=int(hhmmss[0:2]), minute=int(hhmmss[2:4]),
                        second=int(hhmmss[4:6]), microsecond=0)
    if start > end:                      # run crossed midnight
        start -= dt.timedelta(days=1)
    return (end - start).total_seconds()


def _mad_fence(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Keep values within k MADs of the median; always keep >0 values only."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return v
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return v
    return v[np.abs(v - med) <= k * mad]


def cost_per_replica(system: str, mode: str) -> pd.DataFrame:
    """Per-replica cost at each speed: exact MD time and robust wall-clock."""
    rows = []
    for speed in SPEEDS:
        speed_logs = [f for f in logs_for(system, mode)
                      if f"_v{speed}_" in os.path.basename(f)]
        if not speed_logs:
            continue
        md_vals = np.array([_md_ps_from_log(f) for f in speed_logs], dtype=float)
        wall_raw = np.array([_wall_seconds(f) for f in speed_logs], dtype=float)
        wall_keep = _mad_fence(wall_raw)
        n_finite = int(np.sum(np.isfinite(wall_raw) & (wall_raw > 0)))
        rows.append({
            "system": system,
            "mode": mode,
            "speed": speed,
            "md_ps_per_replica": float(np.nanmedian(md_vals)),
            "wall_sec_median": float(np.median(wall_keep)) if wall_keep.size else float("nan"),
            "wall_sec_mad": float(np.median(np.abs(wall_keep - np.median(wall_keep))))
                            if wall_keep.size else float("nan"),
            "n_timed": int(wall_keep.size),
            "n_dropped": int(n_finite - wall_keep.size),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k cost -v
```
Expected: 2 passed

---

### Task 4: Force ladder (multi-speed)

**Files:**
- Modify: `scratch/paper_figures/wdr5_conv_lib.py`
- Test: `scratch/paper_figures/test_wdr5_conv_lib.py`

**Interfaces:**
- Consumes: Task 2 `build_analysis`, `logs_for`
- Produces: `conv_force_ladder(system: str, mode: str, outdir: str, speeds: list[float] | None = None, logs: list[str] | None = None) -> pd.DataFrame`
  returning the same columns as `conv_api` with `speed="ALL"`.

**Background the implementer needs:** `_results_from_running_stats` dispatches
through `ESTIMATOR_REGISTRY`, which already contains `'force'`, so the per-speed
rung machinery works unchanged. Only the cross-speed v→0 step is new.
`extrapolate_to_v0(results, param='dG_weighted', min_speeds=2)` joins on `step`
and uses only steps present at every included speed.

- [ ] **Step 1: Write the failing test**

```python
def test_force_ladder_produces_extrapolated_rungs(tmp_path):
    df = lib.conv_force_ladder("6dy7_A", "murcko", outdir=str(tmp_path))
    assert len(df) > 5
    assert set(df["speed"]) == {"ALL"}
    assert df["estimator"].unique().tolist() == ["force"]
    # ladder depth is capped by the thinnest speed (25 at v=0.01 for this pair)
    assert df["n_replicas"].max() <= 25
    assert df["n_replicas"].is_monotonic_increasing
    for col in ("dG_weighted-rmsd", "barrier_delta", "r_ts_delta", "converged"):
        assert col in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k force_ladder -v
```
Expected: FAIL with `AttributeError: ... has no attribute 'conv_force_ladder'`

- [ ] **Step 3: Write minimal implementation**

Append to `wdr5_conv_lib.py`:

```python
from collections import defaultdict

from autopath.pulling.Estimators import KramersEstimator, extrapolate_to_v0
from autopath.pulling.support import SupportPolicy


def _speed_state(sa, smd, speed, trim_fraction=0.1):
    """Per-speed setup mirroring check_convergence: boundary, clustering.

    Returns
    -------
    (protocol_grid, boundary_cap, speed_data)
        ``protocol_grid`` : Series step -> r_target_protocol
        ``boundary_cap``  : float | None, force-plateau boundary + buffer
        ``speed_data``    : raw_data rows for this speed, with 'path' assigned
    """
    protocol_grid = smd.protocol_grids[speed].set_index("step")["r_target_protocol"]

    boundary_cap = None
    rd = smd.raw_data[smd.raw_data["speed"] == speed]
    if {"r_coord", "force", "speed"}.issubset(rd.columns):
        r_lo, r_hi = float(rd["r_coord"].min()), float(rd["r_coord"].max())
        r_ts = KramersEstimator.force_plateau_boundary(
            rd, speed, r_lo, r_hi, PLATEAU_FRAC)
        if r_ts is not None:
            boundary_cap = min(r_ts * (1.0 + BOUNDARY_BUFFER_FRAC), r_hi)

    feat_df, _ = sa._build_cluster_feature_df(
        smd, group_A=LIG_HA, group_B=None, features=None,
        merge_features=False, ligand_sdf=None,
        geom_features=True, geom_merge="aligned", recompute_distances=False,
    )
    feat_df_cl, fit_r_range = feat_df, 1 - trim_fraction
    if boundary_cap is not None:
        r_of_step = protocol_grid.reindex(feat_df["step"]).to_numpy(dtype=float)
        restricted = feat_df[r_of_step <= boundary_cap]
        if len(restricted) >= 2:
            feat_df_cl, fit_r_range = restricted, None

    clusterer = DTWPathModel(seed=SEED, do_plots=False, outdir=sa.outdir)
    mappings = clusterer.fit_transform(feat_df_cl, r_range=fit_r_range)
    smd.raw_data["path"] = smd.raw_data["trajname"].map(mappings)

    speed_data = smd.raw_data[smd.raw_data["speed"] == speed].copy()
    return protocol_grid, boundary_cap, speed_data


def conv_force_ladder(system: str, mode: str, outdir: str,
                      speeds: list | None = None,
                      logs: list | None = None,
                      trace_min_replicas: int = 3,
                      trim_fraction: float = 0.1,
                      min_common_points: int = 5) -> pd.DataFrame:
    """Multi-speed convergence ladder for the force estimator.

    At rung k, each speed contributes its first k replicas; per-speed weighted
    PMFs are extrapolated to v->0 and consecutive extrapolated PMFs compared
    with deployment's three criteria.
    """
    speeds = speeds or SPEEDS
    logs = logs if logs is not None else logs_for(system, mode)
    tol = get_tolerances()
    sa = build_analysis(system, mode, outdir)

    conv_policy = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)

    # Per-speed state: clustering, ordering, protocol grid, boundary.
    per_speed = {}
    for speed in speeds:
        speed_logs = sorted([f for f in logs if SMDData._speed_from_log(f) == speed],
                            key=SMDData._replica_idx_from_log)
        smd = SMDData(speed_logs, sysname=system, temperature=TEMPERATURE,
                      reference_pdb=equilibrated_pdb(system))
        protocol_grid, boundary_cap, speed_data = _speed_state(
            sa, smd, speed, trim_fraction)
        traj_order = [os.path.basename(f)[:-4] for f in speed_logs]
        per_speed[speed] = {
            "smd": smd,
            "grid": protocol_grid,
            "cap": boundary_cap,
            "data": speed_data,
            "order": traj_order,
            "stats": defaultdict(lambda: {"n": 0, "sum_w": 0.0,
                                          "sum_w2": 0.0, "sum_exp": 0.0}),
            "samples": defaultdict(list),
            "counts": defaultdict(int),
        }

    K = min(len(per_speed[s]["order"]) for s in speeds)
    # RMSD cap for the extrapolated PMF: use the slowest speed's boundary
    slow_cap = per_speed[min(speeds)]["cap"]
    slow_grid = per_speed[min(speeds)]["grid"]

    rows = []
    prev_pmf = prev_barrier = prev_r_ts = None

    for k in range(1, K + 1):
        frames = []
        for speed in speeds:
            st = per_speed[speed]
            trajname = st["order"][k - 1]
            tdf = st["data"][st["data"]["trajname"] == trajname]
            if tdf.empty:
                continue
            paths = tdf["path"].dropna()
            if paths.empty:
                continue
            st["counts"][paths.iloc[0]] += 1
            for _, row in tdf.iterrows():
                key = (int(row["step"]), row["path"])
                s = st["stats"][key]
                w = float(row["work"])
                s["n"] += 1
                s["sum_w"] += w
                s["sum_w2"] += w * w
                s["sum_exp"] += np.exp(-st["smd"].beta * w)
                st["samples"][key].append(w)

            if k < trace_min_replicas:
                continue

            res = sa._results_from_running_stats(
                running_stats=st["stats"], running_samples=st["samples"],
                speed=speed, protocol_grid=st["grid"],
                estimator_name="force", beta=st["smd"].beta, policy=conv_policy,
            )
            if res.empty:
                continue
            series = sa._weighted_series_from_results(
                results_df=res, path_traj_counts=st["counts"],
                value_col="dG", beta=st["smd"].beta,
                trim_fraction=trim_fraction, policy=conv_policy,
            )
            if series.empty:
                continue
            frames.append(pd.DataFrame({
                "step": series.index.astype(int),
                "r_coord": st["grid"].reindex(series.index).to_numpy(),
                "speed": speed,
                "estimator": "force",
                "dG_weighted": series.to_numpy(),
            }))

        if k < trace_min_replicas or len(frames) < 2:
            continue

        extrap = extrapolate_to_v0(pd.concat(frames, ignore_index=True),
                                   param="dG_weighted", speeds=speeds,
                                   min_speeds=2)
        if extrap is None or extrap.empty:
            continue
        pmf_k = (extrap.set_index("step")["dG_weighted"]
                 .astype(float).sort_index())
        if pmf_k.empty:
            continue

        barrier, r_ts = sa._compute_barrier_rts(
            pmf_k, 1.0 / per_speed[min(speeds)]["smd"].beta, slow_grid,
            force_df=None, speed=0.0,
            boundary_method="pmf_peak", plateau_frac=PLATEAU_FRAC,
        )

        if prev_pmf is None:
            prev_pmf, prev_barrier, prev_r_ts = pmf_k, barrier, r_ts
            continue

        common = pmf_k.index.intersection(prev_pmf.index)
        if slow_cap is not None and len(common):
            r_of_step = slow_grid.reindex(common).to_numpy(dtype=float)
            common = common[r_of_step <= slow_cap]

        rows.append(_compare_rung(
            k=k, pmf_k=pmf_k, prev_pmf=prev_pmf,
            barrier=barrier, prev_barrier=prev_barrier,
            r_ts=r_ts, prev_r_ts=prev_r_ts,
            common=common, tol=tol, min_common_points=min_common_points,
            extra={"system": system, "mode": mode,
                   "estimator": "force", "speed": "ALL"},
        ))
        prev_pmf, prev_barrier, prev_r_ts = pmf_k, barrier, r_ts

    return pd.DataFrame(rows)


def _compare_rung(k, pmf_k, prev_pmf, barrier, prev_barrier, r_ts, prev_r_ts,
                  common, tol, min_common_points, extra):
    """Deployment's three-criterion rung comparison.

    Shared by the force ladder and by the single-speed validation harness, so
    the gate in validate_against_api exercises exactly this code.
    NaN handling mirrors check_convergence: both NaN waives the criterion,
    one NaN marks non-convergence (inf).
    """
    if np.isnan(barrier) and np.isnan(prev_barrier):
        barrier_delta = np.nan
    elif np.isnan(barrier) or np.isnan(prev_barrier):
        barrier_delta = np.inf
    else:
        barrier_delta = abs(barrier - prev_barrier)

    if np.isnan(r_ts) and np.isnan(prev_r_ts):
        r_ts_delta = np.nan
    elif np.isnan(r_ts) or np.isnan(prev_r_ts):
        r_ts_delta = np.inf
    else:
        r_ts_delta = abs(r_ts - prev_r_ts)

    if len(common) < min_common_points:
        rmsd, converged, reason = np.nan, False, "insufficient_overlap"
    else:
        rmsd = float(np.sqrt(np.mean(
            (pmf_k.loc[common].to_numpy() - prev_pmf.loc[common].to_numpy()) ** 2)))
        converged = bool(
            (rmsd < tol["dG_weighted-rmsd"]) and
            (np.isnan(barrier_delta) or barrier_delta < tol["barrier_delta"]) and
            (np.isnan(r_ts_delta) or r_ts_delta < tol["r_ts_delta"]))
        reason = ""

    row = {"path": "mixture", "n_replicas": k,
           "dG_weighted-rmsd": rmsd, "barrier_delta": barrier_delta,
           "r_ts_delta": r_ts_delta, "barrier_height": barrier, "r_ts": r_ts,
           "converged": converged, "reason": reason,
           "n_common_points": len(common)}
    row.update(extra)
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k force_ladder -v
```
Expected: 1 passed. If `_weighted_series_from_results` rejects `value_col="dG"`,
inspect its signature with
`/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -c "import inspect;from autopath.pulling import SMDAnalysis;print(inspect.signature(SMDAnalysis._weighted_series_from_results))"`
and pass the column name it expects; do not change package code.

---

### Task 5: Validation gate — ladder machinery reproduces the API on cumulant

**Files:**
- Modify: `scratch/paper_figures/wdr5_conv_lib.py`
- Test: `scratch/paper_figures/test_wdr5_conv_lib.py`

**Interfaces:**
- Consumes: Task 2 `conv_api`, Task 4 internals
- Produces: `first_consecutive_converged(df: pd.DataFrame, k_consec: int = 2) -> float` (NaN when never converged), `validate_against_api(system: str, mode: str, outdir: str) -> dict`

This is the gate that makes the force numbers trustworthy: the ladder's own
comparison arithmetic is exercised on **single-speed cumulant**, where
`check_convergence` provides ground truth.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd


def test_first_consecutive_converged_rules():
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6, 7],
                       "converged": [False, True, False, True, True]})
    assert lib.first_consecutive_converged(df, k_consec=2) == 7
    assert np.isnan(lib.first_consecutive_converged(
        pd.DataFrame({"n_replicas": [3, 4], "converged": [False, False]}),
        k_consec=2))


def test_first_consecutive_k3_is_stricter_than_k2():
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6],
                       "converged": [True, True, False, True]})
    assert lib.first_consecutive_converged(df, k_consec=2) == 4
    assert np.isnan(lib.first_consecutive_converged(df, k_consec=3))


def test_validation_gate_is_not_vacuous():
    """The harness must be a real reimplementation, not an alias for conv_api."""
    assert lib._cumulant_via_ladder_machinery is not lib.conv_api


def test_validation_gate_matches_api_on_cumulant(tmp_path):
    rep = lib.validate_against_api("6dy7_A", "murcko", outdir=str(tmp_path))
    assert rep["n_rows_api"] == rep["n_rows_mine"]
    assert rep["max_abs_rmsd_diff"] < 1e-6
    assert rep["n_to_converge_match"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k "consecutive or validation_gate" -v
```
Expected: FAIL with `AttributeError: ... has no attribute 'first_consecutive_converged'`

- [ ] **Step 3: Write minimal implementation**

Append to `wdr5_conv_lib.py`:

```python
def first_consecutive_converged(df: pd.DataFrame, k_consec: int = 2) -> float:
    """Smallest n_replicas ending a run of k_consec consecutive True rungs.

    Returns NaN when the criterion is never met (a censored observation, which
    callers must report rather than drop).
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


def _cumulant_via_ladder_machinery(system, mode, outdir, speed,
                                   trace_min_replicas=3, trim_fraction=0.1,
                                   min_common_points=5):
    """Run the ladder's own loop on ONE speed with cumulant.

    Identical to conv_force_ladder except that a single speed means no v->0
    extrapolation, and barrier/r_ts use force_plateau with that speed's force
    data -- exactly what check_convergence does. Any divergence from the API
    therefore localises to this shared machinery, which is the point of the gate.
    """
    tol = get_tolerances()
    sa = build_analysis(system, mode, outdir)
    conv_policy = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)

    speed_logs = sorted(
        [f for f in logs_for(system, mode)
         if SMDData._speed_from_log(f) == speed],
        key=SMDData._replica_idx_from_log)
    smd = SMDData(speed_logs, sysname=system, temperature=TEMPERATURE,
                  reference_pdb=equilibrated_pdb(system))
    grid, cap, speed_data = _speed_state(sa, smd, speed, trim_fraction)
    order = [os.path.basename(f)[:-4] for f in speed_logs]

    stats = defaultdict(lambda: {"n": 0, "sum_w": 0.0,
                                 "sum_w2": 0.0, "sum_exp": 0.0})
    samples, counts = defaultdict(list), defaultdict(int)
    rows, prev_pmf, prev_barrier, prev_r_ts = [], None, None, None

    for k, trajname in enumerate(order, start=1):
        tdf = speed_data[speed_data["trajname"] == trajname]
        if tdf.empty or tdf["path"].dropna().empty:
            continue
        counts[tdf["path"].dropna().iloc[0]] += 1
        for _, row in tdf.iterrows():
            key = (int(row["step"]), row["path"])
            s, w = stats[key], float(row["work"])
            s["n"] += 1
            s["sum_w"] += w
            s["sum_w2"] += w * w
            s["sum_exp"] += np.exp(-smd.beta * w)
            samples[key].append(w)

        if k < trace_min_replicas:
            continue

        res = sa._results_from_running_stats(
            running_stats=stats, running_samples=samples, speed=speed,
            protocol_grid=grid, estimator_name="cumulant",
            beta=smd.beta, policy=conv_policy)
        if res.empty:
            continue
        pmf_k = sa._weighted_series_from_results(
            results_df=res, path_traj_counts=counts, value_col="dG",
            beta=smd.beta, trim_fraction=trim_fraction, policy=conv_policy)
        if pmf_k.empty:
            continue

        force_k = speed_data[speed_data["trajname"].isin(order[:k])]
        barrier, r_ts = sa._compute_barrier_rts(
            pmf_k, 1.0 / smd.beta, grid, force_df=force_k, speed=speed,
            boundary_method="force_plateau", plateau_frac=PLATEAU_FRAC)

        if prev_pmf is None:
            prev_pmf, prev_barrier, prev_r_ts = pmf_k, barrier, r_ts
            continue

        common = pmf_k.index.intersection(prev_pmf.index)
        if cap is not None and len(common):
            common = common[grid.reindex(common).to_numpy(dtype=float) <= cap]

        rows.append(_compare_rung(
            k=k, pmf_k=pmf_k, prev_pmf=prev_pmf,
            barrier=barrier, prev_barrier=prev_barrier,
            r_ts=r_ts, prev_r_ts=prev_r_ts, common=common, tol=tol,
            min_common_points=min_common_points,
            extra={"system": system, "mode": mode,
                   "estimator": "cumulant", "speed": speed}))
        prev_pmf, prev_barrier, prev_r_ts = pmf_k, barrier, r_ts

    return pd.DataFrame(rows)


def validate_against_api(system: str, mode: str, outdir: str,
                         speed: float = 0.015) -> dict:
    """Compare ladder-side comparison arithmetic with check_convergence.

    Returns a report dict; callers must refuse to publish force results when
    the tolerances below are exceeded.
    """
    api = conv_api(system, mode, "cumulant", outdir=outdir, speeds=[speed])
    mine = _cumulant_via_ladder_machinery(system, mode, outdir, speed)
    merged = api.merge(mine, on="n_replicas", suffixes=("_api", "_mine"))
    diff = (merged["dG_weighted-rmsd_api"] - merged["dG_weighted-rmsd_mine"]).abs()
    return {
        "n_rows_api": int(len(api)),
        "n_rows_mine": int(len(mine)),
        "max_abs_rmsd_diff": float(np.nanmax(diff)) if len(diff) else float("nan"),
        "n_to_converge_api": first_consecutive_converged(api, 2),
        "n_to_converge_mine": first_consecutive_converged(mine, 2),
        "n_to_converge_match": bool(
            first_consecutive_converged(api, 2)
            == first_consecutive_converged(mine, 2)
            or (np.isnan(first_consecutive_converged(api, 2))
                and np.isnan(first_consecutive_converged(mine, 2)))),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -m pytest test_wdr5_conv_lib.py -k "consecutive or validation_gate" -v
```
Expected: 4 passed, with `max_abs_rmsd_diff < 1e-6`.

**If the gate fails, do not loosen the test.** A mismatch means the ladder's
shared machinery diverges from deployment, so the force results are not
trustworthy. Diagnose by printing the first `n_replicas` where the two differ:

```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -c "
import tempfile, wdr5_conv_lib as lib
d = tempfile.mkdtemp()
a = lib.conv_api('6dy7_A','murcko','cumulant',outdir=d,speeds=[0.015])
m = lib._cumulant_via_ladder_machinery('6dy7_A','murcko',d,0.015)
j = a.merge(m, on='n_replicas', suffixes=('_api','_mine'))
j['d'] = (j['dG_weighted-rmsd_api']-j['dG_weighted-rmsd_mine']).abs()
print(j.loc[j.d>1e-6, ['n_replicas','dG_weighted-rmsd_api','dG_weighted-rmsd_mine']].head())
"
```
Common causes, in order of likelihood: replica ordering differs (must be
`SMDData._replica_idx_from_log`); `trim_fraction` support-trim not applied to
`common`; `value_col` mismatch in `_weighted_series_from_results`.

- [ ] **Step 5: Record the gate result for the notebook**

Run and save the report so Task 7 Step 7 can quote it:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -c "
import json, tempfile, wdr5_conv_lib as lib
rep = lib.validate_against_api('6dy7_A','murcko',outdir=tempfile.mkdtemp())
json.dump(rep, open('validation_gate_report.json','w'), indent=2, default=str)
print(rep)
"
```
Expected: `validation_gate_report.json` written with `max_abs_rmsd_diff` ≈ 0.

---

### Task 6: CLI driver and SLURM array

**Files:**
- Create: `scratch/paper_figures/conv_grid_wdr5.py`
- Create: `scratch/paper_figures/qfiles_conv/wdr5_conv_array.q`

**Interfaces:**
- Consumes: Tasks 2–5 (`conv_api`, `conv_force_ladder`, `cost_per_replica`, `validate_against_api`)
- Produces: `conv_grid/<system>_<mode>.csv`, `cost/<system>_<mode>.csv` under an output root

- [ ] **Step 1: Write the driver**

```python
# scratch/paper_figures/conv_grid_wdr5.py
"""Run the WDR5 convergence grid for one (system, mode) pair.

Usage:
    python conv_grid_wdr5.py --system 6dy7_A --mode murcko --outroot results_conv
    python conv_grid_wdr5.py --index $SLURM_ARRAY_TASK_ID --outroot results_conv
"""
import argparse
import os
import tempfile

import pandas as pd

import wdr5_conv_lib as lib

PAIRS = [(s, m) for s in lib.SYSTEMS for m in lib.MODES]   # 36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system")
    ap.add_argument("--mode")
    ap.add_argument("--index", type=int,
                    help="0-based index into the 36 (system, mode) pairs")
    ap.add_argument("--outroot", default="results_conv")
    args = ap.parse_args()

    if args.index is not None:
        system, mode = PAIRS[args.index]
    else:
        system, mode = args.system, args.mode

    conv_dir = os.path.join(args.outroot, "conv_grid")
    cost_dir = os.path.join(args.outroot, "cost")
    os.makedirs(conv_dir, exist_ok=True)
    os.makedirs(cost_dir, exist_ok=True)

    scratch = tempfile.mkdtemp(prefix=f"conv_{system}_{mode}_")
    frames = []

    for estimator in ("cumulant", "jarzynski"):
        try:
            frames.append(lib.conv_api(system, mode, estimator, outdir=scratch))
        except Exception as exc:                      # noqa: BLE001
            print(f"FAILED {system} {mode} {estimator}: {exc}", flush=True)

    try:
        frames.append(lib.conv_force_ladder(system, mode, outdir=scratch))
    except Exception as exc:                          # noqa: BLE001
        print(f"FAILED {system} {mode} force: {exc}", flush=True)

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(conv_dir, f"{system}_{mode}.csv"), index=False)

    lib.cost_per_replica(system, mode).to_csv(
        os.path.join(cost_dir, f"{system}_{mode}.csv"), index=False)

    print(f"done {system} {mode}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the driver on one pair**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python conv_grid_wdr5.py \
    --system 6dy7_A --mode murcko --outroot /tmp/conv_smoke
head -3 /tmp/conv_smoke/conv_grid/6dy7_A_murcko.csv
cat /tmp/conv_smoke/cost/6dy7_A_murcko.csv
```
Expected: both CSVs exist; the conv CSV contains all three estimators
(`cut -d, -f<estimator col>` shows cumulant, jarzynski, force).

- [ ] **Step 3: Write the SLURM array**

```bash
# scratch/paper_figures/qfiles_conv/wdr5_conv_array.q
#!/bin/bash
#SBATCH --job-name=wdr5_conv
#SBATCH --array=0-35
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --partition=shared,forli
#SBATCH -o logs/wdr5_conv_%A_%a.out
#SBATCH -e logs/wdr5_conv_%A_%a.err

set -euo pipefail
source ~/.bashrc
micromamba activate autopath
export OMP_NUM_THREADS=4

cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
python conv_grid_wdr5.py --index "${SLURM_ARRAY_TASK_ID}" --outroot results_conv
```

- [ ] **Step 4: Submit and confirm**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
mkdir -p logs results_conv
sbatch qfiles_conv/wdr5_conv_array.q
squeue -u $USER -n wdr5_conv -o "%.12i %.9T %.8M %R" | head
```
Expected: 36 array tasks queued. When finished:
```bash
ls results_conv/conv_grid | wc -l   # expect 36
ls results_conv/cost | wc -l        # expect 36
grep -l FAILED logs/*.out || echo "no failures"
```

---

### Task 7: Analysis notebook

**Files:**
- Create: `scratch/paper_figures/sMD_WDR5_convergence_all.ipynb`

**Interfaces:**
- Consumes: `results_conv/conv_grid/*.csv`, `results_conv/cost/*.csv`, `wdr5_conv_lib.first_consecutive_converged`
- Produces: figures under `plots/WDR5/convergence_all/`

Build the notebook with `nbformat` (a script that writes the `.ipynb`), then
execute it with `jupyter nbconvert --to notebook --inplace --execute` using the
**`autopath` env** (it imports `wdr5_conv_lib`, which imports `autopath`).

- [ ] **Step 1: Cell 1 — config and load**

```python
import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import wdr5_conv_lib as lib

RESULTS   = "results_conv"
PLOTFOLDER = "plots/WDR5/convergence_all"
os.makedirs(PLOTFOLDER, exist_ok=True)
K_CONSEC_LIST = [2, 3]          # 2 = deployment rule (headline); 3 = robustness
ESTIMATORS = ["cumulant", "jarzynski", "force"]

conv = pd.concat([pd.read_csv(f) for f in
                  sorted(glob.glob(f"{RESULTS}/conv_grid/*.csv"))],
                 ignore_index=True)
cost = pd.concat([pd.read_csv(f) for f in
                  sorted(glob.glob(f"{RESULTS}/cost/*.csv"))],
                 ignore_index=True)
print(f"conv rows {len(conv)}, pairs {conv.groupby(['system','mode']).ngroups}")
print(f"tolerances in force (introspected): {lib.get_tolerances()}")
```

- [ ] **Step 2: Cell 2 — derive n_to_converge with censoring, for both k_consec**

```python
recs = []
for (system, mode, est, speed), g in conv.groupby(
        ["system", "mode", "estimator", "speed"], dropna=False):
    for kc in K_CONSEC_LIST:
        n = lib.first_consecutive_converged(g, k_consec=kc)
        recs.append({"system": system, "mode": mode, "estimator": est,
                     "speed": speed, "k_consec": kc,
                     "n_to_converge": n, "censored": bool(np.isnan(n)),
                     "n_available": int(g["n_replicas"].max())})
nconv = pd.DataFrame(recs)
cens = nconv.groupby(["estimator", "k_consec"])["censored"].mean().unstack()
print("censoring fraction by estimator / k_consec:\n", cens.round(3))
```

- [ ] **Step 3: Cell 3 — Figure 1, n_to_converge heatmap (estimator x speed)**

```python
for kc in K_CONSEC_LIST:
    sub = nconv[(nconv.k_consec == kc) & (~nconv.censored)]
    piv = sub.pivot_table(index="estimator", columns="speed",
                          values="n_to_converge", aggfunc="median")
    cen = (nconv[nconv.k_consec == kc]
           .pivot_table(index="estimator", columns="speed",
                        values="censored", aggfunc="mean"))
    ann = piv.round(1).astype(str) + "\n(" + (cen * 100).round(0).astype(int).astype(str) + "% cens)"
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(piv, annot=ann, fmt="", cmap="viridis_r", ax=ax)
    ax.set_title(f"Median replicas to converge (k_consec={kc})")
    fig.savefig(f"{PLOTFOLDER}/heatmap_n-to-converge_k{kc}.svg",
                bbox_inches="tight")
    plt.show()
```

- [ ] **Step 4: Cell 4 — Figure 2, cost to converge (MD-ns and wall-clock)**

```python
cost_map = cost.set_index(["system", "mode", "speed"])
def _cost(row, col):
    if row["speed"] == "ALL":                 # force pays for every speed
        return sum(cost_map.loc[(row.system, row["mode"], s), col]
                   for s in lib.SPEEDS)
    return cost_map.loc[(row.system, row["mode"], float(row["speed"])), col]

for kc in K_CONSEC_LIST:
    sub = nconv[(nconv.k_consec == kc) & (~nconv.censored)].copy()
    sub["md_ns"] = sub.apply(lambda r: _cost(r, "md_ps_per_replica"), axis=1) \
                   * sub["n_to_converge"] / 1000.0
    sub["wall_min"] = sub.apply(lambda r: _cost(r, "wall_sec_median"), axis=1) \
                      * sub["n_to_converge"] / 60.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col, lab in zip(axes, ["md_ns", "wall_min"],
                            ["MD ns to converge", "wall-clock min to converge"]):
        piv = sub.pivot_table(index="estimator", columns="speed",
                              values=col, aggfunc="median")
        sns.heatmap(piv, annot=True, fmt=".1f", cmap="magma_r", ax=ax)
        ax.set_title(f"{lab} (k_consec={kc})")
    fig.savefig(f"{PLOTFOLDER}/heatmap_cost-to-converge_k{kc}.svg",
                bbox_inches="tight")
    plt.show()
```

- [ ] **Step 5: Cell 5 — Figure 3, Pareto vs pKd quality**

pKd values are copied verbatim from `sMD_WDR5.ipynb` (`systems_micromolar`).
Note the keys there are 4-character PDB codes without the `_A` chain suffix
used in directory names, so map with `system.split("_")[0]`.

```python
from glob import glob
from scipy.stats import spearmanr

systems_micromolar = {"6e1z": 150.0, "6dy7": 66.2, "6dya": 0.681,
                      "6e1y": 0.022, "6e22": 0.0013, "6e23": 0.0001}
systems_pkd = {k: -np.log10(v * 1e-6) for k, v in systems_micromolar.items()}

# Terminal extrapolated dG per (system, mode, estimator), v->0 rows
rows = []
for f in glob(os.path.join(lib.DATAFOLDER, "*/*/analysis_traces/dG_extrapolated.csv")):
    parts = os.path.relpath(f, lib.DATAFOLDER).split("/")
    system, mode_dir = parts[0], parts[1]
    d = pd.read_csv(f)
    d = d[d["speed"] == 0] if "speed" in d.columns else d
    for est, g in d.groupby("estimator"):
        g = g.sort_values("r_coord")
        rows.append({"system": system, "mode": mode_dir.replace("sMD-", ""),
                     "estimator": est, "dG_end": g["dG"].iloc[-1]})
dg = pd.DataFrame(rows)
dg["pKd"] = dg["system"].str.split("_").str[0].map(systems_pkd)

rho_by_est = {}
for est, g in dg.dropna(subset=["pKd", "dG_end"]).groupby("estimator"):
    per_mode = [spearmanr(gm["dG_end"], gm["pKd"]).statistic
                for _, gm in g.groupby("mode") if gm["system"].nunique() >= 4]
    if per_mode:
        rho_by_est[est] = float(np.nanmedian(per_mode))
print("median Spearman rho (dG_end vs pKd) by estimator:", rho_by_est)

fig, ax = plt.subplots(figsize=(6.5, 5))
for kc, marker in zip(K_CONSEC_LIST, ["o", "s"]):
    sub = nconv[(nconv.k_consec == kc) & (~nconv.censored)].copy()
    sub["md_ns"] = sub.apply(lambda r: _cost(r, "md_ps_per_replica"), axis=1) \
                   * sub["n_to_converge"] / 1000.0
    med = sub.groupby("estimator")["md_ns"].median()
    for est in ESTIMATORS:
        if est in med.index and est in rho_by_est:
            ax.scatter(med[est], rho_by_est[est], s=140, marker=marker)
            ax.annotate(f"{est} (k={kc})", (med[est], rho_by_est[est]),
                        textcoords="offset points", xytext=(8, 4), fontsize=10)
ax.set_xlabel("median MD-ns to converge"); ax.set_ylabel("Spearman rho vs pKd")
ax.set_title("Cost to converge vs pKd predictive quality")
fig.savefig(f"{PLOTFOLDER}/pareto_cost-vs-pkd-rho.svg", bbox_inches="tight")
plt.show()
```

If `dG_extrapolated.csv` lacks a `speed` column or an `estimator` column, print
`pd.read_csv(f).columns` for one file and adapt the two lines that filter on
them; do not silently skip the figure.

- [ ] **Step 6: Cell 6 — Figure 4, per-rung traces for one system**

```python
rep = conv[(conv.system == "6dy7_A") & (conv["mode"] == "murcko")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, est in zip(axes, ESTIMATORS):
    g = rep[rep.estimator == est]
    for speed, gs in g.groupby("speed"):
        ax.plot(gs["n_replicas"], gs["dG_weighted-rmsd"],
                marker="o", ms=3, label=f"v={speed}")
    ax.axhline(lib.get_tolerances()["dG_weighted-rmsd"], ls="--", c="k")
    ax.set_title(est); ax.set_xlabel("n_replicas"); ax.legend(fontsize=8)
axes[0].set_ylabel("PMF RMSD (kJ/mol)")
fig.savefig(f"{PLOTFOLDER}/traces_rmsd_6dy7_A_murcko.svg", bbox_inches="tight")
plt.show()
```

- [ ] **Step 7: Cell 7 — Figure 5, censoring report**

```python
fig, axes = plt.subplots(1, len(K_CONSEC_LIST), figsize=(12, 4), sharey=True)
for ax, kc in zip(np.atleast_1d(axes), K_CONSEC_LIST):
    piv = (nconv[nconv.k_consec == kc]
           .pivot_table(index="estimator", columns="speed",
                        values="censored", aggfunc="mean"))
    sns.heatmap(piv * 100, annot=True, fmt=".0f", vmin=0, vmax=100,
                cmap="Reds", ax=ax, cbar_kws={"label": "% censored"})
    ax.set_title(f"Never converged (k_consec={kc})")
fig.savefig(f"{PLOTFOLDER}/heatmap_censoring.svg", bbox_inches="tight")
plt.show()

# Explicit table so censored cells are never silently dropped downstream
summary = (nconv.groupby(["estimator", "k_consec"])
           .agg(n_cells=("censored", "size"),
                n_censored=("censored", "sum"),
                median_n_to_converge=("n_to_converge", "median"),
                max_available=("n_available", "max"))
           .reset_index())
summary["reported"] = np.where(
    summary.n_censored > 0,
    ">=" + summary.max_available.astype(str) + " (censored "
    + summary.n_censored.astype(str) + "/" + summary.n_cells.astype(str) + ")",
    summary.median_n_to_converge.round(1).astype(str))
print(summary.to_string(index=False))
summary.to_csv(f"{PLOTFOLDER}/censoring_summary.csv", index=False)
```

- [ ] **Step 8: Cell 8 — deviation note (markdown)**

State verbatim in the notebook: cumulant and jarzynski use
`SMDAnalysis.check_convergence` unchanged; the force ladder reuses the same rung
helpers and `extrapolate_to_v0`, but because no force profile exists at v=0 it
uses `boundary_method="pmf_peak"` for barrier/r_ts and caps the RMSD window with
the **slowest speed's** force-plateau boundary. Record the
`validate_against_api` report so readers can see the gate passed.

- [ ] **Step 9: Execute and verify**

Run:
```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/jupyter nbconvert \
  --to notebook --inplace --execute --ExecutePreprocessor.timeout=3600 \
  sMD_WDR5_convergence_all.ipynb
ls -l plots/WDR5/convergence_all/
```
Expected: exit 0 and at least 5 SVGs. Confirm zero error outputs by scanning the
executed notebook's cell outputs for `output_type == "error"`.

---

### Task 8: Strip convergence from the two existing notebooks

**Files:**
- Modify: `scratch/paper_figures/sMD_WDR5.ipynb` (cells 11, 13; repoint cell 16)
- Modify: `scratch/paper_figures/sMD_WDR5_convergence.ipynb` (convergence-metric cells)

- [ ] **Step 1: Back up both notebooks**

```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
mkdir -p /tmp/wdr5_nb_backup
cp sMD_WDR5.ipynb sMD_WDR5_convergence.ipynb /tmp/wdr5_nb_backup/
```

- [ ] **Step 2: Confirm cell indices before deleting**

```bash
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python -c "
import json
nb=json.load(open('sMD_WDR5.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    s=''.join(c['source'])
    if 'n_to_converge' in s or 'minutes' in s or 'min_pivot' in s:
        print('CELL',i,'::',s.splitlines()[0][:90])
"
```
Expected: prints the convergence cells (11, 13) and the consumer (16). Use the
indices actually printed — do not assume.

- [ ] **Step 3: Remove convergence cells from `sMD_WDR5.ipynb`**

Delete by content match, not by hardcoded index (indices shift as cells are
removed, and Step 2 may reveal different ones):

```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python - <<'PY'
import json
p = "sMD_WDR5.ipynb"
nb = json.load(open(p))
MARKERS = ("n_to_converge", "first_consecutive_converged",
           "med_min_per_replica", "minutes_to_converge", "trim_upper")
keep, dropped = [], []
for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"])
    if c["cell_type"] == "code" and any(m in s for m in MARKERS):
        dropped.append(i); continue
    keep.append(c)
nb["cells"] = keep
json.dump(nb, open(p, "w"), indent=1)
print("dropped cell indices:", dropped)
PY
```
Expected: reports the convergence/timing cells as dropped.

Then repoint the combined heatmap that consumed `min_pivot`. Replace that panel's
data source with the cached grid:

```python
# in the combined-heatmap cell of sMD_WDR5.ipynb
import wdr5_conv_lib as lib
_n = pd.concat([pd.read_csv(f) for f in
                sorted(glob(os.path.join("results_conv", "conv_grid", "*.csv")))],
               ignore_index=True)
_rows = [{"mode": m, "speed": sp,
          "n": lib.first_consecutive_converged(g, k_consec=2)}
         for (m, sp), g in _n[_n.estimator == "cumulant"].groupby(["mode", "speed"])]
min_pivot = (pd.DataFrame(_rows)
             .pivot_table(index="mode", columns="speed", values="n",
                          aggfunc="median"))
```
Keep the pKd/Spearman cells untouched.

- [ ] **Step 4: Remove convergence-metric cells from `sMD_WDR5_convergence.ipynb`**

```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
/gpfs/home/mllanos/micromamba/envs/autopath/bin/python - <<'PY'
import json
p = "sMD_WDR5_convergence.ipynb"
nb = json.load(open(p))
MARKERS = ("sMD_conv_vALL_metrics", "sMD_conv_vALL_traces",
           "metrics_combined", "tolerances")
KEEP_ANYWAY = ("fit_transform", "plot_clusters_PCA", "dtw_ndim",
               "sMD_processed_data")
keep, dropped = [], []
for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"])
    if (c["cell_type"] == "code" and any(m in s for m in MARKERS)
            and not any(k in s for k in KEEP_ANYWAY)):
        dropped.append(i); continue
    keep.append(c)
nb["cells"] = keep
json.dump(nb, open(p, "w"), indent=1)
print("dropped cell indices:", dropped)
PY
```
Expected: drops the metric-trace cells while keeping the DTW/PCA deep dive and
the inline cumulant work-profile recompute (both matched by `KEEP_ANYWAY`).

- [ ] **Step 5: Re-execute both and verify**

```bash
cd /gpfs/home/mllanos/forlilab/autopath/scratch/paper_figures
for nb in sMD_WDR5.ipynb sMD_WDR5_convergence.ipynb; do
  /gpfs/home/mllanos/micromamba/envs/autopath/bin/jupyter nbconvert \
    --to notebook --inplace --execute --ExecutePreprocessor.timeout=3600 "$nb"
  echo "EXIT($nb)=$?"
done
```
Expected: both exit 0 with no error outputs. If a downstream cell raised
`NameError` for a deleted variable, that cell was a hidden consumer — repoint it
at the cached CSV rather than restoring the deleted cell.

- [ ] **Step 6: Commit the plan document only**

```bash
cd /gpfs/home/mllanos/forlilab/autopath
git add docs/superpowers/plans/2026-07-30-wdr5-convergence.md
git commit -m "plan: WDR5 multi-estimator convergence analysis"
```
(The scratch files stay untracked by design; `autopath/pulling/Diagnostics.py`
has a pre-existing user edit that must not be committed.)
