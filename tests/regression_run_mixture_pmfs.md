# Regression guard: `run()` `mixture_pmfs` is unchanged by the `SupportPolicy` refactor

## Purpose

Tasks 1–3 introduced a `SupportPolicy` count-based support gate and wired it into
both the convergence check (`check_convergence`) and the production analysis
pipeline (`SMDAnalysis.run`). In `run()` the change is intended to be a **no-op at
the default/production thresholds** (`min_samples_per_step=5`,
`min_replicas_per_path=5`): the policy value that now feeds
`trim_results_by_n_samples_support(min_samples=...)` is sourced from the *same*
`min_samples_per_step` it used before, and `SupportPolicy.usable_path(n)` is
logically identical to the original `n < min_replicas_per_path` path-count drop.

This document records a **repeatable, isolated empirical procedure** that confirms
`run()` produces numerically identical `mixture_pmfs.csv` before and after the
refactor. It is a full-pipeline check that needs real trajectory data, so it
cannot be a fast unit test — hence a documented manual procedure rather than a
`pytest` case.

## This is confirmatory, not the sole guarantee

The `run()` no-op is *already* guaranteed two other, cheaper ways:

1. **Logical equivalence (Task 3):** `not SupportPolicy.usable_path(n)` ≡ the
   original `n < min_replicas_per_path` drop, and the trim's `min_samples` is
   sourced from the identical policy value (`self.support_policy.min_samples_per_step`,
   constructed from `self.min_samples_per_step`).
2. **Unit test (Task 1):** `tests/test_support_policy.py::test_noop_on_fully_supported_data`
   pins that `SupportPolicy.apply` is the identity on already-fully-supported data.

The procedure below is the **empirical, end-to-end confirmation** on real HSP90
SMD data.

## Safety constraints (read before running)

The shared benchmarking tree
`/gpfs/group/forli/mllanos/autopath_benchmarking/kinetics/HSP90_OFF/<SYSTEM>/...`
holds the user's real production outputs. **The commands below never write to it.**
They copy the chosen system's *inputs* (SMD `.dat` logs + the equilibrated
reference PDB) into an isolated `/tmp` working dir and direct every analysis
`outdir` there. Do not point `outdir` at the shared tree, and do not re-run the
full `run_AutoPath.py` (which would overwrite `analysis_traces/`).

## What to compare (important — the robust guard)

Compare **`main`-code output vs worktree-code output produced from the *same*
isolated inputs with the *same* `SMDAnalysis` config.** Do **not** compare the
worktree output against the pre-existing `analysis_traces/mixture_pmfs.csv` in the
shared tree: that stale CSV was produced by an older code revision and a slightly
different production invocation, so it differs from *both* current `main` and the
worktree by a constant `r_coord` binning offset that has nothing to do with
`SupportPolicy`. The only meaningful diff for this task is `main` vs worktree
under identical conditions.

## Procedure

### Step 0 — Choose a completed system and stage isolated inputs

Pick any completed HSP90 system+mode with `trajectories/*.dat` logs and an
equilibrated reference PDB. Example uses `2YKJ_YKJ / sMD-lig_ha`.

```bash
SYS=/gpfs/group/forli/mllanos/autopath_benchmarking/kinetics/HSP90_OFF/2YKJ_YKJ
WT=/gpfs/home/mllanos/forlilab/autopath/.claude/worktrees/fix-convergence-nan-barrier
MAIN=/gpfs/home/mllanos/forlilab/autopath          # user's main checkout (read-only for us)

TMP=$(mktemp -d /tmp/support_policy_regr.XXXXXX)
mkdir -p "$TMP/logs" "$TMP/outdir_main" "$TMP/outdir_worktree"

# Inputs only — never write back to $SYS.
cp "$SYS/sMD-lig_ha/trajectories/"*.dat                 "$TMP/logs/"
cp "$SYS/equilibration/2YKJ_YKJ_equilibrated.pdb"       "$TMP/reference.pdb"
echo "Staged $(ls "$TMP/logs" | wc -l) log files in $TMP"
```

> The reference PDB must be the **equilibrated** system PDB
> (`equilibration/<sys>_equilibrated.pdb`) — the same one `autopath_core.run()`
> passes to `SMDData(...)`. Using the raw `system.pdb` shifts the `r_coord` grid
> and invalidates the comparison.

### Step 1 — Driver script

Write `$TMP/run_check.py`. It mirrors the production `SMDAnalysis` call in
`autopath/autopath_core.py` (estimators `cumulant`+`jarzynski`, seed `42`, T=300,
default trace features, `filter_low_support=True`, thresholds `5/5`,
`min_support_ratio=1.0`). `group_B=None` and `ligand_sdf=None` because the HSP90
production config (`run_AutoPath.py`) leaves `sMD_clust_selection` unset and uses
default trace-only features — so no pocket-distance or ligand-shape features are
computed. `do_plots=False` to avoid needing the aligned DCDs.

```python
# $TMP/run_check.py
import glob, sys, os
from autopath.pulling.SMDData import SMDData
from autopath.pulling.AnalysisSMD import SMDAnalysis

TMP, out_sub = sys.argv[1], sys.argv[2]          # e.g. "outdir_worktree"
logs = sorted(glob.glob(os.path.join(TMP, "logs", "*.dat")))
ref_pdb = os.path.join(TMP, "reference.pdb")
ligand_select = "(resname UNK) and not name H*"

# Customize for your system: adjust ligand_select (resname) and pocket_select
# (residue numbers) to match your system's numbering in the reference PDB.
smd_data = SMDData(logs, "2YKJ_YKJ", temperature=300.0, reference_pdb=ref_pdb)

analysis = SMDAnalysis(
    "2YKJ_YKJ", path_model="dtw",
    estimators=["cumulant", "jarzynski"],
    do_plots=False, seed=42, temperature=300.0,
    ligand_select=ligand_select,
    pocket_select="(resid 145-153 183-190) and name CA",
    outdir=os.path.join(TMP, out_sub),
    filter_low_support=True,
    min_samples_per_step=5, min_support_ratio=1.0,
    min_replicas_per_path=5, min_path_steps_ratio=0.6,
    max_frac_neg_dG_first_half=0.25, min_speeds_for_extrapolation=2,
)
analysis.run(
    smd_data, group_A=ligand_select, group_B=None,
    merge_features=True, cluster_across_speeds=False,
    features=None, ligand_sdf=None,
)
print("DONE ->", os.path.join(TMP, out_sub, "mixture_pmfs.csv"))
```

### Step 2 — Generate both outputs (main and worktree)

Run the **same** driver twice, once with each code tree on `PYTHONPATH`. The user's
main checkout lives at `$MAIN` (branch `main`); the worktree at `$WT`.

```bash
# Baseline: main code
cd "$MAIN"
PYTHONPATH="$PWD:$PYTHONPATH" micromamba run -n autopath \
    python "$TMP/run_check.py" "$TMP" outdir_main

# Candidate: worktree code (SupportPolicy)
cd "$WT"
PYTHONPATH="$PWD:$PYTHONPATH" micromamba run -n autopath \
    python "$TMP/run_check.py" "$TMP" outdir_worktree
```

> If you cannot use the user's `main` checkout, produce the baseline from a clean
> `git worktree add /tmp/autopath_main main` instead — the point is that the
> baseline `mixture_pmfs.csv` comes from code *without* the `SupportPolicy` diff,
> generated from the identical `$TMP` inputs.

### Step 3 — Diff and assert (PASS criterion)

```bash
cd "$WT"
PYTHONPATH="$PWD:$PYTHONPATH" micromamba run -n autopath python - "$TMP" <<'PY'
import sys, pandas as pd, numpy as np
TMP = sys.argv[1]
base = pd.read_csv(f"{TMP}/outdir_main/mixture_pmfs.csv")
new  = pd.read_csv(f"{TMP}/outdir_worktree/mixture_pmfs.csv")
assert base.shape == new.shape, f"shape differs: {base.shape} vs {new.shape}"

key = ["step", "speed", "estimator"]      # stable integer/categorical keys
m = base.merge(new, on=key, suffixes=("_base", "_new"))
assert len(m) == len(base), f"merge lost rows: {len(m)} of {len(base)}"

mism = []
for col in [c for c in base.columns if c not in key and c != "model"]:
    a = pd.to_numeric(m[f"{col}_base"], errors="coerce").fillna(0)
    b = pd.to_numeric(m[f"{col}_new"], errors="coerce").fillna(0)
    if not np.allclose(a, b, rtol=1e-6, atol=1e-6):
        mism.append((col, float((a - b).abs().max())))

if mism:
    print("MISMATCH:", mism)
    raise SystemExit(1)
print("mixture_pmfs numerically unchanged (main vs worktree) — regression guard PASS")
PY
```

**PASS criterion:** the script prints `regression guard PASS` and exits 0 — i.e.
every numeric column of `mixture_pmfs.csv` (`dG`, `Wdiss`, the slope/SE/R² and
`n_speeds` columns, and `r_coord`) matches within `rtol=1e-6, atol=1e-6`, `NaN`s
aligned (compared as `fillna(0)`).

> Notes on the diff: merge on `["step","speed","estimator"]`, not on `r_coord` —
> `r_coord` is a float and joining on it can silently drop all rows. The `model`
> column (a string label) is excluded from the numeric comparison. `r_coord`
> *itself* is compared as a numeric column, so a grid shift would be caught.

**On mismatch — STOP.** The trim-narrowing / path-count gate changed production
output. Revisit **Task 3**: fall back to leaving the trim's `min_samples` on
`self.min_samples_per_step` directly and only delegating the path-count check to
`SupportPolicy.usable_path`. Re-run this procedure until it PASSES.

### Step 4 — Clean up

```bash
rm -rf "$TMP"      # only ever touches /tmp
```

## Last verified result

Executed on 2026-07-07 with system `2YKJ_YKJ / sMD-lig_ha` (127 SMD logs, speeds
0.001 / 0.005 / 0.01 nm/ps, plus the v→0 extrapolated rows):

- `main` vs worktree, both from identical `$TMP` inputs: **PASS** — all numeric
  columns identical (15548 rows, 0 mismatches), confirming the `SupportPolicy`
  refactor is a byte-for-byte no-op in `run()` at production thresholds.
- For reference: comparing either run against the *stale* shared
  `analysis_traces/mixture_pmfs.csv` shows a constant `~0.026 nm` `r_coord` offset
  (and consequent value shifts) — a pre-existing older-revision artifact, **not**
  caused by this branch. This is exactly why the guard compares main-vs-worktree,
  not against the shared CSV.

## Update 2026-07-11: `force` estimator + shared reference `p_eq` (no longer a no-op)

The force-estimator work (spec: shared robustness-selected `p_eq`) changes
`mixture_pmfs.csv` for multi-path systems — this is an **intended** behavior
change, not a regression, but it invalidates the byte-for-byte no-op guarantee
above for any run that adds the `force` estimator or has ≥2 paths per speed.
Recorded here so a future diff against this doc's baseline isn't mistaken for a
bug.

- **New `force` estimator rows.** When `force` is included in `estimators`
  (default-on for runs with ≥2 speeds; auto-dropped for single-speed runs via
  the speed guard), `mixture_pmfs.csv` gains `estimator == "force"` rows: one
  per-speed row per step with `dG == Wmean` (raw mean work, no cumulant/Jensen
  correction) and `Wdiss` left `NaN` (dissipated work is not defined for this
  estimator), plus a `speed == 0` v→0-extrapolated `dG` row. `friction.csv`
  gains `estimator == "force"` rows for **both** `method in
  {"regression","derivative"}` (force-based `dF/dv` friction, `Feq` from the
  same regression), and `koff_kramers.csv` gains a corresponding `force` row:
  `FrictionEstimator.run_kramers_for_all_estimators` (`Estimators.py`) loops
  generically over `mixture_pmfs["estimator"].unique()`, so it is not
  hardcoded to cumulant/jarzynski and picks up `force` automatically. The
  `dG`/`Wdiss`/friction plumbing itself was verified end-to-end on real data
  (`HSP90_OFF/6ELO_BAW/sMD-pocket_com`) in Task 5's integration check: `force`
  rows present in the `dG` v→0 extrapolation, absent from the `Wdiss` v→0
  extrapolation (extrapolate_to_v0 drops NaN per estimator — no crash), and
  `force` friction rows returned by both `gamma_from_force_regression` and
  `gamma_from_force_derivative`.
- **Cumulant/jarzynski values change for multi-path systems.** Path weighting
  used to be per-estimator self-weighting (each estimator picked its own
  per-path weights from its own `p_eq`/robustness diagnostics). It is now a
  single **shared reference `p_eq`**, chosen once by
  `SMDData.choose_reference_estimator` (robustness-selected: fewest
  negative-`dG` bins among the non-`force` estimators, cumulant wins ties) and
  reused by every estimator, including `force`, via the same `ref_weights`
  object. This means cumulant's and jarzynski's `mixture_pmfs` numeric values
  (and downstream `friction.csv` / `koff_kramers.csv`) for any system with ≥2
  paths per speed will **not** match the pre-force-estimator baseline
  byte-for-byte, even at unchanged thresholds — this is expected, not a
  reintroduction of the `SupportPolicy` bug the procedure above guards
  against.
- **Single-path systems are unchanged.** When a speed has exactly one path,
  `p_eq` is trivially `1.0` regardless of which estimator "chooses" it, so the
  shared-reference change is a no-op there — the `2YKJ_YKJ` baseline above (and
  any other single-path-per-speed system) is unaffected by this update.
- **Reference estimator on the 6ELO_BAW benchmark.** Rebuilding cumulant /
  jarzynski / force results directly from
  `HSP90_OFF/6ELO_BAW/sMD-pocket_com/analysis_features/sMD_processed_data.csv`
  and calling `SMDData.choose_reference_estimator` on the combined results
  returns `"jarzynski"` (cumulant has negative-`dG` bins at this system's
  `βσ ≈ 10`, so it loses the robustness comparison) — confirmed by the Task 5
  integration script, which also confirmed `force`'s friction rows
  (`{"Gamma","Gamma_integrated","Feq","method","estimator"}` columns) and the
  dual-pathway `dG`/`Wdiss` v→0 behavior above.
