# Projection Pulling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `SteeredMD` can pull the projection z = n·(COM_lig − COM_pocket) along a protein-frame axis, optionally inside the funnel's lateral restraint, with a ΔG° helper (Limongelli eq. 15) and an axis helper; then a WDR5 6e23 pilot against the existing distance pulls.

**Architecture:** A new `autopath/pulling/projection.py` builds the protein-frame projection CV, the projection spring and the funnel-parameter dict for the sMD groups, reusing `Funnel.py`'s frame code. The metad funnel force gains an `axial_walls` switch so the lateral part can be reused. `SteeredMD` gains a projection mode; `SMDData` reads the CV type from the log header. Eq. 15 moves into one shared function used by both `correct_fe_for_funnel` and the new pull-side helper. The pilot is scratch scripts outside the repo.

**Tech Stack:** OpenMM, cvpack, MDAnalysis, numpy, pandas. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-26-smd-projection-pull-design.md`

## Global Constraints

- Branch `smd-projection` (worktree `~/forlilab/autopath/.claude/worktrees/smd-projection`); `develop` already has the shared-λ0 fix and the HS estimator.
- Python `~/micromamba/envs/autopath/bin/python`; tests `... -m pytest -q <files>` from the worktree root; never install into `autopath`.
- Distance pulls (no axis given) must behave and log exactly as before; the existing sMD and HS test suites keep passing.
- z, the restraint and eq. 15 use the **sMD pull groups** (groupA = ligand, groupB = pocket). Funnel params whose `guest_index`/`host_index` differ from the pull groups → `ValueError("not supported: ...")`.
- Projection mode requires a protein frame (`axis_coeffs`, `frame_b_index`, `frame_c_index`); lab-frame params → "not supported". No positional restraints on the protein.
- The lateral restraint has **no axial walls** during pulls.
- Unsupported → `ValueError` starting `"not supported:"`. No hidden fallbacks.
- Terse comments, numpy docstrings, no system names or measured numbers in repo code.
- Tests run the code (real OpenMM on CPU for sMD tests); never assert on source text.
- Commits: one-line conventional subject ≤ 50 chars, no body, no trailers.
- Never mutate the worktree while a pilot job may import from it: pilot jobs run from a `git archive` copy.
- GPU jobs: slow-node exclude list re-checked with `sinfo -N -h -p gpu,forli -o "%.12N %.12P %.9T %.22G"`, plus a throughput guard aborting below 60 ns/day.

## Review Focus

1. Funnel params built for other atom groups passed to a projection pull → "not supported" (Task 3).
2. Backward pulling with a projection axis → "not supported" (Task 3).
3. Lab-frame funnel params (no protein frame) → "not supported" (Task 3).
4. `SMDData` given a mix of distance and projection logs → "not supported" (Task 4).
5. A pull whose end along n exceeds the PBC half-box limit → capped with a logged warning, never silently wrapped (Task 3).

---

### Task 1: One eq. 15 implementation

**Files:**
- Modify: `autopath/metadynamics/Analysis.py` (`correct_fe_for_funnel`, around the `# Limongelli 2013 Eq. 15` block)
- Create: `autopath/pulling/standard_state.py`
- Test: `tests/test_standard_state.py`

**Interfaces:**
- Produces:
  - `limongelli_dG(site_G_kJ: np.ndarray, dz_ang: float, R_cyl_ang: float, kT: float, C0_per_A3: float = 1/1660) -> dict` in `Analysis.py`: keys `integral_ang`, `dG_basin`, `correction`, `dG_bind_std`, `pKd`. `site_G_kJ` is already zeroed on the unbound reference. Non-positive integral → `ValueError`.
  - `dG_standard_from_profile(coord_nm, G_kJ, site_max_nm, bulk_range_nm, R_cyl_ang, temperature=300.0, C0_per_A3=1/1660) -> dict` in `standard_state.py`: zeroes G on the mean over `bulk_range_nm`, integrates the points with `coord_nm <= site_max_nm` on their own (uniform) grid via `limongelli_dG`, and adds `bulk_ref_kJ`, `bulk_slope_kJ_per_nm` (linear fit over the bulk range) to the result. A non-uniform grid, fewer than 3 bulk points, or a NaN in the site → "not supported".

- [ ] **Step 1: Write the failing tests**

```python
def test_gaussian_well_matches_analytic():
    # W(z) = -D + (k/2) z^2 on the site, 0 in bulk: integral = sqrt(2 pi kT / k) * exp(D/kT) (Å)
    ...  # assert dG_bind_std within 1e-6 kJ/mol of -kT*ln(pi R^2 C0 * integral)

def test_correct_fe_for_funnel_unchanged():
    # synthetic FE array + funnel_params Quantities; values recorded from the pre-refactor code
    # (compute them in Step 2 with the current code and paste them into the test)

def test_bulk_slope_reported():
    ...  # linear bulk tail with slope 3 kJ/mol/nm -> bulk_slope_kJ_per_nm == 3 (rtol 1e-9)

def test_nonuniform_grid_not_supported(): ...
```

- [ ] **Step 2: Record the current `correct_fe_for_funnel` outputs for the synthetic case and run the tests** — `test_correct_fe_for_funnel_unchanged` passes, the others fail with ImportError.
- [ ] **Step 3: Implement `limongelli_dG`, make `correct_fe_for_funnel` call it, implement `dG_standard_from_profile`.**
- [ ] **Step 4: Run** `tests/test_standard_state.py` plus the existing metad funnel tests (`tests/test_funnel_*.py`) — all pass.
- [ ] **Step 5: Commit** — `refactor(funnel): shared eq. 15 helper`

---

### Task 2: Projection CV, spring and parameters

**Files:**
- Create: `autopath/pulling/projection.py`
- Modify: `autopath/metadynamics/Funnel.py` (`create_funnel_force_from_trajectory_analysis`: add `axial_walls: bool = True`; when False the expression omits `U_axial` and `U_lower` and the `k_z`, `z_max`, `z_low` globals)
- Test: `tests/test_projection_cv.py`

**Interfaces:**
- Consumes: `Funnel._frame_definitions`, `Funnel.funnel_frame`, `Funnel.upgrade_funnel_params`, `Funnel.funnel_coords_ang`.
- Produces (all in `projection.py`):
  - `projection_params(axis, ligand_idx, pocket_idx, reference_positions_nm, topology, masses, z_cc_ang, z_max_ang, R_cylinder_ang=2.0, alpha_deg=45.0, k_xy_kcal_per_mol_per_ang2=10.0) -> dict`: a funnel_params dict (`unbinding_axis`, `guest_index=ligand_idx`, `host_index=pocket_idx`, Quantity `z_cc`/`z_max`/`R_cylinder`/`alpha`/`k_xy` as `create_funnel_force_from_trajectory_analysis` expects) upgraded to a protein frame with `upgrade_funnel_params`.
  - `check_projection_params(params, ligand_idx, pocket_idx) -> None`: raises "not supported" for group mismatch or no protein frame.
  - `projection_cv(params)`: cvpack `CentroidFunction` returning unclamped z (nm), groups `[pocket, ligand, frame_b, frame_c]`, mass-weighted, `pbc=False`.
  - `projection_spring(params, k_kj_per_mol_nm2)`: cvpack `CentroidFunction` with energy `0.5*fc_pull*(r_z - r0_smd)^2` and the same groups, global parameters `r0_smd` and `fc_pull` (the names `SteeredMD` already sets).
  - `projection_z_nm(params, positions_nm, masses) -> float`: numpy z for sanity checks (from `funnel_coords_ang`).

- [ ] **Step 1: Write the failing tests** (a small OpenMM system built in the test: 2 pocket particles, 1 ligand particle, 2 + 2 frame particles, held by bonds, periodic box 6 nm; the params dict is built by hand with explicit `frame_b_index`, `frame_c_index`, `axis_coeffs`)

```python
def test_projection_cv_equals_n_dot_dcom(): ...       # cvpack value == projection_z_nm == n·(COM_lig − COM_pocket), atol 1e-6
def test_projection_cv_rotation_invariant(): ...      # rigid rotation of all positions: same z (atol 1e-6)
def test_lab_axis_is_not_rotation_invariant(): ...    # a lab-frame n·Δ changes under the same rotation (documents why the frame is needed)
def test_spring_energy(): ...                         # energy == 0.5 k (z - r0)^2 for r0 = z + 0.1
def test_lateral_force_has_no_axial_walls():
    ...  # create_funnel_force_from_trajectory_analysis(params, axial_walls=False): ligand on the axis at
         # z = z_max + 1 nm and at z = -0.5 nm -> energy 0; with axial_walls=True -> energy > 0
def test_group_mismatch_not_supported(): ...
def test_lab_frame_params_not_supported(): ...         # Review Focus 3
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `projection.py` and the `axial_walls` switch** (default True keeps metad behaviour bit-identical).
- [ ] **Step 4: Run** `tests/test_projection_cv.py` and `tests/test_funnel_*.py` — all pass.
- [ ] **Step 5: Commit** — `feat(pulling): protein-frame projection CV`

---

### Task 3: Projection mode in SteeredMD

**Files:**
- Modify: `autopath/pulling/steered_md.py` (`__init__`, `run`, `pull_single_direction`)
- Test: `tests/test_smd_projection.py`

**Interfaces:**
- Consumes: Task 2 functions.
- Produces:
  - `SteeredMD.__init__(..., projection_params: dict | None = None, lateral_restraint: bool = False)`. `projection_params is None` → distance mode, unchanged. Otherwise `check_projection_params(projection_params, groupA_atoms, groupB_atoms)`; `lateral_restraint=True` adds `create_funnel_force_from_trajectory_analysis(projection_params, axial_walls=False)` in its own force group. `lateral_restraint=True` without params → "not supported".
  - In projection mode `run` builds `self.com_force = projection_spring(...)` and `self.com_dist = projection_cv(...)` instead of the distance CentroidFunctions; the shared-λ0 equilibration, `lambda0_nm` and the logging loop are unchanged (λ0 default = z of the input structure). The restraint is present during the 50 ps and the pull.
  - `pull_single_direction`: projection + `direction="backward"` → "not supported". The PBC move cap uses `z_end_max = min_i (L_i/2)/|n_i| − pbc_safety_nm` (n = the frame axis at the current positions, `funnel_axis_at`) instead of the distance rule, and the sanity check compares the cvpack z with `projection_z_nm` (0.05 nm tolerance, same RuntimeError).
  - Header lines (projection mode only): `# cv=projection`, `# axis_coeffs=a1,a2,a3`, `# restraint=funnel` or `# restraint=none`, and when on `# R_cylinder_ang=`, `# z_cc_ang=`, `# alpha_deg=`, `# k_xy_kcal_per_mol_per_ang2=`. Distance mode writes `# cv=distance`.

- [ ] **Step 1: Write the failing tests** (reuse Task 2's toy system through `SteeredMD.run` on the CPU platform; `restart_velocities=True`, `timestep=0.002`, `autostop_nc=None`, `max_displacement=0.05`, `dx_per_move=0.001`, `sMD_spring_cte=3765.6`)

```python
def test_projection_replicas_share_schedule(): ...    # two replicas: identical r_target; r_target[0] == lambda0 + dx; header cv=projection
def test_explicit_lambda0_below_input(): ...          # lambda0_nm = z_input - 0.15 -> r_target[0] == z_input - 0.15 + dx (1e-9)
def test_work_is_spring_only_with_restraint():
    ...  # lateral_restraint=True: every logged dW_protocol equals
         # 0.5k[(r_before - λ_new)^2 - (r_before - λ_old)^2] recomputed from the logged columns (atol 1e-6)
def test_header_restraint_fields(): ...
def test_backward_projection_not_supported(): ...      # Review Focus 2
def test_mismatched_groups_not_supported(): ...        # Review Focus 1
def test_pbc_cap_along_axis(caplog): ...               # Review Focus 5: max_displacement beyond the half-box along n
                                                      # -> moves capped, warning logged, last r_target < z_end_max
def test_distance_mode_header(): ...                   # no params -> '# cv=distance', columns unchanged
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_smd_projection.py tests/test_smd_shared_lambda0.py tests/test_smd_r0_check.py tests/test_autostop_options.py tests/test_smd_pocket_atoms.py tests/test_steered_geom_logging.py` — all pass.
- [ ] **Step 5: Commit** — `feat(pulling): projection pulls in SteeredMD`

---

### Task 4: SMDData reads the CV type

**Files:**
- Modify: `autopath/pulling/SMDData.py` (`__init__` completion check, metadata)
- Test: `tests/test_smddata_projection.py`

**Interfaces:**
- Produces: `SMDData.cv_type` (`"distance"` or `"projection"`; logs without a `cv` header count as `"distance"`). Mixed types across the loaded logs → "not supported". The forward completion check (`r_after.min() < completion_threshold_nm` drops the pull) applies to distance logs only; for projection logs a pull is kept regardless of z ≤ 0 (no other completion rule in this phase — fixed-length pulls).

- [ ] **Step 1: Write the failing tests** (write small `.dat` files with the header lines from Task 3)

```python
def test_projection_pull_with_negative_z_is_kept(): ...      # r_after down to -0.2 nm, SMDData keeps it; cv_type == "projection"
def test_distance_logs_unchanged(): ...                       # old header-less logs: cv_type == "distance", completion drop still applies
def test_mixed_cv_types_not_supported(): ...                  # Review Focus 4
def test_hs_fit_on_projection_logs(): ...                     # HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=0).fit runs, finite logw
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests plus `tests/test_hs_*.py tests/test_smd_param_logging.py` — all pass.
- [ ] **Step 5: Commit** — `feat(pulling): SMDData projection logs`

---

### Task 5: Axis helper

**Files:**
- Modify: `autopath/metadynamics/Funnel.py` (move the PCA block of `generate_funnel_parameters_from_trajectory` into `pca_unbinding_axis(guest_coms, host_coms) -> np.ndarray` and call it there; behaviour unchanged)
- Modify: `autopath/pulling/projection.py`
- Test: `tests/test_projection_axis.py`

**Interfaces:**
- Produces: `axis_from_pulls(universe, ligand_idx, pocket_idx, ligand_bulk_idx, frames=None, z_cc_percentile=95.0, bulk_cutoff_ang=10.0) -> dict` in `projection.py`, returning `unbinding_axis` (oriented bound → unbound), `z_cc_ang` (percentile of the pulled-group z over the given frames, the funnel rule) and `z_max_ang` (z of the pulled ligand group when every atom of `ligand_bulk_idx`, the whole ligand, clears `bulk_cutoff_ang` from the protein, moving rigidly along the axis from the first frame; reuse `z_where_guest_reaches_bulk`'s scan). `frames` selects one path's frames; None pools all.

- [ ] **Step 1: Write the failing tests** (synthetic MDAnalysis Universe from numpy: a few protein atoms, a 3-atom ligand moving along a known direction plus noise)

```python
def test_axis_recovered(): ...            # |axis · true| > 0.99, oriented bound -> unbound
def test_frames_subset_selects_path(): ... # two exit directions in one universe; frames of path A give axis A
def test_generate_funnel_parameters_unchanged(): ...  # PCA refactor: same unbinding_axis as before on the synthetic universe
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests plus `tests/test_funnel_*.py` — all pass.
- [ ] **Step 5: Commit** — `feat(pulling): axis helper from pull trajectories`

---

### Task 6: WDR5 6e23 pilot setup and submission (scratch, outside the repo)

**Files:**
- Create: `/gpfs/group/forli/mllanos/autopath_benchmarking/WDR5/openFF/6e23_A/sMD-projection_pilot/` with `setup_pilot.py`, `run_pull.py`, `submit.sbatch`, `code/` (a `git archive` of the branch head at submission)

**Interfaces:**
- Consumes: Tasks 2–5.

- [ ] **Step 1: `setup_pilot.py`.** Rebuild the `sMD-murcko` groups exactly as `autopath_core` does (`get_ligand_anchor_atoms(..., mode="murcko", n_atoms=5)`, `get_pocket_atoms_idxs`) from the equilibrated structure, and assert they equal the anchor indices logged in `6e23_A/autopath.log` for that run (abort on mismatch). Fit the axis with `axis_from_pulls` on the `sMD-murcko` `_aligned.dcd` pulls (pooled), compare with the stored metad axis (0.590, 0.504, 0.631) and log the angle, build `projection_params` (R_cyl 2 Å, alpha 45°, k_xy 10), compute z_input, λ0 = z_input − 0.15 nm and the pull length to z_max + 0.5 nm. Write `pilot_params.npz` and a `setup.json` with every number.
- [ ] **Step 2: `run_pull.py <speed> <replica>`.** Throughput guard first: time 2 ps of MD on the job's GPU, abort below 60 ns/day. Then `SteeredMD(..., projection_params=..., lateral_restraint=True, autostop_nc=None, max_displacement=<pull length>, sMD_spring_cte=<as the murcko runs>, dx_per_move=0.001)` and `run(pulling_speed, run_id, state_xml_file or checkpoint as the murcko runs, lambda0_nm=λ0)`. Writes into `trajectories/`.
- [ ] **Step 3: `submit.sbatch`.** Array job 0–49 (25 × 0.005 + 25 × 0.01 nm/ps), one GPU each, `--exclude=` the current slow-node list from `sinfo`, `PYTHONPATH` = `code/`.
- [ ] **Step 4: Submit, and record job ids.** Smoke-test one array element first (`--array=0`), check its log (throughput line, header `cv=projection`, sensible r_target), then submit the rest.

---

### Task 7: Pilot analysis and verdict

**Files:**
- Create: `.../sMD-projection_pilot/analyse_pilot.py` (writes only into `sMD-projection_pilot/analysis/`)

- [ ] **Step 1: Analysis script.** For each arm (projection pilot; distance `sMD-murcko` subsampled to 25 pulls per speed at 0.005 and 0.01 nm/ps, 10 random subsamples): `SMDData` → HS (all four sources, n_boot 200) and force v→0, per-slice ESS and work sd. For the projection arm: `dG_standard_from_profile` on HS G0(z), Jarzynski/cumulant G(λ) and force v→0, site_max = z_cc, bulk_range = [z_max − 0.3, z_max] nm, with bootstrap sd. Convergence curve: HS ΔG° vs number of pulls (5, 10, 15, 20, 25).
- [ ] **Step 2: Figure** (`fig_pilot.png`): G profiles per arm, ESS vs coordinate, F_spread, ΔG° vs pulls with the experimental line (−57.4 kJ/mol, pKd 10.0, flagged as possibly an assay limit).
- [ ] **Step 3: Verdict for the user** against the pre-registered bars (spec "Cleaner / converges faster"): ESS ≥ 5 at least 0.3 nm further; F sources within 2 kT on the bulk plateau; HS ΔG° bootstrap sd < 2 kJ/mol. Report lam0_sd for both arms and the bulk slope. Stop there: no scale-up or workflow change until the user has seen it.
