# Projection pulling in steered MD — design

Status: draft for review (2026-10-26). Branch `smd-projection` off `develop` (includes the shared-λ0 fix and
the Hummer–Szabo estimator).

## Intent

`SteeredMD` pulls the ligand–pocket COM–COM distance r. That leaves the exit direction free, so pulls take
different routes (hence path clustering), the work distributions are wide, and the exponential-work estimators
lose effective sample size within a few tenths of a nm. A distance PMF from pulls also has no usable unbound
reference: the ligand never samples the angular volume of the bulk, so ΔG°_bind cannot be read off it.

This feature adds **projection pulling**: the spring acts on z = n·(COM_lig − COM_pocket) along a given axis n,
optionally with the existing **funnel restraint** (cone over the site, cylinder outside) acting on the lateral
distance from the axis. The two pieces are independent:

- the projection is the CV the spring pulls and the estimators reconstruct (G along z instead of r);
- the funnel is a fixed restraint (no protocol work, part of H0). It is what makes ΔG° computable: the
  cylinder fixes the unbound-state lateral area to πR², and Limongelli 2013 eq. 15 converts a 1D profile
  W(z) into ΔG°_b = −kT ln ∫_site e^{−βW(z)} dz − kT ln(πR²_cyl C°).

The user wants both uses: ΔG°_bind (projection + funnel restraint) and kinetics (barriers / Kramers on G(z)).
This phase implements the feature and tests on one system whether PMFs come out cleaner or converge faster.
How it plugs into the AutoPath workflow (an exploratory fast-distance mode, automatic axes, etc.) is decided
after the test.

## Goals

1. `SteeredMD` can pull along a user-supplied axis; the distance pull stays the default and is unchanged.
2. Optional funnel restraint during projection pulls, reusing the metad funnel code.
3. Axis and restraint live in the **protein frame** (rebuilt each step from CA-group centroids, as the metad
   funnel does), so no positional restraints on the protein are needed.
4. A ΔG° helper applying eq. 15 to any equilibrium profile along z (HS G0(z), Jarzynski / cumulant G(λ),
   force v→0), sharing its maths with `correct_fe_for_funnel` rather than duplicating it.
5. A helper that fits an axis to existing pull trajectories (the PCA used by
   `generate_funnel_parameters_from_trajectory`), pooled or per path.
6. A WDR5 6e23 pilot comparing projection+funnel pulls with the existing distance pulls, with a verdict
   shown to the user before any scale-up.

## Non-goals

- Workflow integration (AutoPathExplore-style fast exploration then slow refinement, choosing the axis
  automatically, config/CLI options in `autopath_core`). Decided after the pilot.
- Drawing the vector interactively (PyMOL) or scanning the site geometrically.
- Positional (CA) restraints on the protein.
- Backward projection pulls.

## Anti-goals

- Changing the behaviour or outputs of existing distance pulls.
- Axial walls of the metad funnel force acting during a pull (the pull's own spring drives z; a z_max wall
  would add unaccounted work).
- Hidden fallbacks: unsupported combinations raise ValueError("not supported: ...").

## Design

### SteeredMD

- New constructor arguments (names are the builder's call):
  - the axis: a 3-vector n (lab frame at the input structure) plus the protein-frame definition, given either
    as a `funnel_params` dict (as produced by `generate_funnel_parameters_from_trajectory` +
    `upgrade_funnel_params`) or built from the vector and the input structure with the same machinery;
  - the restraint: off by default; when on, the funnel force's lateral part (cone up to z_cc, cylinder of
    radius R_cyl beyond, k_xy) built by the existing funnel code, with **no axial walls**.
- The pull CV for projection mode is the funnel-z coordinate (`funnel_z_cv`'s CentroidFunction expression,
  protein frame), unclamped over the pull range. The spring is k/2 (z − λ)², same k and protocol as distance
  pulls, same `dx_per_move` grid, shared-λ0 equilibration (λ0 = z of the input structure, or the explicit
  `lambda0_nm`). The restraint, when on, is present during the 50 ps equilibration and the whole pull.
- Logs keep the same columns (`r_target, r_before, r_after, ...` now hold z) so `SMDData` and every estimator
  run unchanged. The header adds the CV type (`# cv=projection`), the axis, and the restraint parameters
  (`R_cylinder`, `z_cc`, `alpha`, `k_xy`, or `restraint=none`).
- Distance-specific safety logic must be adapted, not silently reused: the PBC move cap (bound the end of the
  pull by the half-box along n) and the cvpack-vs-geometry sanity check (compare z). NC autostop still works
  in projection mode but is **off in the pilot** (see below).

### Atom groups (user decision)

z, the restraint and the eq. 15 integral all use the **same two groups as the sMD pulls**: the pulled ligand
group (e.g. the Murcko anchor subset in the `sMD-murcko` runs) and the sMD pocket group. The funnel geometry
(axis offset, z_cc, z_max, protein frame) is therefore fitted to those groups by the axis helper; funnel
parameters fitted to other groups (e.g. the stored metad `funnel_params.npz`: whole ligand, contact CAs) are
not reused as they are — passing them with mismatched groups is "not supported".

### Analysis side

- `SMDData` reads the header CV type. Its distance-only conditioning must handle z: the forward completion
  check (`r_after.min() < completion_threshold_nm` drops the pull) is meaningless for z, which can be ≤ 0 in
  the bound state. For projection logs, completion means reaching the end of the requested pull range
  (builder decides the exact rule; it must not drop bound-state frames with z ≤ 0).
- Estimators (Jarzynski, cumulant, force, HS from branch `hs-estimator`) are used unchanged.
- New ΔG° helper: input a profile (coordinate in nm, free energy in kJ/mol), the site range, the bulk
  range, R_cyl, T; output ΔG°_b, the site integral, the correction term, and pKd. Shares one implementation
  with `correct_fe_for_funnel` (refactor that function to call it; its tests must keep passing).
  Site = z ≤ z_cc of the funnel; bulk reference = the plateau between the end of the cone and the pull end,
  away from the last ~0.1 nm. For G(λ) profiles the spring-width edge effect (~0.026 nm) is noted, not
  corrected.

### Axis helper

`axis_from_pulls(dat_or_dcd_files, ...)`: PCA of the ligand-COM displacement relative to the pocket COM,
oriented bound → unbound, optionally restricted to one path's pulls, returning a funnel_params-compatible
dict (axis + protein frame). Reuses the code in `generate_funnel_parameters_from_trajectory` (refactor the
PCA into a shared function rather than copying it).

## Tests (repo, CPU, toy-sized)

- Projection CV equals n·(COM_lig − COM_pocket) on a small OpenMM system, and is invariant to a rigid
  rotation of the whole system when the protein frame is used (lab-frame vector is not).
- With the restraint on, the logged `dW_protocol` contains only the spring term (restraint energy changes do
  not enter the work).
- Shared-λ0 behaviour holds in projection mode (two replicas, identical `r_target`).
- No axial wall acts: a pull past z_max proceeds with no restraint force along n.
- Header fields written and parsed; `SMDData` keeps a projection pull whose bound-state z ≤ 0.
- ΔG° helper reproduces eq. 15 on an analytic profile (Gaussian well + flat bulk) to 1e-6, and
  `correct_fe_for_funnel` gives identical results before and after the refactor.
- Distance-mode tests (existing sMD suite) unchanged.

## Pilot: WDR5 6e23 (analysis on CPU, pulls on fast GPUs)

System and inputs: `/gpfs/group/forli/mllanos/autopath_benchmarking/WDR5/openFF/6e23_A`
(`equilibration/` checkpoint and system; ligand group and pocket group as in the `sMD-murcko` runs).
Comparison set: the existing `sMD-murcko` distance pulls (3 speeds × 100: 0.005, 0.01, 0.015 nm/ps), which
were run before the shared-λ0 fix.

Axis: fitted by the axis helper to the `sMD-murcko` distance pulls with the sMD groups (pooled; the stored
metad funnel axis (0.590, 0.504, 0.631), fitted with other groups, is a cross-check only), in the protein frame.
z_cc and z_max re-derived for these groups the same way the metad funnel derives them; R_cyl = 2 Å.

Start (controller's call): λ0 = z_input − 0.15 nm via `lambda0_nm`, equilibrated 50 ps there, so the pull
crosses the whole bound well. Starting at z_input would leave the lower half of the well out of the eq. 15
site integral (≥ kT ln 2 ≈ 1.7 kJ/mol bias).

Pull length: from λ0 to z_max + 0.5 nm. NC autostop off (controller's call): with the pipeline defaults the
distance pulls on this system stopped at r ≈ 2.1–2.4 nm, before any bulk plateau; fixed-length pulls also keep
HS's full-support trimming simple.

Pilot runs: projection + funnel restraint, 25 pulls at each of 0.005 and 0.01 nm/ps. *(Controller's call;
a projection-without-restraint arm is optional and not in the pilot.)*

Execution: sbatch template with the slow-GPU exclude list
(`nodea0110,nodea0111,nodeb0417,nodeb0419,nodeb0423,nodeb0425,nodeb0427,nodeb0429,nodeb0431,nodeb0433,
nodeb1201,nodeb1203,nodeb1205,nodeb1207,nodeb1209,nodeb1215,nodeb1217`, re-checked with `sinfo` before
submitting) and a throughput guard that aborts below ~60 ns/day. Code runs from a frozen copy of the branch
(`git archive`), never from the live worktree. Outputs go to a new `.../6e23_A/sMD-projection_pilot/`.

Comparison (distance pulls subsampled to 25 per speed, several subsamples, same speeds):
- work sd vs coordinate, and the coordinate where the work ESS first drops below 5, per speed;
- HS (all four F sources) and force v→0 G0 vs number of pulls (convergence curve);
- bulk plateau present or not (slope over the last 0.3 nm before the pull end);
- agreement between F sources (F_spread);
- ΔG° (eq. 15) per estimator for the projection arm, with sd; the distance arm has no ΔG° (no restraint).
- lam0_sd should be ~0 for the new pulls (shared λ0), unlike the old distance pulls.

External reference: experimental Kd for 6e23 = 0.0001 µM (pKd 10.0; from the WDR5 paper-figure scripts,
`scratch/paper_figures/build_wdr5_notebook.py`), i.e. ΔG° ≈ −57.4 kJ/mol at 300 K. The round value may be an
assay limit (≤ 0.1 nM); report the comparison as such. The existing funnel-metad run did not converge and is
not a reference.

"Cleaner / converges faster" (controller's call, at 25 pulls per speed for both arms): the work ESS stays ≥ 5
at least 0.3 nm further along the coordinate; the four F sources agree within 2 kT on the bulk plateau; the
HS ΔG° bootstrap sd is below 2 kJ/mol. Reported against those bars, without tuning them after the fact.

Verdict to the user with numbers and figures before scaling up or touching the workflow.

## Left to the builder

- Argument names, where the projection/restraint construction code lives (`steered_md.py` vs a small helper
  module), and how `funnel_params` without a protein frame are upgraded.
- The exact completion rule for projection logs in `SMDData`.
- The pilot scripts' layout.
