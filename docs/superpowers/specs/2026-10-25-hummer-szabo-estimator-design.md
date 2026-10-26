# Hummer–Szabo estimator in `autopath.pulling` — design

Status: draft for review (2026-10-25). Branch `hs-estimator` off `develop`.

## Intent

A Hummer–Szabo (HS) estimator lives in the steered-MD package next to the Jarzynski, cumulant, force and
friction estimators, and runs on the same conditioned data. It serves two uses:

1. **PMF along the real distance**: G0(r) of the pulled COM distance (not of the spring centre λ), for the
   Kramers/barrier pipeline and for comparison with the other estimators.
2. **Per-frame equilibrium weights** for sMD frames (rows of the `.dat` and frames of the DCD), consumed by
   the learned-CV code and pose-population analyses.

The current HS implementation in `autopath/metadynamics/learned/hs.py` (branches `pocket-cv2`, `poseidon`) is
the prototype. It is retired from the learned package once this lands. Known defects of the prototype, which the
port must not inherit:

- Pulls of one speed with different lengths (NC autostop) are truncated to the shortest pull for ln⟨e^{-βW}⟩,
  but frames past that end still get weights (clamped `np.interp`). On an analytic quasistatic toy, one pull of
  ten stopped at 75 % puts 100 % of the weight on the frames past the cut.
- Coordinate is the displacement r − r_target_i(0), which is not a configuration function.
- Speeds are pooled per x-bin by an ad hoc ESS-weighted average of bin populations (`pool_speeds`).
- Requires a `# spring_constant` header; older logs (e.g. `trypsin/3ptb`) have none.

## Theory the implementation follows

At slice t (one protocol step, spring centre λ_t) the biased Hamiltonian is H0 + u(r,t),
u = k/2 (r − λ_t)². Its equilibrium density is p_t(r) = ρ0(r) e^{−β[u(r,t) − F_t]}, with F_t = ΔG(λ_t) the
free energy of the biased ensemble relative to the start. Summing over slices:

    ρ0(r) ∝ Σ_t c_t p_t(r) / Σ_t c_t e^{−β[u(r,t) − F_t]}

with slice weights c_t (classic HS: c_t = 1). Two separate ingredients:

- **p_t(r)** — always from exponential work weights, self-normalised per slice:
  p_t(r) = ⟨δ(r − r_t) e^{−βW_t}⟩ / ⟨e^{−βW_t}⟩. This is the only exact way to turn lagging non-equilibrium
  frames into equilibrium samples. There is no per-frame cumulant analogue.
- **F_t** — any valid estimate of the equilibrium F(λ_t). It is a property of H0 + u(λ), independent of
  speed and path. Three sources, selected explicitly and run side by side:
  - `jarzynski`: F_t = −kT ln⟨e^{−βW_t}⟩ (classic HS);
  - `estimator`: F_t from an existing sMD estimator's result table, either `cumulant` (per speed/path) or
    the `force` v→0 intercept (speed-independent, one F(λ) for all speeds). Mean work at a single speed is
    NOT a valid F_t and is refused;
  - `self_consistent`: F_t solved from slice overlap by binned WHAM iteration on the work-weighted per-slice
    histograms p_t(r), no external ΔG. *(Builder's call: binned WHAM, not pymbar MBAR — a dense MBAR u_kn for
    ~5 speeds × ~1500 slices × ~190k rows per path is ~1e9 entries; binned WHAM is K slices × B bins and needs
    no new dependency. Slices may be strided for this source; consecutive steps are strongly correlated.)*

  Agreement between the three is a reported convergence check, not a gate.

Per-frame weight of row (i, t):
w_it ∝ c_t e^{−βW_i(t)} / ⟨e^{−βW_t}⟩ / Σ_τ c_τ e^{−β[u(r_it, τ) − F_τ]}.

**Multiple speeds** enter one joint estimator: the denominator sums over all (speed, slice) windows, each with
its own ⟨e^{−βW}⟩ and F. `dx_per_move` is speed-independent, so slices are indexed by `step` as elsewhere in the
package. **Slice weights c_t** default to the Kish ESS of the work weights within the slice (WHAM-style N_t);
`c_t = 1` (classic HS) is an option. Classic HS and ESS-weighted are both exact for exact inputs; they differ
in variance.

**Coordinate**: absolute r (`r_after`). The denominator needs λ_τ at every slice, so it uses the shared
per-speed `protocol_grids` (median r0 + step·dx; the logged row 0 already sits at r0 + dx), not each pull's own
r0. *(Builder's call.)* The start-distance spread across replicas is 0.012–0.020 nm (WDR5 6e23, trypsin 3ptb),
about one 0.02 nm bin, so the error this introduces is below the binning. Analytic toys use a common r0 (exact
test) plus one toy with r0 jitter of that size, to show the effect stays inside the bootstrap error.

## Integration with the sMD package

- New `HummerSzaboEstimator` in `autopath/pulling/` that runs as a **separate stage after the existing
  estimators and the v→0 extrapolation** in `SMDAnalysis.run()` (opt-in flag), not as an entry in the
  `_setup_estimators` loop. Reason: the `force` source needs the per-path v→0 intercept, which exists only
  after the other estimators run. *(Builder's call.)*
- **HS results do not go into `smd_data.results`.** They are stored separately (e.g. `smd_data.hs_results`),
  so HS never becomes a candidate in `choose_reference_estimator`, never changes the shared p_eq, and is not
  passed to `_path_filtering`, the v→0 extrapolation or the friction loop. *(Builder's call.)*
- Input is `SMDData` after its existing conditioning: incomplete pulls dropped (`completion_threshold_nm`),
  cumulative work from `dW_protocol`, spring constant from the header or `recover_spring_constant`.
- **Support trimming is mandatory**: HS uses only the steps kept by `trim_results_by_n_samples_support` with
  `min_support_ratio=1.0` per (speed, path), so ⟨e^{−βW_t}⟩ always averages over all pulls. Rows and frames
  beyond the trimmed range get no weight and are counted in the output (never extrapolated).
- **Paths**: run per path like the other estimators; path PMFs are combined by the existing p_eq weights
  (`calculate_weighted_pmf` / `compute_p_eq`).
- **`estimator` F_t source**: `cumulant` reads the per-(step, speed, path) cumulant rows already in
  `smd_data.results`; `force` fits a per-path v→0 intercept from the force rows with the existing
  `extrapolate_to_v0` (realized speeds, OLS, `min_speeds=2`). F_t from `force` exists only at steps where ≥2
  speeds are supported, so with that source HS is restricted to that common step range (rows outside it get no
  weight, and the count is reported). If the requested source has not been run, or there are <2 speeds for
  `force`, raise "not supported" (no fallback).
- **Path combination of G0(r)**: as a density mixture, G0 = −kT ln Σ_p p_eq,p e^{−βG0,p}, using the shared
  reference p_eq from `compute_p_eq`. This is not the linear dG average of `calculate_weighted_pmf`, which is
  not a density mixture. *(Builder's call.)*
- G0(r) is reported on an r-bin grid (bin width option, default 0.02 nm). F_t on the step grid is reported
  alongside it per source.

## Outputs

1. **G0(r) table** per (F-source, path, combined speeds), and per speed for diagnostics: `r_bin`, `G0`,
   bootstrap `G0_sd`, `n_rows`, weighted pull ESS per bin.
2. **Per-row log-weights** for every kept `.dat` row (`trajname, step, r, logw`), normalised per
   (path, F-source), plus one global set across all pulls (p_eq,p × within-path weight) for consumers that
   don't know about paths. *(Builder's call.)*
3. **Per-frame weights** for DCD frames. Frame j of a pull is the state at the end of log row
   (j+1)·save_freq − 1 (the DCD reporter fires every save_freq·steps_per_move MD steps, and row i is written
   after move i; there is no frame of the starting state). `save_freq` is a parameter (default 5, the
   `steered_md` default) because the `.dat` header does not record it. Check: n_frames == n_rows // save_freq
   per pull, autostopped pulls included; a mismatch raises. Frames come from the `_aligned.dcd` next to each
   `.dat` (same frame count as the raw DCD).
   A standalone public function `hs_frame_weights(dat_files, temperature, f_source, save_freq=5, ...)`
   returns `(trajname, frame, logw)` without the full analysis. It builds an `SMDData` internally (so the
   completion drop and k recovery apply), treats all given pulls as one path, applies the 100 %-support trim
   itself, and estimates the given speeds jointly. It accepts `jarzynski`, `self_consistent` and `cumulant`
   (computable from the work alone); `force` is accepted only with ≥2 speeds. A single speed's pulls are a
   valid input (the learned-CV slowest-speed fit uses this).
4. **Diagnostics** per slice: work ESS; mean lag vs equilibrium lag F_eq/k; lag spread vs √(kT/k);
   F_t from the three sources and their pairwise max difference.

The logged `lag_nm` column (older logs) equals r_target − r_after and is not an input; lag is recomputed from
`r_target` and `r_after` for every log.

**Uncertainty**: bootstrap over pulls (resample whole pulls within each speed), default 200 resamples.

## Validation (in this order; a verdict is shown before step 2 feeds anything downstream)

1. **Analytic toys** (overdamped Langevin on a 1D potential with known G0(r), pulled by the same harmonic
   protocol; cheap, in tests):
   - flat potential and a harmonic well (exact Gaussian answers);
   - a double well with a barrier;
   - unequal pull lengths (autostop) → trimming makes the result match; without trimming the toy fails;
   - several speeds; two paths.
   Pass: `jarzynski`, `self_consistent` and `force` v→0 recover G0(r) within the bootstrap error over the
   supported range on every toy. `cumulant` is exact only for Gaussian work, so it is asserted on the flat and
   harmonic toys; on the double well its deviation is reported, not asserted. *(Builder's call.)*
2. **trypsin/3ptb** (`/gpfs/group/forli/mllanos/autopath_benchmarking/trypsin/3ptb/sMD`, 25 pulls × 5
   speeds, no spring-constant header). No external reference: compare HS G0(r) (three F_t sources) with the
   Jarzynski, cumulant and force-v→0 profiles, report ESS and lag diagnostics. Analysis only; no new
   simulations.

## Non-goals

- Backward pulls / Crooks / BAR.
- Fixing the Jarzynski bias: HS uses the same work, so where Jarzynski's ESS collapses so does HS's. With the
  stiff springs in use (√(kT/k) ≈ 0.026 nm) G0(r) is expected to sit close to G(λ); the value is the real
  coordinate, the per-frame weights and the F_t cross-check.
- Changing how `steered_md.py` writes logs.

## Anti-goals

- Hidden fallbacks: an unsupported case (missing source, single speed for `force`, schedule mismatch within a
  speed, frame/step mismatch) raises with "not supported".
- Weights on rows outside the common support.
- Quality gates on ESS: ESS is reported, not used to drop data.

## Follow-ups (separate changes)

- `pocket-cv2` / `poseidon`: delete `learned/hs.py`, import the frame weights from `autopath.pulling`.

## Left to the builder

- Module layout (one class in `Estimators.py` vs a new `hummer_szabo.py`), table column names beyond those
  above, and plotting.
- The WHAM iteration details for `self_consistent` (convergence tolerance, slice stride), as long as one
  iteration with F_t held at the Jarzynski values reproduces the classic HS formula.
- The bootstrap cost: 200 resamples per source is the default; the builder may lower it for
  `self_consistent` if timing needs it, and reports the number used. Analysis is cheap enough to run
  anywhere (no GPU).
