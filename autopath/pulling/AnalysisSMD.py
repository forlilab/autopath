import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Optional, Union
from collections import defaultdict, deque
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from autopath.pulling import SMDData
from autopath.pulling.Estimators import BaseEstimator, FrictionEstimator
from autopath.pulling.PathModel import DTWPathModel, NullPathModel, PathModel
from autopath.pulling.Estimators import (
    JarzynskiEstimator,
    CumulantEstimator,
    ForceEstimator,
    KramersEstimator,
    ESTIMATOR_REGISTRY,
    trim_results_by_n_samples_support,
    calculate_weighted_pmf,
    extrapolate_to_v0,
    _find_pmf_peak,
    recover_spring_constant,
)
from autopath.pulling.Convergence import window_mean_pmf, window_reference_scalar
from autopath.pulling.Diagnostics import (
    plot_work_profiles,
    plot_profile,
    plot_friction,
    plot_extrapolated_param,
    make_unbinding_paths_visualization,
)
from autopath.pulling.LigandFeatures import (
    LigandTrajectoryFeatures,
    SUPPORTED_FEATURES as LIGAND_FEATURE_POOL,
)
from autopath.pulling.support import SupportPolicy

logger = logging.getLogger("autopath.pulling")

# Trace-feature names that exist on SMDData.raw_data after __init__:
# the .dat columns (force, U_cvpack, dW_protocol, m_eff, r_before, r_after)
# plus 'lag' (computed in SMDData.__init__) and 'work' (added by
# integrate_force_dx). 'NC' is contact-related and forward-pull-specific,
# so deliberately excluded.
TRACE_FEATURE_POOL = frozenset({
    "lag", "work",
    "r_before", "r_after",
    "force", "U_cvpack", "dW_protocol", "m_eff",
})

# Default clustering feature set (hybrid) when the caller passes features=None.
# Two trace channels — reaction-coordinate progress (r_before) and spring
# fluctuation (lag) — plus inline-geom columns selected by name from the sidecar
# (geom_features=True): ligand shape (rog, npr1, npr2) and exit-direction
# (exit_1/2/3, the unbinding-route discriminator). All of these are PCA'd
# TOGETHER in DTWPathModel (pca_all_features=True → n_geom_pcs components).
#
# geom_* names here are selected from the loaded sidecar (no DCD read). Shape is
# NOT requested as bare lig_* names (that would use the un-PCA'd post-processing
# path). The pocket-distance dist_* PCA is a separate reference route and is
# never merged into this set.
_DEFAULT_TRACE_FEATURES = (
                        #    "lag", 
                           "r_before",
                           'work',
                           )
_DEFAULT_GEOM_FEATURES = (
                          "geom_rog",
                          "geom_npr1", "geom_npr2",
                        #   "geom_spherocity", 
                        #   "geom_pbf",
                        #   'geom_mindist',
                          "geom_exit_1", "geom_exit_2", "geom_exit_3"
                          )
DEFAULT_FEATURES = list(_DEFAULT_TRACE_FEATURES) + list(_DEFAULT_GEOM_FEATURES)

assert TRACE_FEATURE_POOL.isdisjoint(LIGAND_FEATURE_POOL), (
    "TRACE_FEATURE_POOL and LIGAND_FEATURE_POOL must be disjoint; got overlap: "
    f"{sorted(TRACE_FEATURE_POOL & LIGAND_FEATURE_POOL)}"
)

class SMDAnalysis:
    """Orchestrates the full dcTMD/SMD post-processing pipeline.

    Pipeline stages
    ---------------
    1. Load SMD traces from log files via :class:`SMDData`.
    2. Cluster trajectories into unbinding paths using DTW
       (:class:`DTWPathModel`) or a user-supplied path model.
    3. Estimate ΔG and W_diss per (speed, path, step) using
       Jarzynski and/or Cumulant estimators.
    4. Build a mixture PMF weighted by equilibrium path probabilities
       (p_eq), then extrapolate to v→0 when multiple speeds are present.
    5. Compute friction (γ) from the dissipative work profile and
       estimate k_off via Kramers/Pontryagin MFPT.
    6. Write CSV outputs and optional diagnostic plots.

    Feature design: ``TRACE_FEATURE_POOL``
    ---------------------------------------
    Trace features (RC progress + energetics: ``lag``, ``work``,
    ``r_before``, …) capture what distinguishes genuinely different
    unbinding routes.  Shape features (``npr1``, ``npr2``, ``pbf``) are
    optional — they are useful only when trajectories diverge spatially
    AND ligand conformation varies across those routes.  Including shape
    features by default causes paths to split on conformation rather than
    exit route when trajectories are kinematically similar (e.g. all
    autostop at the same barrier).  Pass them explicitly via the
    ``features`` argument to :meth:`run` when needed.

    Parameters
    ----------
    replica_imbalance_threshold : float, optional
        Warn when the max/min replica count ratio across paths exceeds this
        value (default 3.0).  Logged in :meth:`_path_filtering` after the
        replica-balance report per speed.
    """

    def __init__(self,
        sysname: str = 'system',
        path_model: Union[PathModel, str] = 'dtw',
        estimators:Union[list[BaseEstimator], list[str]] = ['jarzynski', 'cumulant', 'force'],
        temperature: float = 300.0,
        outdir: str = 'sMD_analysis',
        do_plots: bool = True,
        reference_pdb: Optional[str] = None,
        ligand_select: Optional[str] = 'resname UNK',
        pocket_select: Optional[str] = None,
        seed: int = 42,
        # --- filtering / weighting thresholds (all visible at construction time) ---
        filter_low_support: bool = True,
        min_support_ratio: float = 1.0,
        min_samples_per_step: int = 5,
        min_replicas_per_path: int = 5,
        n_paths: int | None = None,   # fixed K; None = silhouette-selected (deployment default)
        min_path_steps_ratio: float = 0.6,
        max_frac_neg_dG_first_half: float = 0.25,
        min_speeds_for_extrapolation: int = 2,
        replica_imbalance_threshold: float = 3.0,
    ):
        self.sysname = sysname
        self.n_paths = n_paths
        self.seed = seed
        self.temperature = temperature
        self.do_plots = do_plots
        self.reference_pdb = reference_pdb
        self.ligand_select = ligand_select
        self.pocket_select = pocket_select
        self.outdir = outdir
        self.filter_low_support = filter_low_support
        self.min_support_ratio = min_support_ratio
        self.min_samples_per_step = min_samples_per_step
        self.min_replicas_per_path = min_replicas_per_path
        self.min_path_steps_ratio = min_path_steps_ratio
        self.max_frac_neg_dG_first_half = max_frac_neg_dG_first_half
        self.min_speeds_for_extrapolation = min_speeds_for_extrapolation
        self.replica_imbalance_threshold = replica_imbalance_threshold
        self.support_policy = SupportPolicy(
            min_samples_per_step=self.min_samples_per_step,
            min_trajs_per_path=self.min_replicas_per_path,
        )
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "path_analysis"), exist_ok=True)
    
        self._setup_path_model(path_model)
        self._setup_estimators(estimators)

        return None
    
    @staticmethod
    def _ligand_resname_from_select(ligand_select: str | None) -> str:
        """Extract the residue name from an MDAnalysis-style selection string.

        Accepts strings like ``"resname UNK"`` or ``"resname UNK and not name H*"``
        and returns ``"UNK"``. Falls back to ``"UNK"`` if the selection is
        missing or doesn't follow the expected pattern.
        """
        if not ligand_select:
            return "UNK"
        toks = ligand_select.split()
        for i, t in enumerate(toks):
            if t.lower() == "resname" and i + 1 < len(toks):
                return toks[i + 1]
        return "UNK"

    def _setup_estimators(self, estimators: Union[list[BaseEstimator], list[str]]):
        estimator_map = {
            'jarzynski': JarzynskiEstimator(),
            'cumulant': CumulantEstimator(),
            'force': ForceEstimator(),
        }
        self.estimators = []
        for est in estimators:
            if isinstance(est, str):
                if est in estimator_map:
                    self.estimators.append(estimator_map[est])
                else:
                    raise ValueError(f"Unknown estimator: {est}")
            elif isinstance(est, BaseEstimator):
                self.estimators.append(est)
            else:
                raise ValueError(f"Estimator must be a string or BaseEstimator instance, got: {type(est)}")
        return

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
    
    def _setup_path_model(self, path_model: Union[PathModel, str]):
        if isinstance(path_model, str):
            if path_model == 'dtw':
                self.path_model = DTWPathModel(
                    seed=self.seed,
                    do_plots=self.do_plots,
                    outdir=self.outdir,
                )
            elif path_model == 'null':
                self.path_model = NullPathModel()
            else:
                raise ValueError(f"Unknown path model: {path_model}")
        elif isinstance(path_model, PathModel):
            self.path_model = path_model
        else:
            raise ValueError(f"path_model must be a string or PathModel instance, got: {type(path_model)}")
        return

    def _build_cluster_feature_df(self, smd, group_A=None, group_B=None,
                                  features=None, merge_features=True, ligand_sdf=None,
                                  geom_features=False, geom_merge="aligned",
                                  recompute_distances=False):
        """Build the clustering feature DataFrame for an ``SMDData``.

        Shared by :meth:`run` and :meth:`check_convergence` so both cluster on the
        same feature set. Supports three modes (mirroring ``run``):

        * **trace-only** — no clustering selection (``group_A``/``group_B`` unset).
        * **distance-only** — a clustering selection is given and ``merge_features``
          is ``False``: cluster purely on the pocket-ligand ``dist_*`` features.
        * **merged** — trace + pocket-distance + ligand-shape features combined.

        Returns a DataFrame with the ``trajname/speed/step/time`` index columns plus
        the selected feature columns, ready for ``DTWPathModel.fit_transform``.
        """
        # Resolve & validate the flat feature list. Names may be:
        #   - trace features (TRACE_FEATURE_POOL),
        #   - bare ligand-shape names (LIGAND_FEATURE_POOL -> un-PCA'd lig_* path), or
        #   - inline-geom column names prefixed 'geom_' (selected from the sidecar).
        if features is None:
            features = list(DEFAULT_FEATURES)
        geom_named = [f for f in features if f.startswith("geom_")]
        unknown = set(features) - TRACE_FEATURE_POOL - LIGAND_FEATURE_POOL - set(geom_named)
        if unknown:
            raise ValueError(
                f"Unsupported feature name(s): {sorted(unknown)}.\n"
                f"  Trace pool:  {sorted(TRACE_FEATURE_POOL)}\n"
                f"  Ligand pool: {sorted(LIGAND_FEATURE_POOL)}\n"
                f"  Or inline-geom columns prefixed 'geom_' (from the sidecar)."
            )
        trace_feats  = [f for f in features if f in TRACE_FEATURE_POOL]
        ligand_feats = [f for f in features if f in LIGAND_FEATURE_POOL]

        if not trace_feats:
            logger.warning(
                "No trace features requested; clustering will rely on ligand "
                "and/or pocket-distance features only."
            )
            # get_trace_features still emits the four index columns
            # (trajname/speed/step/time) — needed for merge_feature_sets.
            traces_feat_df = smd.get_trace_features(features=[])
        else:
            traces_feat_df = smd.get_trace_features(features=trace_feats)
        logger.info(
            f"Trace features: {trace_feats or '(none — index columns only)'}; "
            f"DataFrame columns: {list(traces_feat_df.columns)}"
        )

        dist_feat_df = None
        if group_A is not None and group_B is not None:
            dist_feat_df = smd.calculate_pocket_distances(
                group_A=group_A,
                group_B=group_B,
                recompute=recompute_distances,
            )

        # Compute ligand trajectory features (RoG and/or RDKit 3D shape
        # descriptors) once — used for clustering when requested.
        ligand_feat_dfs = []
        if ligand_feats:
            ref_pdb = self.reference_pdb or smd.reference_pdb
            if ref_pdb is None:
                logger.warning(
                    "Ligand features requested but no reference PDB is available; "
                    "skipping ligand features for clustering."
                )
            else:
                ligand_resname = self._ligand_resname_from_select(self.ligand_select)
                lig_calc = LigandTrajectoryFeatures(
                    lig_resname=ligand_resname,
                    sdf_file=ligand_sdf,
                    features=ligand_feats,
                    stride=1,
                )
                lf = lig_calc.compute(smd.traj_files, ref_pdb)
                if not lf.empty:
                    ligand_feat_dfs.append(lf)

        distance_only = dist_feat_df is not None and not merge_features
        if distance_only:
            # Distance-only clustering: when a clustering selection is provided
            # but merge_features is False, cluster purely on the pocket–ligand
            # distance features (PCA-reduced downstream in DTWPathModel). Trace
            # and ligand/shape features are deliberately excluded so paths split
            # on geometric exit route alone. dist_feat_df already carries the
            # trajname/speed/step/time index columns required for clustering.
            feat_df = dist_feat_df
            _feat_cols = [c for c in feat_df.columns
                          if c not in ('trajname', 'speed', 'step', 'time')]
            logger.info(f'Clustering on distance features only: {len(_feat_cols)} dist_* columns')
        else:
            all_feat_dfs = [traces_feat_df]
            if dist_feat_df is not None and merge_features:
                all_feat_dfs.append(dist_feat_df)
            for lf in ligand_feat_dfs:
                all_feat_dfs.append(lf)

            if len(all_feat_dfs) > 1:
                feat_df = SMDData.merge_feature_sets(*all_feat_dfs)
                _feat_cols = [c for c in feat_df.columns
                              if c not in ('trajname', 'speed', 'step', 'time')]
                logger.info(f'Clustering features: {_feat_cols}')
            else:
                feat_df = traces_feat_df
                logger.info('Clustering will be performed using trace features only')

        # Inline geom features logged during pulling (sMD_*_geom.dat sidecars).
        # A near-free, pre-computed alternative to post-processing the DCDs. When
        # explicit geom_ names are given (geom_named), only those columns are used;
        # otherwise (geom_features=True with no names) all geom_ columns are merged.
        # Skipped entirely in distance-only mode: geom features are computed on a
        # different time stride than dist_feat_df, and merging them in would also
        # contradict the "pure distance" clustering this mode is meant to provide.
        if not distance_only and (geom_features or geom_named):
            geom_df = smd.load_geom_features()
            if geom_df.empty:
                logger.warning(
                    "geom features requested but no _geom.dat sidecars found; "
                    "proceeding without inline geom features."
                )
            else:
                if geom_named:
                    missing = [c for c in geom_named if c not in geom_df.columns]
                    if missing:
                        logger.warning(
                            f"Requested geom features absent from sidecar, skipped: "
                            f"{sorted(missing)}"
                        )
                    idx_cols = [c for c in ('trajname', 'speed', 'step', 'time')
                                if c in geom_df.columns]
                    present = [c for c in geom_named if c in geom_df.columns]
                    geom_df = geom_df[idx_cols + present]
                geom_cols_final = [c for c in geom_df.columns if c.startswith("geom_")]
                if geom_cols_final:
                    feat_df = SMDData.merge_geom_features(feat_df, geom_df, mode=geom_merge)
                    logger.info(f"Merged inline geom features ({geom_merge}): {geom_cols_final}")
                else:
                    logger.warning("No usable geom_ columns after selection; skipping geom merge.")

        return feat_df, ligand_feat_dfs

    def run(self,
            sMDDdata: SMDData,
            r_range: tuple[float, float] | None = None,
            group_A: str = None,
            group_B: str = None,
            merge_features: bool = True,
            cluster_across_speeds: bool = False,
            features: list[str] | None = None,
            ligand_sdf: str | None = None,
            geom_features: bool = True,
            geom_merge: str = "aligned",
            plateau_frac: float = 0.4,
            cluster_to_boundary: bool = True,
            boundary_buffer_frac: float = 0.1,
            recompute_distances: bool = False,
            ) -> SMDData:
        """
        Parameters
        ----------
        features : list[str] | None
            Flat list of clustering feature names to compute. Each name must
            be in ``TRACE_FEATURE_POOL`` (e.g. ``"work"``, ``"lag"``,
            ``"r_before"``, ``"force"``, ``"U_cvpack"``, ``"dW_protocol"``,
            ``"m_eff"``, ``"r_after"``) or in ``LIGAND_FEATURE_POOL`` (e.g.
            ``"rog"``, ``"asphericity"``, ``"npr1"``, …). Pocket distances
            are still controlled by ``group_A``/``group_B``, not this list.
            If ``None`` (default), uses :data:`DEFAULT_FEATURES`
            (``['lag','work','r_before']``). To add shape descriptors pass
            e.g. ``features=['lag','work','r_before','npr1','npr2','pbf']``.
        ligand_sdf : str | None
            Path to an SDF with the ligand and correct bond orders.
            Required if any RDKit feature is in ``features``.
        """

        if self.reference_pdb is None:
            self.reference_pdb = sMDDdata.reference_pdb
            logger.warning(f"No reference PDB provided, using from SMDData: {self.reference_pdb}")

        # Derive the clustering boundary (bound -> rupture) from the force-plateau
        # so clustering ignores the bulk-solvent tail. This is applied to the
        # CLUSTERING feature matrix ONLY (below) — never to raw_data, so the
        # estimators/PMF keep the full pull. An explicit user r_range still
        # restricts everything via filter_by_r_range (legacy behavior).
        boundary_cap = None
        if r_range is None and cluster_to_boundary:
            rd = sMDDdata.raw_data
            if {'r_coord', 'force', 'speed'}.issubset(rd.columns):
                r_lo, r_hi = float(rd['r_coord'].min()), float(rd['r_coord'].max())
                r_ts = KramersEstimator.force_plateau_boundary(
                    rd, 0.0, r_lo, r_hi, plateau_frac,
                )
                if r_ts is not None and r_ts > r_lo:
                    boundary_cap = min(r_ts * (1.0 + boundary_buffer_frac), r_hi)
                    logger.info(
                        f"Clustering restricted to force-plateau boundary: r <= "
                        f"{boundary_cap:.2f} nm (rupture {r_ts:.2f} + "
                        f"{boundary_buffer_frac:.0%} buffer, plateau_frac={plateau_frac}); "
                        f"excludes bulk-solvent tail (estimators keep the full range). "
                        f"Set cluster_to_boundary=False to disable."
                    )
                else:
                    logger.warning(
                        "cluster_to_boundary: force-plateau boundary not found; "
                        "clustering over full r range."
                    )

        if r_range is not None:
            sMDDdata.filter_by_r_range(r_range, sMDDdata.r_column)

        if (group_A is not None and group_B is not None
                and merge_features and cluster_across_speeds):
            logger.warning("Clustering using all speeds together on trace features. Those depend on speed, so this may lead to suboptimal clustering. Consider setting cluster_across_speeds=False or using only distance features for clustering.")

        # Build the clustering feature DataFrame (trace-only / distance-only /
        # merged) — shared with check_convergence so both cluster identically.
        feat_df, ligand_feat_dfs = self._build_cluster_feature_df(
            sMDDdata, group_A=group_A, group_B=group_B,
            features=features, merge_features=merge_features, ligand_sdf=ligand_sdf,
            geom_features=geom_features, geom_merge=geom_merge,
            recompute_distances=recompute_distances,
        )

        # Apply the force-plateau boundary to the CLUSTERING matrix only. raw_data
        # (and hence the estimators/PMF) is left at full range. Map r via raw_data's
        # (trajname, step) -> r_coord so it works regardless of which features are used.
        if boundary_cap is not None and not feat_df.empty:
            _rc = sMDDdata.raw_data[['trajname', 'step', 'r_coord']]
            feat_df = feat_df.merge(_rc, on=['trajname', 'step'], how='left')
            _n0 = len(feat_df)
            feat_df = feat_df[feat_df['r_coord'] <= boundary_cap].drop(columns='r_coord')
            logger.info(
                f"Clustering feature rows after boundary cut: {_n0} -> {len(feat_df)} "
                f"(r <= {boundary_cap:.2f} nm)."
            )

        trajectory_files = {}
        for traj in sMDDdata.traj_files:
            trajname = SMDData._traj_to_log_name(traj)
            trajectory_files[trajname] = (self.reference_pdb, traj)

        path_mappings = self.path_model.fit_transform(
            feat_df,
            reference_pdb=self.reference_pdb,
            ligand_select=self.ligand_select,
            trajectory_files=trajectory_files,
            cluster_across_speeds=cluster_across_speeds,
            pocket_select=group_B,
            n_paths=self.n_paths,      # None -> silhouette search (deployment default)
        )

        sMDDdata.raw_data['path'] = sMDDdata.raw_data['trajname'].map(path_mappings)

        if self.filter_low_support:
            sMDDdata.raw_data = trim_results_by_n_samples_support(
                sMDDdata.raw_data,
                min_samples=self.support_policy.min_samples_per_step,
                min_support_ratio=self.min_support_ratio,
            )

        # Fit estimators sequentially; some may rely on path assignments,
        # so this must run before any path filtering.
        n_speeds = int(sMDDdata.raw_data['speed'].nunique())
        self.estimators = self._drop_force_if_single_speed(self.estimators, n_speeds)
        for estimator in self.estimators:
            logger.info(f"Fitting estimator: {estimator.name}")
            sMDDdata = estimator.fit_transform(sMDDdata)

        
        sMDDdata = self._path_filtering(sMDDdata)

        if sMDDdata.results.empty:
            raise RuntimeError(
                "All (speed, path) groups were removed by path filtering — "
                "no data remains for PMF construction. "
                "Consider running more SMD replicas or using slower pulling speeds."
            )

        # Determine which estimators still have results after path filtering.
        # Path filtering may drop cumulant rows (e.g. frac_neg_dG_first_half filter)
        # while leaving jarzynski intact — continue with whatever survives.
        results_estimators = set(sMDDdata.results['estimator'].dropna().unique())
        active_estimators = [e for e in self.estimators if e.name in results_estimators]
        if len(active_estimators) < len(self.estimators):
            dropped = [e.name for e in self.estimators if e.name not in results_estimators]
            logger.warning(
                f"Estimator(s) {dropped} have no results after path filtering — "
                f"continuing with {[e.name for e in active_estimators]} only."
            )

        # Compute p_eq once from a robustness-selected reference estimator and
        # share it across all active estimators (including 'force', which has
        # no dG of its own to derive weights from). Paths with negative dG
        # values trigger a warning; those bins are excluded from the
        # integrand (contribute 0 to Z), which smoothly downweights
        # artifact-heavy paths in the mixture.
        ref_estimator = SMDData.choose_reference_estimator(sMDDdata.results)
        logger.info(f"[p_eq] shared reference estimator (robustness): {ref_estimator}")
        ref_weights = self.compute_p_eq(sMDDdata, estimator=ref_estimator)
        weights_by_estimator = {est.name: ref_weights for est in active_estimators}

        self._write_path_quality(sMDDdata, weights_by_estimator)

        self.mixture_pmfs = calculate_weighted_pmf(
            smd_data=sMDDdata,
            weights_by_estimator=weights_by_estimator,
        )

        if self.mixture_pmfs.empty or 'speed' not in self.mixture_pmfs.columns:
            raise RuntimeError(
                "Weighted PMF calculation produced no output. "
                "All paths may have been excluded during PMF construction "
                "(e.g. no common support across paths at any speed)."
            )

        weighted_pmf_v0 = {}
        if self.mixture_pmfs['speed'].nunique() >= 2:
            for pcol in ['dG', 'Wdiss']:
                v0_df = extrapolate_to_v0(self.mixture_pmfs, param=pcol,
                                           min_speeds=self.min_speeds_for_extrapolation,
                                           realized_speed_map=sMDDdata.realized_speed_map)
                if not v0_df.empty:
                    weighted_pmf_v0[pcol] = v0_df
        else:
            logger.warning("Not enough speeds to perform v=0 extrapolation. Skipping.")

        # Fold v=0 extrapolated rows into mixture_pmfs (speed=0.0 rows with stat columns).
        # Rename R2 to {param}_R2 to distinguish dG vs Wdiss fit quality.
        if weighted_pmf_v0:
            _key = ['step', 'r_coord', 'estimator', 'speed']
            _parts = [df.rename(columns={'R2': f'{pcol}_R2'})
                      for pcol, df in weighted_pmf_v0.items()]
            _v0 = _parts[0]
            for _part in _parts[1:]:
                _new = [c for c in _part.columns if c not in _v0.columns]
                _v0 = _v0.merge(_part[_key + _new], on=_key, how='outer')
            self.mixture_pmfs = pd.concat([self.mixture_pmfs, _v0], ignore_index=True)

        self.mixture_pmfs.to_csv(f'{self.outdir}/mixture_pmfs.csv', index=False)

        friction_est = FrictionEstimator(use_spline=False,
                                         realized_speed_map=sMDDdata.realized_speed_map)
        df = self.mixture_pmfs.copy()

        friction_deriv_results = []
        friction_regress_results = []
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

        friction_df = pd.concat(friction_deriv_results + friction_regress_results, ignore_index=True)
        friction_df.to_csv(os.path.join(self.outdir, 'friction.csv'), index=False)

        # --- Mean-force TI PMF on the deconvolved coordinate z = lambda - Feq/k (force only) ---
        if any(e.name == 'force' for e in active_estimators) and \
           sMDDdata.raw_data['speed'].nunique() >= 2:
            try:
                _k = recover_spring_constant(sMDDdata.raw_data,
                                             logged_k=sMDDdata.spring_constant)
                _force_rows = sMDDdata.results[sMDDdata.results['estimator'] == 'force']
                _ti_df, _ti_diag = friction_est.meanforce_ti_pmf(_force_rows, ref_weights, _k)
                if _ti_df is not None:
                    _ti_df.to_csv(os.path.join(self.outdir, 'force_meanforce_ti.csv'), index=False)
                    # lambda-vs-z endpoint diagnostic (finite-spring smearing)
                    _lam_end = None
                    _fv0 = self.mixture_pmfs[(self.mixture_pmfs['estimator'] == 'force') &
                                             (self.mixture_pmfs['speed'] == 0.0)].sort_values('r_coord')
                    if not _fv0.empty:
                        _lam_end = float(_fv0['dG'].iloc[-1] - _fv0['dG'].iloc[0])
                    logger.info(f"[meanforce_ti] k={_ti_diag['k']:.1f} | dG_z endpoint="
                                f"{_ti_diag['dG_z_endpoint']:.0f} kJ/mol | dG_lambda endpoint="
                                f"{_lam_end} | z-divergence median/max="
                                f"{_ti_diag['z_divergence_nm_median']*1e3:.1f}/"
                                f"{_ti_diag['z_divergence_nm_max']*1e3:.1f} pm | "
                                f"fell_back_paths={_ti_diag['fell_back_paths']}")
            except Exception as _exc:
                logger.warning(f"[meanforce_ti] skipped: {_exc}")

        # --- Kramers / Pontryagin MFPT k_off ---
        _dG_v0 = weighted_pmf_v0.get('dG')   # v→0 extrapolated PMF (None if single speed)
        _kramers = KramersEstimator(temperature=self.temperature)
        try:
            _koff_df = _kramers.compute(
                mixture_pmfs=self.mixture_pmfs,
                friction_df=friction_df,
                dG_extrapolated=_dG_v0,
                force_df=sMDDdata.raw_data,          # restraint force for boundary detection
                boundary_method="force_plateau",     # kinetics-motivated abs_r (HSP90-validated)
                plateau_frac=plateau_frac,
            )
            if not _koff_df.empty:
                _koff_df.to_csv(os.path.join(self.outdir, 'koff_kramers.csv'), index=False)
                logger.info(f"[Kramers] koff_kramers.csv written ({len(_koff_df)} rows)")
        except Exception as _exc:
            logger.warning(f"[Kramers] k_off estimation skipped: {_exc}")

        # --- per-path k_off (diagnostic) ---
        try:
            _koff_per_path = _kramers.compute_per_path(
                per_path_results=sMDDdata.results,
                friction_df=friction_df,
                weights_by_estimator=weights_by_estimator,
            )
            if not _koff_per_path.empty:
                _koff_per_path.to_csv(
                    os.path.join(self.outdir, 'koff_per_path.csv'), index=False
                )
                logger.info(f"[Kramers/per-path] koff_per_path.csv written ({len(_koff_per_path)} rows)")
        except Exception as _exc:
            logger.warning(f"[Kramers/per-path] skipped: {_exc}")

        # Regenerate the unbinding-paths PSE with friction colouring now that friction.csv is ready.
        # path_model.fit_transform ran earlier (before friction was computed) so the first PSE has
        # no friction overlay. We rebuild paths_dict from medoid_to_path and overwrite that PSE.
        if hasattr(self.path_model, 'medoid_to_path') and self.path_model.medoid_to_path:
            try:
                friction_csv_path = os.path.join(self.outdir, 'friction.csv')
                _paths_dict: dict = {}
                for medoid_name, path_id in self.path_model.medoid_to_path.items():
                    if path_id not in _paths_dict:
                        _paths_dict[path_id] = []
                    if medoid_name in trajectory_files:
                        _paths_dict[path_id].append(trajectory_files[medoid_name])
                if _paths_dict:
                    make_unbinding_paths_visualization(
                        paths=_paths_dict,
                        reference_pdb=self.reference_pdb,
                        ligand_select=self.ligand_select,
                        outdir=os.path.join(self.path_model.outdir, "path_analysis"),
                        color_by_friction=False,
                        friction_csv=friction_csv_path,
                        pocket_select=group_B if group_B is not None else self.pocket_select,
                        # Mirror the path model's figure settings so this rebuild
                        # does not silently revert to the legacy PSE defaults.
                        output_format=getattr(self.path_model, 'viz_output_format', 'pse'),
                        n_lig_conformations=getattr(
                            self.path_model, 'viz_n_lig_conformations', 5),
                        stick_transparency_ladder=getattr(
                            self.path_model, 'viz_stick_transparency_ladder', None),
                    )
                    logger.info("Friction-coloured unbinding paths PSE regenerated.")
            except Exception as _exc:
                logger.warning(f"Could not regenerate friction-coloured PSE: {_exc}")

        if self.do_plots:
            # Two dashed markers on the profiles: the transition state (mean|force|
            # peak / rupture) and the Kramers absorbing boundary (force-plateau
            # decay point that the MFPT integrates to). Both from the same
            # per-speed force profile as the kinetics.
            ts_r = abs_r = None
            _rd = sMDDdata.raw_data
            if {'r_coord', 'force', 'speed'}.issubset(_rd.columns):
                _lo, _hi = float(_rd['r_coord'].min()), float(_rd['r_coord'].max())
                ts_r = KramersEstimator.force_peak_r(_rd, 0.0, _lo, _hi)
                abs_r = KramersEstimator.force_plateau_boundary(_rd, 0.0, _lo, _hi, plateau_frac)

            for estimator in active_estimators:
                plot_work_profiles(sMDDdata.results, estimator=estimator.name, outdir=self.outdir)
            for vcol in ['Wdiss', 'dG']:
                plot_profile(
                    self.mixture_pmfs,
                    value_col=vcol,
                    hue='estimator',
                    ylabel='Energy (kJ/mol)',
                    outdir=self.outdir,
                    ts_line=ts_r,
                    abs_line=abs_r,
                )
            if friction_df is not None and not friction_df.empty:
                plot_friction(friction_df, outdir=self.outdir, ts_line=ts_r, abs_line=abs_r)
            for pcol, v0_df in weighted_pmf_v0.items():
                plot_extrapolated_param(
                    v0_df,
                    param=pcol,
                    outfname=os.path.join(self.outdir, f'{pcol}_extrapolated.svg'),
                    ts_line=ts_r,
                    abs_line=abs_r,
                )

        # Merge trajectory-derived features into raw_data so they appear in the CSV.
        # ligand_feat_dfs was populated early (before clustering) and is reused here
        # to avoid redundant trajectory reads.
        _BASE_COLS = {
            "step", "time", "r_target", "r_before", "r_after", "NC",
            "force", "U_cvpack", "dW_protocol", "m_eff",
            "trajname", "speed", "repid", "work", "lag", "r_coord", "path",
            "n_samples", "support_frac", "support_ok", "n_ref",
        }
        if ligand_feat_dfs:
            for lf in ligand_feat_dfs:
                sMDDdata.raw_data = SMDData.merge_feature_sets(sMDDdata.raw_data, lf)
            new_cols = [c for c in sMDDdata.raw_data.columns if c not in _BASE_COLS]
            if new_cols:
                sMDDdata.raw_data[new_cols].describe().to_csv(
                    os.path.join(self.outdir, "trajectory_features_summary.csv")
                )
                logger.info(f"Trajectory features written to CSV: {new_cols}")

        sMDDdata.raw_data.to_csv(f'{self.outdir}/sMD_processed_data.csv', index=False)

        return sMDDdata
        
    def check_convergence(self,
        logs: list[str],
        speeds: list[float] = None,
        quantities: list[str] = ['dG_weighted'],
        estimator_name: str = 'cumulant',
        group_A: str = None,
        group_B: str = None,
        features: list | None = None,   # clustering feature list (mirrors run())
        merge_features: bool = False,   # merge trace + pocket-distance + ligand feats
        ligand_sdf: str | None = None,  # needed only if ligand-shape features requested
        geom_features: bool = True,     # merge inline geom sidecars into clustering (hybrid default)
        geom_merge: str = "aligned",     # 'impute' (nearest-fill) or 'aligned' (exact)
        min_replicas: int = 5,
        trace_min_replicas: int = 3,  # start building PMF traces before convergence checking begins
        tol_rmsd: float = 4.0,     # kJ/mol
        tol_barrier: float = 3.0,  # kJ/mol
        tol_r_ts: float = 0.1,     # nm — 1 Å change in TS position
        trim_fraction: float = 0.1,      # drop steps where fewer than (1-trim_fraction) of replicas contributed; 0=no trimming
        min_common_points: int = 5,
        conv_window: int = 5,      # rungs averaged to form the comparison reference; 1 == previous-rung
        boundary_method: str = "force_plateau",  # TS/barrier detector: "force_plateau" (restraint-force, default) or "pmf_peak"
        plateau_frac: float = 0.4,
        restrict_rmsd_to_boundary: bool = True,   # compute PMF-RMSD only up to the force-plateau boundary (+buffer), not the noisy solvent tail
        cluster_to_boundary: bool = True,         # restrict clustering to r <= boundary (mirrors run()); else fraction-trim
        boundary_buffer_frac: float = 0.1,        # extend the boundary cap by this fraction of r_ts
        min_samples_per_step_conv: int = 3,
        min_trajs_per_path_conv: int = 2,
        recompute_distances: bool = False,
        trace_per_path: bool = False,  # emit per-path PMF trace rows alongside "mixture" (expensive on disk; opt-in)
    ):
        """Check PMF convergence as replica count grows, per pulling speed.

        For each speed, replicas are added one at a time in sorted order.
        After every addition the mixture PMF is rebuilt from the running
        estimator statistics.  PMF(k) is compared against a reference formed
        by averaging over the last ``conv_window`` rungs (``conv_window=1``
        reduces to the previous rung) on three criteria: RMSD over the
        common r-range, change in barrier height, and change in
        transition-state position.

        Returns
        -------
        convergence_df : pd.DataFrame
            One row per (speed, k) with columns: ``n_replicas``,
            ``{quantity}-rmsd``, ``barrier_delta``, ``r_ts_delta``,
            ``barrier_height``, ``r_ts``, ``converged``, and ``reason``
            (when convergence criteria cannot be evaluated).
        traces_df : pd.DataFrame
            Long-form PMF traces — one row per (speed, k, step) — for
            plotting PMF evolution with increasing replica count. Contains
            only ``path="mixture"`` rows unless ``trace_per_path=True``, in
            which case per-path rows are added alongside the mixture rows
            (expensive on disk — see ``trace_per_path``).
        """

        if isinstance(quantities, str):
            quantities = [quantities]

        main_quantity = quantities[0]
        value_col = main_quantity[:-9] if main_quantity.endswith('_weighted') else main_quantity

        conv_policy = SupportPolicy(
            min_samples_per_step=min_samples_per_step_conv,
            min_trajs_per_path=min_trajs_per_path_conv,
        )

        allowed_estimators = {
            'jarzynski',
            'cumulant',
            'force',
        }
        if estimator_name not in allowed_estimators:
            raise ValueError(
                f"Unknown estimator '{estimator_name}'. "
                f"Allowed values: {sorted(allowed_estimators)}"
            )

        if speeds is None:
            speeds = sorted(set(SMDData._speed_from_log(fn) for fn in logs))

        # 'force' needs >=2 speeds simultaneously for its v->0 extrapolation,
        # so it cannot use the per-speed loop below; dispatch to the
        # multi-speed ladder instead. No per-rung PMF traces are produced
        # (the ladder mixes speeds, not a single-speed running PMF), so the
        # traces frame returned here is deliberately empty rather than
        # fabricated.
        if estimator_name == 'force':
            from .ConvergenceLadder import force_convergence_ladder
            conv_df = force_convergence_ladder(
                self, logs, speeds,
                trace_min_replicas=trace_min_replicas,
                trim_fraction=trim_fraction,
                min_common_points=min_common_points,
                conv_window=conv_window,
            )
            return conv_df, pd.DataFrame()

        convergence_all_speeds = []
        traces_all_speeds = []

        for speed in speeds:
            
            # filter logs by speed
            speed_logs = [fn for fn in logs if SMDData._speed_from_log(fn) == speed]

            if len(speed_logs) < min_replicas:
                raise ValueError(
                    f"Not enough replicas for speed={speed}: {len(speed_logs)}"
                )

            # Chronological, not by filename ID: the ID is a bare HHMMSS with
            # no date, so it would order multi-day cells by time of day and
            # rung k would not be "the first k replicas run".
            speed_logs = sorted(speed_logs, key=SMDData._replica_start_datetime)
            logger.info(f"Checking convergence for speed={speed} with {len(speed_logs)} replicas")

            smd = SMDData(
                speed_logs,
                sysname=self.sysname,
                temperature=self.temperature,
                reference_pdb=self.reference_pdb,
            )

            # Build clustering features the same way run() does (trace-only /
            # distance-only / merged), so convergence clustering matches the
            # main analysis for every classification.
            feat_df, _ = self._build_cluster_feature_df(
                smd, group_A=group_A, group_B=group_B,
                features=features, merge_features=merge_features, ligand_sdf=ligand_sdf,
                geom_features=geom_features, geom_merge=geom_merge,
                recompute_distances=recompute_distances,
            )

            protocol_grid = smd.protocol_grids[speed].set_index('step')['r_target_protocol']

            # Force-plateau boundary (+buffer) for this speed, computed once from the
            # full per-speed force data (stable across replica counts). Used to (a)
            # restrict clustering to the unbinding route and (b) cap the PMF-RMSD
            # window — both excluding the noisy, dissipation-dominated solvent tail.
            boundary_cap = None
            if cluster_to_boundary or restrict_rmsd_to_boundary:
                _rd = smd.raw_data[smd.raw_data['speed'] == speed]
                if {'r_coord', 'force', 'speed'}.issubset(_rd.columns):
                    _r_lo, _r_hi = float(_rd['r_coord'].min()), float(_rd['r_coord'].max())
                    _rts = KramersEstimator.force_plateau_boundary(
                        _rd, speed, _r_lo, _r_hi, plateau_frac,
                    )
                    if _rts is not None:
                        boundary_cap = min(_rts * (1.0 + boundary_buffer_frac), _r_hi)
                        logger.info(
                            f"[convergence] force-plateau boundary r <= {boundary_cap:.2f} nm "
                            f"(rupture {_rts:.2f} nm + {boundary_buffer_frac:.0%} buffer, "
                            f"plateau_frac={plateau_frac})."
                        )
                    else:
                        logger.warning(
                            "[convergence] force-plateau boundary not found; using full "
                            "PMF range for clustering and RMSD."
                        )

            # Cluster (mirroring run()): when cluster_to_boundary and the boundary is
            # found, restrict clustering to r <= boundary; otherwise fall back to the
            # fraction-trim that keeps the first (1-trim_fraction) of frames.
            feat_df_cl = feat_df
            fit_r_range = 1 - trim_fraction
            if cluster_to_boundary and boundary_cap is not None:
                _r_of_step = protocol_grid.reindex(feat_df['step']).to_numpy(dtype=float)
                _restricted = feat_df[_r_of_step <= boundary_cap]
                if len(_restricted) >= 2:
                    feat_df_cl = _restricted
                    fit_r_range = None   # boundary replaces the fraction-trim
                    logger.info(
                        f"[convergence] clustering restricted to r <= {boundary_cap:.2f} nm "
                        f"(excludes bulk-solvent tail)."
                    )

            clusterer = DTWPathModel(
                seed=self.seed,
                do_plots=False,
                outdir=self.outdir,
            )
            path_mappings = clusterer.fit_transform(feat_df_cl, r_range=fit_r_range)

            smd.raw_data['path'] = smd.raw_data['trajname'].map(path_mappings)

            speed_data = smd.raw_data[smd.raw_data['speed'] == speed].copy()

            rmsd_r_cap = boundary_cap if restrict_rmsd_to_boundary else None

            traj_order = [os.path.basename(fn)[:-4] for fn in speed_logs]
            traj_data_map = {
                traj: speed_data[speed_data['trajname'] == traj][['step', 'path', 'work']].copy()
                for traj in traj_order
            }

            running_stats = defaultdict(lambda: {'n': 0, 'sum_w': 0.0, 'sum_w2': 0.0, 'sum_exp': 0.0})
            running_samples = defaultdict(list)
            path_traj_counts = defaultdict(int)

            rows = []
            pmf_records = []
            _w = max(1, int(conv_window))
            hist_pmf = deque(maxlen=_w)
            hist_barrier = deque(maxlen=_w)
            hist_r_ts = deque(maxlen=_w)

            for k, trajname in enumerate(traj_order, start=1):
                traj_df = traj_data_map.get(trajname)
                if traj_df is None or traj_df.empty:
                    continue

                current_path = traj_df['path'].dropna().iloc[0] if not traj_df['path'].dropna().empty else None
                if current_path is None:
                    continue

                path_traj_counts[current_path] += 1

                for _, row in traj_df.iterrows():
                    step = int(row['step'])
                    path = row['path']
                    work = float(row['work'])
                    key = (step, path)
                    stats = running_stats[key]
                    stats['n'] += 1
                    stats['sum_w'] += work
                    stats['sum_w2'] += work * work
                    stats['sum_exp'] += np.exp(-smd.beta * work)
                    running_samples[key].append(work)

                # Stage 1: skip entirely below trace threshold
                if k < trace_min_replicas:
                    continue

                # Stage 2: compute PMF for all k >= trace_min_replicas
                results_df = self._results_from_running_stats(
                    running_stats=running_stats,
                    running_samples=running_samples,
                    speed=speed,
                    protocol_grid=protocol_grid,
                    estimator_name=estimator_name,
                    beta=smd.beta,
                    policy=conv_policy,
                )

                pmf_k, per_path_pmf_k, n_paths_k = self._weighted_series_from_results(
                    results_df=results_df,
                    path_traj_counts=path_traj_counts,
                    value_col=value_col,
                    beta=smd.beta,
                    trim_fraction=trim_fraction,
                    policy=conv_policy,
                    return_paths=True,
                )

                # Empty PMF at this k: skip without touching the history — appending an
                # empty Series would make hist_pmf non-empty and bypass the
                # initialization guard below, leaving hist_barrier/hist_r_ts unset.
                if pmf_k.empty:
                    continue

                # always store PMF trace (from trace_min_replicas onwards)
                for step, val in pmf_k.items():
                    r_coord = protocol_grid.loc[step]
                    pmf_records.append({
                        "step": step,
                        "r_coord": r_coord,
                        "speed": speed,
                        "path": "mixture",
                        "n_replicas": k,
                        "quantity": main_quantity,
                        "value": val,
                    })
                    # Per-path PMF at this step, one row per path that actually
                    # contributed weight to the mixture value above (same
                    # gating as _weighted_series_from_results). Lets a stalled
                    # RMSD be diagnosed as genuine under-sampling vs. a DTW
                    # path-model relabeling between rungs. Opt-in: doubles (or
                    # more) the size of the traces frame, so gated behind
                    # trace_per_path (default False).
                    if trace_per_path:
                        for path_label, path_val in per_path_pmf_k.get(step, {}).items():
                            pmf_records.append({
                                "step": step,
                                "r_coord": r_coord,
                                "speed": speed,
                                "path": path_label,
                                "n_replicas": k,
                                "quantity": main_quantity,
                                "value": path_val,
                            })

                # Running per-k restraint-force profile (first k replicas) for the
                # force-plateau TS detector; None when using the PMF-peak method.
                force_k = (speed_data[speed_data['trajname'].isin(traj_order[:k])]
                           if boundary_method == "force_plateau" else None)

                # Stage 3: first PMF in the trace window — initialize history, no comparison yet
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

                # === convergence comparison (all k > trace_min_replicas) ===
                # Metrics are computed from here regardless of min_replicas, so plots
                # show early trends even when convergence is declared just after the floor.

                barrier_height, r_ts = self._compute_barrier_rts(
                    pmf_k, 1.0/smd.beta, protocol_grid,
                    force_df=force_k, speed=speed,
                    boundary_method=boundary_method, plateau_frac=plateau_frac,
                )

                # compare PMF(k) vs the window-averaged reference over the last _w rungs
                ref_pmf = window_mean_pmf(hist_pmf)
                ref_barrier = window_reference_scalar(hist_barrier)
                ref_r_ts = window_reference_scalar(hist_r_ts)

                common_r = pmf_k.index.intersection(ref_pmf.index)

                # Cap the RMSD window at the force-plateau boundary (+buffer): drop
                # steps past rupture whose noisy plateau otherwise dominates the RMSD.
                if rmsd_r_cap is not None and len(common_r) > 0:
                    _r_of_step = protocol_grid.reindex(common_r).to_numpy()
                    common_r = common_r[_r_of_step <= rmsd_r_cap]

                # NaN-aware barrier/r_ts deltas between PMF(k) and the window
                # reference (the median over the last conv_window rungs; with
                # conv_window=1 that reference is the previous rung):
                # - both NaN: no peak in either → criterion waived (True)
                # - one NaN: peak appeared/disappeared → not converged (inf)
                # - both finite: normal absolute difference
                # The two NaN branches are defensive only and are currently
                # unreachable: _compute_barrier_rts always returns finite values
                # (force_plateau interpolates onto the PMF; pmf_peak falls back to
                # nanargmax(dG) when no prominent peak is found), and the PMF
                # itself is NaN-free, so window_reference_scalar never sees an
                # all-NaN window either. They are kept in case a future TS
                # detector is allowed to report "no peak".
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

                if len(common_r) < min_common_points:
                    rows.append({
                        "speed": speed,
                        "path": "mixture",
                        "n_replicas": k,
                        f"{main_quantity}-rmsd": np.nan,
                        "barrier_delta": barrier_delta,
                        "r_ts_delta": r_ts_delta,
                        "barrier_height": barrier_height,
                        "r_ts": r_ts,
                        "converged": False,
                        "reason": "insufficient_overlap",
                        "n_common_points": len(common_r),
                        "n_paths": n_paths_k,
                    })
                    hist_pmf.append(pmf_k)
                    hist_barrier.append(barrier_height)
                    hist_r_ts.append(r_ts)
                    continue

                # Trim to well-supported steps: keep only steps where the fraction
                # of replicas that reached that step >= (1 - trim_fraction).
                # This is adaptive — low-coverage tails (noisy region) are excluded
                # automatically regardless of where they fall in the r range.
                # Set trim_fraction=0.0 to use the full range (e.g. membrane pulls).
                if trim_fraction > 0.0 and not results_df.empty and 'n_samples' in results_df.columns:
                    step_support = results_df.groupby('step')['n_samples'].max()
                    n_ref = float(step_support.max())
                    if n_ref > 0:
                        supported_steps = step_support[
                            step_support / n_ref >= (1.0 - trim_fraction)
                        ].index
                        common_r = common_r[common_r.isin(supported_steps)]

                if len(common_r) < min_common_points:
                    rows.append({
                        "speed": speed,
                        "path": "mixture",
                        "n_replicas": k,
                        f"{main_quantity}-rmsd": np.nan,
                        "barrier_delta": barrier_delta,
                        "r_ts_delta": r_ts_delta,
                        "barrier_height": barrier_height,
                        "r_ts": r_ts,
                        "converged": False,
                        "reason": "insufficient_support",
                        "n_common_points": len(common_r),
                        "n_paths": n_paths_k,
                    })
                    hist_pmf.append(pmf_k)
                    hist_barrier.append(barrier_height)
                    hist_r_ts.append(r_ts)
                    continue

                yN = pmf_k.loc[common_r].values
                yNm1 = ref_pmf.loc[common_r].values

                pmf_rmsd = np.sqrt(np.mean((yN - yNm1) ** 2))

                converged = (
                    (pmf_rmsd < tol_rmsd) and
                    (np.isnan(barrier_delta) or barrier_delta < tol_barrier) and
                    (np.isnan(r_ts_delta)    or r_ts_delta   < tol_r_ts)
                )

                rows.append({
                    "speed": speed,
                    "path": "mixture",
                    "n_replicas": k,
                    f"{main_quantity}-rmsd": pmf_rmsd,
                    "barrier_delta": barrier_delta,
                    "r_ts_delta": r_ts_delta,
                    "barrier_height": barrier_height,
                    "r_ts": r_ts,
                    "converged": converged,
                    "decision_quantity": main_quantity,
                    "n_common_points": len(common_r),
                    "n_paths": n_paths_k,
                })

                # append to bounded history (window reference for next comparison)
                hist_pmf.append(pmf_k)
                hist_barrier.append(barrier_height)
                hist_r_ts.append(r_ts)

            conv_df = pd.DataFrame(rows)
            convergence_all_speeds.append(conv_df)
            traces_df = pd.DataFrame(pmf_records)
            traces_all_speeds.append(traces_df)
            
        convergence_all_speeds = pd.concat(convergence_all_speeds, ignore_index=True) if convergence_all_speeds else pd.DataFrame()
        traces_all_speeds = pd.concat(traces_all_speeds, ignore_index=True) if traces_all_speeds else pd.DataFrame()

        return convergence_all_speeds, traces_all_speeds

    @staticmethod
    def _results_from_running_stats(
        running_stats: dict,
        running_samples: dict,
        speed: float,
        protocol_grid: pd.Series,
        estimator_name: str,
        beta: float,
        policy: "SupportPolicy | None" = None,
    ) -> pd.DataFrame:
        estimator_cls = ESTIMATOR_REGISTRY.get(estimator_name)
        if estimator_cls is None:
            raise ValueError(
                f"Unknown estimator '{estimator_name}'. "
                f"Allowed: {sorted(ESTIMATOR_REGISTRY)}"
            )

        rows = []

        for (step, path), stats in running_stats.items():
            n = stats['n']
            if n <= 0:
                continue
            if policy is not None and not policy.estimable_step(n):
                continue

            raw_W = np.asarray(running_samples.get((step, path), []), dtype=float)
            if raw_W.size == 0:
                continue

            result = estimator_cls.estimate_dG(raw_W, beta)
            if result is None:
                continue

            rows.append({
                'step': step,
                'speed': speed,
                'path': path,
                'n_samples': n,
                'Wmean': result['Wmean'],
                'Wdiss': result['Wdiss'],
                'dG': result['dG'],
                'r_coord': protocol_grid.loc[step],
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_barrier_rts(
        pmf_k: pd.Series,
        kBT: float,
        protocol_grid: pd.Series,
        force_df: pd.DataFrame | None = None,
        speed: float | None = None,
        boundary_method: str = "pmf_peak",
        plateau_frac: float = 0.3,
    ):
        """Return (barrier_height_kJ, r_ts_nm) for the TS/barrier of PMF(k).

        ``boundary_method``:
        - ``"pmf_peak"`` (default): highest-prominence PMF peak via
          ``_find_pmf_peak`` (shared with ``KramersEstimator.detect_ts``); falls
          back to the position/height of ``max(dG)`` when no peak is found.
        - ``"force_plateau"``: transition state from the restraint-force decay via
          ``KramersEstimator.force_plateau_boundary`` (estimator-independent, from
          the running per-k force in *force_df*); ``barrier_height`` is then
          ``dG(r_ts) − dG(start)``. Falls back to ``"pmf_peak"`` when the force
          profile is unusable. This keeps the convergence TS consistent with the
          Kramers k_off absorbing boundary.
        """
        steps = pmf_k.index.to_numpy()
        r = np.array([float(protocol_grid.loc[s]) for s in steps], dtype=float)
        dG = pmf_k.values.astype(float)
        if boundary_method == "force_plateau" and force_df is not None and len(r) >= 2:
            r_ts = KramersEstimator.force_plateau_boundary(
                force_df, speed if speed is not None else 0.0,
                float(r.min()), float(r.max()), plateau_frac,
            )
            if r_ts is not None:
                return float(np.interp(r_ts, r, dG) - dG[0]), float(r_ts)
            # force profile unusable → fall through to PMF-peak detection
        r_ts, barrier = _find_pmf_peak(r, dG, kBT)
        if r_ts is None:
            # No prominent peak — fall back to the position and height of max(dG),
            # consistent with KramersEstimator's r_coord.max() fallback.
            best = int(np.nanargmax(dG))
            return float(dG[best] - dG[0]), float(r[best])
        return barrier, r_ts

    @staticmethod
    def _weighted_series_from_results(
        results_df: pd.DataFrame,
        path_traj_counts: dict,
        value_col: str,
        beta: float,
        trim_fraction: float = 0.0,
        policy: "SupportPolicy | None" = None,
        return_paths: bool = False,
    ) -> "pd.Series | tuple[pd.Series, dict, int]":
        """Build the mixture PMF series (unchanged default behaviour).

        ``return_paths=False`` (default) reproduces the original single-Series
        return exactly, so existing callers (``ConvergenceLadder``, scratch
        notebooks) are unaffected.

        ``return_paths=True`` additionally returns ``(per_path, n_paths)``:
        ``per_path`` is ``{step: {path: value}}`` for every path that carries
        non-zero equilibrium weight (``p_eq > 0``) at that step — the same set
        used to build the mixture value — and ``n_paths`` is the count of such
        paths overall (paths that survived the count-based support policy
        *and* contribute non-zero weight to the mixture, i.e. excludes paths
        whose dG is entirely negative and therefore Zk=0).
        """
        empty = (pd.Series(dtype=float), {}, 0) if return_paths else pd.Series(dtype=float)

        if results_df.empty or value_col not in results_df.columns:
            return empty

        if policy is not None:
            # Belt-and-suspenders: _results_from_running_stats skips below-threshold
            # steps (estimable_step); this apply() gate re-enforces the same floor
            # on n_samples + path counts. Both gates are intentional — this apply()
            # is authoritative for weighting computation downstream.
            results_df, path_traj_counts = policy.apply(results_df, path_traj_counts)
            if results_df.empty or not path_traj_counts:
                return empty

        p_neq = SMDData._compute_p_neq(path_traj_counts)
        if not p_neq:
            return empty

        p_eq = SMDAnalysis._compute_p_eq(results_df, p_neq, beta)
        if not p_eq:
            return empty

        # Restrict to the shortest path's extent so all convergence traces
        # are compared over the same r-range at every replica count.
        # Singleton paths (one trajectory) from an outlier/short traj must not
        # chop the extent of all other paths — use only multi-traj paths to
        # determine the common step ceiling.
        path_last = results_df.groupby('path')['step'].max()
        if policy is not None:
            usable = set(path_traj_counts)   # already gated to usable paths
        else:
            usable = {p for p, c in path_traj_counts.items() if c >= 2}
        extent_series = path_last[path_last.index.isin(usable)] if usable else path_last
        max_common_step = extent_series.min() if not extent_series.empty else path_last.min()
        grid_steps = sorted(results_df.loc[results_df['step'] <= max_common_step, 'step'].unique())

        # trim low-support tail — mirrors trim_results_by_n_samples_support used in run()
        if trim_fraction > 0.0 and 'n_samples' in results_df.columns:
            step_support = results_df.groupby('step')['n_samples'].max()
            n_ref = float(step_support.max())
            if n_ref > 0:
                supported = step_support[step_support / n_ref >= (1.0 - trim_fraction)].index
                grid_steps = [s for s in grid_steps if s in supported]

        # Paths with non-zero equilibrium weight are the ones that actually move
        # the mixture value (a p_eq==0 path — e.g. all-negative dG, Zk=0 — is
        # still gated-in above but contributes nothing to any weighted sum).
        contributing_paths = {p for p, w in p_eq.items() if w > 0}

        out = {}
        per_path_out: dict = {}
        for step in grid_steps:
            slice_step = results_df[results_df['step'] == step]
            vals = []
            ws = []

            for _, row in slice_step.iterrows():
                path = row['path']
                if path not in p_eq:
                    continue
                val = row[value_col]
                if pd.isna(val):
                    continue
                vals.append(float(val))
                ws.append(float(p_eq[path]))
                if return_paths and path in contributing_paths:
                    per_path_out.setdefault(int(step), {})[path] = float(val)

            if vals and sum(ws) > 0:
                out[int(step)] = float(np.sum(np.array(vals) * np.array(ws)) / np.sum(ws))

        series = pd.Series(out).sort_index()
        if return_paths:
            return series, per_path_out, len(contributing_paths)
        return series
    
    def _write_path_quality(
        self,
        sMDDdata: SMDData,
        weights_by_estimator: dict,
    ) -> None:
        """Write a per-(estimator, speed, path) quality summary to disk.

        Captures negative-dG counts, replica counts, dG endpoints and the
        final p_eq weight so the user can audit which paths the cumulant
        estimator is struggling with.

        Cumulant reliability diagnostics (cumulant rows only — the jarzynski
        ``Wdiss`` is ``<W> - dG_exp``, not a variance, and force reports NaN):

        ``beta_sigma``
            ``sqrt(2 * beta * Wdiss)``, the dimensionless work spread, since
            ``Wdiss == beta*Var(W)/2`` for the cumulant.
            ``beta_sigma_advisory_exceeded`` compares it to
            a threshold.
            ``Wdiss/Wmean``. Threshold-free and the more direct warning: since
            ``dG = Wmean - Wdiss``, a ratio > 1 is identically ``dG < 0``, and a
            ratio near 1 means dG is a difference of two large, near-equal numbers.
        """
        max_steps_per_speed = sMDDdata.results.groupby('speed')['step'].max().to_dict()
        beta = sMDDdata.beta
        rows = []
        for est in self.estimators:
            df_est = sMDDdata.results[sMDDdata.results['estimator'] == est.name]
            weights = weights_by_estimator.get(est.name, {})
            for (speed, path), gpath in df_est.groupby(['speed', 'path']):
                gpath_sorted = gpath.sort_values('step')
                dG = gpath_sorted['dG'].dropna().to_numpy(dtype=float)
                if dG.size == 0:
                    continue
                n_neg = int((dG < 0).sum())

                r_vals = gpath_sorted['r_coord'].dropna().to_numpy(dtype=float)
                r_mid = (r_vals.min() + r_vals.max()) / 2
                dG_first = gpath_sorted.loc[gpath_sorted['r_coord'] <= r_mid, 'dG'].dropna()
                dG_second = gpath_sorted.loc[gpath_sorted['r_coord'] > r_mid, 'dG'].dropna()

                last_step = int(gpath_sorted['step'].max())
                last_row = gpath_sorted[gpath_sorted['step'] == last_step]
                global_max = max_steps_per_speed.get(speed, last_step)

                # Cumulant-only: Wdiss == beta*Var(W)/2  =>  beta*sigma_W.
                bs_final = corr_ratio = float('nan')
                if est.name == 'cumulant' and not last_row.empty:
                    w_last = last_row['Wdiss'].iloc[0]
                    wm_last = last_row['Wmean'].iloc[0]
                    if np.isfinite(w_last):
                        bs_final = float(np.sqrt(2.0 * beta * max(float(w_last), 0.0)))
                        # Threshold-free: Wdiss/Wmean > 1 is identically dG < 0.
                        if np.isfinite(wm_last) and float(wm_last) != 0.0:
                            corr_ratio = float(w_last) / float(wm_last)

                rows.append({
                    'estimator': est.name,
                    'speed': speed,
                    'path': path,
                    'n_replicas': int(gpath_sorted['n_samples'].median()),
                    'n_steps': int(dG.size),
                    'n_neg_dG': n_neg,
                    'frac_neg_dG': n_neg / dG.size,
                    'frac_neg_dG_first_half': float((dG_first < 0).mean()) if len(dG_first) > 0 else float('nan'),
                    'frac_neg_dG_second_half': float((dG_second < 0).mean()) if len(dG_second) > 0 else float('nan'),
                    'dG_peak': float(dG.max()),
                    'dG_min': float(dG.min()),
                    'dG_final': float(dG[-1]),
                    'Wmean_final': float(last_row['Wmean'].iloc[0]) if not last_row.empty else float('nan'),
                    'Wdiss_final': float(last_row['Wdiss'].iloc[0]) if not last_row.empty else float('nan'),
                    'beta_sigma': bs_final,
                    'cumulant_correction_ratio': corr_ratio,
                    'n_steps_ratio': float(last_step / global_max) if global_max > 0 else float('nan'),
                    'p_eq': float(weights.get(speed, {}).get(path, 0.0)),
                })
        if rows:
            pd.DataFrame(rows).to_csv(
                f'{self.outdir}/path_quality.csv', index=False
            )

    def _path_filtering(self, sMDDdata: SMDData) -> SMDData:
        """Drop (speed, path) groups that are under-sampled or prematurely terminated.

        Must be called after path assignments have been written to
        ``sMDDdata.raw_data['path']`` and estimators have been fitted,
        because the filtering logic reads both ``raw_data`` and
        ``sMDDdata.results``.

        Three passes, controlled by instance attributes:

        Pass 1 — replica count (``self.min_replicas_per_path``): drop paths
        with fewer than that many trajectories.

        Pass 2 — step ratio (``self.min_path_steps_ratio``): drop paths whose
        ``n_steps < ratio × max_steps_at_that_speed``.  A ratio of 0 disables
        this filter.  This catches clusters formed by trajectories that hit an
        early stopping condition (e.g. premature unbinding at high speed),

        Pass 3 — binding-well quality (``self.max_frac_neg_dG_first_half``):
        drop (speed, path, estimator) rows where the fraction of negative-dG
        bins in the **first half** of the r-range exceeds the threshold.  The
        check is performed per estimator so that a cumulant failure at slow
        speed does not falsely condemn a valid jarzynski profile for the same
        path.  If all estimators fail for a (speed, path), the path is also
        removed from raw_data.  Paths where negativity is confined to the
        second half are kept: the TS lies in the first half and p_eq is
        naturally suppressed by the negative-bin exclusion in
        ``_compute_p_eq``.  Set to ≤ 0 to disable.

        Applied after estimator fitting and before p_eq computation and PMF
        construction.  Also logs a per-speed replica imbalance warning when
        the ratio of max/min replica counts exceeds ``self.replica_imbalance_threshold``.
        """
        sMDDdata.raw_data['path_ok'] = True

        def _flag_raw(keys):
            for speed, path in keys:
                mask = (
                    (sMDDdata.raw_data['speed'] == speed) &
                    (sMDDdata.raw_data['path'] == path)
                )
                sMDDdata.raw_data.loc[mask, 'path_ok'] = False

        def _drop_results(keys):
            for speed, path in keys:
                sMDDdata.results = sMDDdata.results.drop(
                    sMDDdata.results[
                        (sMDDdata.results['speed'] == speed) &
                        (sMDDdata.results['path'] == path)
                    ].index
                )

        if sMDDdata.results is None or sMDDdata.results.empty \
           or not {'speed', 'path', 'n_samples'}.issubset(sMDDdata.results.columns):
            raise RuntimeError(
                "No estimator results to filter — sMDDdata.results is empty. "
                "This usually means trim_results_by_n_samples_support removed all "
                "rows from raw_data before the estimators ran. Lower "
                "min_samples_per_step / min_support_ratio (or run more replicas "
                "per (speed, path) group) and try again."
            )

        # --- pass 1: minimum replica count ---
        keys_to_drop: list[tuple[float, str]] = []
        for (speed, path), group in sMDDdata.results.groupby(['speed', 'path']):
            n_replicas = int(group['n_samples'].median())
            if not self.support_policy.usable_path(n_replicas):
                logger.warning(
                    f"Excluding path '{path}' at speed={speed} nm/ps: "
                    f"only {n_replicas} replicas < {self.min_replicas_per_path}"
                )
                keys_to_drop.append((speed, path))
        _flag_raw(keys_to_drop)
        _drop_results(keys_to_drop)

        # --- pass 2: minimum step ratio (premature-termination filter) ---
        if self.min_path_steps_ratio > 0 and not sMDDdata.results.empty:
            keys_to_drop = []
            for speed, grp in sMDDdata.results.groupby('speed'):
                path_max_step = grp.groupby('path')['step'].max()
                threshold = float(path_max_step.max()) * self.min_path_steps_ratio
                for path, n_steps in path_max_step.items():
                    if n_steps < threshold:
                        logger.warning(
                            f"Excluding path '{path}' at speed={speed} nm/ps: "
                            f"n_steps={n_steps:.0f} < {self.min_path_steps_ratio:.0%} × "
                            f"{path_max_step.max():.0f} (premature termination)"
                        )
                        keys_to_drop.append((speed, path))
            _flag_raw(keys_to_drop)
            _drop_results(keys_to_drop)

        # --- pass 3: corrupted binding-well filter ---
        if self.max_frac_neg_dG_first_half > 0 and not sMDDdata.results.empty:
            est_keys_to_drop = []  # (speed, path, estimator)
            for (speed, path, estimator), grp in sMDDdata.results.groupby(
                    ['speed', 'path', 'estimator']):
                sorted_grp = grp.sort_values('r_coord')
                mid = len(sorted_grp) // 2
                valid = sorted_grp['dG'].iloc[:mid].dropna()
                if valid.empty:
                    continue
                frac = float((valid < 0).sum() / len(valid))
                if frac > self.max_frac_neg_dG_first_half:
                    logger.info(
                        f"[filter/pass3] dropping {estimator} path '{path}' "
                        f"at speed={speed} nm/ps: "
                        f"frac_neg_dG_first_half={frac:.2f} > {self.max_frac_neg_dG_first_half}"
                    )
                    est_keys_to_drop.append((speed, path, estimator))

            if est_keys_to_drop:
                # Remove per-estimator rows from results
                for speed, path, estimator in est_keys_to_drop:
                    sMDDdata.results = sMDDdata.results.drop(
                        sMDDdata.results[
                            (sMDDdata.results['speed'] == speed) &
                            (sMDDdata.results['path'] == path) &
                            (sMDDdata.results['estimator'] == estimator)
                        ].index
                    )
                # Flag in raw_data any (speed, path) now absent from results entirely
                touched_sp = {(s, p) for s, p, _ in est_keys_to_drop}
                surviving_sp = set(zip(sMDDdata.results['speed'], sMDDdata.results['path']))
                _flag_raw(touched_sp - surviving_sp)
                _drop_results(touched_sp - surviving_sp)

        # Replica balance report per speed
        for speed, gspeed in sMDDdata.results.groupby('speed'):
            counts = {
                path: int(g['n_samples'].median())
                for path, g in gspeed.groupby('path')
            }
            if not counts:
                continue
            logger.info(f"  Speed={speed} nm/ps  replicas: {counts}")
            if max(counts.values()) / max(min(counts.values()), 1) > self.replica_imbalance_threshold:
                logger.warning(
                    f"Replica imbalance at speed={speed} nm/ps: {counts}. "
                    f"Cumulant variance scales with 1/N — under-sampled paths "
                    f"may produce unreliable dG."
                )

        return sMDDdata

    @staticmethod
    def _compute_p_eq(
        results_df: pd.DataFrame,
        p_neq: dict,
        beta: float,
    ) -> dict:
        """Compute normalised equilibrium path probabilities for a single speed.

        Negative dG bins are treated as missing data: the integrand is set to
        ``0`` at those bins (equivalent to "infinite barrier, contributes
        nothing to Z").  This naturally downweights paths whose cumulant
        produced artifact bins, instead of inflating Z by ``exp(0)=1`` per bin
        as a clip-to-0 strategy would.  A path whose dG is entirely negative
        ends up with Zk=0 and p_eq=0.  Diagnostics for negative bins are
        reported separately by :meth:`compute_p_eq` and ``path_quality.csv``.

        Parameters
        ----------
        results_df :
            DataFrame with 'path', 'step', 'dG', 'r_coord' for a single speed.
        p_neq :
            Non-equilibrium path probabilities {path: p_neq}.
        beta :
            Inverse thermal energy (1 / k_B T).
        """
        weights = {}
        for path, gpath in results_df.groupby('path'):
            if path not in p_neq:
                continue
            gpath = gpath.sort_values('step')
            dG = gpath['dG'].to_numpy(dtype=float)
            x = gpath['r_coord'].to_numpy(dtype=float)
            if len(dG) < 2:
                continue
            mask = dG >= 0
            if mask.sum() < 2:
                weights[path] = 0.0
                continue
            log_integrand = -beta * dG[mask]
            shift = log_integrand.max()
            integrand = np.zeros_like(dG)
            integrand[mask] = np.exp(log_integrand - shift)
            # abs() handles backward pulling where r_coord decreases with step,
            # causing trapz to return a negative value for a non-negative integrand.
            Zk = abs(np.trapz(integrand, x)) * np.exp(shift)
            p_eq_raw = p_neq[path] * Zk
            if not np.isfinite(p_eq_raw) or p_eq_raw < 0.0:
                p_eq_raw = 0.0
            weights[path] = p_eq_raw
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {path: w / total for path, w in weights.items()}

    def compute_p_eq(
        self,
        sMDDdata: 'SMDData',
        estimator: str = 'auto',
    ) -> dict:
        """Compute equilibrium path probabilities.

        Negative dG bins are excluded from the partition-function integral
        (see :meth:`_compute_p_eq`): they contribute 0 to Z, so paths with
        many artifact bins are smoothly downweighted in the mixture rather
        than inflating it.  The negative-bin counts are reported per path
        via warnings and ``path_quality.csv``.

        Parameters
        ----------
        sMDDdata : SMDData
            Data object with fitted estimator results.
        estimator : str
            Which estimator's dG to use ('auto', 'cumulant', 'jarzynski', ...).

        Returns
        -------
        dict
            ``{speed: {path: p_eq}}`` normalised per speed.
        """
        results = sMDDdata.results
        if results is None or results.empty:
            return {}

        estimator_name = SMDData._choose_estimator_for_weights(results, estimator)
        results_for_weights = results[results['estimator'] == estimator_name]

        p_neq_dic = sMDDdata.get_p_neq(byspeed=True)

        p_eq_dic = {}
        for speed, gspeed in results_for_weights.groupby('speed'):
            p_neq = p_neq_dic.get(speed, {})

            # Detect and warn about negative dG paths before computing weights
            for path, gpath in gspeed.groupby('path'):
                dG = gpath['dG'].dropna().to_numpy(dtype=float)
                n_neg = int(np.sum(dG < 0))
                if n_neg > 0:
                    logger.warning(
                        f"Path '{path}' at speed={speed} nm/ps: {n_neg}/{len(dG)} dG "
                        f"values are negative (min={np.nanmin(dG):.1f} kJ/mol). "
                        f"Likely a cumulant artifact from insufficient sampling. "
                        f"Negative bins excluded from the p_eq integrand "
                        f"(contribute 0 to Z), so this path is downweighted."
                    )

            p_eq_dic[speed] = self._compute_p_eq(gspeed, p_neq, sMDDdata.beta)

            for path, w in p_eq_dic[speed].items():
                logger.info(
                    f"  Speed={speed} nm/ps  path={path}  "
                    f"p_neq={p_neq.get(path, 0):.4f}  p_eq={w:.4f}"
                )

        return p_eq_dic
