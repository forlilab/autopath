import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Optional, Union
from collections import defaultdict
import logging

from autopath.sMDAnalysis import SMDData
from autopath.sMDAnalysis.Estimators import BaseEstimator, FrictionEstimator
from autopath.sMDAnalysis.PathModel import DTWPathModel, NullPathModel, PathModel
from autopath.sMDAnalysis.Estimators import (
    JarzynskiEstimator,
    CumulantEstimator,
    JarzynskiGMMEstimator,
    CumulantGMMEstimator,
    CumulantGMMComponentwiseEstimator,
    ESTIMATOR_REGISTRY,
    trim_results_by_n_samples_support,
    calculate_weighted_pmf,
    extrapolate_to_v0,
)
from autopath.sMDAnalysis.Diagnostics import (
    plot_work_profiles,
    plot_profile,
    plot_friction,
    plot_extrapolated_param,
    make_unbinding_paths_visualization,
)

logger = logging.getLogger("autopath.sMDAnalysis")

class SMDAnalysis:
    def __init__(self,
        sysname: str = 'system',
        path_model: Union[PathModel, str] = 'dtw',
        estimators:Union[list[BaseEstimator], list[str]] = ['jarzynski', 'cumulant'],
        temperature: float = 300.0,
        outdir: str = 'sMD_analysis',
        do_plots: bool = True,
        reference_pdb: Optional[str] = None,
        ligand_select: Optional[str] = 'resname UNK',
        seed: int = 42,
    ):
        self.sysname = sysname
        self.seed = seed
        self.temperature = temperature
        self.do_plots = do_plots
        self.reference_pdb = reference_pdb
        self.ligand_select = ligand_select
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        
        self._setup_path_model(path_model)
        self._setup_estimators(estimators)

        return None
    
    def _setup_estimators(self, estimators: Union[list[BaseEstimator], list[str]]):
        estimator_map = {
            'jarzynski': JarzynskiEstimator(),
            'cumulant': CumulantEstimator(),
            'jarzynski_gmm': JarzynskiGMMEstimator(),
            'cumulant_gmm': CumulantGMMEstimator(),
            'cumulant_gmm_componentwise': CumulantGMMComponentwiseEstimator(),
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
    
    def run(self,
            sMDDdata: SMDData,
            r_range: tuple[float, float] | None = None,
            group_A: str = None,
            group_B: str = None,
            merge_features: bool = True,
            trim_low_support_results: bool = True,
            trim_min_support_ratio: float = 0.9,
            cluster_across_speeds: bool = False,
            ) -> SMDData:

        if self.reference_pdb is None:
            self.reference_pdb = sMDDdata.reference_pdb
            logger.warning(f"No reference PDB provided, using from SMDData: {self.reference_pdb}")

        if r_range is not None:
            sMDDdata.filter_by_r_range(r_range, sMDDdata.r_column)

        traces_feat_df = sMDDdata.get_trace_features(
            features=['lag','work','r_before'], # names tracesV2
        )

        dist_feat_df = None
        if group_A is not None and group_B is not None:
            dist_feat_df = sMDDdata.calculate_pocket_distances(
                group_A=group_A,
                group_B=group_B,
                recompute=True,
            )

        if dist_feat_df is not None and cluster_across_speeds:
            logger.warning("Clustering using all speeds together on trace features. Those depend on speed, so this may lead to suboptimal clustering. Consider setting cluster_across_speeds=False or using only distance features for clustering.")

        if group_A is not None and group_B is not None and merge_features:
            feat_df = SMDData.build_merged_features(
                trace_df=traces_feat_df,
                geom_df=dist_feat_df,
            )
            logger.info('Clustering will be performed using merged (trace + distance) features')
        elif group_A is not None and group_B is not None:
            feat_df = dist_feat_df
            logger.info('Clustering will be performed using distance features only')
        else:
            feat_df = traces_feat_df
            logger.info('Clustering will be performed using trace features only')

        # assemble trajectory files for path model (if needed)
        trajectory_files = {}
        for traj in sMDDdata.traj_files:
            trajname = SMDData._traj_to_log_name(traj)
            trajectory_files[trajname] = (self.reference_pdb, traj)

        # fit path model and assign paths to trajectories
        path_mappings = self.path_model.fit_transform(
            feat_df,
            # r_range=0.75,  # use only the first 75% of frames for clustering to avoid noisy end states
            reference_pdb=self.reference_pdb,
            ligand_select=self.ligand_select,
            trajectory_files=trajectory_files,
            cluster_across_speeds=cluster_across_speeds,
            pocket_select=group_B
        )

        sMDDdata.raw_data['path'] = sMDDdata.raw_data['trajname'].map(path_mappings)

        # trim (speed, path) groups with insufficient sample support for reliable estimation
        if trim_low_support_results:
            sMDDdata.raw_data = trim_results_by_n_samples_support(
                sMDDdata.raw_data,
                min_samples=3,
                min_support_ratio=trim_min_support_ratio,
            )
        
        # fit estimators sequentially (some may rely on the path assignments, so do this before any path filtering)
        for estimator in self.estimators:
            logger.info(f"Fitting estimator: {estimator.name}")
            sMDDdata = estimator.fit_transform(sMDDdata)

        
        sMDDdata = self._path_filtering(sMDDdata, min_replicas=3)

        if sMDDdata.results.empty:
            raise RuntimeError(
                "All (speed, path) groups were removed by path filtering — "
                "no data remains for PMF construction. "
                "Consider running more SMD replicas or using slower pulling speeds."
            )

        # Compute p_eq per estimator explicitly.
        # Paths with negative dG values trigger a warning; those bins are
        # excluded from the integrand (contribute 0 to Z), which smoothly
        # downweights artifact-heavy paths in the mixture.
        weights_by_estimator = {
            est.name: self.compute_p_eq(sMDDdata, estimator=est.name)
            for est in self.estimators
        }

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

        self.mixture_pmfs.to_csv(f'{self.outdir}/mixture_pmfs.csv', index=False)

        weighted_pmf_v0 = {}
        for pcol in ['dG', 'Wdiss']:
            if self.mixture_pmfs['speed'].nunique() < 2:
                logger.warning(f"Not enough speeds to perform v=0 extrapolation for '{pcol}'. Skipping.")
                continue
            v0_df = extrapolate_to_v0(self.mixture_pmfs, param=pcol)
            if not v0_df.empty:
                weighted_pmf_v0[pcol] = v0_df
                v0_df.to_csv(f'{self.outdir}/{pcol}_extrapolated.csv', index=False)

        friction_est = FrictionEstimator(use_spline=False)
        df = self.mixture_pmfs.copy()

        friction_deriv_results = []
        friction_regress_results = []
        for estimator in self.estimators:
            f_deriv = friction_est.gamma_from_wdiss_derivative(df, estimator=estimator.name)
            f_regress = friction_est.gamma_from_wdiss_regression(df, estimator=estimator.name)
            friction_deriv_results.append(f_deriv)
            friction_regress_results.append(f_regress)

        friction_df = pd.concat(friction_deriv_results + friction_regress_results, ignore_index=True)
        friction_df.to_csv(os.path.join(self.outdir, 'friction.csv'), index=False)

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
                        outdir=os.path.join(self.path_model.outdir, "unbinding_paths"),
                        friction_csv=friction_csv_path,
                    )
                    logger.info("Friction-coloured unbinding paths PSE regenerated.")
            except Exception as _exc:
                logger.warning(f"Could not regenerate friction-coloured PSE: {_exc}")

        if self.do_plots:
            for estimator in self.estimators:
                plot_work_profiles(sMDDdata.results, estimator=estimator.name, outdir=self.outdir)
            for vcol in ['Wdiss', 'dG']:
                plot_profile(
                    self.mixture_pmfs,
                    value_col=vcol,
                    hue='estimator',
                    ylabel='Energy (kJ/mol)',
                    outdir=self.outdir,
                )
            if friction_df is not None and not friction_df.empty:
                plot_friction(friction_df, outdir=self.outdir)
            for pcol, v0_df in weighted_pmf_v0.items():
                plot_extrapolated_param(
                    v0_df,
                    param=pcol,
                    outfname=os.path.join(self.outdir, f'{pcol}_extrapolated.svg'),
                )

        sMDDdata.raw_data.to_csv(f'{self.outdir}/sMD_processed_data.csv', index=False)

        return sMDDdata
        
    def check_convergence(self,
        logs: list[str],
        speeds: list[float] = None,
        quantities: list[str] = ['dG_weighted'],
        estimator_name: str = 'cumulant',
        group_A: str = None,
        group_B: str = None,
        min_replicas: int = 5,
        tol_rmsd: float = 3.0,     # kJ/mol
        tol_barrier: float = 3.0,  # kJ/mol
        trim_fraction: float = 0.25,     # fraction of PMF to trim from the end (noisy region)
        min_common_points: int = 5,
        return_gmm_diagnostics: bool = False,
    ):
        """
        Sequential convergence check for a single pulling speed using
        the weighted PMF (mixture over paths).

        The PMF for k replicas is assumed to be invariant once computed;
        only PMF(k) vs PMF(k-1) is compared.
        """

        if isinstance(quantities, str):
            quantities = [quantities]

        main_quantity = quantities[0]
        value_col = main_quantity[:-9] if main_quantity.endswith('_weighted') else main_quantity

        allowed_estimators = {
            'jarzynski',
            'cumulant',
            'jarzynski_gmm',
            'cumulant_gmm',
            'cumulant_gmm_componentwise',
        }
        if estimator_name not in allowed_estimators:
            raise ValueError(
                f"Unknown estimator '{estimator_name}'. "
                f"Allowed values: {sorted(allowed_estimators)}"
            )

        if speeds is None:
            speeds = sorted(set(self._speed_from_log(fn) for fn in logs))

        convergence_all_speeds = []
        traces_all_speeds = []
        gmm_diag_all_speeds = []
        
        for speed in speeds:
            
            #filter logs by speed
            speed_logs = [fn for fn in logs if self._speed_from_log(fn) == speed]

            if len(speed_logs) < min_replicas:
                raise ValueError(
                    f"Not enough replicas for speed={speed}: {len(speed_logs)}"
                )

            # sort replicas sequentially
            speed_logs = sorted(speed_logs, key=self._replica_idx_from_log)
            logger.info(f"Checking convergence for speed={speed} with {len(speed_logs)} replicas")

            # Build SMDData and clustering ONCE for this speed
            smd = SMDData(
                speed_logs,
                sysname=self.sysname,
                temperature=self.temperature,
                reference_pdb=self.reference_pdb,
            )

            if group_A is not None and group_B is not None:
                feat_df = smd.calculate_pocket_distances(group_A=group_A, group_B=group_B)
            else:
                # feat_df = smd.get_trace_features(features=['work', 'force', 'lag', 'r_before', 'r_after'])
                feat_df = smd.get_trace_features(features=['work', 'lag', 'r_before']) # tracesV2
                
            clusterer = DTWPathModel(
                seed=self.seed,
                do_plots=False,
                outdir=self.outdir,
            )
            path_mappings = clusterer.fit_transform(feat_df,
                                                    r_range=1-trim_fraction,  # use only the first (1-trim_fraction)% of frames for clustering to avoid noisy end states
            )
                                                    
            smd.raw_data['path'] = smd.raw_data['trajname'].map(path_mappings)

            speed_data = smd.raw_data[smd.raw_data['speed'] == speed].copy()
            protocol_grid = smd.protocol_grids[speed].set_index('step')['r_target_protocol']

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
            prev_pmf = None

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

                if k < min_replicas:
                    continue

                results_df = self._results_from_running_stats(
                    running_stats=running_stats,
                    running_samples=running_samples,
                    speed=speed,
                    protocol_grid=protocol_grid,
                    estimator_name=estimator_name,
                    beta=smd.beta,
                )

                if not results_df.empty and {'gmm_n_components', 'gmm_bic'}.issubset(results_df.columns):
                    gmm_diag_k = results_df[['step', 'r_coord', 'path', 'gmm_n_components', 'gmm_bic']].copy()
                    gmm_diag_k['speed'] = speed
                    gmm_diag_k['n_replicas'] = k
                    gmm_diag_k['estimator'] = estimator_name
                    gmm_diag_k = gmm_diag_k.dropna(subset=['gmm_n_components'], how='any')
                    if not gmm_diag_k.empty:
                        gmm_diag_all_speeds.append(gmm_diag_k)

                pmf_k = self._weighted_series_from_results(
                    results_df=results_df,
                    path_traj_counts=path_traj_counts,
                    value_col=value_col,
                    beta=smd.beta,
                )

                if pmf_k.empty:
                    prev_pmf = pmf_k
                    continue

                # store PMF trace
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

                # first PMF: initialize
                if prev_pmf is None:
                    prev_pmf = pmf_k
                    continue

                #compare PMF(k) vs PMF(k-1)
                common_r = pmf_k.index.intersection(prev_pmf.index)

                if len(common_r) < min_common_points:
                    rows.append({
                        "speed": speed,
                        "path": "mixture",
                        "n_replicas": k,
                        f"{main_quantity}-rmsd": np.nan,
                        f"{main_quantity}-deltaMax": np.nan,
                        "converged": False,
                        "reason": "insufficient_overlap",
                        "n_common_points": len(common_r),
                    })
                    prev_pmf = pmf_k
                    continue

                yN = pmf_k.loc[common_r].values
                yNm1 = prev_pmf.loc[common_r].values
                
                # trim last chunk of the PMF as it is noisy
                trim_points = max(1, int(len(common_r) * trim_fraction))
                yN = yN[:-trim_points]
                yNm1 = yNm1[:-trim_points]
                # print(f'Trimmed points: {trim_points}, remaining points: {len(yN)}')
                
                pmf_rmsd = np.sqrt(np.mean((yN - yNm1) ** 2))
                delta_barrier = abs(yN.max() - yNm1.max())

                converged = (
                    (pmf_rmsd < tol_rmsd) and
                    (delta_barrier < tol_barrier)
                )

                rows.append({
                    "speed": speed,
                    "path": "mixture",
                    "n_replicas": k,
                    f"{main_quantity}-rmsd": pmf_rmsd,
                    f"{main_quantity}-deltaMax": delta_barrier,
                    "converged": converged,
                    "decision_quantity": main_quantity,
                    "n_common_points": len(common_r),
                })

                # overwrite previous PMF (incremental logic)
                prev_pmf = pmf_k

            conv_df = pd.DataFrame(rows)
            convergence_all_speeds.append(conv_df)
            traces_df = pd.DataFrame(pmf_records)
            traces_all_speeds.append(traces_df)
            
        convergence_all_speeds = pd.concat(convergence_all_speeds, ignore_index=True) if convergence_all_speeds else pd.DataFrame()
        traces_all_speeds = pd.concat(traces_all_speeds, ignore_index=True) if traces_all_speeds else pd.DataFrame()
        gmm_diag_df = pd.concat(gmm_diag_all_speeds, ignore_index=True) if gmm_diag_all_speeds else pd.DataFrame()

        # keep diagnostics accessible after the call
        self.convergence_gmm_diagnostics = gmm_diag_df

        if return_gmm_diagnostics:
            return convergence_all_speeds, traces_all_speeds, gmm_diag_df

        return convergence_all_speeds, traces_all_speeds

    @staticmethod
    def _results_from_running_stats(
        running_stats: dict,
        running_samples: dict,
        speed: float,
        protocol_grid: pd.Series,
        estimator_name: str,
        beta: float,
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
                'gmm_n_components': result.get('gmm_n_components', np.nan),
                'gmm_bic': result.get('gmm_bic', np.nan),
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    @staticmethod
    def _weighted_series_from_results(
        results_df: pd.DataFrame,
        path_traj_counts: dict,
        value_col: str,
        beta: float,
    ) -> pd.Series:
        if results_df.empty or value_col not in results_df.columns:
            return pd.Series(dtype=float)

        p_neq = SMDData._compute_p_neq(path_traj_counts)
        if not p_neq:
            return pd.Series(dtype=float)

        p_eq = SMDAnalysis._compute_p_eq(results_df, p_neq, beta)
        if not p_eq:
            return pd.Series(dtype=float)

        # common support over paths (same criterion used in calculate_weighted_pmf)
        path_last = results_df.groupby('path')['step'].max()
        max_common_step = path_last.min()
        grid_steps = sorted(results_df.loc[results_df['step'] <= max_common_step, 'step'].unique())

        out = {}
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

            if vals and sum(ws) > 0:
                out[int(step)] = float(np.sum(np.array(vals) * np.array(ws)) / np.sum(ws))

        return pd.Series(out).sort_index()
    
    def _write_path_quality(
        self,
        sMDDdata: SMDData,
        weights_by_estimator: dict,
    ) -> None:
        """Write a per-(estimator, speed, path) quality summary to disk.

        Captures negative-dG counts, replica counts, dG endpoints and the
        final p_eq weight so the user can audit which paths the cumulant
        estimator is struggling with.
        """
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
                rows.append({
                    'estimator': est.name,
                    'speed': speed,
                    'path': path,
                    'n_replicas': int(gpath_sorted['n_samples'].median()),
                    'n_steps': int(dG.size),
                    'n_neg_dG': n_neg,
                    'frac_neg_dG': n_neg / dG.size,
                    'dG_min': float(dG.min()),
                    'dG_final': float(dG[-1]),
                    'p_eq': float(weights.get(speed, {}).get(path, 0.0)),
                })
        if rows:
            pd.DataFrame(rows).to_csv(
                f'{self.outdir}/path_quality.csv', index=False
            )

    def _path_filtering(
        self,
        sMDDdata: SMDData,
        min_replicas: int = 3,
    ) -> SMDData:
        """Drop (speed, path) groups with fewer than ``min_replicas`` trajectories.

        Applied after estimator fitting and before p_eq computation and PMF
        construction.  Negative-dG bins are excluded from the p_eq integrand
        by :meth:`compute_p_eq`; per-(speed, path) quality details are
        written to ``path_quality.csv``.  Also logs a per-speed replica
        imbalance warning when the ratio of max/min replica counts exceeds 3.
        """
        keys_to_drop: list[tuple[float, str]] = []
        for (speed, path), group in sMDDdata.results.groupby(['speed', 'path']):
            n_replicas = int(group['n_samples'].median())
            if n_replicas < min_replicas:
                logger.warning(
                    f"Excluding path '{path}' at speed={speed} nm/ps: "
                    f"only {n_replicas} replicas < {min_replicas}"
                )
                keys_to_drop.append((speed, path))

        for speed, path in keys_to_drop:
            sMDDdata.results = sMDDdata.results.drop(
                sMDDdata.results[
                    (sMDDdata.results['speed'] == speed) &
                    (sMDDdata.results['path'] == path)
                ].index
            )
            sMDDdata.raw_data = sMDDdata.raw_data.drop(
                sMDDdata.raw_data[
                    (sMDDdata.raw_data['speed'] == speed) &
                    (sMDDdata.raw_data['path'] == path)
                ].index
            )

        # Replica balance report per speed
        for speed, gspeed in sMDDdata.results.groupby('speed'):
            counts = {
                path: int(g['n_samples'].median())
                for path, g in gspeed.groupby('path')
            }
            if not counts:
                continue
            logger.info(f"  Speed={speed} nm/ps  replicas: {counts}")
            if max(counts.values()) / max(min(counts.values()), 1) > 3:
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

    @staticmethod
    def _speed_from_log(fn):
        # Example log name: sMD_replica-182557_v0.005_forward.dat
        base = os.path.basename(fn)
        for token in base.split("_"):
            if token.startswith("v"):
                return float(token[1:])
        return None
    
    @staticmethod
    def _replica_idx_from_log(fn):
        base = os.path.basename(fn)[:-4]
        rep = base.split("_")[-3]
        return int(rep.split("-")[1])