import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Optional, Union
from collections import defaultdict
import logging

from autopath.sMDAnalysis import SMDData
from autopath.sMDAnalysis.Estimators import BaseEstimator
from autopath.sMDAnalysis.PathModel import DTWPathModel, PathModel
from autopath.sMDAnalysis.Estimators import (
    JarzynskiEstimator,
    CumulantEstimator,
    JarzynskiGMMEstimator,
    CumulantGMMEstimator,
    CumulantGMMComponentwiseEstimator,
    ESTIMATOR_REGISTRY,
    calculate_weighted_pmf,
    extrapolate_to_v0,
)
from autopath.sMDAnalysis.Diagnostics import (
    plot_work_profiles,
    plot_weighted_pmf,
    plot_extrapolated_param,
)

logger = logging.getLogger("autopath.sMDAnalysis.core")

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
            group_B: str = None
            ) -> SMDData:
        
        if self.reference_pdb is None:
            self.reference_pdb = sMDDdata.reference_pdb
            logger.warning(f"No reference PDB provided, using from SMDData: {self.reference_pdb}")
        
        # filter by r_range if provided
        if r_range is not None:
            sMDDdata.filter_by_r_range(r_range, sMDDdata.r_column)

        if group_A is not None and group_B is not None:
            # extract geometrical (distance) features
            logger.info(f'Using distance features for path clustering')
            feat_df = sMDDdata.calculate_pocket_distances(
                        group_A=group_A,
                        group_B=group_B,
                        recompute=False,  # force recomputation to ensure we have the latest data
            )
        else:
            # extract trace features
            logger.info(f'Using trace features for path clustering')
            feat_df = sMDDdata.get_trace_features(
                                features=['work', 'lag', 'r_before' ],
                                # features=['force', 'lag', 'r_before', 'r_after'],
            )
        
        # build trajectory files mapping for clustering
        trajectory_files = {}
        for traj in sMDDdata.traj_files:
            trajname = os.path.basename(traj).split(".dcd")[0]
            trajectory_files[trajname] = (self.reference_pdb, traj)

        # cluster trajectories into pathways using the path model
        path_mappings = self.path_model.fit_transform(feat_df,
                                                      reference_pdb=self.reference_pdb,
                                                      ligand_select=self.ligand_select,
                                                      trajectory_files=trajectory_files
                                                      )

        # Map back to full raw_data
        sMDDdata.raw_data['path'] = (sMDDdata.raw_data['trajname'].map(path_mappings))
        
        # fit the estimators
        for estimator in self.estimators:
            logger.info(f"Fitting estimator: {estimator.name}")
            sMDDdata = estimator.fit_transform(sMDDdata) 
            
        # optional path filtering
        sMDDdata = self._path_filtering(sMDDdata, min_replicas=3)
            
        #Calculate weighted PMF across paths using p_eq weights for multiple columns.
        self.weighted_pmf = calculate_weighted_pmf(sMDDdata, weight_cols=['dG', 'Wdiss'])
        self.weighted_pmf['path'] = 'mixture'  # indicate mixed paths
        self.weighted_pmf.to_csv(f'{self.outdir}/weighted_pmf.csv', index=False)
        
        # extrapolate to v=0 for each estimator
        self.weighted_pmf_v0 = {}
        for pcol in ['dG_weighted', 'Wdiss_weighted']:
            if pcol in self.weighted_pmf.columns:
                v0_df = extrapolate_to_v0(self.weighted_pmf, param=pcol)
                if not v0_df.empty:
                    self.weighted_pmf_v0[pcol] = v0_df
                    v0_df.to_csv(f'{self.outdir}/v0_extrapolation_{pcol}.csv', index=False)
        
        if self.do_plots:
            for estimator in self.estimators:
                plot_work_profiles(sMDDdata.results, estimator=estimator.name, outdir=self.outdir)
            plot_weighted_pmf(self.weighted_pmf, outdir=self.outdir)
            for pcol, v0_df in self.weighted_pmf_v0.items():
                plot_extrapolated_param(
                    v0_df, param=pcol,
                    outfname=os.path.join(self.outdir, f'{pcol}_v0_extrapolation.png'),
                )
        
        # save processed data
        sMDDdata.raw_data.to_csv(f'{self.outdir}/sMD_processed_data.csv', index=False)
        
        return sMDDdata
        
    def check_convergence(self,
        logs: list[str],
        speeds: list[float] = None,
        quantities: list[str] = ['dG_weighted'],
        estimator_name: str = 'cumulant',
        group_A: str = None,
        group_B: str = None,
        min_replicas: int = 3,
        tol_rmsd: float = 3.0,     # kJ/mol
        tol_barrier: float = 2.0,  # kJ/mol
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
                feat_df = smd.get_trace_features(features=['work', 'lag', 'r_before'])

            clusterer = DTWPathModel(
                seed=self.seed,
                do_plots=False,
                outdir=self.outdir,
            )
            path_mappings = clusterer.fit_transform(feat_df)
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
                trim_fraction = 0.2
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

        p_eq = SMDData._compute_p_eq(results_df, p_neq, beta)
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
    
    def _path_filtering(self, 
                        sMDDdata: SMDData,
                        min_replicas: int = 3,
                        min_dG_allowed: float = -10.0,
                        max_path_p_eq: float = 1.00) -> SMDData:
        """A simple path filtering based on minimum number of replicas speed and per path.
        This could be extended in the future to more sophisticated criteria.
        """

        # sampling filter by number of replicas per path (minimum sampling guard)
        for (speed, path), group in sMDDdata.results.groupby(['speed', 'path']):
            n_replicas = len(group)
            if n_replicas <= min_replicas:
                logger.warning(
                    f"Excluding path '{path}' at speed={speed} nm/ps: "
                    f"only {n_replicas} replicas < {min_replicas}"
                )
                sMDDdata.results = sMDDdata.results.drop(group.index)
                sMDDdata.raw_data = sMDDdata.raw_data.drop(
                    sMDDdata.raw_data[
                        (sMDDdata.raw_data['speed'] == speed) &
                        (sMDDdata.raw_data['path'] == path)
                    ].index
                )
    
        bad_keys: set[tuple[float, str]] = set()

        # # dG floor filter (artifact guard)
        # dG_min_by_path = (
        #     sMDDdata.results
        #     .groupby(['speed', 'path'])['dG']
        #     .min()
        #     .dropna()
        # )
        # for (speed, path), dgmin in dG_min_by_path.items():
        #     if dgmin < min_dG_allowed:
        #         logger.warning(
        #             f"Excluding path '{path}' at speed={speed} nm/ps: "
        #             f"min(dG)={dgmin:.3f} < {min_dG_allowed:.3f} kJ/mol"
        #         )
        #         bad_keys.add((speed, path))

        # dominant p_eq filter (single-path domination guard)
        try:
            p_eq_dic = sMDDdata.get_p_eq(byspeed=True, results=sMDDdata.results)
        except Exception as exc:
            logger.warning(f"Could not compute p_eq for quality filtering: {exc}")
            p_eq_dic = None

        if p_eq_dic is not None:
            for speed, path_weights in p_eq_dic.items():
                for path, p_eq in path_weights.items():
                    if p_eq > max_path_p_eq:
                        logger.warning(
                            f"Excluding path '{path}' at speed={speed} nm/ps: "
                            f"p_eq={p_eq:.4f} > {max_path_p_eq:.4f}"
                        )
                        bad_keys.add((speed, path))

        if bad_keys:
            bad_df = pd.DataFrame(list(bad_keys), columns=['speed', 'path'])

            sMDDdata.results = (
                sMDDdata.results
                .merge(bad_df, on=['speed', 'path'], how='left', indicator=True)
                .query("_merge == 'left_only'")
                .drop(columns=['_merge'])
            )

            sMDDdata.raw_data = (
                sMDDdata.raw_data
                .merge(bad_df, on=['speed', 'path'], how='left', indicator=True)
                .query("_merge == 'left_only'")
                .drop(columns=['_merge'])
            )

            logger.warning(f"Path quality filtering removed {len(bad_keys)} (speed, path) groups.")

        return sMDDdata    
    
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