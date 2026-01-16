import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Optional, Union

from autopath.sMDAnalysis import SMDData
from autopath.sMDAnalysis.Estimators import BaseEstimator
from autopath.sMDAnalysis.PathModel import DTWPathModel, PathModel
from autopath.sMDAnalysis.Estimators import JarzynskiEstimator, CumulantEstimator, calculate_weighted_pmf, extrapolate_to_v0
from autopath.sMDAnalysis.Diagnostics import plot_work_profiles, plot_weighted_pmf

class SMDAnalysis:
    def __init__(self,
        sysname: str = 'system',
        path_model: Union[PathModel, str] = 'dtw',
        estimators:Union[list[BaseEstimator], list[str]] = ['jarzynski', 'cumulant'],
        temperature: float = 300.0,
        outdir: str = 'sMD_analysis',
        do_plots: bool = True,
        seed: int = 42,
    ):
        self.sysname = sysname
        self.seed = seed
        self.temperature = temperature
        self.do_plots = do_plots
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        
        self._setup_path_model(path_model)
        self._setup_estimators(estimators)

        return None
    
    def _setup_estimators(self, estimators: Union[list[BaseEstimator], list[str]]):
        estimator_map = {
            'jarzynski': JarzynskiEstimator(),
            'cumulant': CumulantEstimator(),
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
            ) -> SMDData:
        
        # filter by r_range if provided
        if r_range is not None:
            sMDDdata.filter_by_r_range(r_range, sMDDdata.r_column)

        # extract features
        traces_feat_df = sMDDdata.get_trace_features(
                            # features=['work', 'lag', 'r_before' ],
                            features=['force', 'lag', 'r_before', 'r_after'],
                            # features=['force', 'lag', 'r_before', 'r_after']
        )

        # cluster trajectories into pathways
        clusterer = DTWPathModel(seed=self.seed, 
                                do_plots=self.do_plots,
                                outdir=self.outdir
                                )
        path_mappings = clusterer.fit_transform(traces_feat_df)

        # Map back to full raw_data
        sMDDdata.raw_data['path'] = (sMDDdata.raw_data['trajname'].map(path_mappings))
        
        # fit the estimators
        for estimator in self.estimators:
            sMDDdata = estimator.fit_transform(sMDDdata) 
            
        # optional path filtering
        sMDDdata = self._path_filtering(sMDDdata, min_replicas=3)
            
        #Calculate weighted PMF across paths using p_eq weights for multiple columns.
        self.weighted_pmf = calculate_weighted_pmf(sMDDdata, weight_cols=['dG', 'Wdiss'])
        self.weighted_pmf['path'] = 'mixture'  # indicate mixed paths
        self.weighted_pmf.to_csv(f'{self.outdir}/weighted_pmf.csv', index=False)
        
        # extrapolate to v=0 for each estimator
        # if self.weighted_pmf['speed'].nunique() > 2:
        #     self.weighted_pmf_v0 = extrapolate_to_v0(self.weighted_pmf, param_cols=['Wdiss_weighted', 'dG_weighted'])      
        
        if self.do_plots:
            plot_work_profiles(sMDDdata.results, estimator='jarzynski', outdir=self.outdir)
            plot_work_profiles(sMDDdata.results, estimator='cumulant', outdir=self.outdir)
            plot_weighted_pmf(self.weighted_pmf, outdir=self.outdir)
        
        # save processed data
        sMDDdata.raw_data.to_csv(f'{self.outdir}/sMD_processed_data.csv', index=False)
        
        return sMDDdata
        
    def check_convergence(self,
        logs: list[str],
        speeds: list[float] = None,
        quantities: list[str] = ['dG_weighted'],
        estimator_name: str = 'cumulant',
        min_replicas: int = 3,
        tol_rmsd: float = 3.0,     # kJ/mol
        tol_barrier: float = 2.0,  # kJ/mol
        min_common_points: int = 5,
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

        if speeds is None:
            speeds = sorted(set(self._speed_from_log(fn) for fn in logs))

        convergence_all_speeds = []
        traces_all_speeds = []
        
        for speed in speeds:
            
            #filter logs by speed
            speed_logs = [fn for fn in logs if self._speed_from_log(fn) == speed]

            if len(speed_logs) < min_replicas:
                raise ValueError(
                    f"Not enough replicas for speed={speed}: {len(speed_logs)}"
                )
            #ort replicas sequentially
            speed_logs = sorted(speed_logs, key=self._replica_idx_from_log)
            print(f"Checking convergence for speed={speed} with {len(speed_logs)} replicas")

            rows = []
            pmf_records = []

            prev_pmf = None

            for k in range(min_replicas, len(speed_logs) + 1):

                current_logs = speed_logs[:k]

                smd = SMDData(current_logs, sysname=self.sysname)
                
                # extract features and cluster
                traces_feat_df = smd.get_trace_features(
                    features=['force', 'lag', 'r_before', 'r_after'],
                    # features=['work', 'lag', 'r_before' ],
                )
                
                clusterer = DTWPathModel(seed=self.seed,
                                        do_plots=False,
                                        outdir=self.outdir,
                                        )
                path_mappings = clusterer.fit_transform(traces_feat_df)

                # Map back to full raw_data
                smd.raw_data['path'] = smd.raw_data['trajname'].map(path_mappings)
                
                # fit free-energy estimator
                if estimator_name == 'jarzynski':
                    smd = JarzynskiEstimator().fit_transform(smd)
                elif estimator_name == 'cumulant':
                    smd = CumulantEstimator().fit_transform(smd)
                else:
                    raise ValueError(f"Unknown estimator: {estimator_name}")

                #compute weighted PMF
                weighted_pmf_df = calculate_weighted_pmf(smd, weight_cols=['dG', 'Wdiss'])

                pmf_k = (
                    weighted_pmf_df
                    .set_index("step")[main_quantity]
                    .sort_index()
                )

                # store PMF trace
                # for r, val in pmf_k.items():
                for step, val in pmf_k.items():
                    r_coord = smd.protocol_grids[speed].loc[
                        smd.protocol_grids[speed]["step"] == step,
                        "r_target_protocol"
                    ].values[0]
                    
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
                
                # trimm last chunk of the PMF as its noisy
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
            
        convergence_all_speeds = pd.concat(convergence_all_speeds, ignore_index=True)
        traces_all_speeds = pd.concat(traces_all_speeds, ignore_index=True)
        
        return convergence_all_speeds, traces_all_speeds
    
    def _path_filtering(self, 
                        sMDDdata: SMDData,
                        min_replicas: int = 3):
        """A simple path filtering based on minimum number of replicas speed and per path.
        This could be extended in the future to more sophisticated criteria.
        """
        
        sMDDdata.results = sMDDdata.results[sMDDdata.results['n_samples'] >= min_replicas]
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