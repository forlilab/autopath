import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.integrate import cumulative_trapezoid

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import tqdm

import logging
logger = logging.getLogger("autopath.sMDAnalysis.SMDData")
from collections import defaultdict

class SMDData:
    def __init__(self, 
                log_files: list[str],
                sysname: str = 'autopath',
                exclude_speeds: list[float] = None,
                r_column: str = 'r_target',
                temperature: float = 300.0,
                reference_pdb: str = None
                ):
        
        self.sysname = sysname
        self.log_files = log_files
        self.traj_files = [self._log_to_traj_file(fn) for fn in log_files]
        self.reference_pdb = reference_pdb # for topology if needed and reference
        
        self.temperature = temperature
        self.R = 0.008314462618  # kJ/(mol*K)
        self.RT = self.R * self.temperature
        self.beta = 1.0 / self.RT
        
        # get it from the first log file
        self.pulling_direction = self.log_files[0].split("_")[-1].replace(".dat","")
        self.force_column = 'force'  # assuming the log files have a 'force' column
        self.outdir = os.path.dirname(self.log_files[0])
        
        # r_target is the theoretical target distance grid, the one that dcTMD uses
        # r_before is the actual distance before applying the constraint force. This one requires
        # binning because different replicas will have different grids.
        self.r_column = r_column
        if r_column not in ['r_target', 'r_before']:
            logger.error(f'Unknown r_column: {r_column}. Must be "r_target" or "r_before".')
            exit(1)
            
        # load the data        
        raw_data = self.load_logs()
        
        # Build protocol grids BEFORE any filtering
        self.protocol_grids = self.build_protocol_grids(raw_data)
        
        # filter out unwanted speeds
        if exclude_speeds is not None:
            logger.warning(f'The following speeds will be excluded from analysis: {exclude_speeds}')
            raw_data = raw_data[~raw_data['speed'].isin(exclude_speeds)]
            
        # filter out trajectories that did not reach end state
        to_drop = []
        for traj_name, traj_data in raw_data.groupby('trajname'):
            if self.pulling_direction == 'forward':
                if traj_data["r_after"].min() < 0.1:
                    logger.warning(f'Dropping {traj_name}, min distance {traj_data["r_after"].min():.2f} nm')
                    to_drop.append(traj_name)
            else:  # backward pulling
                if traj_data["r_after"].min() > 0.1:
                    logger.warning(f'Dropping {traj_name}, min distance {traj_data["r_after"].min():.2f} nm')
                    to_drop.append(traj_name)
        
        raw_data = raw_data[~raw_data['trajname'].isin(to_drop)]

        # integrate force over distance to get work
        raw_data = self.integrate_force_dx(raw_data, r_column)
        
        raw_data['lag'] = raw_data['r_after'] - raw_data['r_target']
        
        # build common r_coord grid across replicas for a given speed.
        # THIS GRID SHOULD NOT BE USED FOR BINNING OR INDEXING, ONLY FOR ANALYSIS COORDINATE
        raw_data = self.build_analysis_coord(raw_data)

        raw_data['path'] = 1  # default single path
                
        self.raw_data = raw_data
        self.results = None
        
        return None
        
    def load_logs(self) -> pd.DataFrame:

        # compile raw log files
        count = 0
        raw_data = []
        for fn in self.log_files:
            try:
                df = pd.read_csv(fn, comment='#')
                df['trajname'] = os.path.basename(fn)[:-4]  # remove .dat extension
                df['speed'] = self._speed_from_log(fn)
                df['repid'] = self._replica_idx_from_log(fn)
                raw_data.append(df)
                count += 1
            except Exception as e:
                logger.error(f"Error loading {fn}: {e}")
                continue
        if count == 0:
            logger.warning("No valid log files found.")
            return None
        logger.info(f"Loaded {count} log files'.")

        if not raw_data:
            logger.warning("No data loaded from log files.")
            return None
        
        return pd.concat(list(raw_data))

    def build_protocol_grids(self, raw_data: pd.DataFrame) -> dict[float, pd.DataFrame]:
        """
        Reconstruct the ideal protocol grid per speed analytically.

        Returns
        -------
        protocol_grids : dict
            protocol_grids[speed] = DataFrame with columns:
            ['step', 'r_target_protocol']
        """
        protocol_grids = {}

        for speed, g in raw_data.groupby("speed"):
            # Use first trajectory as reference
            ref = g.sort_values("step").iloc[0]

            # Identify protocol parameters
            step0 = g["step"].min()
            stepN = g["step"].max()

            # Initial r0 from first step
            r0 = g.loc[g["step"] == step0, "r_target"].iloc[0]

            # Infer dx_per_move robustly
            dx_vals = (
                g.groupby("trajname")["r_target"]
                .diff()
                .dropna()
                .values
            )
            dx = np.median(dx_vals)

            steps = np.arange(step0, stepN + 1)
            r_target_protocol = r0 + (steps - step0) * dx

            protocol_grids[speed] = pd.DataFrame({
                "step": steps,
                "r_target_protocol": r_target_protocol,
            })

        return protocol_grids
    
    def build_analysis_coord(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Attach an analysis coordinate r_coord derived from r_target,
        without redefining the protocol grid. In theory r_target should be the same
        across replicas for a given speed, but in practice there are tiny numerical differences,
        mostly becuase r0 is not exactly the same. Here we just take the median
        """
        df = raw_data.copy()

        df["r_coord"] = (
            df.groupby(["speed", "step"])["r_target"]
            .transform("median")
        )

        return df
    
    def integrate_force_dx(self, raw_data, r_column) -> pd.DataFrame:
        """Integrate the force over distance to compute work done.
        """
        grouped = raw_data.groupby("trajname")
        for traj, group in grouped:
            # group = group.sort_values(by=r_column)  # ensure sorted by distance
            group = group.sort_values(by='time')  # ensure sorted by time
            work = cumulative_trapezoid(group[self.force_column], group[r_column], initial=0.0)
            raw_data.loc[raw_data['trajname'] == traj, 'work'] = work
        
        return raw_data
    
    def filter_by_r_range(self, r_range, r_column) -> pd.DataFrame:
        """Filter raw_data by r_column within r_range.
        """

        logger.warning(f'Filtering data by x-range: {r_range}')
        self.raw_data = self.raw_data[(self.raw_data[r_column] >= r_range[0]) & 
                                        (self.raw_data[r_column] <= r_range[1])]
        return
        
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
    
    @staticmethod
    def _traj_to_log_name(traj_path: str) -> str:
        """
        Map a trajectory file path to the corresponding trajname used in raw_data.
        """
        base = os.path.basename(traj_path)
        root, ext = os.path.splitext(base)
        log_name = root.replace('traj', 'log')
        return log_name
    
    @staticmethod
    def _log_to_traj_file(log_path: str) -> str:
        """
        Map a log file path to the corresponding trajectory file name.
        """
        traj_name = log_path.replace('log', 'traj')
        traj_name = traj_name.replace('.dat', '.dcd')
        return traj_name
        
    def get_trace_features(self, 
                           features: list[str]=[ 'work', 'lag', 'r_before'],
                           stride: int=1, 
                           rescale_by_speed: bool = False,
                           zscore_by_speed: bool = False) -> pd.DataFrame:
        """
        Build a feature table from trace-like quantities (force, lag, work, ...),
        organized in r-space (x_col) for each trajectory. Optional rescaling/normalization
        by speed (in case of clustering multi-speed data).
        """

        df = self.raw_data.copy()
        
        # make sure trajs are ordered by step
        df = df.sort_values(['trajname', 'step']).reset_index(drop=True)

        # rescale trace features by speed
        if rescale_by_speed:
            for col in df.columns:
                if col in features:
                    df[col] = df[col] / df['speed']

        # z-score features within each speed 
        if zscore_by_speed:
            def _zscore_speed(group):
                for col in df.columns:
                    if col in features:
                        mean = group[col].mean()
                        std = group[col].std(ddof=0)
                        if std == 0 or np.isnan(std):
                            group[col] = 0.0
                        else:
                            group[col] = (group[col] - mean) / std
                return group

            df = df.groupby('speed', group_keys=False).apply(_zscore_speed)

        # apply a stride to the data to reduce size
        df = df.groupby(['speed', 'trajname']).apply(lambda g: g.iloc[::stride]).reset_index(drop=True)
        
        # round up values
        for col in df.columns:
            if col in features:
                df[col] = df[col].round(3)
                
        # keep only relevant columns
        keep_cols = ['trajname', 'speed', 'step', 'time'] + features
        df = df[keep_cols]
        
        return df
    
    def calculate_pocket_distances(self, 
                          group_A: str = None,
                          group_B: str = None,
                          recompute: bool = False,
                          stride: int = 2,
                          ) -> pd.DataFrame:
        """
        Compute (or load) distance features between pocket and ligand.

        Returns a DataFrame with columns:
            ['trajname', 'step', 'time'] + dist_* feature columns

        'step' is the frame index; 'time' is taken from the trajectory if available,
        otherwise time = step.
        """
        
        distance_file = f"{self.outdir}/{self.sysname}_pocketDistances.csv"

        if (not recompute) and os.path.exists(distance_file):
            df = pd.read_csv(distance_file)
            return df

        all_rows = []

        for traj in tqdm.tqdm(self.traj_files, desc="Calculating distances.."):
            u = mda.Universe(self.reference_pdb, traj)
            
            pocket_atoms = u.select_atoms(group_A)
            ligand_atoms = u.select_atoms(group_B)
            if ligand_atoms.n_atoms == 0 or pocket_atoms.n_atoms == 0:
                print(f"Warning: No atoms found for selection in trajectory {traj}. Skipping.")
                continue
            
            traj_name = self._traj_to_log_name(traj)
            speed = traj_name.split("_")[-2].strip("v")
        
            for ts in u.trajectory[::stride]:
                distances = distance_array(
                    pocket_atoms, ligand_atoms,
                    result=np.ndarray((len(pocket_atoms), len(ligand_atoms)))
                )
                dist_flat = distances.flatten() / 10.0  # nm
                time_ps = getattr(ts, "time", ts.frame)  # ts.time in ps

                row = {
                    "trajname": traj_name,
                    "speed": float(speed),
                    "step": ts.frame,   # index within this trajectory
                    "time": time_ps,
                }
                for i, v in enumerate(dist_flat):
                    row[f"dist_{i}"] = v
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.to_csv(distance_file, index=False)
        return df
    
    
    
    def add_estimator_results(self, estimator_name: str, results_df: pd.DataFrame):
        """Store estimator results in the SMDData object.
        """
        if self.results is None:
            self.results = pd.DataFrame()
        results_df['estimator'] = estimator_name
        self.results = pd.concat([self.results, results_df], ignore_index=True)
        return None    

    @staticmethod
    def _compute_p_neq(path_traj_counts: dict) -> dict:
        """Compute non-equilibrium path probabilities from trajectory counts.

        Parameters
        ----------
        path_traj_counts :
            Mapping of path label -> number of trajectories assigned to that path.

        Returns
        -------
        dict
            Mapping of path label -> p_neq (fraction of trajectories).  Empty
            dict when the total count is zero.
        """
        total_trajs = sum(path_traj_counts.values())
        if total_trajs <= 0:
            return {}
        return {
            path: count / total_trajs
            for path, count in path_traj_counts.items()
            if count > 0
        }

    @staticmethod
    def _compute_p_eq(results_df: pd.DataFrame, p_neq: dict, beta: float) -> dict:
        """Compute normalised equilibrium path probabilities for a single speed.

        Each path's unnormalised weight is p_neq[path] * Z_k, where Z_k is the
        path partition function obtained by numerically integrating
        exp(-beta * dG) over the reaction coordinate.

        Parameters
        ----------
        results_df :
            DataFrame with at least ``path``, ``step``, ``dG`` and ``r_coord``
            columns for a **single speed** (typically the per-step estimator output).
        p_neq :
            Non-equilibrium path probabilities as returned by
            :meth:`_compute_p_neq`.
        beta :
            Inverse thermal energy (1 / k_B T).

        Returns
        -------
        dict
            Mapping of path label -> normalised p_eq.  Empty dict when weights
            cannot be computed.
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

            dG0 = np.nanmin(dG)
            # dG0 = 0.0  # alternative: set reference to zero for each path (relative PMF)
            
            integrand = np.exp(-beta * (dG - dG0))
            Zk = np.trapz(integrand, x) * np.exp(-beta * dG0)
            p_eq_raw = p_neq[path] * Zk
            if not np.isfinite(p_eq_raw) or p_eq_raw < 0.0:
                logger.warning(
                    f"p_eq_raw={p_eq_raw} for path '{path}' is non-finite or negative. "
                    "Setting weight to 0."
                )
                p_eq_raw = 0.0
            weights[path] = p_eq_raw

        total_weight = sum(weights.values())
        if total_weight <= 0:
            return {}
        return {path: w / total_weight for path, w in weights.items()}

    def get_p_neq(self,
                  byspeed: bool = True
                  ) -> dict[str, float]:
        """p_neq Non-equilibrium path probabilities from sMD analysis.
        Returns a dictionary mapping path labels to p_neq values."""

        data = self.raw_data.copy()

        # decide grouping strategy
        if byspeed:
            grouping_iter = data.groupby("speed")
        else:
            grouping_iter = [(None, data)]

        p_neq_dic = {}
        for speed, g in grouping_iter:
            logger.info(f"Speed {speed} nm/ps has {g['path'].nunique()} unique paths.")
            path_traj_counts = g.groupby('path')['trajname'].nunique().to_dict()
            speed_key = speed if byspeed is not None else "all_speeds"
            p_neq_dic[speed_key] = SMDData._compute_p_neq(path_traj_counts)

        return p_neq_dic

    def get_p_eq(self,
                 byspeed: bool = True,
                 results: pd.DataFrame = None
                 ) -> dict[str, float]:
        """p_eq Equilibrium path probabilities from sMD analysis.
        Returns a dictionary mapping path labels to p_eq values.
        As described in https://doi.org/10.1063/5.0138761
        If weights are too different means the CV is not good enough.
        """
        if results is None:
            results = self.results
        if results is None:
            logger.error("No results available to compute p_eq.")
            return None

        preferred_estimators = ['cumulant', 'jarzynski'] # order of preference for which estimator to use for weights
        available_estimators = results['estimator'].dropna().unique().tolist()

        estimator_for_weights = None
        for est in preferred_estimators:
            if est in available_estimators:
                estimator_for_weights = est
                break

        if estimator_for_weights is None:
            if len(available_estimators) == 0:
                logger.error("No estimator results available to compute p_eq.")
                exit(1)
                        
        logger.info(f"Computing p_eq from estimator '{estimator_for_weights}'.")
        results = results[results['estimator'] == estimator_for_weights]

        # get p_neq first
        p_neq_dic = self.get_p_neq(byspeed=byspeed)

        p_eq_dic = defaultdict(dict)
        for speed, gspeed in results.groupby('speed'):
            speed_p_neq = p_neq_dic.get(speed, {})
            p_eq_dic[speed] = SMDData._compute_p_eq(gspeed, speed_p_neq, self.beta)

        # log details
        for speed in p_eq_dic:
            logger.info(f"Speed {speed} nm/ps path weights:")
            for path in p_eq_dic[speed]:
                logger.info(
                    f"  Path {path}: p_neq = {p_neq_dic[speed].get(path, 0):.6f}, "
                    f"p_eq = {p_eq_dic[speed][path]:.6f}"
                )

        return p_eq_dic
        
    # def _bin_data(self,
    #             x_col:str='r_before',
    #             use_quantiles: bool = True,
    #             n_bins: int=50,
    #             min_points: int = 1       # drop bins with < min_points
    #             ) -> tuple[pd.DataFrame, np.ndarray]:
    #     """
    #     Bins r_column into n_bins using pd.cut (equal width) or pd.qcut (equal count).
    #     After optional filtering of low-count bins, bins are reindexed to 0..M-1 and centers
    #     are returned only for surviving bins. 'r_coord' holds the center for each row.

    #     Returns
    #     -------
    #     raw_data : DataFrame with columns ['bin', 'r_coord'] added
    #     centers  : np.ndarray of bin centers aligned with bin indices 0..M-1
    #     """
    #     rvals = self.raw_data[x_col].to_numpy()
    #     raw_data = self.raw_data.copy()

    #     if use_quantiles:
    #         # equal-count bins
    #         codes, edges = pd.qcut(rvals, q=n_bins, labels=False, retbins=True, duplicates='drop', precision=3)
    #     else:
    #         # equal-width bins
    #         codes, edges = pd.cut(rvals, bins=n_bins, labels=False, include_lowest=True, right=False, retbins=True, precision=3)

    #     # assign bins; drop anything not assigned (NaN)
    #     raw_data['bin'] = pd.Series(codes, index=raw_data.index, dtype='Int64')
    #     raw_data = raw_data.dropna(subset=['bin']).copy()
    #     raw_data['bin'] = raw_data['bin'].astype(int)

    #     # pre-centers from edges
    #     centers = 0.5 * (edges[:-1] + edges[1:])

    #     # optional filter: remove bins with toso few points
    #     if min_points > 1:
    #         counts = raw_data['bin'].value_counts()
    #         keep = set(counts[counts >= min_points].index.tolist())
    #         raw_data = raw_data[raw_data['bin'].isin(keep)].copy()

    #     # reindex surviving bins to 0..M-1 and shrink centers accordingly
    #     present_bins = np.sort(raw_data['bin'].unique())
    #     bin_map = {old: i for i, old in enumerate(present_bins)}
    #     raw_data['bin'] = raw_data['bin'].map(bin_map)
    #     centers = np.asarray(centers)[present_bins]

    #     # these are the center each point belongs to
    #     raw_data['r_coord'] = raw_data['bin'].map(lambda b: centers[b] if 0 <= b < len(centers) else np.nan)

    #     return raw_data, centers
    
    # def _filter_low_count_bins(self, raw_data, min_points):
    #     """Filter low count bins per speed. This function filters out bins 
    #     that have fewer than `min_points` data points for each speed.
    #     """
    #     #TODO add this to bin_data and group by path too.
    #     # Im not dropping bins any more
        
    #     all_data = []
    #     for speed in raw_data['speed'].unique():

    #         df = raw_data[raw_data['speed'] == speed]

    #         # Count points per bin and filter out bins with too few points
    #         points_per_bin = df.groupby('bin').size()
    #         points_per_bin = points_per_bin[points_per_bin > min_points]  
    #         df = df[df['bin'].isin(points_per_bin.index)]
    #         df['speed'] = speed  # add speed column
    #         all_data.append(df)

    #     return pd.concat(all_data)
