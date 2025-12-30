import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.integrate import cumulative_trapezoid

import logging
logger = logging.getLogger("autopath")

class SMDData:
    def __init__(self, 
                sysname: str, 
                log_files: list[str],
                exclude_speeds: list[float] = None,
                x_range: tuple[float, float] | None = None,
                x_column: str = 'r_target',
                temperature: float = 300.0,
                 ):
        self.sysname = sysname
        self.log_files = log_files
        
        self.temperature = temperature
        self.R = 0.008314462618  # kJ/(mol*K)
        self.RT = self.R * self.temperature
        self.beta = 1.0 / self.RT
        
        # get it from the first log file
        self.pulling_direction = self.log_files[0].split("_")[-1].replace(".dat","")
        self.force_column = 'force'  # assuming the log files have a 'force' column
        
        # r_target is the theoretical target distance grid, the one that dcTMD uses
        # r_before is the actual distance before applying the constraint force. This one requires
        # binning because different replicas will have different grids.
        if x_column not in ['r_target', 'r_before']:
            logger.error(f'Unknown x_column: {x_column}. Must be "r_target" or "r_before".')
            exit(1)
            
        # load the data        
        raw_data = self.load_logs()
        
        # filter out unwanted speeds
        if exclude_speeds is not None:
            logger.warning(f'The following speeds will be excluded from analysis: {exclude_speeds}')
            raw_data = raw_data[~raw_data['speed'].isin(exclude_speeds)]

        # filter by x_range
        if x_range is not None:
            logger.warning(f'Filtering data by x-range: {x_range}')
            raw_data = raw_data[(raw_data[x_column] >= x_range[0]) & 
                                          (raw_data[x_column] <= x_range[1])]
        
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
        raw_data = self.integrate_force_dx(raw_data, x_column)
        
        raw_data['lag'] = raw_data['r_after'] - raw_data['r_target']
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
        logger.info(f"Loaded {count} log files for system '{self.sysname}'.")

        if not raw_data:
            logger.warning("No data loaded from log files.")
            return None
        
        return pd.concat(list(raw_data))
    
    def integrate_force_dx(self, raw_data, x_column) -> pd.DataFrame:
        """Integrate the force over distance to compute work done.
        """
        grouped = raw_data.groupby("trajname")
        for traj, group in grouped:
            # group = group.sort_values(by=x_column)  # ensure sorted by distance
            group = group.sort_values(by='time')  # ensure sorted by time
            work = cumulative_trapezoid(group[self.force_column], group[x_column], initial=0.0)
            raw_data.loc[raw_data['trajname'] == traj, 'work'] = work
        
        return raw_data

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
    
    def _bin_data(self,
                x_col:str='r_before',
                use_quantiles: bool = True,
                n_bins: int=50,
                min_points: int = 1       # drop bins with < min_points
                ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Bins x_column into n_bins using pd.cut (equal width) or pd.qcut (equal count).
        After optional filtering of low-count bins, bins are reindexed to 0..M-1 and centers
        are returned only for surviving bins. 'r_coord' holds the center for each row.

        Returns
        -------
        raw_data : DataFrame with columns ['bin', 'r_coord'] added
        centers  : np.ndarray of bin centers aligned with bin indices 0..M-1
        """
        rvals = self.raw_data[x_col].to_numpy()
        raw_data = self.raw_data.copy()

        if use_quantiles:
            # equal-count bins
            codes, edges = pd.qcut(rvals, q=n_bins, labels=False, retbins=True, duplicates='drop', precision=3)
        else:
            # equal-width bins
            codes, edges = pd.cut(rvals, bins=n_bins, labels=False, include_lowest=True, right=False, retbins=True, precision=3)

        # assign bins; drop anything not assigned (NaN)
        raw_data['bin'] = pd.Series(codes, index=raw_data.index, dtype='Int64')
        raw_data = raw_data.dropna(subset=['bin']).copy()
        raw_data['bin'] = raw_data['bin'].astype(int)

        # pre-centers from edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        # optional filter: remove bins with toso few points
        if min_points > 1:
            counts = raw_data['bin'].value_counts()
            keep = set(counts[counts >= min_points].index.tolist())
            raw_data = raw_data[raw_data['bin'].isin(keep)].copy()

        # reindex surviving bins to 0..M-1 and shrink centers accordingly
        present_bins = np.sort(raw_data['bin'].unique())
        bin_map = {old: i for i, old in enumerate(present_bins)}
        raw_data['bin'] = raw_data['bin'].map(bin_map)
        centers = np.asarray(centers)[present_bins]

        # these are the center each point belongs to
        raw_data['r_coord'] = raw_data['bin'].map(lambda b: centers[b] if 0 <= b < len(centers) else np.nan)

        return raw_data, centers
    
    def _filter_low_count_bins(self, raw_data, min_points):
        """Filter low count bins per speed. This function filters out bins 
        that have fewer than `min_points` data points for each speed.
        """
        #TODO add this to bin_data and group by path too.
        # Im not dropping bins any more
        
        all_data = []
        for speed in raw_data['speed'].unique():

            df = raw_data[raw_data['speed'] == speed]

            # Count points per bin and filter out bins with too few points
            points_per_bin = df.groupby('bin').size()
            points_per_bin = points_per_bin[points_per_bin > min_points]  
            df = df[df['bin'].isin(points_per_bin.index)]
            df['speed'] = speed  # add speed column
            all_data.append(df)

        return pd.concat(all_data)