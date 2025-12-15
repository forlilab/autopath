import os
import glob
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import linregress
from scipy.integrate import cumulative_trapezoid
from scipy import special

import statsmodels.formula.api as smf

from dtaidistance import dtw_ndim
import kmedoids

import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array  
from MDAnalysis.analysis import align, density

import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# style.use("fivethirtyeight")

from pymbar import timeseries

import statsmodels.api as sm

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging
logger = logging.getLogger("autopath")

class SteeredMDAnalysis:
    """Class to analyze Steered Molecular Dynamics (sMD) data, mostly within the dcTMD framework.
    For more information on the dcTMD see...

    
    """
    def __init__(self, 
                 log_files: list[str] = None,
                 sysname: str = None,
                 n_bins: int = 50,
                 temperature: float = 300, #K
                 timestep: float = 0.004, #ps
                 dist_minmax: tuple = None, #nm
                 cluster_paths: bool = False,
                 cluster_range: tuple = None, #nm
                 trajectories: list[str] = None,
                 reference_pdb: str = None,
                 pocket_select: str = 'protein and (around 6.0 resname UNK) and name CA',
                 ligand_select: str = 'resname UNK and not name H*',
                 pulling_direction: str = 'forward', # 'forward' or 'backward'
                 outdir: str = None,
                 seed: int = 42
                 ):

        """Initialize the SteeredMDAnalysis class with log files and parameters."""
        
        if log_files is None or len(log_files) == 0:
            raise ValueError("No log files provided for analysis.")
        
        if sysname is not None:
            self.sysname = sysname
        else:
            self.sysname = log_files[0].split('/')[0]  # Extract system name from the first log file path

        if outdir is not None:
            self.outdir = outdir
        else:
            self.outdir = os.path.join(os.path.dirname(log_files[0]), 'analysis')  # Output directory is the same as the first log file
        os.makedirs(self.outdir, exist_ok=True)
        
        self.temp = temperature
        self.R = 0.008314462618  # kJ/(mol*K)
        self.RT = self.R * self.temp
        self.beta = 1.0 / self.RT
                
        self.timestep = timestep

        self.cluster_paths = cluster_paths
        self.cluster_range = cluster_range
        self.trajectories = trajectories
        if cluster_paths in ['geometric', 'full']:
            if (trajectories is None or len(trajectories) == 0):
                raise ValueError(f"{cluster_paths} clustering of paths is enabled, but no trajectories provided.")
        
        # this will be used for topology and pocket selection, so should be pdb before pulling
        self.reference_pdb = reference_pdb
        self.ligand_select = ligand_select
        self.pocket_select = pocket_select

        self.work_column = 'work'
        self.force_column = 'force'

        self.n_bins = int(n_bins)
        self.dist_minmax = dist_minmax  # default min/max for distance bins
        self.log_files = log_files

        if pulling_direction not in ['forward', 'backward']:
            raise ValueError("pulling_direction must be either 'forward' or 'backward'.")

        self.pulling_direction = pulling_direction
        self.seed = seed

        return

    def run_analysis(self,
                     use_target_grid: bool = True,
                     speeds: list[float] = None,
                     temperature: float = None,
                     fit_GMM: bool = True
                     )-> pd.DataFrame:
        
        """Run the analysis on the raw data.
        This method computes the free energy difference using Jarzynski's equality
        and the dissipated work approximation.
        """
        
        GMM_WEIGHT_CUTOFF = 0.0 # # cutoff for GMM weights, below which we ignore the component

        # r_target is the theoretical target distance grid, the one that dcTMD uses
        # r_before is the actual distance before applying the constraint force. This one requires
        # binning because different replicas will have different grids.
        if use_target_grid:
            self.dist_column = 'r_target'
        else:
            self.dist_column = 'r_before'  # binned distance before applying constraint

        # assemble the master dataframe loading log files
        raw_data = self.load_logs()

        # sometimes you wanna run the analysis with a different temperature or exlude some speeds
        if temperature is not None:
            self.temp = temperature
            self.RT = self.R * self.temp
            self.beta = 1.0 / self.RT

        if speeds is not None:
            print(f'WARNING: Filtering data by speeds: {speeds}')
            raw_data = raw_data[raw_data['speed'].isin(speeds)]

        if self.dist_minmax is not None:
            print(f'WARNING: Filtering data by distance range: {self.dist_minmax}')
            raw_data = raw_data[(raw_data[self.dist_column] >= self.dist_minmax[0]) & 
                                          (raw_data[self.dist_column] <= self.dist_minmax[1])]

        # Integrate force to get work
        raw_data = self.integrate_force_dx(raw_data)

        # bin the data if not using target grid
        if not use_target_grid:
            raw_data, centers = self.bin_data(raw_data, 
                                            self.n_bins, 
                                            use_quantiles=True, 
                                            min_points=1)
        
        
        # export the raw data after work calculation and save it to csv     
        self.raw_data = raw_data
        self.raw_data.to_csv(f"{self.outdir}/sMD_data_raw.csv")
        
        # filter out trajectories that did not reach close contact
        to_drop = []
        for traj_name, traj_data in raw_data.groupby('trajname'):
            if self.pulling_direction == 'forward':
                if traj_data["r_after"].min() < 0.1:
                    print(f'WARNING: Dropping {traj_name}, min distance {traj_data["r_after"].min():.2f} nm')
                    to_drop.append(traj_name)
            else:  # backward pulling
                if traj_data["r_after"].min() > 0.1:
                    print(f'WARNING: Dropping {traj_name}, min distance {traj_data["r_after"].min():.2f} nm')
                    to_drop.append(traj_name)
                    
        processed_data = raw_data[~raw_data['trajname'].isin(to_drop)]
        
        # cluster the pulling paths if requested
        if self.cluster_paths == False:
            processed_data['path'] = 1 # default to single cluster if no clustering method is specified            
        else:
            processed_data = self.find_paths_smd(processed_data, recompute_geom=False, do_plots=True)
        
        # filter out paths with too few trajectories
        to_drop = []
        for path in processed_data.groupby(['path'])[['trajname']].nunique().iterrows():
            if path[1]['trajname'] <= 3:
                to_drop.append(path[0])
                print(f"WARNING: Dropping path: {path[0]} due to insufficient number of trajectories ({path[1].trajname}).")        
        processed_data = processed_data[~processed_data['path'].isin(to_drop)]

        # decorrelate work values using statistical inefficiency g or by replica averaging
        # If we dont decorrelate, we should use the per-replica aggregated work. i.e. each replica contributes one work value per bin ENSEMBLE AVERAGE OVER REPLICAS
        if use_target_grid:
            processed_data = self.build_common_target_grid(processed_data)
            group_keys = ["step", "speed", "path"]  # integer key avoids fragmentation
        else:
            processed_data = self.decorrelate_work_data(processed_data, use_g=True)
            group_keys = ["r_coord", "speed", "path"]  # no 'step' on the binning path
                
        # processed_data['work'] = processed_data['work'] * self.beta
        # processed_data['work'] = processed_data['work'] / 2.476  # convert to KT
        
        self.processed_data = processed_data
        self.processed_data.to_csv(f"{self.outdir}/sMD_data_processed.csv")
        
        results = []
        gmm_results = defaultdict(dict)  # to store GMM results for plotting
        
        for (coord, speed, path), group in processed_data.groupby(group_keys):
            
            r_coord = float(group['r_coord'].iloc[0])
            
            #coord is the index or step, r_coord is the actual distance value
            # print(f'analyzing coord={coord:.2f}, r_bin={r_coord:.2f}, speed={speed:.5f}, path={path} with {len(group)} points.')
            raw_W = group[self.work_column].astype(float).values

            Wmean_raw = raw_W.mean()
            var_raw = raw_W.var(ddof=1)
            Wdiss_raw = 0.5 * self.beta * var_raw

            # plain Jarzynski on the samples
            dG_Jarzynski = -(1.0/self.beta) * (
                special.logsumexp(-self.beta * raw_W) - np.log(raw_W.size)
            )
            
            # GMM branch
            dG_mix_exact = Wdiss_neq = Wmean_neq = var_neq= np.nan
            if fit_GMM:
                if len(raw_W) < 2:
                    # print(f"Not enough data points for GMM fitting at r_bin={r_coord:.2f}, speed={speed:.5f}, path={path:.2f}. Skipping GMM fit.")
                    continue

                gmm_dict = self.fit_gmm_to_work_values(raw_W,
                                                        max_K=3,
                                                        covariance_type='spherical',
                                                        random_state=self.seed)

                #These have shape (K,) for K components
                w = np.asarray(gmm_dict["GMM_weights"])          # α_k
                mu = np.asarray(gmm_dict["GMM_means"]).ravel()    # μ_k
                sig2 = np.asarray(gmm_dict["GMM_variances"])       # σ_k²

                # cumulant (second order) per component
                dG_k = mu - 0.5 * self.beta * sig2 # now this is exact for each Gaussian

                # mask small nonequilibrium weights, which means ignore small gaussians
                if np.any(w < GMM_WEIGHT_CUTOFF):
                    print(f'WARNING: Filtering out {np.sum(w < GMM_WEIGHT_CUTOFF)} '
                        f'components at grid={r_coord:.3f}, v={speed:.5g}, path={path}.')
                           
                w = np.where(w < GMM_WEIGHT_CUTOFF, 0.0, w)
                w /= w.sum()  # normalize weights

                # Transfor the weights from the non-equilibrium populations.
                # This is for path impotance, but I should use w becuase 
                # jarzynski and dcTMD are based on equilibrium weights (I think).
                # p_eq = w * np.exp(-self.RT * dG_k)
                # p_eq /= p_eq.sum()

                # exact mixture Jarzynski
                # This is the exact Jarzynski estimator for the Gaussian mixture in each bin.
                # If k=1 this collapses to the exact Jarzynski estimator for the single gaussian
                # and this can be approximated by cumulant expansion to the second order, what the dcTMD paper does.
                log_terms = -self.beta * mu + 0.5 * (self.beta**2) * sig2
                log_Z = special.logsumexp(log_terms, b=w) # log ⟨e^{-βW}⟩
                dG_mix_exact = -(1.0/self.beta) * log_Z
                
                # non-equilibrium mean of W
                Wmean_neq = np.dot(w, mu)
                
                Wdiss_neq = Wmean_neq - dG_mix_exact
                
                # Non-eq variance of W
                var_neq = np.dot(w, sig2 + (mu - Wmean_neq)**2)
                # this one should match Wdiss_neq
                # Wdiss_neq = 0.5 * self.beta * var_neq

                # Collect results in a dictionary for plotting 
                key = (float(coord), int(path))                
                gmm_results[key][speed] = {'GMM_neq_weights': w,
                                            'replica_W': raw_W,
                                            'GMM_means': mu,
                                            'GMM_variances': sig2,
                                            'Wmean_mix': Wmean_neq
                                            }
            # build this partial dataframe
            results.append({
                'step': coord,
                'r_coord': r_coord,
                'speed': speed,
                'path': path,
                'Wmean': Wmean_raw,
                'Wdiss': Wdiss_raw,
                'dG': Wmean_raw - Wdiss_raw,
                'dG_Jarzynski': dG_Jarzynski, # exact from samples, normal Jarzynski
                'dG_Jarzynski_gmm': dG_mix_exact, # exact mixture Jarzynski
                'Wmean_gmm': Wmean_neq,
                'Wdiss_gmm': Wdiss_neq,
                'dG_gmm': Wmean_neq - Wdiss_neq,
                'Wdiss_Jarzynski_gmm': Wmean_neq - dG_mix_exact,
                'Wdiss_Jarzynski': Wmean_raw - dG_Jarzynski,
                'varW_gmm_neq': var_neq
            })

        results = pd.DataFrame(results)

        return results, gmm_results

    def load_logs(self) -> pd.DataFrame:

        # compile raw log files
        count = 0
        raw_data = []
        for fn in self.log_files:
            try:
                df = pd.read_csv(fn, comment='#')
                df['trajname'] = os.path.basename(fn)[:-4]  # remove .dat extension
                df['speed'] = self.speed_from_log(fn)
                df['replica'] = self.replica_idx_from_log(fn)
                raw_data.append(df)
                count += 1
            except Exception as e:
                print(f"Error loading {fn}: {e}")
                continue
        if count == 0:
            logger.warning("No valid log files found.")
            return None
        logger.info(f"Loaded {count} log files for system '{self.sysname}'.")

        if not raw_data:
            logger.warning("No data loaded from log files.")
            return None
        return pd.concat(list(raw_data))
    
    def integrate_force_dx(self, raw_data) -> pd.DataFrame:
        """Integrate the force over distance to compute work done.
        """
        grouped = raw_data.groupby("trajname")
        for traj, group in grouped:
            # group = group.sort_values(by=self.dist_column)
            group = group.sort_values(by='time')  # ensure sorted by time
            work = cumulative_trapezoid(group[self.force_column], group[self.dist_column], initial=0.0)
            raw_data.loc[raw_data['trajname'] == traj, self.work_column] = work
        
        return raw_data

    def build_common_target_grid(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Trim each (speed, path) group so all replicas share the same number of frames.
        This is required to build a common target grid across replicas.
        """
        df = raw_data.copy()            
        new = []
        for (speed, path), g in df.groupby(["speed", "path"], sort=False):
            # Intersection length across replicas
            nmin = g.groupby("trajname")["step"].max().add(1).min()
            gg = g[g["step"] < nmin].copy()
            # average grid (for reference/plots)
            grid = gg.pivot_table(index="step", values="r_target", aggfunc="mean").reset_index()
            gg = gg.merge(grid, on="step", suffixes=("", "_mean"))
            gg["r_coord"] = gg["r_target_mean"].values
            new.append(gg.drop(columns=["r_target_mean"]))
            
            new_df = pd.concat(new, ignore_index=True)
            
            # do this to avoid missmatch of float r_coord values due to numerical precision
            new_df["r_coord"] = (new_df
                                .groupby(["speed"])["r_coord"]
                                .transform(lambda x: np.round(x, 3))
                                )
        return new_df

    def bin_data(self,
                raw_data: pd.DataFrame,
                n_bins: int,
                use_quantiles: bool = True,
                min_points: int = 1       # drop bins with < min_points
                ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Bins self.dist_column into n_bins using pd.cut (equal width) or pd.qcut (equal count).
        After optional filtering of low-count bins, bins are reindexed to 0..M-1 and centers
        are returned only for surviving bins. 'r_coord' holds the center for each row.

        Returns
        -------
        raw_data : DataFrame with columns ['bin', 'r_coord'] added
        centers  : np.ndarray of bin centers aligned with bin indices 0..M-1
        """
        rvals = raw_data[self.dist_column].to_numpy()
        raw_data = raw_data.copy()

        if use_quantiles:
            # equal-count bins
            codes, edges = pd.qcut(rvals, q=n_bins, labels=False, retbins=True, duplicates='drop', precision=2)
        else:
            # equal-width bins
            codes, edges = pd.cut(rvals, bins=n_bins, labels=False, include_lowest=True, right=False, retbins=True, precision=2)

        # assign bins; drop anything not assigned (NaN)
        raw_data['bin'] = pd.Series(codes, index=raw_data.index, dtype='Int64')
        raw_data = raw_data.dropna(subset=['bin']).copy()
        raw_data['bin'] = raw_data['bin'].astype(int)

        # pre-centers from edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        # # optional filter: remove bins with toso few points
        # if min_points > 1:
        #     counts = raw_data['bin'].value_counts()
        #     keep = set(counts[counts >= min_points].index.tolist())
        #     raw_data = raw_data[raw_data['bin'].isin(keep)].copy()

        # reindex surviving bins to 0..M-1 and shrink centers accordingly
        present_bins = np.sort(raw_data['bin'].unique())
        bin_map = {old: i for i, old in enumerate(present_bins)}
        raw_data['bin'] = raw_data['bin'].map(bin_map)
        centers = np.asarray(centers)[present_bins]

        # these are the center each point belongs to
        raw_data['r_coord'] = raw_data['bin'].map(lambda b: centers[b] if 0 <= b < len(centers) else np.nan)

        return raw_data, centers

    def filter_low_count_bins(self, raw_data, min_points):
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
    
    @staticmethod
    def smooth_columns(df: pd.DataFrame,
                       columns: list[str]=None,
                       sigma: float = 2.0
                        ) -> pd.DataFrame:
        new_df = df.copy()
        if columns is None or len(columns) == 0:
            columns = [c for c in new_df.columns if c not in ['speed','path','r_coord','r_bin','replica','trajname']]
        for (speed, path), group in df.groupby(['speed','path']):
            for col in columns:
                mask = (new_df['speed'] == speed) & (new_df['path'] == path)
                new_df.loc[mask, f'{col}_smooth'] = gaussian_filter1d(group[col], sigma=sigma)
        return new_df

    @staticmethod
    def fit_gmm_to_work_values(raw_work,
                                max_K:int=3, 
                                max_iter:int=1000,
                                covariance_type:str='spherical',
                                random_state:int=42
                                ):   
        """Fit GMMs to the work values and return the main parameters."""

        # TODO maybe we dont need the elbow loop and can approximate K with the dirichlet process

        raw_work = raw_work.reshape(-1, 1) # reshape for GMM
        n_samples = raw_work.shape[0]

        scores = []
        models = []

        for n in range(1, max_K+1):
            if n > n_samples:
                break  # can't fit more components than points

            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,  # 'full' may overfit for small datasets
                max_iter=max_iter,
                init_params="k-means++",
                random_state=random_state,
                # reg_covar=1e-6  # regularization to avoid singular covariance matrices
            )
            gmm.fit(raw_work)
            models.append(gmm)

            scores.append(gmm.bic(raw_work))
    
        best_gmm = models[np.argmin(scores)]

        variances = best_gmm.covariances_.reshape(best_gmm.n_components, -1).flatten()

        augmented_data = {
            "GMM_n_components": best_gmm.n_components,
            "GMM_scores": scores,
            "GMM_weights": best_gmm.weights_,
            "GMM_means": best_gmm.means_,
            "GMM_variances": variances,
            "GMM_posteriors": best_gmm.predict_proba(raw_work),
            "GMM_labels": best_gmm.predict(raw_work),
        }
        return augmented_data
        
    def plot_gmm_per_speed_overlay_paths(
        self,
        results: dict,
        ncols: int = 8,
        figsize: tuple = (22, 4),
        outdir: str | None = None,
        hist_bins: int = 30,
        path_order: list[int] | None = None,   # optional explicit ordering of paths
        gray_range: tuple[float, float] = (0.20, 0.80),  # darkest..lightest gray
    ):
        """
        Overlay work histograms + GMM PDFs for ALL paths in the same panel (one panel per r_bin).
        'results' must be keyed by (r_bin, path) -> { speed: info_dict }.
        Each info_dict must contain 'replica_W', 'GMM_neq_weights', 'GMM_means', 'GMM_variances'.
        """

        if outdir is None:
            outdir = os.path.join(os.getcwd(), self.sysname)
        os.makedirs(outdir, exist_ok=True)

        # Collect speeds and r_bins present
        all_speeds = sorted({s for sd in results.values() for s in sd.keys()})
        all_rbins  = sorted({k[0] for k in results.keys()})
        all_paths  = sorted({k[1] for k in results.keys()})
        # Optional explicit path order; otherwise use sorted unique paths
        if path_order is None:
            path_order = list(all_paths)

        # Build grayscale palette for the number of paths we will display
        def build_gray_map(paths: list[int]) -> dict[int, tuple[float,float,float]]:
            n = max(1, len(paths))
            gmin, gmax = gray_range
            shades = np.linspace(gmin, gmax, n)
            return {p: (g, g, g) for p, g in zip(paths, shades)}

        for spd in all_speeds:
            # r_bins with at least one path having data at this speed
            rbins_present = []
            for rbin in all_rbins:
                has_any = False
                for p in path_order:
                    info = results.get((rbin, p), {}).get(spd, None)
                    if info is None:
                        continue
                    work = info.get("replica_W", np.array([]))
                    if isinstance(work, list):
                        work = np.concatenate([np.asarray(w).ravel() for w in work]) if work else np.array([])
                    work = np.asarray(work).ravel()
                    if work.size > 0:
                        has_any = True
                        break
                if has_any:
                    rbins_present.append(rbin)

            if not rbins_present:
                continue

            rbins_present = sorted(rbins_present)
            nplots = len(rbins_present)
            nrows  = math.ceil(nplots / ncols)

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(figsize[0], figsize[1] * nrows),
                sharex=False, sharey=False, squeeze=False
            )
            axes = axes.flatten()

            # Determine which paths actually appear at this speed to set colors consistently
            paths_this_speed = sorted({p for (r,p), sd in results.items() if spd in sd})
            # Maintain user-specified order but drop missing
            paths_this_speed = [p for p in path_order if p in paths_this_speed]
            gray_map = build_gray_map(paths_this_speed)

            for ax, rbin in zip(axes, rbins_present):
                # Collect per-path data in this (speed, rbin)
                per_path = {}
                all_work_for_xlim = []
                for p in paths_this_speed:
                    info = results.get((rbin, p), {}).get(spd, None)
                    if info is None:
                        continue
                    work = info.get("replica_W", np.array([]))
                    if isinstance(work, list):
                        work = np.concatenate([np.asarray(w).ravel() for w in work]) if work else np.array([])
                    work = np.asarray(work).ravel()
                    if work.size == 0:
                        continue
                    per_path[p] = dict(
                        work=work,
                        weights=np.asarray(info["GMM_neq_weights"]),
                        means=np.asarray(info["GMM_means"]),
                        variances=np.asarray(info["GMM_variances"]),
                    )
                    all_work_for_xlim.append(work)

                if not per_path:
                    ax.set_visible(False)
                    continue

                wmin = min(float(w.min()) for w in all_work_for_xlim)
                wmax = max(float(w.max()) for w in all_work_for_xlim)
                if not np.isfinite(wmin) or not np.isfinite(wmax) or np.isclose(wmin, wmax):
                    ax.axvline(wmin, color='k', lw=1, alpha=0.7)
                    ax.set_title(f"r_bin={rbin:.2f} (degenerate)")
                    continue

                x = np.linspace(wmin, wmax, 256)

                # Overlay: histogram + components + total per path
                for p in paths_this_speed:
                    if p not in per_path:
                        continue
                    color = gray_map[p]
                    dat   = per_path[p]
                    work, weights, means, variances = dat["work"], dat["weights"], dat["means"], dat["variances"]

                    # Histogram
                    ax.hist(work, bins=hist_bins, density=True, alpha=0.35, color=color, label=f"path {p}")

                    # Mixture components (solid) and total (dashed)
                    stds = np.sqrt(np.clip(variances, 1e-12, None))
                    for w, m, s in zip(weights, means, stds):
                        if w <= 0:
                            continue
                        ax.plot(x, w * norm.pdf(x, loc=m, scale=s), lw=1.5, color=color)
                    total_pdf = np.sum([w * norm.pdf(x, loc=m, scale=s)
                                        for w, m, s in zip(weights, means, stds)], axis=0)
                    ax.plot(x, total_pdf, ls="--", lw=2, color=color)

                ax.set_title(f"r_bin={rbin:.2f}")
                ax.set_xlabel("Work (kJ/mol)")
                ax.set_ylabel("Density")

            # Hide leftover axes
            for ax in axes[nplots:]:
                ax.set_visible(False)

            # One legend for all paths
            handles, labels = [], []
            for p in paths_this_speed:
                ph = plt.Line2D([0], [0], color=gray_map[p], lw=6)
                handles.append(ph)
                labels.append(f"path {p}")
            if handles:
                fig.legend(handles, labels, loc="upper right", frameon=False)

            fig.suptitle(f"GMM fits at speed = {spd:.5f}  (overlaid paths)", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = os.path.join(outdir, f"gmm_fits_overlay_speed_{spd:.5f}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            return      

    def plot_gmm_per_speed(self, results, ncols=8, figsize=(22, 4), outdir:str=None):
        """
        For each pulling speed, plot a grid of GMM fits across r_bins.
        
        """
        if outdir is None:
            outdir = os.path.join(os.getcwd(), self.sysname)
        os.makedirs(outdir, exist_ok=True)

        # Identify all unique speeds
        speeds = sorted({speed for speed_dict in results.values() for speed in speed_dict.keys()})
        for speed in speeds:

            r_bins_for_speed = [r for r, speed_dict in results.items() if speed in speed_dict]
            nplots = len(r_bins_for_speed)
            nrows = math.ceil(nplots / ncols)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows), 
                                    sharex=False, sharey=False,
                                    squeeze=False)
            axes = axes.flatten()
            
            for ax, r_bin in zip(axes, r_bins_for_speed):
                info = results[r_bin][speed]

                work = info['replica_W'].flatten()
                weights = info["GMM_neq_weights"]
                means = info["GMM_means"]
                variances = info["GMM_variances"]
                
                ax.hist(work, bins=30, density=True, alpha=0.4, color="C0")
                
                # PDF grid
                x = np.linspace(work.min(), work.max(), 30)
                
                for w, m, v in zip(weights, means, variances):
                    pdf_comp = w * norm.pdf(x, loc=m, scale=np.sqrt(v))
                    # ax.plot(x, pdf_comp, lw=2)
                
                # Plot total mixture PDF
                total_pdf = np.sum([w * norm.pdf(x, loc=m, scale=np.sqrt(v))
                                    for w, m, v in zip(weights, means, variances)], axis=0)
                # ax.plot(x, total_pdf, "k--", lw=2)
                
                ax.set_title(f"r_bin={r_bin:.2f}")
                ax.set_xlabel("Work (kJ/mol)")
                ax.set_ylabel("Density")
            
            for ax in axes[nplots:]:
                ax.set_visible(False)
            
            fig.suptitle(f"GMM Fits at speed = {speed:.5f}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f'{outdir}/gmm_fits_speed_{speed:.5f}.png', dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()
        return
    
    def decorrelate_work_data(self, 
                              data: pd.DataFrame,
                              use_g:bool=True) -> pd.DataFrame:    
        decorrelated_data = []
        g = None
        for (r_bin, v, path), group in data.groupby(["r_coord", "speed", "path"]):
            # decorrelate using the statistical inefficiency
            if use_g:
                work_arrays = []
                for replica, traj in group.groupby("replica"):
                    W = traj[self.work_column].values
                    work_arrays.append(W)
                # get the statistical inefficiency for this set of work arrays
                g = timeseries.statistical_inefficiency_multiple(work_arrays, return_correlation_function=False)
                
            for replica, traj in group.groupby("replica"):
                W = traj[self.work_column].values
                if use_g:
                    indices = timeseries.subsample_correlated_data(W, g, conservative=False)
                    W_decorrelated = W[indices]
                    # Use iloc to select rows by integer position
                    selected_rows = traj.iloc[indices]
                else:
                    # decorrelate by getting the mean per replica
                    W_decorrelated = np.array([np.mean(W)])
                    # For mean, take the first row as representative
                    selected_rows = traj.iloc[[0]]
                
                # Append each decorrelated point with its corresponding metadata
                for idx, w_val in enumerate(W_decorrelated):
                    row = selected_rows.iloc[idx]
                    decorrelated_data.append({
                        'r_coord': r_bin,
                        'r_target': row['r_target'],
                        'r_before': row['r_before'],
                        'r_after': row['r_after'],
                        'NC': row['NC'],
                        'force': row['force'],
                        'U_cvpack': row['U_cvpack'],
                        self.work_column: w_val,
                        'speed': v,
                        'replica': replica,
                        'trajname': traj['trajname'].iloc[0],
                        'path': path,
                })

        decorrelated_df = pd.DataFrame(decorrelated_data)      
        return decorrelated_df
        
    @staticmethod
    def friction_from_wdiss(df: pd.DataFrame, 
                            w_col:str='Wdiss', # could use smoothed column here
                            x_col:str='r_coord',
                            use_spline:bool=True) -> pd.DataFrame:
        rows = []
        for (speed, path), g in df.groupby(["speed","path"]):
            g = g.sort_values(x_col, ascending=True)
            if use_spline:
                # Use a cubic spline fit for derivative
                spline = UnivariateSpline(g[x_col].values, g[w_col].values, k=3)
                Gamma = spline.derivative()(g[x_col].values) / speed
            else:
                dWdiss_dx = np.gradient(g[w_col].values, g[x_col].values)
                Gamma = dWdiss_dx / speed
            rows.append(pd.DataFrame({x_col: g[x_col].values,
                                    "Gamma": Gamma,
                                    "speed": speed, 
                                    "path": path}))
        gamma_df = pd.concat(rows, ignore_index=True)
        # merge back to original df
        df = df.merge(gamma_df, on=[x_col, 'speed', 'path'], how='left')
        return df
    
    def extrapolate_to_v0(self,
                        df: pd.DataFrame = None,
                        x_col:str='r_coord',
                        param_cols: list[str]=['Wdiss_gmm'],
                        speeds: list[float] = None,
                        mixed_models: bool = False
                        ) -> pd.DataFrame:

        """Extrapolate a given parameter to zero speed using linear regression. 
        This method groups the data by speed and fits a linear regression to the
        param vs speed for each bin."""

        # Filter out speeds if provided. You could only want to fit specific (low) speeds
        if speeds is not None:
            speeds = [float(s) for s in speeds]
            df = df[df['speed'].isin(speeds)]

        if df['speed'].nunique() < 2:
            print("Not enough speeds for extrapolation.")
            return pd.DataFrame()
        
        results = []
        for param_col in param_cols:
            # FIXME 
            if mixed_models:
                df = df.dropna(subset=[param_col])
                model  = smf.mixedlm(f"{param_col} ~ speed", df,
                                    groups=df[x_col],
                                    re_formula="~speed")
                result = model.fit(reml=False)
                for r_bin, rand_eff in result.random_effects.items():
                    intercept = result.fe_params["Intercept"] + rand_eff["Group"]
                    slope = result.fe_params["speed"] + rand_eff["speed"]
                    results.append({
                    x_col: r_bin,
                    f"{param_col}_v0_intercept": intercept,
                    f"{param_col}_v0_slope": slope,
                    "R2": 1.0
                })
                    
            else:
                results = []
                for param_col in param_cols:
                    _df = []
                    for (r_bin, path), group in df.groupby([x_col,'path']):
                        speeds = group['speed'].values
                        means = group[param_col].values

                        lr_results = linregress(speeds, means)
                        _df.append({
                            x_col: r_bin,
                            'path': path,
                            f"{param_col}_v0_intercept": lr_results.intercept,
                            f"{param_col}_v0_slope": lr_results.slope,
                            f"{param_col}_v0_intercept_se": lr_results.intercept_stderr,
                            f"{param_col}_v0_slope_se": lr_results.stderr,
                            'R2': lr_results.rvalue**2,
                            'n_speeds': len(speeds)
                        })

                    if len(_df) == 0:
                        print(f"No valid extrapolation results found for {param_col}.")

                    results.append(pd.DataFrame(_df))

                results = pd.concat(results, axis=1)

        return results
        
    def plot_extrapolated_param(self, 
                                df: pd.DataFrame = None, 
                                param: str = 'Wdiss',
                                x_col: str = 'step',
                                ):
        """Plot the extrapolated parameter vs x_col with R2 color mapping and error bands.
        
        If a 'path' column is present, creates one subplot per path in a single figure.
        """
        outfname = os.path.join(self.outdir, f'{self.sysname}_{param}_extrapolated.png')
        color_col = 'R2'  # Column for color mapping
        se_col = f'{param}_se'

        if df is None or df.empty:
            return

        # Figure out x for the whole df (only used if x_col not present)
        if x_col in df.columns:
            x_global = df[x_col]
        else:
            x_global = df.index

        # Normalize R2 for colormap across all paths
        norm = mcolors.Normalize(vmin=df[color_col].min(), vmax=df[color_col].max())
        cmap = cm.get_cmap('coolwarm')

        # Determine paths
        if 'path' in df.columns:
            paths = sorted(df['path'].unique())
        else:
            paths = [None]  # single "path" (no splitting)

        n_paths = len(paths)
        fig, axes = plt.subplots(
            n_paths, 1,
            figsize=(6, 4 * n_paths),
            sharex=True if x_col in df.columns else False
        )
        if n_paths == 1:
            axes = [axes]  # make iterable

        for ax, path_val in zip(axes, paths):
            if path_val is not None:
                df_p = df[df['path'] == path_val].copy()
            else:
                df_p = df.copy()

            # Sort by x for nice plotting
            if x_col in df_p.columns:
                df_p = df_p.sort_values(x_col)
                x = df_p[x_col]
            else:
                df_p = df_p.sort_index()
                x = df_p.index

            y = df_p[param].values
            yerr = df_p[se_col].values if se_col in df_p.columns else None

            # Plot shaded error bands and colored lines segment-wise
            for i in range(len(df_p) - 1):
                # x may be Series or Index
                xi = x.iloc[i:i+2] if hasattr(x, 'iloc') else x[i:i+2]
                yi = y[i:i+2]
                yerri = yerr[i:i+2] if yerr is not None else None
                r2_val = df_p[color_col].iloc[i]

                color = cmap(norm(r2_val))
                ax.plot(xi, yi, color=color, lw=4)

                if yerri is not None:
                    ax.fill_between(xi, yi - yerri, yi + yerri, color=color, alpha=0.3)

            # Ax labels/titles per subplot
            # if param == 'dG_v0_intercept':
            #     ax.axhline(27, color='k', lw=2, ls='--')

            ax.set_xlabel(x_col)
            ax.set_ylabel(param)
            if path_val is not None:
                ax.set_title(f'{self.sysname} - path {path_val}')
            else:
                ax.set_title(f'{self.sysname} - path 1')
            ax.grid(True)

        # Add a single colorbar for the whole figure
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.7])  # Position for colorbar
        cbar = fig.colorbar(sm, orientation='vertical', cax=cbar_ax)
    
        cbar.set_label('$R^2$ of extrapolation')

        plt.tight_layout()
        plt.savefig(outfname, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        return
        
    def get_trace_features(self,
                        processed_data: pd.DataFrame,
                        x_col: str = 'r_coord',   # or 'r_coord'
                        outdir: str | None = None,
                        rescale_by_speed: bool = False,
                        zscore_by_speed: bool = False) -> pd.DataFrame:
        """
        Build a feature table from trace-like quantities (force, lag, work, ...),
        organized in r-space (x_col) for each trajectory.
        """
        
        if outdir is None:
            outdir = os.path.join(self.outdir, 'path_clustering')
        os.makedirs(outdir, exist_ok=True)

        outfname = os.path.join(outdir, f'{self.sysname}_trace_features_{x_col}.csv')

        df = processed_data.copy()
            
        # sort so each trajectory is an ordered trace in r-space
        sort_cols = ['speed', 'trajname', x_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # rescale trace features by speed
        skip_normalization = ['speed', 'trajname', x_col, 
                              'path', 'replica', 'step', 'replica', 'm_eff',
                              'time', 'r_before', 'r_after', 'NC', 'r_target']
        if rescale_by_speed:
            for col in df.columns:
                if col not in skip_normalization:
                    df[col] = df[col] / df['speed']

        # z-score features within each speed 
        if zscore_by_speed:
            def _zscore_speed(group):
                for col in df.columns:
                    if col not in skip_normalization:
                        mean = group[col].mean()
                        std = group[col].std(ddof=0)
                        if std == 0 or np.isnan(std):
                            group[col] = 0.0
                        else:
                            group[col] = (group[col] - mean) / std
                return group

            df = df.groupby('speed', group_keys=False).apply(_zscore_speed)

        df.to_csv(outfname, index=False)

        return df

    def get_geom_features(self, 
                          recompute: bool = False,
                          trajectories: list[str] = None,
                          topology: str = None,
                          group_A: str = None,
                          group_B: str = None,
                          stride: int = 1,
                          outdir: str = None
                          ) -> pd.DataFrame:
        """
        Compute (or load) geometric distance features between pocket and ligand.

        Returns a DataFrame with columns:
            ['trajname', 'step', 'time'] + dist_* feature columns

        'step' is the frame index; 'time' is taken from the trajectory if available,
        otherwise time = step.
        """
        
        distance_file = f"{outdir}/{self.sysname}_raw_distances.csv"

        if (not recompute) and os.path.exists(distance_file):
            df = pd.read_csv(distance_file)
            return df

        if trajectories is None:
            trajectories = self.trajectories
        if topology is None:
            topology = self.reference_pdb
        if group_A is None:
            group_A = self.pocket_select
        if group_B is None:
            group_B = self.ligand_select

        all_rows = []

        for traj in tqdm.tqdm(trajectories, desc="Calculating distances.."):
            u = mda.Universe(topology, traj)
            
            pocket_atoms = u.select_atoms(group_A)
            ligand_atoms = u.select_atoms(group_B)
            if ligand_atoms.n_atoms == 0 or pocket_atoms.n_atoms == 0:
                print(f"Warning: No atoms found for selection in trajectory {traj}. Skipping.")
                continue
            
            traj_name = self.traj_to_log_name(traj)
        
            for ts in u.trajectory[::stride]:
                distances = distance_array(
                    pocket_atoms, ligand_atoms,
                    result=np.ndarray((len(pocket_atoms), len(ligand_atoms)))
                )
                dist_flat = distances.flatten() / 10.0  # nm
                time_ps = getattr(ts, "time", ts.frame)  # ts.time in ps

                row = {
                    "trajname": traj_name,
                    "step": ts.frame,   # index within this trajectory
                    "time": time_ps,
                }
                for i, v in enumerate(dist_flat):
                    row[f"dist_{i}"] = v
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.to_csv(distance_file, index=False)
        return df
    
    @staticmethod
    def traj_to_log_name(traj_path: str) -> str:
        """
        Map a trajectory file path to the corresponding trajname used in raw_data.
        """
        base = os.path.basename(traj_path)
        root, ext = os.path.splitext(base)
        log_name = root.replace('traj', 'log')
        return log_name
    
    @staticmethod
    def build_merged_features(raw_data: pd.DataFrame,
                            geom_df: pd.DataFrame,
                            tolerance_ps: float | None = None) -> pd.DataFrame:
        """
        Merge geometry features (coarse) with raw_data (fine-grained) using time,
        done *per trajectory* to avoid merge_asof sorting headaches.

        Returns one row per geometry frame, augmented with nearest raw_data row.
        """

        for name, df in (("geom_df", geom_df), ("raw_data", raw_data)):
            if 'trajname' not in df.columns:
                raise ValueError(f"{name} is missing 'trajname' column")
            if 'time' not in df.columns:
                raise ValueError(f"{name} is missing 'time' column")

        g = geom_df.copy()
        r = raw_data.copy()

        g['time'] = pd.to_numeric(g['time'], errors='coerce')
        r['time'] = pd.to_numeric(r['time'], errors='coerce')

        g = g.dropna(subset=['time'])
        r = r.dropna(subset=['time'])

        merged_chunks = []

        # only trajectories present in both
        common_traj = sorted(set(g['trajname'].unique()) & set(r['trajname'].unique()))

        for traj in common_traj:
            g_traj = g[g['trajname'] == traj].sort_values('time').reset_index(drop=True)
            r_traj = r[r['trajname'] == traj].sort_values('time').reset_index(drop=True)

            if len(g_traj) == 0 or len(r_traj) == 0:
                continue

            kwargs = dict(
                left=g_traj,
                right=r_traj,
                on='time',
                direction='nearest',
                allow_exact_matches=True,
            )
            if tolerance_ps is not None:
                kwargs['tolerance'] = tolerance_ps

            merged_traj = pd.merge_asof(**kwargs)
            merged_traj['trajname'] = traj  # ensure trajname is set
            merged_chunks.append(merged_traj)

        if not merged_chunks:
            raise RuntimeError("No trajectories could be merged. Check trajname/time consistency.")

        merged = pd.concat(merged_chunks, ignore_index=True)
        return merged

    def cluster_time_series(self,
                            feature_df: pd.DataFrame,
                            feature_cols: list,
                            r_coord: str = 'r_target',
                            r_range: tuple = None,
                            use_silhouette: bool = True,
                            max_k: int = 5,
                            seed: int = 42,
                            outdir: str = '.',
                            method: str = 'full'):

        """
        Cluster trajectories using Dynamic Time Warping (DTW) and k-medoids.
        For more information on DTW see: https://doi.org/10.1073/pnas.231354212
                                         https://dtaidistance.readthedocs.io/en/latest/index.html
        """
        df = feature_df.copy()

        # r-range filtering
        if r_range is not None:
            low, high = map(float, r_range)
            print(f'WARNING: Filtering trajectories to r_target in [{low}, {high}] for clustering.')
            df = df[(df[r_coord] >= low) & (df[r_coord] <= high)]

        # Build per-trajectory arrays
        data_struct = {}
        for trajname, traj_df in df.groupby("trajname"):
            traj_df = traj_df.sort_values('time')
            if traj_df.shape[0] < 2:
                print(f"Skipping trajectory {trajname} due to insufficient data points.")
                continue
            data_struct[trajname] = traj_df[feature_cols].to_numpy()

        if len(data_struct) < 2:
            raise RuntimeError("Not enough trajectories with data to cluster.")

        names = list(data_struct.keys())
        vectors_stacked = [data_struct[name] for name in names]

        # Scale features across all frames / trajectories
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(np.vstack(vectors_stacked))
        vectors_stacked_scaled = [scaler.transform(arr) for arr in vectors_stacked]

        # DTW distance matrix
        distmatrix = dtw_ndim.distance_matrix_fast(s=vectors_stacked_scaled)

        plt.figure(figsize=(6, 5))
        sns.heatmap(distmatrix, cmap="viridis")
        plt.xlabel("Trajectory index"); plt.ylabel("Trajectory index")
        plt.title(f"DTW Distance Matrix ({method})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"distmatrix_{method}.png"))
        plt.close()

        # Choose K
        K_MAX = min(max_k, len(names))
        if K_MAX < 2:
            raise RuntimeError("Not enough trajectories to form at least 2 clusters.")

        scores = {}
        for k in range(2, K_MAX + 1):
            c = kmedoids.fasterpam(distmatrix, k, random_state=seed)
            if use_silhouette:
                scores[k] = silhouette_score(distmatrix, c.labels, metric="precomputed")
            else:
                scores[k] = -c.loss  # higher is better if we flip the sign

        K = max(scores, key=scores.get)
        print(f"Found {K} paths with score {scores[K]:.2f}")

        # elbow plot, comment out if not do_plots
        plt.figure(figsize=(6, 5))
        sns.lineplot(x=list(scores.keys()), y=list(scores.values()))
        plt.title(f"Optimal number of paths: {K}")
        plt.axvline(x=K, color='red', linestyle='--', label=f'Optimal K={K}')
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score" if use_silhouette else "Score")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"elbowplot_{method}.png"))
        plt.close()

        # Final clustering
        cluster = kmedoids.fasterpam(distmatrix, K, random_state=seed)

        labels_dict = {name: label for name, label in zip(names, cluster.labels)}
        medoid_indices = cluster.medoids
        medoid_names = [names[idx] for idx in medoid_indices]

        print("Medoid trajectories:", medoid_names)
        print("cluster counts:")
        unique, counts = np.unique(cluster.labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f" Cluster {u}: {c} trajectories")

        # Map cluster labels back to full df (including any rows filtered out earlier)
        trajname_map = pd.DataFrame(
            {"trajname": names, "cluster": cluster.labels}
        ).set_index('trajname')['cluster'].to_dict()

        return feature_df, labels_dict, trajname_map, medoid_names, vectors_stacked_scaled

    def find_paths_smd(self, data: pd.DataFrame,
                            do_plots: bool = True, 
                            recompute_geom: bool = False) -> pd.DataFrame:

        outdir = os.path.join(self.outdir, 'path_clustering')
        os.makedirs(outdir, exist_ok=True)
        
        traces_feat = geom_feat = None
        
        if self.cluster_paths not in ['geometric', 'traces', 'full']:
            raise ValueError("cluster_paths must be one of: False, 'geometric', 'traces', 'full'")
        
        if self.cluster_paths == 'geometric':
            geom_feat = self.get_geom_features(recompute=recompute_geom, outdir=outdir)
            # I do this to have r_target in feature_df for r_range filtering
            feature_df = self.build_merged_features(
            raw_data=self.raw_data[['trajname','time','r_target']],
            geom_df=geom_feat,
            tolerance_ps=None
        )
            feature_cols = [c for c in feature_df.columns if c.startswith('dist_')]
        elif self.cluster_paths == 'traces':
            data['lag'] = data['r_target'] - data['r_after']
            feature_df = self.get_trace_features(data,
                                                x_col='r_target',
                                                rescale_by_speed=True,
                                                zscore_by_speed=True,
                                                )
            feature_cols = ['lag', 'force', 'r_before']# or ['work','lag']
        else:  # 'full'
            geom_feat = self.get_geom_features(recompute=recompute_geom, outdir=outdir)
            traces_feat = self.get_trace_features(data,
                                                x_col='r_target',
                                                rescale_by_speed=True,
                                                zscore_by_speed=True,
                                                )
            traces_feat['lag'] = traces_feat['r_target'] - traces_feat['r_after']
            feature_df = self.build_merged_features(traces_feat, geom_feat)
            geom_cols = [c for c in feature_df.columns if c.startswith('dist_')]
            trace_cols = ['force', 'lag']
            feature_cols = geom_cols + trace_cols
            
        feature_df, labels_dict, trajname_map, medoid_names, vectors_stacked_scaled = self.cluster_time_series(
                    feature_df, feature_cols, r_range=self.cluster_range, outdir=outdir, method=self.cluster_paths, seed=self.seed)
        # finally map back to raw_data
        feature_df['path'] = feature_df['trajname'].map(trajname_map)
        data['path'] = data['trajname'].map(trajname_map)
        
        print(data.groupby(['path', 'speed'])[['trajname']].nunique())
        if do_plots: 
            # generate pymol sesh for the paths
            paths = {}
            for trajname in medoid_names:
                for traj in self.trajectories:
                    if trajname == os.path.basename(traj)[:-4]:
                        #{'path_0': [(protein_pdb, traj1), (protein_pdb, traj2)], ...}
                        trajcode = trajname.split('_')[1]
                        paths[f'path_{labels_dict[trajname]}_{trajcode}'] = [(self.reference_pdb, traj)]

            self.make_unbinding_paths_pml(paths, outdir=outdir)
             # generate PCA plot of the clustered paths
            # Get the minimum number of frames across all trajectories
            min_len = min(arr.shape[0] for arr in vectors_stacked_scaled)

            # Trim all arrays to this length, required for PCA
            vectors_trimmed = [arr[:min_len, :] for arr in vectors_stacked_scaled]
            X = np.vstack(vectors_trimmed)   # shape (N_traj * min_len, d)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # --- figure out trajectory order matching vectors_stacked_scaled ---
            # groupby preserves the order of appearance of trajname in feature_df,
            # which is what cluster_time_series used when building vectors_stacked_scaled
            traj_order = [name for name, _ in feature_df.groupby('trajname')]
            n_traj = len(traj_order)
            assert n_traj == len(vectors_trimmed), "traj_order and vectors_stacked_scaled misaligned"

            # base scatter: all points in light gray
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                            alpha=0.75, color='lightgray', s=50, linewidth=0)

            # color palette by path label
            n_paths = len(set(labels_dict.values()))
            palette = sns.color_palette("tab10", n_colors=n_paths)

            # overlay medoid trajectories, colored by path
            used_labels = set()
            for i, trajname in enumerate(traj_order):
                if trajname not in medoid_names:
                    continue

                start = i * min_len
                end = start + min_len
                path_label = labels_dict[trajname]
                color = palette[path_label]

                label = f'path-{path_label}_{trajname}'
                # avoid duplicate legend entries
                if path_label in used_labels:
                    label = None
                else:
                    used_labels.add(path_label)

                plt.scatter(X_pca[start:end, 0],
                            X_pca[start:end, 1],
                            s=20, alpha=0.9,
                            label=label)

            if used_labels:
                plt.legend(frameon=False)

            plt.xlabel("PC1");             plt.ylabel("PC2")
            plt.title(f"Medoids in PCA space")
            plt.tight_layout()
            plt.savefig(f"{outdir}/clustering_PCA.png")
            plt.close()
           
            # # generate PCA plot of the clustered paths
            # # Get the minimum number of frames across all trajectories
            # min_len = min(arr.shape[0] for arr in vectors_stacked_scaled)

            # # Trim all arrays to this length
            # vectors_trimmed = [arr[:min_len, :] for arr in vectors_stacked_scaled]

            # # Stack for PCA
            # X = np.vstack(vectors_trimmed)   # shape (N_traj * min_len, d)

            # from sklearn.decomposition import PCA
            # pca = PCA(n_components=2)
            # X_pca = pca.fit_transform(X)

            # plt.figure(figsize=(6, 5))
            # # sns.scatterplot(data=feature_df, x=feature_df.iloc[:, 2], y=feature_df.iloc[:, 3], alpha=0.2, c='gray', s=2, linewidth=0)
            # sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.3, c='gray', s=10, linewidth=0)
            # # for medoid in cluster.medoids:
            # #     medoid_name = list(paths.keys())[medoid]
            # #     distance_df_medoid = distance_df[distance_df["trajname"] == medoid_name]
            # #     sns.scatterplot(data=distance_df_medoid, x=distance_df_medoid.iloc[:, 2], y=distance_df_medoid.iloc[:, 3], 
            # #                     label=medoid_name, alpha=1, s=25, linewidth=0#, edgecolor='black', st
            # #                                                                     )
            # # plt.title(f"Medoids in PCA space for {K} paths")
            # plt.xlabel("PC1");  plt.ylabel("PC2")
            # plt.tight_layout()
            # plt.savefig(f"{outdir}/clustering_PCA.png")
            # # plt.show()
            # plt.close()
            
        return data
   
    def make_unbinding_paths_pml(self, 
        paths: Dict[str, List[Tuple[str, str]]],
        outdir: str = "unbinding_paths_vis",
        align_sel: str = "protein and backbone",
        grid_spacing: float = 0.5,
        cartoon_color: str = "palecyan",
        sample_stride: int = 1,
    ) -> str:
        """Generate ligand-path density maps and a PyMOL .pml that uses only relative paths."""
                
        level = 0.000002
        surface_transparency = 0.35
        cartoon_transparency = 0.25
        
        os.makedirs(outdir, exist_ok=True)
        outdir = Path(outdir)
        protein_pdb = self.reference_pdb
        ligand_sel = self.ligand_select
        # Copy the reference PDB into OUTDIR so the .pml can run anywhere
        prot_copy = outdir / os.path.basename(protein_pdb)
        shutil.copy2(protein_pdb, prot_copy)

        protein_abs = str(prot_copy.resolve())
        u_ref = mda.Universe(protein_abs)

        default_palette = ["violetpurple", "marine", "forest", "deepsalmon", "gold", "tv_red", "tv_blue"]
        path_colors = {name: default_palette[i % len(default_palette)] for i, name in enumerate(paths)}

        dx_files_rel = {}
        for path_name, traj_list in paths.items():
            dens_sum = None
            total_frames = 0

            for top, traj in traj_list:
                # if not aligned, align to reference
                u = mda.Universe(top, traj)
                align.AlignTraj(u, u_ref, select=align_sel, in_memory=True).run()

                lig = u.select_atoms(ligand_sel)
                if lig.n_atoms == 0:
                    raise ValueError(f"No atoms found for '{ligand_sel}' in {traj}.")

                da = density.DensityAnalysis(lig, delta=grid_spacing,padding=50.0)
                da.run(step=sample_stride)
                rho = da.results.density
                
                dens_sum = rho if dens_sum is None else dens_sum._replace(grid=dens_sum.grid + rho.grid) or dens_sum
                total_frames += len(u.trajectory[::sample_stride])

            # Normalize and write DX
            if total_frames > 0:
                dens_sum.grid /= float(total_frames)

              #smooth the density a bit
            dens_sum.grid = gaussian_filter(dens_sum.grid, sigma=1.0)
                
            dx_path = str((outdir / f"{path_name}_density.dx").resolve())
            dens_sum.export(dx_path)
            dx_files_rel[path_name] = dx_path

        # dx_05 = np.quantile(dens_sum.grid, 0.05)
        # print(f"0.05 quantile of last path density: {dx_05}")
        
        # Write the .pml using ONLY filenames (relative to outdir)
        pml_path = os.path.join(outdir, "unbinding_paths.pml")
        with open(pml_path, "w") as pml:
            pml.write("# Relative-path PyMOL visualization for ligand unbinding paths\n")
            pml.write("reinitialize\n")
            pml.write("bg_color white\n")
            pml.write("set ray_opaque_background, 0\n")
            pml.write("set antialias, 2\n")
            pml.write("set specular, 0.2\n")
            pml.write("set ray_shadow, off\n")
            pml.write(f"set cartoon_transparency, {cartoon_transparency:.2f}\n")
            pml.write(f"load {protein_abs}, prot\n")
            pml.write("hide everything, prot\n")
            pml.write("show cartoon, prot\n")
            pml.write(f"color {cartoon_color}, prot\n")

            for path_name, dx_filename in dx_files_rel.items():
                map_obj = f"map_{path_name}"
                surf_obj = f"surf_{path_name}"
                col = path_colors[path_name]
                pml.write(f"load {dx_filename}, {map_obj}\n")
                pml.write(f'map_double {map_obj}\n')
                pml.write(f"isosurface {surf_obj}, {map_obj}, {level}\n")
                pml.write(f"color {col}, {surf_obj}\n")
                pml.write(f"set transparency, {surface_transparency:.2f}, {surf_obj}\n")
                pml.write(f"set two_sided_lighting, on, {surf_obj}\n")

            pml.write(f"select lig_ref, ({ligand_sel}) and prot\n")
            pml.write("if sele count lig_ref > 0:\n")
            pml.write("    create lig, lig_ref\n")
            # pml.write("    show stick, lig\n")
            pml.write("    show sphere, lig\n")
            pml.write("    color yellow, lig\n")
            pml.write("    set sphere_transparency, 0.9, lig\n")
            pml.write("orient lig\n")
            pml.write("zoom prot, 10.0\n")
            pml.write("png preview.png, ray=1, dpi=300\n")

        return str(pml_path)
    
    @staticmethod
    def plot_work_profiles(
        results: pd.DataFrame,
        r_coord: str = 'r_coord',
        cols_to_plot: list = ['Wmean', 'dG', 'Wdiss'],
        title_suffix: str = 'dcTMD',
        outdir: str = 'work_profiles'
    ):
        if outdir is None:
            outdir = self.outdir
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"work_profiles_{title_suffix}.png")
        
        speeds = sorted(results['speed'].unique())
        fig, ax = plt.subplots(
            figsize=(12, 4),
            ncols=len(speeds),
            nrows=1,
            sharey=True,
            sharex=True
        )
        axes = ax.flatten() if len(speeds) > 1 else [ax]

        legend_handles, legend_labels = None, None

        for i, speed in enumerate(speeds):
            speed_df = results.loc[results['speed'] == speed, [r_coord, 'path', 'speed'] + cols_to_plot]

            # Long format: metric is {Wmean, dG, Wdiss}, value = corresponding y
            long_df = speed_df.melt(
                id_vars=[r_coord, 'path', 'speed'],
                value_vars=cols_to_plot,
                var_name=title_suffix,
                value_name='value'
            )

            sns.lineplot(
                data=long_df,
                x=r_coord, y='value',
                hue=title_suffix, style='path',
                estimator=None, errorbar=None,  # don't aggregate across paths
                ax=axes[i]
            )
            axes[i].grid(True)
            
            # # just reference for trypsin
            # axes[i].axhline(27, color='k', lw=1, ls='--')
            
            axes[i].set_title(f'Speed: {speed} nm/ps', fontsize=10)
            axes[i].set_xlabel(f'{r_coord} (nm)')
            if i == 0:
                axes[i].set_ylabel('dG (kJ/mol)')
            else:
                axes[i].set_ylabel('')

            # Capture legend once, then remove per-axes legends
            if legend_handles is None:
                legend_handles, legend_labels = axes[i].get_legend_handles_labels()
            axes[i].legend_.remove()

        # Figure-level legend combining hue (metrics) and style (paths)
        if legend_handles:
            fig.legend(
                legend_handles, legend_labels,
                title='',
                bbox_to_anchor=(1.01, 0.8), loc='upper left',
                borderaxespad=0.0
            )

        # plt.title(f'Work Profiles {title_suffix}', fontsize=16)
        plt.tight_layout()
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        return
    
    def add_acf_column(self,
                    df: pd.DataFrame,
                    param: str = 'lag',
                    x_col: str = 'time',
                    max_lag: int | None = None,
                    plot: bool = True) -> pd.DataFrame:
        """
        Compute the autocorrelation function (ACF) of `param` along `x_col`,
        treating each (speed, path) group separately. Adds 'acf_<param>' and
        plots one figure with subplots per speed, colored by path.
        """
        columns_needed = [param, x_col, 'speed', 'path']
        for col in columns_needed:
            if col not in df.columns:
                raise ValueError(f"DataFrame is missing required column '{col}'")

        df = df.copy()
        # ensure global ordering is consistent
        df = df.sort_values(['speed', 'path', x_col])

        acf_colname = f'acf_{param}'
        df[acf_colname] = np.nan

        group_iter = df.groupby(['speed', 'path'])

        # compute ACF per group
        for key, df_g in group_iter:
            df_g = df_g.sort_values(x_col)

            y = df_g[param].to_numpy(dtype=float)
            n = len(y)
            if n < 2:
                continue

            y_centered = y - y.mean()
            var = np.dot(y_centered, y_centered)
            if var == 0.0:
                acf_vals = np.zeros(n)
                acf_vals[0] = 1.0
            else:
                this_max_lag = max_lag
                if this_max_lag is None or this_max_lag >= n:
                    this_max_lag = n - 1

                acf_short = np.empty(this_max_lag + 1, dtype=float)
                for lag in range(this_max_lag + 1):
                    if lag == 0:
                        acf_short[lag] = 1.0
                    else:
                        acf_short[lag] = np.dot(y_centered[:-lag], y_centered[lag:]) / var

                if this_max_lag + 1 < n:
                    acf_vals = np.concatenate(
                        [acf_short, np.full(n - (this_max_lag + 1), np.nan)]
                    )
                else:
                    acf_vals = acf_short

            df.loc[df_g.index, acf_colname] = acf_vals

        if plot:
            speeds = sorted(df['speed'].dropna().unique())
            n_speeds = len(speeds)
            fig, axes = plt.subplots(1, n_speeds,
                                    figsize=(6 * n_speeds, 4.5),
                                    sharex=False)
            if n_speeds == 1:
                axes = [axes]

            for ax, spd in zip(axes, speeds):
                df_s = df[df['speed'] == spd]
                if df_s.empty:
                    ax.set_visible(False)
                    continue

                paths = sorted(df_s['path'].dropna().unique())

                for p in paths:
                    # sort by x_col so ACF sequence matches lag order
                    df_p = df_s[df_s['path'] == p].sort_values(x_col)
                    if df_p.empty:
                        continue

                    acf_vals = df_p[acf_colname].to_numpy()
                    valid = ~np.isnan(acf_vals)
                    if not np.any(valid):
                        continue
                    acf_vals = acf_vals[valid]

                    lags = np.arange(len(acf_vals), dtype=float)

                    x_vals = df_p[x_col].to_numpy()
                    if len(x_vals) > 1:
                        dx = np.median(np.diff(x_vals))
                        if np.isfinite(dx) and dx > 0:
                            lags = lags * dx
                            x_label = f'Lag in {x_col} units'
                        else:
                            x_label = 'Lag (frames)'
                    else:
                        x_label = 'Lag (frames)'

                    label = f'path {p}'
                    ax.plot(lags, acf_vals, lw=2, label=label)

                ax.axhline(0.0, color='k', lw=1)
                title = f'ACF of {param} (speed = {spd})'
                ax.set_title(title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(f'ACF({param})')
                ax.grid(True)
                if len(paths) > 1:
                    ax.legend(frameon=False)

            plt.tight_layout()
            outname = os.path.join(
                self.outdir,
                f'{self.sysname}_acf_{param}_vs_{x_col}_by_speed_path.png'
            )
            plt.savefig(outname, dpi=300)
            plt.show()
            plt.close()

        return df

    def assess_sequential_replica_convergence(
        self,
        speed: float,
        quantities: list[str] | str = "Wdiss",
        min_replicas: int = 4,
        tol_rmsd: float = 2.0,     # kJ/mol
        tol_barrier: float = 2.0,  # kJ/mol
        save_pmfs: bool = True,
        ):
        """
        Sequential convergence check for a single pulling speed.

        Compares PMF(N) vs PMF(N-1). Paths are ignored.
        Multiple quantities can be monitored; the first one
        is used to decide convergence.

        Parameters
        ----------
        quantities : list[str] or str
            Quantities to monitor (e.g. ['Wdiss', 'dG', 'dG_gmm']).
            The first entry is used for convergence criteria.
        save_pmfs : bool
            If True, save PMFs to CSV for later plotting.
        """

        if isinstance(quantities, str):
            quantities = [quantities]

        main_quantity = quantities[0]

        #determine whether GMM is needed. This saves time
        fit_GMM = any("gmm" in q for q in quantities)

        # filter logs by speed
        all_logs = self.log_files.copy()
        speed_logs = [fn for fn in all_logs if self.speed_from_log(fn) == speed]

        if len(speed_logs) < min_replicas:
            raise ValueError(
                f"Not enough replicas for speed={speed}: {len(speed_logs)}"
            )

        # sort replicas by timestamp, so its sequential
        speed_logs = sorted(speed_logs, key=self.replica_idx_from_log)

        rows = []
        pmf_records = []  # for optional CSV output
        prev_pmfs = None

        for k in range(min_replicas, len(speed_logs) + 1):

            self.log_files = speed_logs[:k]
            results_k, _ = self.run_analysis(fit_GMM=fit_GMM)

            if results_k.empty:
                continue

            # build PMFs for all quantities
            pmfs_k = {}
            for q in quantities:
                pmfs_k[q] = (
                    results_k
                    .groupby("r_coord")[q]
                    .mean()
                    .sort_index()
                )

            if prev_pmfs is None:
                prev_pmfs = pmfs_k
                continue

            # trim + align using the main quantity. Only compare common r-coords in rmsd
            pmf_k = pmfs_k[main_quantity]
            pmf_km1 = prev_pmfs[main_quantity]

            r_min = max(pmf_k.index.min(), pmf_km1.index.min())
            r_max = min(pmf_k.index.max(), pmf_km1.index.max())

            pmf_k = pmf_k[(pmf_k.index >= r_min) & (pmf_k.index <= r_max)]
            pmf_km1 = pmf_km1[(pmf_km1.index >= r_min) & (pmf_km1.index <= r_max)]

            common_r = pmf_k.index.intersection(pmf_km1.index)

            if len(common_r) < 5:
                prev_pmfs = pmfs_k
                continue

            yN = pmf_k.loc[common_r].values
            yNm1 = pmf_km1.loc[common_r].values

            pmf_rmsd = np.sqrt(np.mean((yN - yNm1) ** 2))
            delta_barrier = abs(yN.max() - yNm1.max())

            converged = (
                pmf_rmsd < tol_rmsd and
                delta_barrier < tol_barrier
            )

            row = {
                "speed": speed,
                "n_replicas": k,
                f"{main_quantity}-rmsd": pmf_rmsd,
                f"{main_quantity}-deltaMax": delta_barrier,
                "converged": converged,
                "decision_quantity": main_quantity,
            }

            # compute auxiliary metrics for other quantities
            for q in quantities:
                if q == main_quantity:
                    continue

                pmf_q = pmfs_k[q]
                pmf_qm1 = prev_pmfs[q]

                pmf_q = pmf_q.loc[common_r]
                pmf_qm1 = pmf_qm1.loc[common_r]

                row[f"{q}-rmsd"] = np.sqrt(
                    np.mean((pmf_q.values - pmf_qm1.values) ** 2)
                )
                row[f"{q}-deltaMax"] = abs(
                    pmf_q.values.max() - pmf_qm1.values.max()
                )

            rows.append(row)

            #store PMFs for optional CSV output
            if save_pmfs:
                for q in quantities:
                    for r, val in pmfs_k[q].items():
                        pmf_records.append({
                            "speed": speed,
                            "n_replicas": k,
                            "quantity": q,
                            "r_coord": r,
                            "value": val,
                        })

            prev_pmfs = pmfs_k

        #restore full log list
        self.log_files = all_logs

        conv_df = pd.DataFrame(rows)

        # write PMF CSV
        if save_pmfs:
            pmf_df = pd.DataFrame(pmf_records)
            pmf_df.to_csv(f"{self.outdir}/pmfs_speed_{speed}.csv", index=False)

        return conv_df

    
    @staticmethod
    def speed_from_log(fn):
        # Example log name: sMD_replica-182557_v0.005_forward.dat
        base = os.path.basename(fn)
        for token in base.split("_"):
            if token.startswith("v"):
                return float(token[1:])
        return None
    
    @staticmethod
    def replica_idx_from_log(fn):
        base = os.path.basename(fn)[:-4]
        rep = base.split("_")[-3]
        return int(rep.split("-")[1])