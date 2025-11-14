import os
import glob
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
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

class SteeredMDAnalysis:
    """Class to analyze Steered Molecular Dynamics (sMD) data, mostly within the dcTMD framework.
    For more information on the dcTMD see...

    
    """
    def __init__(self, 
                 log_files: list[str] = None,
                 sysname: str = None,
                 n_bins: float = 100,
                 temperature: float = 300, #K
                 timestep: float = 0.004, #ps
                 dist_minmax: tuple = None, #nm
                 cluster_paths: bool = True,
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
            sysname = log_files[0].split('/')[0]  # Extract system name from the first log file path

        if outdir is not None:
            self.outdir = outdir
        else:
            self.outdir = os.path.dirname(log_files[0])  # Output directory is the same as the first log file

        self.temp = temperature
        self.R = 0.008314462618  # kJ/(mol*K)
        self.RT = self.R * self.temp
        self.beta = 1.0 / self.RT
                
        self.timestep = timestep

        self.cluster_paths = cluster_paths
        self.cluster_range = cluster_range
        self.trajectories = trajectories
        if cluster_paths and (trajectories is None or len(trajectories) == 0):
            raise ValueError("Clustering paths is enabled, but no trajectories provided.")
        
        # this will be used for topology and pocket selection, so should be pdb before pulling
        self.reference_pdb = reference_pdb
        self.ligand_select = ligand_select
        self.pocket_select = pocket_select

        self.work_column = 'work'
        self.force_column = 'force'

        self.n_bins = n_bins
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
            self.dist_column = 'r_before'

        # assemble the master dataframe loading log files
        raw_data = self.load_logs(self.log_files)

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

        # bin the data
        if use_target_grid:
            self.raw_data = raw_data # no binning when using target grid
        else:
            self.raw_data, centers = self.bin_data(raw_data, 
                                                   self.n_bins, 
                                                   use_quantiles=True, 
                                                   min_points=1)
        
        # filter out trajectories that did not reach close contact
        to_drop = []
        for traj_name, traj_data in self.raw_data.groupby('trajname'):
            if self.pulling_direction == 'forward':
                if traj_data["r_after"].min() < 0.1:
                    print(f'WARNING: Dropping {traj_name}, min distance {traj_data["r_before"].min():.2f} nm')
                    to_drop.append(traj_name)
            else:  # backward pulling
                if traj_data["r_after"].min() > 0.1:
                    print(f'WARNING: Dropping {traj_name}, min distance {traj_data["r_before"].min():.2f} nm')
                    to_drop.append(traj_name)
        
        self.raw_data = self.raw_data[~self.raw_data['trajname'].isin(to_drop)]
        
        if self.cluster_paths == None:
            self.raw_data['path'] = 1 # default to single cluster if no clustering method is specified
        # cluster trajectories into paths if specified
        elif self.cluster_paths == 'geometric':
            self.raw_data, labels_dict, medoid_names = self.cluster_trajectories()
        elif self.cluster_paths == 'traces':
            self.raw_data, labels_dict, medoid_names = self.cluster_raw_traces(self.raw_data, r_range=self.cluster_range, outdir=self.outdir)
        else:
            print(f'ERROR: Unknown clustering method {self.cluster_paths}. No clustering will be performed.')
   
        if self.raw_data['path'].nunique() > 1:
            print(self.raw_data.groupby(['path', 'speed'])[['trajname']].nunique())
            # generate pymol sesh for the paths
            paths = {}
            for trajname in medoid_names:
                for traj in self.trajectories:
                    if trajname == os.path.basename(traj)[:-4]:
                        #{'path_0': [(protein_pdb, traj1), (protein_pdb, traj2)], ...}
                        paths[f'path_{labels_dict[trajname]}'] = [(self.reference_pdb, traj)]
            outdir = os.path.join(self.outdir,'path_clustering')
            self.make_unbinding_paths_pml(paths, outdir=outdir)

        # decorrelate work values using statistical inefficiency g or by replica averaging
        # If we dont decorrelate, we should use the per-replica aggregated work. i.e. each replica contributes one work value per bin ENSEMBLE AVERAGE OVER REPLICAS
        if use_target_grid:
            # ensure 'step' exists (per-trajectory running index)
            if "step" not in self.raw_data.columns:
                self.raw_data = (self.raw_data.sort_values(["trajname"])
                                .assign(step=lambda d: d.groupby("trajname").cumcount()))
            self.processed_data = self.build_common_target_grid(self.raw_data)
            x_col = "r_target_grid" # x-axis coordinate
            group_keys = ["step", "speed", "path"]  # integer key avoids fragmentation
        else:
            self.processed_data = self.decorrelate_work_data(use_g=True)
            x_col = "r_bin"
            group_keys = ["r_bin", "speed", "path"]  # no 'step' on the binning path
                
        # self.processed_data['work'] = self.processed_data['work'] * self.beta
        # self.processed_data['work'] = self.processed_data['work'] / 2.476  # convert to KT

        results = []
        gmm_results = defaultdict(dict)  # to store GMM results for plotting
        
        to_drop = []
        for path in self.processed_data.groupby(['path'])[['trajname']].nunique().iterrows():
            if path[1]['trajname'] < 3:
                to_drop.append(path[0])
        if to_drop:
            print(f"WARNING: Dropping paths: {to_drop} due to insufficient number of trajectories ({path[1]['trajname']}).")
        
        processed_data = self.processed_data[~self.processed_data['path'].isin(to_drop)]

        for (coord, speed, path), group in processed_data.groupby(group_keys):
            
            r_coord = float(group[x_col].iloc[0])
            
            #coord is the index
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
                gmm_results[r_coord][speed] = {'GMM_neq_weights': w,
                                             'replica_W': raw_W,
                                             'GMM_means': mu,
                                             'GMM_variances': sig2,
                                             'Wmean_mix': Wmean_neq
                                            }
            # build this partial dataframe
            results.append({
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

    
    def load_logs(self, log_files: list[str]) -> pd.DataFrame:

        # compile raw log files
        count = 0
        raw_data = []
        for fn in log_files:
            try:
                base = os.path.basename(fn)[:-4]  # remove .dat extension
                # print(f"Loading {base}...")
                speed = float(base.split('_')[-2].strip('v'))
                df = pd.read_csv(fn, comment='#')
                df['trajname'] = base
                df['speed'] = speed  # add speed column
                df['replica'] = base.split('_')[-3]  # extract replica number from filename
                raw_data.append(df)
                count += 1
            except Exception as e:
                print(f"Error loading {fn}: {e}")
                continue
        if count == 0:
            print("No valid log files found.")
            return None
        print(f"Loaded {count} log files for system '{self.sysname}'.")

        if not raw_data:
            print("No data loaded from log files.")
            return None
        return pd.concat(list(raw_data))
    
    def integrate_force_dx(self, raw_data) -> pd.DataFrame:
        """Integrate the force over distance to compute work done.
        """
        grouped = raw_data.groupby("trajname")
        for traj, group in grouped:
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
            gg["r_target_grid"] = gg["r_target_mean"].values
            new.append(gg.drop(columns=["r_target_mean"]))
        return pd.concat(new, ignore_index=True)

    def bin_data(self,
                raw_data: pd.DataFrame,
                n_bins: int,
                use_quantiles: bool = True,
                min_points: int = 1       # drop bins with < min_points
                ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Bins self.dist_column into n_bins using pd.cut (equal width) or pd.qcut (equal count).
        After optional filtering of low-count bins, bins are reindexed to 0..M-1 and centers
        are returned only for surviving bins. 'r_bin' holds the center for each row.

        Returns
        -------
        raw_data : DataFrame with columns ['bin', 'r_bin'] added
        centers  : np.ndarray of bin centers aligned with bin indices 0..M-1
        """
        rvals = raw_data[self.dist_column].to_numpy()
        raw_data = raw_data.copy()

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
        raw_data['r_bin'] = raw_data['bin'].map(lambda b: centers[b] if 0 <= b < len(centers) else np.nan)
        # carry over

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
            
    def cluster_trajectories(self, do_PCA:bool=True, do_plots:bool=True):
        """Cluster trajectories using Dynamic Time Warping (DTW) and k-medoids.
        For more information on DTW see: https://doi.org/10.1073/pnas.231354212
                                         https://dtaidistance.readthedocs.io/en/latest/index.html
        """
        #FIXME hardcoded traj extension .xtc
        raw_data = self.raw_data.copy()

        outdir = os.path.join(self.outdir,'path_clustering')
        os.makedirs(outdir, exist_ok=True)
        distance_file = f"{outdir}/{self.sysname}_raw_distances.csv"

        if os.path.exists(distance_file):
            df = pd.read_csv(distance_file)
            print(f"Loaded raw distances from {distance_file}")
        else:
            all_distances = {}
            for traj in tqdm.tqdm(self.trajectories, desc="Calculating distances.."):
                u = mda.Universe(self.reference_pdb, traj)
                pocket_atoms = u.select_atoms(self.pocket_select)
                ligand_atoms = u.select_atoms(self.ligand_select)
                data_array = np.zeros((len(u.trajectory),(len(pocket_atoms)*len(ligand_atoms))))
                for ts in u.trajectory:
                    distances = distance_array(pocket_atoms, ligand_atoms, 
                                                result=np.ndarray((len(pocket_atoms), len(ligand_atoms))))
                    data_array[ts.frame] = distances.flatten() / 10.0  # Convert to nm
                all_distances[os.path.basename(traj)] = data_array
                
            df_list = []
            for traj_name, distances in all_distances.items():
                num_frames = distances.shape[0]
                traj_name = traj_name.replace('.xtc', '').replace('traj', 'log') 
                frame_numbers = np.arange(num_frames)
                traj_df = pd.DataFrame(distances, columns=[f"dist_{i}" for i in range(distances.shape[1])])
                traj_df.insert(0, "frame", frame_numbers)
                traj_df.insert(1, "trajname", traj_name)
                df_list.append(traj_df)
            df = pd.concat(df_list, ignore_index=True)
            df.to_csv(distance_file, index=False)

        distances = df.iloc[:,2:].values
        scaler = StandardScaler()
        distances = scaler.fit_transform(distances)  # Scale the distances

        if do_PCA:
            pca = PCA(n_components=2, random_state=self.seed)
            distances = pca.fit_transform(distances)
            print(f'The first 2 PC explain {sum(pca.explained_variance_ratio_)*100:.2f}% of the variance')

        distance_df = pd.DataFrame(distances)
        distance_df.insert(0,"frame", df['frame'])
        distance_df.insert(1,"trajname", df['trajname'])

        # Create a distance matrix using DTW
        paths = defaultdict(np.ndarray)
        for traj_name in distance_df.groupby("trajname").groups:
            traj_df = distance_df[distance_df["trajname"] == traj_name]
            paths[traj_name] = traj_df.iloc[:, 2:].values

        stacked = [paths[traj_name] for traj_name in paths.keys()]

        distmatrix = dtw_ndim.distance_matrix_fast(s=stacked)#, ndim=stacked[0].shape[1])

        if do_plots:
            plt.figure(figsize=(6, 5))
            sns.heatmap(distmatrix, cmap="viridis")
            plt.title("DTW Distance Matrix")
            plt.tight_layout()
            plt.savefig(f"{outdir}/{self.sysname}_distmatrix.png")
            # plt.show()
            plt.close()

        # Find optimal number of paths using Elbow method and Silhouette score
        maxK = min(5, len(paths))
        silloutte_scores = {}
        for i in range(2,maxK):
            c = kmedoids.fasterpam(distmatrix, i) # c.loss
            silloutte_scores[i] = silhouette_score(distmatrix, c.labels, metric="precomputed")

        K = max(silloutte_scores, key=silloutte_scores.get)
        print(f"Found {K} paths with Silhouette score {silloutte_scores[K]:.2f}")
        
        if do_plots:
            plt.figure(figsize=(6, 5))
            sns.lineplot(x=list(silloutte_scores.keys()), y=list(silloutte_scores.values()))
            plt.axvline(x=K, color='red', linestyle='--', label=f'Optimal K={K}')
            plt.xlabel("Number of paths"); plt.ylabel("Silhouette score")
            plt.title(f"Optimal number of paths: {K}")
            plt.tight_layout()
            plt.savefig(f"{outdir}/{self.sysname}_elbowplot.png")
            # plt.show()
            plt.close()

        # K-Medoids clustering using the optimal number of paths
        cluster = kmedoids.fasterpam(distmatrix, K, random_state=self.seed)

        # visualize the medoids in the PCA space
        if do_PCA and do_plots:
            plt.figure(figsize=(6, 5))
            sns.scatterplot(data=distance_df, x=distance_df.iloc[:, 2], y=distance_df.iloc[:, 3], alpha=0.2, c='gray', s=2, linewidth=0)
            for medoid in cluster.medoids:
                medoid_name = list(paths.keys())[medoid]
                distance_df_medoid = distance_df[distance_df["trajname"] == medoid_name]
                sns.scatterplot(data=distance_df_medoid, x=distance_df_medoid.iloc[:, 2], y=distance_df_medoid.iloc[:, 3], 
                                label=medoid_name, alpha=1, s=25, linewidth=0#, edgecolor='black', st
                                                                                )
            plt.title(f"Medoids in PCA space for {K} paths")
            plt.xlabel("PC1");  plt.ylabel("PC2")
            plt.tight_layout()
            plt.savefig(f"{outdir}/{self.sysname}_clustering_K-{K}.png")
            # plt.show()
            plt.close()

        names = [os.path.basename(name).replace('.xtc', '').replace('traj', 'log') for name in paths.keys()]
        medoid_names = [names[medoid] for medoid in cluster.medoids]
        labels_dict = {k: v for k, v in zip(names, cluster.labels)}

        trajname_map = pd.DataFrame({"trajname": list(paths.keys()),
                        "cluster": cluster.labels}).set_index('trajname')['cluster'].to_dict()
        raw_data['path'] = raw_data['trajname'].map(trajname_map)

        return raw_data, labels_dict, medoid_names

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
                    ax.plot(x, pdf_comp, lw=2)
                
                # Plot total mixture PDF
                total_pdf = np.sum([w * norm.pdf(x, loc=m, scale=np.sqrt(v))
                                    for w, m, v in zip(weights, means, variances)], axis=0)
                ax.plot(x, total_pdf, "k--", lw=2)
                
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
    
    def decorrelate_work_data(self, use_g:bool=True) -> pd.DataFrame:    
        decorrelated_data = []
        g = None
        for (r_bin, v, path), group in self.raw_data.groupby(["r_bin", "speed", "path"]):
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
                    indices = timeseries.subsample_correlated_data(W, g, conservative=True)
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
                        'r_bin': r_bin,
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
        return pd.concat(rows, ignore_index=True)   

    def extrapolate_to_v0(self,
                            df: pd.DataFrame = None,
                            param_cols: list[str]=['Wdiss_gmm'],
                            speeds: list[float] = None,
                            mixed_models: bool = False
                            ) -> pd.DataFrame:

        """Extrapolate a given parameter to zero speed using linear regression. 
        This method groups the data by speed and fits a linear regression to the
        param vs speed for each bin."""

        # Filter out speeds if provided. You could only want to fit specific (low) speeds
        if speeds is not None:
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
                                    groups=df["r_bin"],
                                    re_formula="~speed")
                result = model.fit(reml=False)
                for r_bin, rand_eff in result.random_effects.items():
                    intercept = result.fe_params["Intercept"] + rand_eff["Group"]
                    slope = result.fe_params["speed"] + rand_eff["speed"]
                    results.append({
                    "r_bin": r_bin,
                    f"{param_col}_v0_intercept": intercept,
                    f"{param_col}_v0_slope": slope,
                    "R2": 1.0
                })
                    
            else:
                results = []
                for param_col in param_cols:
                    _df = []
                    for (r_bin, path), group in df.groupby(['r_bin','path']):
                        speeds = group['speed'].values
                        means = group[param_col].values

                        lr_results = linregress(speeds, means)
                        _df.append({
                            'r_bin': r_bin,
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
                # drop duplicate 'r_bin' adn speed columns
                # results = results.loc[:, ~results.columns.duplicated()]

            # Calculate the diffusion coefficient D(x) using the friction coefficient F(x)
            # df['D(x)'] = self.R * self.temp / df['F(x)']

        return results
    
    def cluster_raw_traces(self,
                           data:pd.DataFrame, 
                           r_range:tuple=None,
                           columns:list=['work', 'r_before'], 
                           use_silhouette:bool=True,
                           outdir:str='.',
                           seed:int=42):
        
        df = data.copy()
        
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in dataframe.")

        columns = columns + ['lag']
        df['lag'] = df['r_target'] - df['r_after']
        
        data_struct = defaultdict(np.ndarray)
        for trajname in df.groupby("trajname").groups.keys():
            traj_df = df[df["trajname"]==trajname]
            if r_range is not None:
                traj_df = traj_df[(traj_df["r_target"]>=float(r_range[0])) 
                                  & (traj_df["r_target"]<=float(r_range[1]))]

            if traj_df.shape[0] < 2:
                print(f"Skipping trajectory {trajname} due to insufficient data points in range.")
                continue
            # Scale the work and force columns to [0,1] range because they depend on pulling speed
            # if 'force' in columns:
            #     traj_df['force'] = MinMaxScaler().fit_transform(traj_df['force'].values.reshape(-1,1))
            # if 'work' in columns:
            data_struct[trajname] = traj_df[columns].to_numpy()
        vectors_stacked = [data_struct[traj_name] for traj_name in data_struct.keys()]
        names = list(data_struct.keys())

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(np.vstack(vectors_stacked))
        vectors_stacked_scaled = [scaler.transform(arr) for arr in vectors_stacked]

        distmatrix = dtw_ndim.distance_matrix_fast(s=vectors_stacked_scaled)#, ndim=stacked[0].shape[1])
        sns.heatmap(distmatrix, cmap="viridis")
        plt.xlabel("Trajectory index"); plt.ylabel("Trajectory index")
        plt.title("DTW Distance Matrix for Raw Traces")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{self.sysname}_rawtrace_distmatrix.png")
        # plt.show()
        plt.close()

        K_MAX = min(5, df.groupby("trajname").ngroups)
        scores = {}
        for i in range(2, K_MAX):
            c = kmedoids.fasterpam(distmatrix, i, random_state=seed)
            if use_silhouette:
                scores[i] = silhouette_score(distmatrix, c.labels, 
                                             random_state=seed, metric="precomputed")
            else:
                scores[i] = c.loss
                
        K = max(scores, key=scores.get)
        print(f"Found {K} paths with score {scores[K]:.2f}")

        plt.figure(figsize=(6, 5))
        sns.lineplot(x=list(scores.keys()), y=list(scores.values()))
        plt.title(f"Optimal number of paths: {K}")
        plt.axvline(x=K, color='red', linestyle='--', label=f'Optimal K={K}')
        plt.xlabel("Number of clusters"); plt.ylabel("Silhouette score" if use_silhouette else "Loss")
        plt.savefig(f"{outdir}/{self.sysname}_elbowplot.png")
        # plt.show()
        plt.close()

        # K-Medoids clustering using the optimal number of paths
        cluster = kmedoids.fasterpam(distmatrix, K, random_state=seed)

        labels_dict = {k: v for k, v in zip(names, cluster.labels)}
        medoid_indices = cluster.medoids
        medoid_names = [names[idx] for idx in medoid_indices]
        print("Medoid trajectories:", medoid_names)
        print(f"cluster counts:")
        unique, counts = np.unique(cluster.labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Cluster {u}: {c} trajectories")

        trajname_map = pd.DataFrame({"trajname": list(data_struct.keys()),
                        "cluster": cluster.labels}).set_index('trajname')['cluster'].to_dict()
        data['path'] = df['trajname'].map(trajname_map)
        
        #statistics by cluster
        return data, labels_dict, medoid_names

    def make_unbinding_paths_pml(self, 
        paths: Dict[str, List[Tuple[str, str]]],
        outdir: str = "unbinding_paths_vis",
        align_sel: str = "protein and backbone",
        grid_spacing: float = 1.0,
        cartoon_color: str = "palecyan",
        sample_stride: int = 1,
    ) -> str:
        """Generate ligand-path density maps and a PyMOL .pml that uses only relative paths."""
                
        level = 0.000003
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

        default_palette = ["deepsalmon", "marine", "forest", "violetpurple", "gold", "tv_red", "tv_blue"]
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

                da = density.DensityAnalysis(lig, delta=grid_spacing,padding=20.0)
                da.run(step=sample_stride)
                rho = da.results.density

                dens_sum = rho if dens_sum is None else dens_sum._replace(grid=dens_sum.grid + rho.grid) or dens_sum
                total_frames += len(u.trajectory[::sample_stride])

            # Normalize and write DX
            if total_frames > 0:
                dens_sum.grid /= float(total_frames)

            dx_path = str((outdir / f"{path_name}_ligand_density.dx").resolve())
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
            pml.write("    show spheres, lig\n")
            pml.write("    color lightpink, lig\n")
            pml.write("    set sphere_transparency, 0.35, lig\n")
            pml.write("orient lig\n")
            pml.write("zoom prot, 10.0\n")
            # pml.write("png preview.png, ray=1, dpi=300\n")

        return str(pml_path)
    
    @staticmethod
    def plot_work_profiles(
        results: pd.DataFrame,
        r_coord: str = 'r_coord',
        cols_to_plot: list = ['Wmean', 'dG', 'Wdiss'],
        title_suffix: str = 'dcTMD',
        outdir: str = 'work_profiles'
    ):
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

            axes[i].set_title(f'Speed: {speed} nm/ps', fontsize=12)
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
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"work_profiles_{title_suffix}.png")
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        return
