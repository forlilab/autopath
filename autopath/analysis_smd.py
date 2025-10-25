import os
import glob
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
                 pocket_select: str = 'protein within 6.0 of resname UNK and name CA',
                 ligand_select: str = 'resname UNK and not name H*',
                 dist_column: str = 'r_before',
                 work_column: str = 'work',
                 force_column: str = 'force',
                 seed: int = 42
                 ):

        """Initialize the SteeredMDAnalysis class with log files and parameters."""
        if log_files is None or len(log_files) == 0:
            raise ValueError("No log files provided for analysis.")
        
        if sysname is None:
            sysname = log_files[0].split('/')[0]  # Extract system name from the first log file path
        self.sysname = sysname

        self.outdir = os.path.dirname(log_files[0])  # Output directory is the same as the first log file

        self.temp = temperature # Kelvin
        self.kB = 0.0083144621 # kJ/(mol*K)
        self.beta = 1.0 / (self.kB * self.temp)
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

        self.dist_column = dist_column
        self.work_column = work_column
        self.force_column = force_column
        self.meff_column = 'm_eff'

        self.n_bins = n_bins
        self.dist_minmax = dist_minmax  # default min/max for distance bins
        self.log_files = log_files

        self.seed = seed

        return

    def run_analysis(self,
                     speeds: list[float] = None,
                     temperature: float = None,
                     fit_GMM: bool = True
                     )-> pd.DataFrame:
        
        """Run the analysis on the raw data.
        This method computes the free energy difference using Jarzynski's equality
        and the dissipated work approximation.
        """
        
        GMM_WEIGHT_CUTOFF = 0.0 # # cutoff for GMM weights, below which we ignore the component

        # sometimes you wanna run the analysis with a different temperature or intervals
        if temperature is not None:
            self.temp = temperature
            self.beta = 1.0 / (self.kB * self.temp)

        # assemble the master dataframe
        raw_data = self.load_logs(self.log_files)

        if speeds is not None:
            print(f'WARNING: Filtering data by speeds: {speeds}')
            raw_data = raw_data[raw_data['speed'].isin(speeds)]

        if self.dist_minmax is not None:
            print(f'WARNING: Filtering data by distance range: {self.dist_minmax}')
            raw_data = raw_data[(raw_data[self.dist_column] >= self.dist_minmax[0]) & 
                                          (raw_data[self.dist_column] <= self.dist_minmax[1])]

        # this will overwrite the work column in the raw_data DataFrame
        raw_data = self.integrate_force_dx(raw_data)

        # bin the data
        self.raw_data, centers = self.bin_data(raw_data, self.n_bins, 
                                               use_quantiles=True,
                                               min_points=1)
        # cluster trajectories into paths if specified
        if self.cluster_paths:
            self.raw_data = self.cluster_trajectories()
        else:
            self.raw_data['path'] = 1 # default to single cluster if no clustering method is specified

        # decorrelate work values using statistical inefficiency g or by replica averaging
        # If we dont decorrelate, we should use the per-replica aggregated work. i.e. each replica contributes one work value per bin ENSEMBLE AVERAGE OVER REPLICAS
        self.decorrelated_data = self.decorrelate_work_data(use_g=True)

        results = []
        gmm_results = defaultdict(dict)  # to store GMM results per bin and speed
        for (r_bin, speed, path), group in self.decorrelated_data.groupby(["r_bin", "speed", "path"]):

            raw_W = group[self.work_column].values.astype(np.float64) # shape (N_points,)
                        
            Wmean_raw = raw_W.mean()
            var_raw = raw_W.var(ddof=1)
            Wdiss_raw = 0.5 * self.beta * var_raw

            # plain Jarzynski on the samples
            dG_Jarzynski = -(1/self.beta) * np.log(np.exp(-self.beta * raw_W).mean())

            # GMM branch
            dG_mix_exact = Wdiss_neq = Wmean_neq = np.nan

            if fit_GMM:
                if len(raw_W) < 2:
                    print(f"Not enough data points for GMM fitting at r_bin={r_bin:.2f}, speed={speed:.5f}, path={path:.2f}. Skipping GMM fit.")
                    continue

                gmm_dict = self.fit_gmm_to_work_values(raw_W,
                                                        max_K=3,
                                                        covariance_type='diag',
                                                        random_state=self.seed)

                #These have shape (K,) for K components
                w = np.asarray(gmm_dict["GMM_weights"])          # α_k
                mu = np.asarray(gmm_dict["GMM_means"]).ravel()    # μ_k
                sig2 = np.asarray(gmm_dict["GMM_variances"])       # σ_k²

                # cumulant (second order) per component
                dG_k = mu - 0.5 * self.beta * sig2 # now this is exact for each Gaussian

                # mask small nonequilibrium weights, which means ignore small gaussians
                if np.any(w < GMM_WEIGHT_CUTOFF):
                    print(f'WARNING: Filtering out {len(w[w < GMM_WEIGHT_CUTOFF])} components from bin {r_bin} - {speed} - {path} w/ weights {w[w < GMM_WEIGHT_CUTOFF]}')
                
                w = np.where(w < GMM_WEIGHT_CUTOFF, 0.0, w)
                w /= w.sum()  # normalize weights

                # Transfor the weights from the non-equilibrium populations.
                # This is for path impotance, but I should use w becuase 
                # jarzynski and dcTMD are based on equilibrium weights (I think).
                # p_eq = w * np.exp(-self.beta * dG_k)
                # p_eq /= p_eq.sum()

                # exact mixture Jarzynski
                # This is the exact Jarzynski estimator for the Gaussian mixture in each bin.
                # If k=1 this collapses to the exact Jarzynski estimator for the single gaussian
                # and this can be approximated by cumulant expansion to the second order, what the dcTMD paper does.
                log_terms = -self.beta * mu + 0.5 * self.beta**2 * sig2
                log_Z = special.logsumexp(log_terms, b=w)          # log ⟨e^{-βW}⟩
                dG_mix_exact = -log_Z / self.beta
               
                # non-equilibrium mean of W
                Wmean_neq = np.dot(w, mu)
                
                Wdiss_neq = Wmean_neq - dG_mix_exact
                
                # Non-eq variance of W
                var_neq = np.dot(w, sig2 + (mu - Wmean_neq)**2)
                # this one should match Wdiss_neq
                # Wdiss_neq = 0.5 * self.beta * var_neq

                # Collect results in a dictionary for plotting 
                gmm_results[r_bin][speed] = {'GMM_neq_weights': w,
                                             'replica_W': raw_W,
                                             'GMM_means': mu,
                                             'GMM_variances': sig2,
                                             'Wmean_mix': Wmean_neq
                                            }
                # build this partial dataframe
                results.append({
                    'r_bin': r_bin,
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
        This will overwrite the work column in the raw_data DataFrame.
        """
        grouped = raw_data.groupby("trajname")
        for traj, group in grouped:
            work = cumulative_trapezoid(group[self.force_column], group[self.dist_column], initial=0.0)
            raw_data.loc[raw_data['trajname'] == traj, self.work_column] = work
        
        return raw_data
    
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
            columns = [c for c in df.columns if c not in ['speed','path','r_bin','replica','trajname']]
        for (speed,path), group in df.groupby(['speed','path']):
            for col in columns:
                mask = (df['speed'] == speed) & (df['path'] == path)
                new_df.loc[mask, f'{col}_smooth'] = gaussian_filter1d(group[col], sigma=sigma)
        return new_df
            
    def cluster_trajectories(self, do_PCA:bool=True, do_plots:bool=True):
        """Cluster trajectories using Dynamic Time Warping (DTW) and k-medoids.
        For more information on DTW see: https://doi.org/10.1073/pnas.231354212
                                         https://dtaidistance.readthedocs.io/en/latest/index.html
        """

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
                traj_name = traj_name.replace('.dcd', '').replace('traj', 'log') 
                frame_numbers = np.arange(num_frames)
                traj_df = pd.DataFrame(distances, columns=[f"dist_{i}" for i in range(distances.shape[1])])
                traj_df.insert(0, "frame", frame_numbers)
                traj_df.insert(1, "trajname", traj_name)
                df_list.append(traj_df)
            df = pd.concat(df_list, ignore_index=True)
            df.to_csv(distance_file, index=False)

        # Ensure the keys in trajname_map align with the values in df['trajname']
        # trajname_map = raw_data.set_index('trajname')[self.work_column].to_dict()
        # df[self.work_column] = df['trajname'].map(trajname_map)
        # df.dropna(inplace=True) # Ensure no NaN values in work column
        # trajname_map = raw_data.set_index('trajname')[self.dist_column].to_dict()
        # df[self.dist_column] = df['trajname'].map(trajname_map)

        # # filter by r_bin
        # df = df[(df['r_bin'] >= 0.75) & (df['r_bin'] <= 1.5)]

        distances = df.iloc[:, 2:].values
        scaler = StandardScaler()
        distances = scaler.fit_transform(distances)  # Scale the distances

        if do_PCA:
            pca = PCA(n_components=2, random_state=self.seed)
            distances = pca.fit_transform(distances)
            print(f'The first 2 PC explain {sum(pca.explained_variance_ratio_)*100:.2f}% of the variance')

        distance_df = pd.DataFrame(distances)
        distance_df.insert(0, "frame", df['frame'])
        distance_df.insert(1, "trajname", df['trajname'])

        # Create a distance matrix using DTW
        paths = defaultdict(np.ndarray)
        for traj_name in distance_df.groupby("trajname").groups:
            traj_df = distance_df[distance_df["trajname"] == traj_name]
            paths[traj_name] = traj_df.iloc[:, 2:].values

        stacked = [paths[traj_name] for traj_name in paths.keys()]

        distmatrix = dtw_ndim.distance_matrix_fast(s=stacked, ndim=stacked[0].shape[1])
        
        # if self.cluster_range is not None:
        #     # filter by distance range
        #     valid_indices = []
        #     for i, traj_name in enumerate(paths.keys()):
        #         traj_df = distance_df[distance_df["trajname"] == traj_name]
        #         r_values = traj_df[self.dist_column].values
        #         if np.any((r_values >= self.cluster_range[0]) & (r_values <= self.cluster_range[1])):
        #             valid_indices.append(i)
        #     distmatrix = distmatrix[np.ix_(valid_indices, valid_indices)]
        #     paths = {list(paths.keys())[i]: paths[list(paths.keys())[i]] for i in valid_indices}

        if do_plots:
            # sort the distance matrix by the average distance of each trajectory for plotting
            avg_distances = distmatrix.mean(axis=1)
            sorted_indices = np.argsort(avg_distances)
            sorted_distances = distmatrix[sorted_indices][:, sorted_indices]

            plt.figure(figsize=(6, 5))
            sns.heatmap(sorted_distances, cmap="viridis")
            plt.title("DTW Distance Matrix")
            plt.tight_layout()
            plt.savefig(f"{outdir}/{self.sysname}_distmatrix.png")
            plt.show()
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
            plt.show()
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
            plt.show()
            plt.close()

        trajname_map = pd.DataFrame({"trajname": list(paths.keys()),
                        "cluster": cluster.labels}).set_index('trajname')['cluster'].to_dict()
        raw_data['path'] = raw_data['trajname'].map(trajname_map)

        return raw_data

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
            plt.show()
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
        decorrelated_df = decorrelated_df.explode([self.work_column, 'r_target', 'r_before', 'r_after', 'NC', 'force', 'U_cvpack'], ignore_index=True)
        return decorrelated_df
        
    ########### FUNCTIONS BELOW ARE EXPERIMENTAL #############
    # I just keep them here for now, but they are not used in the main analysis pipeline
    ##########################################################
        
    @staticmethod
    def compute_acf(series, nlags=100):
        """Compute autocorrelation function of a series"""
        return sm.tsa.acf(series, nlags=nlags, fft=True)
    @staticmethod
    def integrated_autocorrelation(acf_vals, cutoff=0.01):
        """Compute integrated autocorrelation time (IACT)"""
        acf_cut = acf_vals[1:]
        # truncate where ACF first becomes negative or drops below cutoff
        mask = acf_cut > cutoff
        return 1 + 2 * np.sum(acf_cut[mask])
    @staticmethod
    def block_average(series, block_size):
        """Returns block-averaged values from a time series. 
        This splits the series into non-overlapping blocks of size block_size and averages each."""
        n_blocks = len(series) // block_size
        trimmed = series[:n_blocks * block_size]
        blocks = trimmed.reshape(n_blocks, block_size)
        return blocks.mean(axis=1)

    def _decorrelate_work(self,
                         raw_data: pd.DataFrame = None,
                         work_column = 'work(kJ/mol)'
                        )-> pd.DataFrame:

        #FIXME: I'm not sure if this is the best way 

        if raw_data is None:
            raw_data = self.raw_data.copy()

        block_sizes = {}

        for (r_bin, v), group in raw_data.groupby(["r_bin", "speed"]):
            for replica, traj in group.groupby("replica"):
                W = traj[work_column].values
                acf_vals = self.compute_acf(W, nlags=200)
                tau_int = self.integrated_autocorrelation(acf_vals, cutoff=0.05)
                block_size = int(np.ceil(tau_int)) * 2  # use 2*tau_int as block size
                block_sizes[(r_bin, v, replica)] = block_size
                # print(f"r_bin: {r_bin:.2f}, speed: {speed}, IACT: {tau_int:.2f}, W_lenght: {len(W)} block_size: {block_size}")

        block_samples = []

        for (r_bin, v), bin_group in raw_data.groupby(["r_bin", "speed"]):
            for replica, traj in bin_group.groupby("replica"):
                trajname = traj['trajname'].iloc[0]  # Get the trajectory name from the first row
                path = traj['path'].iloc[0]  # Get the path from the first row
                block_size = block_sizes.get((r_bin, v, replica), 1)
                W = traj[work_column].values
                W_blocks = self.block_average(W, block_size)

                for w, w_eff in zip(W_blocks, W):
                    block_samples.append({
                        "r_bin": r_bin,
                        "speed": v,
                        "replica": replica,
                        work_column: w_eff,
                        f"{work_column}_raw": w,
                        "block_size": block_size,
                        'trajname': trajname,
                        'path': path,
                    })

        return pd.DataFrame(block_samples)


    def estimate_dG_Jarzynski(self,
                              df: pd.DataFrame = None,
                              work_column:str=None,
                              speeds: list[float] = None,
                              )-> pd.DataFrame:
        """Estimate the free energy difference using Jarzynski's equality.
        This method computes the free energy difference for each speed
        based on the work done during the pulling simulation"""


        if df is None:
            df = self.raw_data.copy()
        # Filter out speeds if provided. You could only want to fit specific (low) speeds
        if speeds is not None:
            df = df[df['speed'].isin(speeds)]

        if work_column is None:
            work_column = self.work_column
            
        # Ensure centers are sorted in ascending order
        centers = df['r_bin'].unique()
        centers = np.sort(centers)

        records = []
        for speed, speed_data in df.groupby('speed'):
            try:
                exp_av = (speed_data.groupby('r_bin')[work_column]
                        .apply(lambda w: np.exp(-self.beta * w).mean()))

                dG = -(1.0 / self.beta) * np.log(exp_av)
                Wmean = speed_data.groupby('r_bin')[work_column].mean()
                W_diss_Jarz = Wmean - dG

                df = pd.DataFrame({
                    'r_bin': centers[dG.index],
                    'dG_Jarz': dG.values,
                    'W_diss_Jarz': W_diss_Jarz.values,
                    'speed': speed,
                })
                records.append(df)
            except Exception as e:
                print(f"Error estimating dG_Jarzynski for speed {speed}: {e}")
                continue
        if len(records) == 0:
            return pd.DataFrame()
        
        return pd.concat(records, ignore_index=True)
    
    def estimate_dG_Cumulative(self, 
                            data: pd.DataFrame = None,
                            smooth_Wdiss: bool = True,
                            fit_spline: bool = True,
                            path_column: str = 'path',
                            replica_column: str = 'replica',
                            inertial_correction: str = None, # "per_replica" or "per_bin"
                            ) -> pd.DataFrame:
        """Estimate the free energy difference using the dissipated work approximation 
        Cumulant expansion to the second order.
        If you clustered the trajectories before, like using any disance based method, 
        you can specify the path_column to group by.
        Optionally apply inertial correction via meff.
        """
        if data is None:
            data = self.raw_data.copy()

        # Ensure centers are sorted in ascending order
        centers = data['r_bin'].unique()
        centers = np.sort(centers)

        all_data = []
        for path in data[path_column].unique():
            # Filter data for the current path
            path_data = data[data[path_column] == path]
            per_speed_dfs = []
            for speed in path_data['speed'].unique():
                df = path_data[path_data['speed'] == speed]

                # first we aggregate per (replica, bin)
                df_replica_bin = (
                    df.groupby([replica_column, 'r_bin'])[self.work_column]
                    .mean()
                    .reset_index()
                )

                # group across replicas
                g_work = df_replica_bin.groupby('r_bin')[self.work_column]
                count = g_work.count().reindex(centers, fill_value=0)
                Wmean = g_work.mean().reindex(centers, fill_value=np.nan)
                Wvar = g_work.var(ddof=1).reindex(centers, fill_value=0.0)

                # Collect m_eff
                g_meff = pd.Series(np.nan, index=centers)
                if self.meff_column is not None:
                    g_meff = df.groupby('r_bin')[self.meff_column].mean().reindex(centers, fill_value=np.nan)
                elif inertial_correction:
                    raise ValueError("Inertial correction requested but no m_eff column provided.")

                # Apply inertial correction
                # This is very small, only meaningful for v high speeds
                if inertial_correction == "per_replica":
                    df['work_corr'] = df[self.work_column] - 0.5 * df[self.meff_column] * speed**2

                    df_replica_bin = (
                        df.groupby([replica_column, 'r_bin'])['work_corr']
                        .mean()
                        .reset_index()
                    )

                    g_corr = df_replica_bin.groupby('r_bin')['work_corr']
                    Wmean = g_corr.mean().reindex(centers, fill_value=np.nan)
                    Wvar = g_corr.var(ddof=1).reindex(centers, fill_value=0.0)

                elif inertial_correction == "per_bin":
                    Wvar = Wvar - g_meff * speed**2  # Var[W_corr] = Var[W] - Var[inertial term]
                    Wvar = np.clip(Wvar, 0.0, None)  # make sure no negatives

                # Raw dissipated work
                Wdiss = 0.5 * self.beta * Wvar.values

                # Optional smoothing
                if smooth_Wdiss:
                    Wdiss = gaussian_filter1d(Wdiss, sigma=1)
                    # smooth the dissipation using a Savitzky-Golay filter. This might be better
                    # Wdiss = savgol_filter(Wdiss, window_length=9, polyorder=3, mode='interp')

                # Optional spline fit for derivative
                if fit_spline:
                    spline = UnivariateSpline(centers, Wdiss, k=3)
                    Gamma = spline.derivative()(centers) / speed
                else:
                    Gamma = np.gradient(Wdiss, self.bin_width) / speed

                # Remove bad bins
                Gamma = np.where(Gamma < 1e-3, np.nan, Gamma)
                dG = Wmean.values - Wdiss

                df_out = pd.DataFrame({
                    'speed': speed,
                    'count': count.values,
                    'W_mean': Wmean.values,
                    'W_var': Wvar.values,
                    'W_diss_Cum': Wdiss,
                    'dG_Cum': dG,
                    'Gamma': Gamma,
                    'm_eff': g_meff.values,
                    'tau_inertia': g_meff.values / Gamma,
                }, index=pd.Index(centers, name='r_bin'))

                per_speed_dfs.append(df_out)

            speed_df = pd.concat(per_speed_dfs).reset_index()
            speed_df[path_column] = path
            all_data.append(speed_df)

        result_df = pd.concat(all_data, ignore_index=True)

        return result_df

    
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

        # if df['speed'].nunique() < 2:
        #     print("Not enough speeds for extrapolation.")
        #     return pd.DataFrame()
        
        # FIXME 
        if mixed_models:
            results = []
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
        # df['D(x)'] = self.kB * self.temp / df['F(x)']

        return results
    
    def compute_gamma_from_fac(self, 
                               force_col='force(kJ/mol/nm)',
                                )-> pd.DataFrame:
        """
        Estimate Gamma(x) from force autocorrelation at each reaction coordinate bin.
        df must contain 'replica', 'step', 'force(kJ/mol/nm)', and r-bin variable.
        In the dcTMD framework the position-dependent friction coefficient Γ(x) 
        is obtained from a Green-Kubo-type relation that connects the integral 
        of the constraint-force fluctuations to the friction experienced 
        """
        # #FIXME: check that steps per move should be the same so time make sense 
        df = self.raw_data.copy()
        
        gamma_results = []
        for (r_bin, speed), group in df.groupby(['r_bin', 'speed']):
            replicas = group['replica'].unique()
            acfs = []

            for replica in replicas:
                traj = group[group['replica'] == replica].sort_values('step')
                forces = traj[force_col].values

                f_centered = forces - np.mean(forces)
                acf = np.correlate(f_centered, f_centered, mode='full')
                acf = acf[acf.size // 2:]  # keep positive lags only
                # acf /= acf[0]  # normalize to ACF(0) = 1
                acfs.append(acf)

            # Truncate to shortest ACF length and average
            min_len = min(len(a) for a in acfs)
            acfs = [a[:min_len] for a in acfs]
            mean_acf = np.mean(acfs, axis=0)

            # Estimate variance of the force across all replicas
            # all_forces = group[force_col].values
            # var_f = np.var(all_forces)

            # Integrate the normalized ACF and rescale
            integral = np.trapz(mean_acf, dx=self.timestep)  # result in ps
            # gamma = integral * var_f / (kB * temperature)  # result in (kJ·ps)/(mol·nm²)
            gamma = integral / (self.kB * self.temp)  # result in (kJ·ps)/(mol·nm²)

            gamma_results.append({
                'r_bin': r_bin,
                'speed': speed,
                'F(x)_ACF': gamma,
                'n_replicas': len(replicas)
            })

        return pd.DataFrame(gamma_results)
    

    def estimate_permeability_ISD(self,
                                df: pd.DataFrame= None,
                                dG_col:str = 'dG_extrapolated',
                                diffusion_col:str = 'D(x)',
                                )-> pd.DataFrame:
        """Estimate the permeability using the extrapolated free energy difference
        and the diffusion coefficient.
        As described here: https://pmc.ncbi.nlm.nih.gov/articles/PMC6506413/
        """
        #FIXME I neeed to fix the units

        if df is None:
            df = self.extrapolate_work_v0(self.estimate_dG_dcWork())

        if dG_col not in df.columns or diffusion_col not in df.columns:
            raise ValueError(f"DataFrame must contain columns '{dG_col}' and '{diffusion_col}'")
        
        # Calculate the centers from the index
        centers = np.sort(df.index.values)

        P_inv = np.trapz(np.exp(self.beta*df[dG_col])/df[diffusion_col], x=centers)
        df['P(x)'] = 1 / P_inv

        return df

    def plot_dcWork_profiles(self, 
                             dG_dcWork:pd.DataFrame, 
                             outfname:str=None
                             )-> None:
        """Plot the dissipated work and free energy difference profiles."""

        if dG_dcWork is None:
            dG_dcWork = self.estimate_dG_dcWork()

        g = sns.FacetGrid(dG_dcWork, col="speed", 
                        col_wrap=len(dG_dcWork['speed'].unique()), 
                        height=4,
                        sharex=True, sharey=True)

        g.map(sns.lineplot, "r_bin", "W_diss", label="W_diss", color='red', linestyle='-')
        g.map(sns.lineplot, "r_bin", "W_mean", label="W_mean", color='black', linestyle='--')
        g.map(sns.lineplot, "r_bin", "dG", label="dG", color='green')#, linestyle='-')
        g.set_axis_labels("r_bin", self.work_column)
        g.set_titles("Speed: {col_name}")
        plt.title(f"{self.sysname} - Wdiss and dG vs Target Distance")
        g.add_legend()
        plt.grid(True)
        plt.tight_layout()
        plt.grid(True)
        if outfname is not None:
            plt.savefig(outfname, dpi=300)
        plt.show()
        plt.close() 
        return
    
    def plot_gamma_NEQ(self,
                       dG_dcWork:pd.DataFrame=None,
                       outfname:str=None
                          )-> None:
        """Plot the Gamma profile from the dissipated work."""
        if dG_dcWork is None:
            dG_dcWork = self.estimate_dG_dcWork()

        g = sns.FacetGrid(dG_dcWork, col="speed", 
            col_wrap=len(dG_dcWork['speed'].unique()), 
            height=4, 
            sharex=True, sharey=True) 
        g.map(sns.scatterplot, "r_bin", "Gamma", label="Γ_NEQ(x)", color='red', linestyle='-')
        g.set_axis_labels("r_bin", "Γ_NEQ(x) (J/mol/nm²/ps)")
        plt.title(f"{self.sysname} - Γ_NEQ(x) vs Target Distance")
        g.set_titles("Speed: {col_name}")
        g.add_legend()
        plt.tight_layout()
        plt.grid(True)
        if outfname is not None:
            plt.savefig(outfname, dpi=300)
        plt.show()
        plt.close() 
        return
    
    def plot_force_profiles(self, outfname:str=None)-> None:
        """Plot the force profile from the raw data."""
  
        palette = sns.color_palette("flare", n_colors=len(self.raw_data['replica'].unique()))
        g = sns.FacetGrid(self.raw_data, 
                        col="speed", 
                        col_wrap=len(self.raw_data['speed'].unique()), 
                        hue='replica',
                        palette=palette,
                        height=4, 
                        sharex=True, sharey=True)
        g.map(sns.lineplot, self.dist_column, self.force_column, alpha=0.4)
        plt.title(f"{self.sysname} - Force vs Target Distance")
        g.set_axis_labels(self.dist_column, self.force_column)
        g.set_titles("Speed: {col_name}")
        plt.tight_layout()
        plt.grid(True)
        if outfname is not None:
            plt.savefig(outfname, dpi=300)
        plt.show()
        plt.close()
        return
    
    def plot_work_profiles(self, outfname:str=None)-> None:
        """Plot the work profile from the raw data."""
        
        palette = sns.color_palette("mako", n_colors=len(self.raw_data['replica'].unique()))
        g = sns.FacetGrid(self.raw_data, 
                        col="speed", 
                        col_wrap=len(self.raw_data['speed'].unique()), 
                        hue='replica',
                        palette=palette,
                        height=4, 
                        sharex=True, sharey=True)
        g.map(sns.lineplot, self.dist_column, self.work_column, label="Gamma", alpha=0.4)
        plt.title(f"{self.sysname} - Work vs Target Distance")
        g.set_axis_labels(self.dist_column, self.work_column)
        g.set_titles("Speed: {col_name}")
        plt.tight_layout()
        plt.grid(True)
        if outfname is not None:
            plt.savefig(outfname, dpi=300)
        plt.show()
        plt.close()
        return
    
    def plot_lag_profiles(self, 
                        outfname:str=None,
                        target_col:str = 'r_target(nm)',
                        current_col:str = 'r_after(nm)',
                        )-> None:
        """Plot the drift profile from the raw data."""
        
        raw_data = self.raw_data.copy()
        raw_data['lag(nm)'] = raw_data[target_col] - raw_data[current_col]

        palette = sns.color_palette("crest", n_colors=len(raw_data['replica'].unique()))
        g = sns.FacetGrid(raw_data, 
                        col="speed", 
                        col_wrap=len(raw_data['speed'].unique()), 
                        hue='replica',
                        palette=palette,
                        height=4, 
                        sharex=True, sharey=True)
        g.map(sns.lineplot, 'r_target(nm)', 'lag(nm)', alpha=0.4)
        plt.title(f"{self.sysname} - Lag vs Target Distance")
        g.set_axis_labels(target_col, 'lag(nm)')
        g.set_titles("Speed: {col_name}")
        plt.tight_layout()
        plt.grid(True)
        if outfname is not None:
            plt.savefig(outfname, dpi=300)
        plt.show()
        plt.close()
        return

    def plot_potential_profiles(self,
                                df: pd.DataFrame = None,
                                potential_col:str = 'U_cvpack(kJ/mol)',
                                outfname:str = None
                                )-> None:
        """Plot the potential energy profile from the raw data."""
        
        if df is None:
            df = self.raw_data

        # Plot using seaborn lineplot with dashed lines for each speed
        # plt.figure(figsize=(10, 6))
        sns.lineplot(df, x='r_bin', y=potential_col, hue='speed', palette='tab10',linestyle='--')
        plt.title(f"{self.sysname} - Potential Energy vs r_bin")
        plt.xlabel('r_bin (nm)'); plt.ylabel(potential_col)
        plt.grid(True)
        plt.tight_layout()
        if outfname is not None:
            plt.savefig(outfname, dpi=300)
        plt.show()
        plt.close()
        return


    def plot_extrapolated_param(self, 
                                df_v0: pd.DataFrame = None, 
                                param_col:str = 'dG_extrapolated',
                                dG_Jarzynski: pd.DataFrame = None,
                                outfname:str = None
                                ):

        if df_v0 is None:
            df_v0 = self.extrapolate_work_v0(self.estimate_dG_dcWork())

        try:
            color_col = 'R2'  # Column for color mapping
            se_col = f'se_{param_col.split("_")[0]}'

            x = df_v0['r_bin'] if 'r_bin' in df_v0 else df_v0.index
            y = df_v0[param_col].values

            yerr = df_v0[se_col].values if se_col in df_v0.columns else None

            # Normalize R2 for colormap
            norm = mcolors.Normalize(vmin=df_v0[color_col].min(), vmax=df_v0[color_col].max())
            cmap = cm.get_cmap('coolwarm')

            # Plot shaded error bands and colored lines segment-wise
            fig, ax = plt.subplots(figsize=(6, 4))
            for i in range(len(df_v0) - 1):
                xi = x.iloc[i:i+2] if hasattr(x, 'iloc') else x[i:i+2]
                yi = y[i:i+2]
                yerri = yerr[i:i+2] if yerr is not None else None
                r2_val = df_v0[color_col].iloc[i]

                color = cmap(norm(r2_val))
                ax.plot(xi, yi, color=color, lw=3)

                if yerri is not None:
                    ax.fill_between(xi, yi - yerri, yi + yerri, color=color, alpha=0.4)

            # Plot the Jarzynski data if provided
            if dG_Jarzynski is not None:
                palette = sns.color_palette("mako", n_colors=len(dG_Jarzynski['speed'].unique()))
                sns.lineplot(data=dG_Jarzynski, x='r_center(nm)', y='dG_Jarz(kJ/mol)', hue='speed', palette=palette)

            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('$R^2$ of extrapolation')

            ax.set_xlabel('r_bin(nm)');        ax.set_ylabel(param_col)
            ax.set_title(f'{self.sysname} - {param_col}_v0 vs. r_bin(nm)')
            ax.grid(True)
            # plt.ylim((0,700))
            # plt.xticks(np.arange(0, 2.1, 0.2))
            plt.tight_layout()
            if outfname is not None:
                plt.savefig(outfname, dpi=300)
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Error plotting extrapolated parameter: {e}")
            return None
        return
    
