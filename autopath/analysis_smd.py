import os
import glob
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import statsmodels.formula.api as smf

from dtaidistance import dtw_ndim
import kmedoids
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# style.use("fivethirtyeight")


class SteeredMDAnalysis:
    """Class to analyze Steered Molecular Dynamics (sMD) data, mostly within the dcTMD framework.
    For more information on the dcTMD see...

    
    
    """
    def __init__(self, 
                 log_files: list[str] = None,
                 sysname: str = "system",
                 bin_width: float = 0.05, #nm
                 min_points: int = 10,
                 temperature: float = 300, #K
                 timestep: float = 0.004, #ps
                 dist_column: str = 'r_target(nm)',
                 work_column: str = 'work(kJ/mol)',
                 force_column: str = 'force(kJ/mol/nm)',
                 meff_column: str = 'm_eff(dalton)'
                 ):

        """Initialize the SteeredMDAnalysis class with log files and parameters."""
        if log_files is None or len(log_files) == 0:
            raise ValueError("No log files provided for analysis.")
        
        if sysname is None:
            sysname = log_files[0].split('/')[0]  # Extract system name from the first log file path

        self.sysname = sysname

        self.temp = temperature # Kelvin
        self.kB = 0.0083144621 # kJ/(mol*K)
        self.beta = 1.0 / (self.kB * self.temp)
        self.timestep = timestep

        self.dist_column = dist_column
        self.work_column = work_column
        self.force_column = force_column
        self.meff_column = meff_column

        self.bin_width = bin_width

        # assemble the master dataframe
        raw_data = self.load_logs(log_files)
        
        self.raw_data = self.bin_data(raw_data, bin_width, min_points)

        return

    def load_logs(self, log_files: list[str]) -> pd.DataFrame:

        # compile raw log files
        count = 0
        raw_data = []
        for fn in log_files:
            try:
                base = os.path.basename(fn).strip('.dat')
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

    def bin_data(self, 
                 raw_data: pd.DataFrame,
                 bin_width: float=0.05, #nm
                 min_points: int=10
                 )-> pd.DataFrame:
                 
        raw_data = raw_data.loc[raw_data['r_target(nm)'] < 2.5]  # filter for r_target < 2.0 nm

        # get overall centers and edges for histogram
        rmin, rmax = min(raw_data[self.dist_column]), max(raw_data[self.dist_column])
        edges  = np.arange(rmin, rmax + bin_width, bin_width)
        centers = edges[:-1] + bin_width / 2

        # Use shared edges and centers
        raw_data['bin'] = np.digitize(raw_data[self.dist_column], edges) - 1
        raw_data = raw_data[(raw_data['bin'] >= 0) & (raw_data['bin'] < len(centers))]

        raw_data = self.filter_low_count_bins(raw_data, min_points)

        #these are the center each point belongs to
        raw_data['r_bin'] = raw_data['bin'].map(lambda b: centers[b] if b >= 0 and b < len(centers) else np.nan)

        return raw_data

    def filter_low_count_bins(self, raw_data, min_points):
        """Filter low count bins per speed. This function filters out bins 
        that have fewer than `min_points` data points for each speed.
        """

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

    def cluster_trajectories_DTW(self, 
                                data:pd.DataFrame=None, 
                                features_dict:dict=None, 
                                K:int=None,
                                outdir:str=None):

        """Cluster trajectories using Dynamic Time Warping (DTW) and k-medoids.
        For more information on DTW see: https://doi.org/10.1073/pnas.231354212
                                         https://dtaidistance.readthedocs.io/en/latest/index.html
        """
        if outdir is None:
            outdir = os.path.join(os.getcwd(), self.sysname)
        os.makedirs(outdir, exist_ok=True)
        
        if features_dict is None:
            features_dict = {'r_before(nm)': (0.25, 2.5),
                            'C':(5,70),
                            'work(kJ/mol)': (10, None)
                            }
            
        features = list(features_dict.keys())

        if data is None:
            data = self.raw_data.copy()

        # check if features are in the data
        for feature in features:
            if feature not in data.columns:
                raise ValueError(f"Feature '{feature}' not found in data columns.")
            
        data[features] = data[features].round(3)  # round features to 3 decimal places
        if 'work(kJ/mol)' in features:
            data['work(kJ/mol)'] = data['work(kJ/mol)'] / data['speed']  # normalize work by speed

        # Filter data based on feature ranges
        filters = []
        for f, (l, h) in features_dict.items():
            if l is not None and h is not None:
                filters.append((data[f] > l) & (data[f] < h))
            elif l is not None:
                filters.append(data[f] > l)
            elif h is not None:
                filters.append(data[f] < h)
        if filters:
            data = data[np.logical_and.reduce(filters)]
        data[features] = StandardScaler().fit_transform(data[features])

        paths = []
        # Extract paths for each trajectory and measure distance
        for traj_name in data.groupby("trajname").groups:
            traj_df = data[data["trajname"] == traj_name][features]
            paths.append(traj_df[features].values)
            
        distmatrix = dtw_ndim.distance_matrix_fast(s=paths, ndim=paths[0].shape[1])
        # plot the distance matrix
        sns.heatmap(distmatrix, cmap='viridis')
        plt.title(f"Distance Matrix for {self.sysname} - DTW Clustering")
        plt.xlabel("Trajectories"); plt.ylabel("Trajectories")
        plt.tight_layout()
        plt.savefig(f"{outdir}/distmatrix.png", dpi=300)
        plt.show()
        plt.close()

        # Perform clustering using k-medoids
        if K is None:
            silloutte_scores = {}
            for i in range(2, data['trajname'].nunique()):
                c = kmedoids.fasterpam(distmatrix, i)
                silloutte_scores[i] = silhouette_score(distmatrix, c.labels)

            K = max(silloutte_scores, key=silloutte_scores.get)
            print(f"Optimal number of clusters: {K}")

            # Plot silhouette scores
            sns.lineplot(x=list(silloutte_scores.keys()), y=list(silloutte_scores.values()))
            plt.xlabel("Number of clusters (K)"); plt.ylabel("Silhouette Score")
            plt.title(f"Silhouette Scores for {self.sysname}")
            plt.tight_layout()
            plt.savefig(f"{outdir}/silhouette_scores.png", dpi=300)
            plt.show()
            plt.close()

        cluster = kmedoids.fasterpam(distmatrix, K)

        df_labels = []
        for traj_name, label in zip(data.groupby("trajname").groups, cluster.labels):
            df_labels.append([traj_name, label])

        df = data.groupby(["trajname"]).first().reset_index()
        df['label'] = cluster.labels
        df = df.set_index('trajname')  # Restore original index
        # Merge labels back to the original data
        data['label'] = data['trajname'].map(df['label'])

        if len(features) == 2:
            sns.scatterplot(data=data, x=features[0], y=features[1], hue='label', alpha=0.3, palette='Set1')
        else:
            sns.PairGrid(data,
                vars=features,
                hue='label', # 'label','speed'
                palette='viridis',
                height=3).map_lower(sns.scatterplot).map_diag(sns.kdeplot)
            
        # plt.title(f"Clustering with {K} clusters")
        plt.xlabel(features[0]); plt.ylabel(features[1])
        plt.legend(title='Cluster', loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{outdir}/trajs_clustered_k-{K}.png", dpi=300, bbox_inches='tight')
        plt.show()   
        plt.close()

        return data

    def estimate_dG_Jarzynski(self, )-> pd.DataFrame:
        """Estimate the free energy difference using Jarzynski's equality.
        This method computes the free energy difference for each speed
        based on the work done during the pulling simulation"""

        centers = self.raw_data['r_bin'].unique()
        # Ensure centers are sorted in ascending order
        centers = np.sort(centers)
        records = []
        for speed, speed_data in self.raw_data.groupby('speed'):
            try:
                exp_av = (speed_data.groupby('bin')[self.work_column]
                        .apply(lambda w: np.exp(-self.beta * w).mean()))

                dG = -(1.0 / self.beta) * np.log(exp_av)

                df = pd.DataFrame({
                    'r_center(nm)': centers[dG.index],
                    'dG_Jarz(kJ/mol)': dG.values,
                    'speed': speed,
                })
                records.append(df)
            except Exception as e:
                print(f"Error estimating dG_Jarzynski for speed {speed}: {e}")
                continue
        if len(records) == 0:
            return pd.DataFrame()
        return pd.concat(records, ignore_index=True)
    
    def estimate_dG_dcWork(self, 
                            smooth_Wdiss: bool = True,
                            fit_spline: bool = True,
                            inertial_correction: str = None, # "per_replica" or "per_bin"
                            ) -> pd.DataFrame:
        """Estimate the free energy difference using the dissipated work.
        Optionally apply inertial correction via meff.
        """

        # Ensure centers are sorted in ascending order
        centers = self.raw_data['r_bin'].unique()
        centers = np.sort(centers)

        per_speed_dfs = []
        for speed in self.raw_data['speed'].unique():
            df = self.raw_data[self.raw_data['speed'] == speed]

            # Group by bin
            g_work = df.groupby('bin')[self.work_column]
            count = g_work.count().reindex(range(len(centers)), fill_value=0)
            Wmean = g_work.mean().reindex(range(len(centers)), fill_value=np.nan)
            Wvar = g_work.var(ddof=1).reindex(range(len(centers)), fill_value=0.0)

            # Collect m_eff
            g_meff = pd.Series(np.nan, index=range(len(centers)))
            if self.meff_column is not None:
                g_meff = df.groupby('bin')[self.meff_column].mean().reindex(range(len(centers)), fill_value=np.nan)
            elif inertial_correction:
                raise ValueError("Inertial correction requested but no m_eff column provided.")

            # Apply inertial correction
            # This is very small, only meaningful for v high speeds
            if inertial_correction == "per_replica":
                df['work_corr'] = df[self.work_column] - 0.5 * df[self.meff_column] * speed**2
                g_corr = df.groupby('bin')['work_corr']
                Wmean = g_corr.mean().reindex(range(len(centers)), fill_value=np.nan)
                Wvar = g_corr.var(ddof=1).reindex(range(len(centers)), fill_value=0.0)

            elif inertial_correction == "per_bin":
                Wvar = Wvar - g_meff * speed**2  # use Var[W_corr] = Var[W] - Var[inertial term]
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
                'W_diss': Wdiss,
                'dG': dG,
                'Gamma': Gamma,
                'm_eff': g_meff.values,
                'tau_inertia': g_meff.values / Gamma,
            }, index=pd.Index(centers, name='r_bin'))

            per_speed_dfs.append(df_out)

        result_df = pd.concat(per_speed_dfs).reset_index()
        return result_df

    def extrapolate_to_v0(self,
                            df: pd.DataFrame = None,
                            param_col: str = 'W_mean',
                            speeds: list[float] = None,
                            mixed_models: bool = False
                            ) -> pd.DataFrame:

        """Extrapolate a the mean work to zero speed using linear regression. 
        This method groups the data by speed and fits a linear regression to the
        param vs speed for each bin."""

        # param_col = 'W_mean'

        if df is None:
            df = self.estimate_dG_dcWork()

        # Filter out speeds if provided. You could only want to fit specific (low) speeds
        if speeds is not None:
            df = df[df['speed'].isin(speeds)]

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
                "dG_extrapolated": intercept,
                "F(x)": slope,
                "R2": 1.0
            })
                
        else:
            results = []
            for r_bin, group in df.groupby('r_bin'):
                speeds = group['speed'].values
                means = group[param_col].values

                # if len(speeds) < 2:
                #     continue

                lr_results = linregress(speeds, means)
                results.append({
                    'r_bin': r_bin,
                    'dG_extrapolated': lr_results.intercept,
                    'F(x)': lr_results.slope,
                    'se_dG': lr_results.intercept_stderr,
                    'se_F(x)': lr_results.stderr,
                    'R2': lr_results.rvalue**2,
                    'n_speeds': len(speeds)
                })

        if len(results) == 0:
            print("No valid extrapolation results found.")
            return pd.DataFrame()
        
        df = pd.DataFrame(results).set_index("r_bin")

        # Calculate the diffusion coefficient D(x) using the friction coefficient F(x)
        df['D(x)'] = self.kB * self.temp / df['F(x)']

        return df
    
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