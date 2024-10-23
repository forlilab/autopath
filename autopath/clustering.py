import os
import logging
import numpy as np
import pandas as pd
from glob import glob
from typing import Union, List

import matplotlib.pyplot as plt
import seaborn as sns

from deeptime.decomposition import VAMP, KVAD
from deeptime.kernels import GaussianKernel
from deeptime.clustering import KMeans, RegularSpace, BoxDiscretization
from sklearn.decomposition import PCA
import MDAnalysis as mda
import pyemma

from autopath.utils import save_model, load_model
import umap

class ClusterTrajectories:
    def __init__(self, 
                data_fname:str=None,
                prmtop_file:str=None,
                traj_list:List[str]=None,
                out_dir:str='milestones',
                seed:int=42
                ):

        self.data_fname = data_fname
        self.prmtop_file = prmtop_file
        self.traj_list = traj_list
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.seed = seed
        np.random.seed(seed)

        return        
    
    def _get_raw_data(self):

        if os.path.exists(self.data_fname):
            logging.info(f'Loading precomputed features from {self.data_fname}')
            return np.load(self.data_fname)
        else:
            logging.error(f'No data found at {self.data_fname}')
            exit(1)
            return None
        
    @staticmethod
    def fit_vamp_model(data, lagtime, embedding_dim):
        logging.info('Fitting VAMP model..')
        if isinstance(embedding_dim, float):
            var_cutoff = embedding_dim
            dim = None
        else:
            var_cutoff = None
            dim = embedding_dim

        vamp_estimator = VAMP(lagtime=lagtime, dim=dim, var_cutoff=var_cutoff, scaling=None)#'kinetic_map')
        vamp_model = vamp_estimator.fit(data).fetch_model()

        return vamp_model

    @staticmethod
    def fit_kvad_model(data, lagtime, embedding_dim):

        logging.info('Fitting KVAD model..')
        kvad_estimator = KVAD(kernel=GaussianKernel(1.),
            lagtime=lagtime, epsilon=1e-5, dim=embedding_dim,
            )
        kvad_model = kvad_estimator.fit(data).fetch_model()

        return kvad_model
    
    @staticmethod
    def fit_umap_model(data, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        logging.info('Fitting UMAP model..')
        X = np.concatenate(data, axis=0)
        umap_estimator = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
        umap_model = umap_estimator.fit(X)

        return umap_model
    
    @staticmethod
    def fit_pca_model(data, n_components=2):
        logging.info('Fitting PCA model..')
        X = np.concatenate(data, axis=0)
        print(X.shape)
        pca_estimator = PCA(n_components=n_components)
        pca_model = pca_estimator.fit(X)

        return pca_model

    def plot_cumulative_variance(self, vamp_model):

        vamp1_score = vamp_model.score(r=1)
        vamp2_score = vamp_model.score(r=2)
        vampE_score = vamp_model.score(r="E")

        # Plot the cumulative kinetic variance
        plt.figure(figsize=(5, 5))
        plt.plot(vamp_model.cumulative_kinetic_variance)
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative kinetic variance')
        plt.title('VAMP cumulative kinetic variance')
        plt.legend([f'VAMP1 score: {vamp1_score:.2f}\nVAMP2 score: {vamp2_score:.2f}\nVAMP-E score: {vampE_score:.2f}'])
        plt.savefig(f'{self.out_dir}/cumulative_kinetic_variance.png')
        plt.show()
        plt.close()
        return None


    def fit_clustering_model(self, model:str='kmeans', projection:np.ndarray=None, n_clusters:int=100, dmin:float=0.1):

        projection_concatenated = np.concatenate(projection, axis=0)

        if model == 'kmeans':
            estimator = KMeans(
                n_clusters=n_clusters,  # place 100 cluster centers
                init_strategy='kmeans++',  # uniform initialization strategy
                # max_iter=0,  # don't actually perform the optimization, just place centers
                fixed_seed=self.seed,
                n_jobs=None,
            )
        elif model == 'regular_space':
            estimator = RegularSpace(
                        dmin=dmin,  # minimum distance between cluster centers
                        max_centers=n_clusters,  # maximum number of cluster centers
                        n_jobs=None
            )
        elif model == 'box_discretization':
            estimator = BoxDiscretization(
                        dim=2,
                        n_boxes=n_clusters # Number of boxes per dimension
            )
        else:
            logging.error(f'Invalid clustering model {model}')
            return None
            
        fitted_model = estimator.fit(projection_concatenated).fetch_model()       
        cluster_labels = [fitted_model.transform(run) for run in projection] #Transform each run to cluster labels separately
        cluster_centers = fitted_model.cluster_centers

        return fitted_model, cluster_labels, cluster_centers
    
    def plot_projection(self, projection, centroids, lagtime):
        projection_concatenated = np.concatenate(projection, axis=0)
        n_clusters = centroids.shape[0]

        pyemma.plots.plot_density(*projection_concatenated.T, alpha=0.2)
        # plot_density(*projection.T, contourf_kws={'norm':'logit'})
        plt.xlabel('comp 1')
        plt.ylabel('comp 2')
        if centroids is not None:
            plt.scatter(*(centroids.T), s=15, c='C1')
            plt.title(f'k = {n_clusters} | lag = {lagtime}')
            plt.savefig(f'{self.out_dir}/projection_{n_clusters}K_{lagtime}lag.png')
        else:
            plt.title(f'lag = {lagtime}')
            plt.savefig(f'{self.out_dir}/projection_{lagtime}lag.png')
        plt.show()
        plt.close()
        return
    
    def plot_individual_projections(self, projection, plots_per_row:int=4):
        num_plots = len(projection)
        num_rows = (num_plots + plots_per_row - 1) // plots_per_row
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(5 * plots_per_row, 4 * num_rows))
        
        if num_rows == 1:
            axes = axes.flatten()  # Ensure axes is always a list of axes objects
        
        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))
        
        # Determine the limits for the axes
        all_data = np.concatenate(projection, axis=0)
        x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
        y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
        
        for i, (ax, p, color) in enumerate(zip(axes, projection, colors)):
            ax.scatter(p[:, 0], p[:, 1], s=20, alpha=0.2, label=f'Trajectory {i}', color=color)
            ax.set_xlabel('Comp 1')
            ax.set_ylabel('Comp 2')
            ax.set_title(f'Traj {i}')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # ax.legend()
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/vamp_individual_projections.png')
        plt.show()
        plt.close()
        return None
    
    @staticmethod
    def find_closest_points(X, centroids, N=1):
        closest_points = []
        for centroid in centroids:
            distances = np.linalg.norm(X - centroid, axis=1)
            closest_point_indices = np.argsort(distances)[:N]
            closest_points.append(closest_point_indices)
        return closest_points

    def write_centroids_pdb(self, projection, centers, N=1):

        n_clusters = centers.shape[0]
        os.makedirs(f'{self.out_dir}/centroids_{n_clusters}K', exist_ok=True)

        projection_concatenated = np.concatenate(projection, axis=0)
        closest_points_indices = self.find_closest_points(projection_concatenated, centers, N)

        # This is a bit tricky, but we need to map the indices of the concatenated projection back to the original runs
        index_mapping = np.concatenate([np.arange(proj.shape[0]) for proj in projection])
        indexes = [index_mapping[indices] for indices in closest_points_indices]

        traj_lengths = [len(mda.Universe(self.prmtop_file, traj).trajectory) for traj in self.traj_list]
        cumulative_lengths = np.cumsum(traj_lengths)

        for centroid_idx, centroid_indexes in enumerate(indexes):
            centroid_x, centroid_y = centers[centroid_idx]
            for index in centroid_indexes:
                try:
                    traj_idx = np.searchsorted(cumulative_lengths, index, side='right')
                    if traj_idx == 0:
                        frame_idx = index
                    else:
                        frame_idx = index - cumulative_lengths[traj_idx - 1]

                    traj = self.traj_list[traj_idx]
                    u = mda.Universe(self.prmtop_file, traj)

                    u.trajectory[frame_idx]
                    u.atoms.write(f"{self.out_dir}/centroids_{n_clusters}K/milestone_{centroid_x:.2f}_{centroid_y:.2f}_frame{index}.pdb")
                except Exception as e:
                    logging.error(f'Error writing pdb for index {index} in traj {traj_idx}\n{e}')
        return None
   
    def score_cv(self, data, dim, lag, n_splits=10, val_frac=0.5):
        """Compute a cross-validated VAMP2 score.

        We randomly split the list of independent trajectories into
        a training and a validation set, compute the VAMP2 score,
        and repeat this process several times.

        Parameters
        ----------
        data : list of numpy.ndarrays
            The input data.
        dim : int
            Number of processes to score; equivalent to the dimension
            after projecting the data with VAMP2.
        lag : int
            Lag time for the VAMP2 scoring.
        n_splits : int, optional, default=10
            How often do we repeat the splitting and score calculation.
        val_frac : int, optional, default=0.5
            Fraction of trajectories which should go into the validation
            set during a split.
        """
        nval = int(len(data) * val_frac)
        scores = np.zeros(n_splits)
        for n in range(n_splits):
            ival = np.random.choice(len(data), size=nval, replace=False)

            vamp_estimator = VAMP(lagtime=lag, dim=dim)
            # vamp_estimator.scaling = "kinetic_map"
            model = vamp_estimator.fit([d for i, d in enumerate(data) if i not in ival]).fetch_model()
            test_model = vamp_estimator.fit([d for i, d in enumerate(data) if i in ival]).fetch_model()
            scores[n] = model.score(r=2, test_model=test_model)
        return scores
    
    def run_vamp_cv(self, 
                    dims:List[int], 
                    lags:List[int], 
                    n_splits:int=10, 
                    val_frac:float=0.5
                    ):

        data = self._get_raw_data()

        lista = []
        for lag in lags:
            for dim in dims:
                scores = self.score_cv(data, dim, lag, n_splits=n_splits, val_frac=val_frac)
                df = pd.DataFrame(scores, columns=['VAMP2 score'])
                df['lag'] = lag
                df['dim'] = dim
                lista.append(df)
        df = pd.concat(lista)
        df.to_csv(f'{self.out_dir}/vamp2_scores_CV.csv', index=True)

        # Plot the results
        sns.boxplot(x='lag', y='VAMP2 score', hue='dim', data=df)
        plt.title('VAMP cross-validation')
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/vamp2_cv.png')
        plt.close()

        return df

    def run(self,
                dim_model:str='vamp',
                dim:Union[int, float]=2, 
                lagtime:int=20,
                clustering_model:str='regular_space',
                dmin:float=0.5,
                n_clusters:int=100,
                write_pdbs:bool=False,
                plot_projection:bool=True
                ):

        self.lagtime = lagtime
        data = self._get_raw_data()

        if os.path.exists(f'{self.out_dir}/{dim_model}_model_{dim}d_{lagtime}lag.pkl'):
            logging.info(f'Loading precomputed {dim_model} model from {self.out_dir}')
            fitted_model = load_model(f'{self.out_dir}/{dim_model}_model_{dim}d_{lagtime}lag.pkl')
        else:
            if dim_model == 'vamp':
                fitted_model = self.fit_vamp_model(data, lagtime, dim)
                self.plot_cumulative_variance(fitted_model)
            elif dim_model == 'kvad':
                fitted_model = self.fit_kvad_model(data, lagtime, dim)
            elif dim_model == 'umap':
                fitted_model = self.fit_umap_model(data, n_components=dim)
            elif dim_model == 'pca':
                fitted_model = self.fit_pca_model(data, n_components=dim)

            save_model(fitted_model, f'{self.out_dir}/{dim_model}_model_{dim}d_{lagtime}lag.pkl')
        
        projection = fitted_model.transform(data)
        np.save(f'{self.out_dir}/projection_{dim_model}_{dim}d_{lagtime}lag.npy', projection)

        fitted_clustering_model, dtrajs, centers = self.fit_clustering_model(clustering_model, projection, n_clusters, dmin)
        save_model(fitted_clustering_model, f'{self.out_dir}/{clustering_model}_model_K{n_clusters}.pkl')
            
        if plot_projection:
            self.plot_projection(projection, centers, lagtime)
            self.plot_individual_projections(projection)
        
        if write_pdbs:
            self.write_centroids_pdb(projection, centers, N=1)

        return