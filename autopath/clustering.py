import os
import logging
import numpy as np
import pandas as pd
from glob import glob
from typing import Union, List

import matplotlib.pyplot as plt
import seaborn as sns

from deeptime.decomposition import TICA, VAMP, KVAD
from deeptime.kernels import GaussianKernel
from deeptime.clustering import KMeans, RegularSpace, BoxDiscretization
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

import MDAnalysis as mda
# import pyemma

from autopath.utils import save_model, load_model
from sklearn.metrics import silhouette_score

import mosaic
import umap

class ClusterTrajectories:
    def __init__(self, 
                mosaic_top_clusters:int=3,
                embedding_model:str='vamp',
                embedding_dim:Union[int, float]=2, 
                embedding_lagtime:int=20,
                clustering_model:str='kmeans',
                n_clusters:int=100,
                dmin:float=0.5, #only for regular_space
                write_pdbs:bool=False,
                out_dir:str='milestones',
                seed:int=42,
                ):
        
        self.mosaic_top_clusters = mosaic_top_clusters
        self.embedding_model = embedding_model.upper()
        self.embedding_dim = embedding_dim
        self.embedding_lagtime = embedding_lagtime
        self.clustering_model = clustering_model
        self.n_clusters = n_clusters
        self.dmin = dmin
        self.write_pdbs = write_pdbs

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.seed = seed
        np.random.seed(seed)

        return   
         
    @staticmethod
    def _get_raw_data(data_fname):

        if os.path.exists(data_fname):
            data = np.load(data_fname)
            logging.info(f'Loading data from {data_fname}')
            logging.info(f"Trajectories: {data.shape[0]}")
            logging.info(f"Frames per trajectory: {data.shape[1]}")
            logging.info(f"Features: {data.shape[2]}")
            return data
        else:
            logging.error(f'No data found at {data_fname}')
            exit(1)
            return None
    
    ### MoSAIC analysis ###
    @staticmethod
    def mosaic_clustering(X, similarity_metric='correlation', 
                        clustering_mode='CPM', weighted=True, 
                        resolution_parameter=0.5):

        # Merge all trajs together
        X = np.concatenate(X, axis=0)

        sim = mosaic.Similarity(
            metric=similarity_metric,  # or 'NMI', 'GY', 'JSD'
        )
        sim.fit(X)
        correlation_matrix = sim.matrix_

        # Cluster the correlation matrix
        clustering = mosaic.Clustering(
            mode=clustering_mode,  # or 'modularity
            weighted=weighted,
            resolution_parameter=resolution_parameter,
        )
        clustering.fit(correlation_matrix)

        clusters = clustering.clusters_
        clustered_X = clustering.matrix_

        return correlation_matrix, clusters, clustered_X
    
    @staticmethod
    def plot_clusters(correlation_matrix, clusters, out_dir):

        idxs = np.argsort(
            [len(cluster) for cluster in clusters],
        )[::-1]
        clusters_sorted = clusters[idxs]
        clusters_sorted_flattened = np.concatenate(clusters[idxs])

        # sort the matrix accordingly
        matrix_sorted = correlation_matrix[
            np.ix_(clusters_sorted_flattened, clusters_sorted_flattened)
        ]

        ticks = np.cumsum([len(cluster) for cluster in clusters[idxs]])
        ticks = [0, *ticks[:-1]]  # ticks start with 0 

        # Perform the same plot again, but with sorted clusters
        fig, ax = plt.subplots()
        im = ax.pcolormesh(
            matrix_sorted,
            snap=True,
            vmin=0,
            vmax=1,
        )
        ax.invert_yaxis()  # origin to the upper left
        ax.set_aspect('equal')  # 1:1 ratio
        ax.set_xticks(ticks[:4])  # we focus only on the first three clusters
        ax.set_yticks(ticks[:4])
        ax.set_xticklabels(np.arange(4)+1)
        ax.set_yticklabels(np.arange(4)+1)
        ax.set_xlabel('clusters')
        ax.set_ylabel('clusters')
        ax.grid(True)
        plt.colorbar(im, label=r'$|\rho|$')
        plt.show()
        plt.savefig(f"{out_dir}/mosaic_clustering.png")
        plt.close()
        return clusters_sorted
    
    @staticmethod
    def filter_data_mosaic(data, clusters_sorted, top_clusters=3):
        features = set()
        for cluster in range(top_clusters):
            for feats in clusters_sorted[cluster]:
                features.add(feats)

        print(f"Number of unique features: {len(features)}")

        cols = list(features)
        filtered_data = data[:,:,cols]

        return filtered_data

    def fit_tica_model(self, data):

        # Scaling parameter
        # ‘kinetic_map’: Eigenvectors will be scaled by eigenvalues. As a result, Euclidean distances in the transformed data approximate kinetic distances [2]. This is a good choice when the data is further processed by clustering.
        # ‘commute_map’: Eigenvector i will be scaled by sqrt(timescale_i / 2). As a result, Euclidean distances in the transformed data will approximate commute distances [3].

        if isinstance(self.embedding_dim, float):
            var_cutoff = self.embedding_dim
            dim = None
        else:
            var_cutoff = None
            dim = self.embedding_dim

        tica_estimator = TICA(lagtime=self.embedding_lagtime, dim=dim, var_cutoff=var_cutoff, scaling='kinetic_map')
        tica_model = tica_estimator.fit(data).fetch_model()

        return tica_model
            
    def fit_vamp_model(self, data, plot_cumulative_variance:bool=True):

        if isinstance(self.embedding_dim, float):
            var_cutoff = self.embedding_dim
            dim = None
        else:
            var_cutoff = None
            dim = self.embedding_dim

        vamp_estimator = VAMP(lagtime=self.embedding_lagtime, dim=dim, var_cutoff=var_cutoff, scaling='kinetic_map')
        vamp_model = vamp_estimator.fit(data).fetch_model()

        if plot_cumulative_variance:
            self._plot_cumulative_kinetic_variance(vamp_model)

        return vamp_model

    def fit_kvad_model(self, data, kernel_width:float=0.75, epsilon:float=1e-6):

        kvad_estimator = KVAD(kernel=GaussianKernel(kernel_width),
            lagtime=self.embedding_lagtime, epsilon=epsilon, dim=self.embedding_dim,
            )
        kvad_model = kvad_estimator.fit(data).fetch_model()

        return kvad_model
    
    def fit_umap_model(self, data, n_neighbors=15, min_dist=0.5, metric='euclidean'):

        X = np.concatenate(data, axis=0)
        umap_estimator = umap.UMAP(n_components=self.embedding_dim, 
                                   n_neighbors=n_neighbors, 
                                   min_dist=min_dist, 
                                   metric=metric, 
                                   densmap=False)
        umap_model = umap_estimator.fit(X)

        return umap_model
    
    def fit_pca_model(self, data, plot_cumulative_variance:bool=True):

        if data.ndim == 2:
            X = data
        else:
            X = np.concatenate(data, axis=0)

        pca_estimator = PCA(n_components=self.embedding_dim)
        pca_model = pca_estimator.fit(X)

        if plot_cumulative_variance:
            self._plot_cumulative_variance(pca_model)
        
        return pca_model
    
    def _plot_cumulative_variance(self, pca_model):

        # Plot the cumulative variance
        plt.figure(figsize=(5, 5))
        plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('PCA cumulative explained variance')
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/pca_cumulative_variance.png')
        plt.close()
        return None

    def _plot_cumulative_kinetic_variance(self, vamp_model):

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
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/vamp_cumulative_kinetic_variance.png')
        plt.close()
        return None
            
    def fit_embedding_model(self, data):

        logging.info(f'Fitting {self.embedding_model} model..')

        if os.path.exists(f'{self.out_dir}/{self.embedding_model}_model_{self.embedding_dim}d_{self.embedding_lagtime}lag.pkl'):
            logging.info(f'Loading precomputed {self.embedding_model} model from {self.out_dir}')
            fitted_model = load_model(f'{self.out_dir}/{self.embedding_model}_model_{self.embedding_dim}d_{self.embedding_lagtime}lag.pkl')
            projection = np.array([fitted_model.transform(run) for run in data])
        else:
            if self.embedding_model == 'VAMP':
                fitted_model = self.fit_vamp_model(data)
                projection = fitted_model.transform(data)
            elif self.embedding_model == 'KVAD':
                fitted_model = self.fit_kvad_model(data)
                projection = fitted_model.transform(data)
            elif self.embedding_model == 'TICA':
                fitted_model = self.fit_tica_model(data)
                projection = fitted_model.transform(data)
            elif self.embedding_model == 'UMAP':
                fitted_model = self.fit_umap_model(data)
                projection = np.array([fitted_model.transform(run) for run in data])
            elif self.embedding_model == 'PCA':
                fitted_model = self.fit_pca_model(data)
                projection = np.array([fitted_model.transform(run) for run in data])
        
        np.save(f'{self.out_dir}/projection_{self.embedding_model}_{self.embedding_dim}d_{self.embedding_lagtime}lag.npy', projection)

        return fitted_model, projection
    
    def fit_clustering_model(self, projection:np.ndarray=None):

        projection_concatenated = np.concatenate(projection, axis=0)

        if self.clustering_model == 'kmeans':
            estimator = KMeans(
                n_clusters=self.n_clusters,  # place 100 cluster centers
                init_strategy='kmeans++',  # uniform initialization strategy
                # max_iter=0,  # don't actually perform the optimization, just place centers
                fixed_seed=self.seed,
                n_jobs=None,
            )
        elif self.clustering_model == 'regular_space':
            estimator = RegularSpace(
                        dmin=self.dmin,  # minimum distance between cluster centers
                        max_centers=self.n_clusters,  # maximum number of cluster centers
                        n_jobs=None
            )
        elif self.clustering_model == 'box_discretization':
            nbox = int(self.n_clusters**(1/self.embedding_dim))
            estimator = BoxDiscretization(
                        dim=self.embedding_dim,  # dimension of the space
                        n_boxes=nbox # Number of boxes per dimension
            )
        else:
            logging.error(f'Invalid clustering model {self.clustering_model}')

            return None
            
        fitted_model = estimator.fit(projection_concatenated).fetch_model()       
        cluster_labels = [fitted_model.transform(run) for run in projection] #Transform each run to cluster labels separately
        cluster_centers = fitted_model.cluster_centers

        return fitted_model, cluster_labels, cluster_centers
    
    def plot_projection(self, projection, labels, centroids):
        """
        Plot the projection of the data.
        Depending on the dimensionality of the projection, either plot a 2D density plot or a pairplot.
        If clustering has been done, use labels to color by cluster labels and also plot centroids, otherwise just plot the projection.
        """
        projection_concatenated = np.concatenate(projection, axis=0)
        df = pd.DataFrame(projection_concatenated, columns=[f'comp {i+1}' for i in range(projection_concatenated.shape[1])])
        if labels is not None and centroids is not None:
            df['cluster'] = np.concatenate(labels, axis=0)
            centroids_df = pd.DataFrame(centroids, columns=[f'comp {i+1}' for i in range(centroids.shape[1])])
            n_clusters = centroids.shape[0]
            out_fname = f'{self.out_dir}/{self.embedding_model}_projection-{self.embedding_dim}d_{n_clusters}K_{self.embedding_lagtime}lag.png'
        else:
            out_fname = f'{self.out_dir}/{self.embedding_model}_projection-{self.embedding_dim}d_{self.embedding_lagtime}lag.png'

        if projection_concatenated.shape[1] == 2:
        #     # pyemma.plots.plot_density(*projection_concatenated.T, alpha=0.2)
            if labels is not None and centroids is not None:
                sns.scatterplot(data=df, x='comp 1', y='comp 2', s=5, alpha=0.2, hue='cluster', palette='Set1')
                sns.scatterplot(data=centroids_df,  x='comp 1', y='comp 2', s=25, c='black', marker='X')#, color='black', label='Centroids')
            else:
                sns.scatterplot(df, s=5, alpha=0.2)
            plt.xlabel('comp 1'); plt.ylabel('comp 2')

        elif projection_concatenated.shape[1] > 2:    
            _df = df.iloc[:, :5]  # Only plot the first 5 components
            _df['cluster'] = df['cluster']
            if labels is not None and centroids is not None:
                g = sns.PairGrid(_df, hue='cluster', corner=True, palette='Set1')
            else:
                g = sns.PairGrid(_df, corner=True)
            g.map_lower(sns.scatterplot, alpha=0.2, s=5)
            g.map_diag(sns.kdeplot, hue=None, color=".3")
            # g.map_upper(sns.kdeplot)
            g.add_legend(title="", adjust_subtitles=True)

        if n_clusters > 5:
            plt.legend().remove()

        plt.tight_layout()
        plt.savefig(out_fname)
        plt.close()
        return
    
    def plot_individual_projections(self, projection, plots_per_row:int=4):
        num_plots = projection.shape[0]
        if num_plots < plots_per_row:
            plots_per_row = num_plots
        num_rows = (num_plots + plots_per_row - 1) // plots_per_row
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(5 * plots_per_row, 4 * num_rows))
                
        if num_rows == 1:
            axes = np.array([axes])  # Ensure axes is always a list of axes objects
        else:
            axes = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 1, num_plots))
        
        # Determine the limits for the axes
        all_data = np.concatenate(projection, axis=0)
        x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
        y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
        
        for i, (ax, p, color) in enumerate(zip(axes, projection, colors)):
            ax.scatter(p[:, 0], p[:, 1], s=20, alpha=0.2, label=f'Trajectory {i}', color=color)
            ax.set_xlabel('Comp 1'); ax.set_ylabel('Comp 2')
            ax.set_title(f'Traj {i}')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # ax.legend()
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/{self.embedding_model}_{self.embedding_dim}d_{self.embedding_lagtime}lag_individual_projections.png')
        plt.close()
        return None
    
    @staticmethod
    def find_closest_points(X, centroids, N=1):
        """A function to find the N closest points to each centroid in the dataset X.
        Centroids may not be real data points, so we need to find the closest real data points to them.
        """
        kdtree = KDTree(X)
        closest_points = []
        for centroid in centroids:
            _, indices = kdtree.query(centroid, k=N)
            closest_points.append(indices)

        return closest_points
    
    def write_centroids_pdb(self, projection, centers, N=1):
        """A function to write the closest points to the cluster centers to PDB files.
        It is a bit tricky, because we need to map the indices of the concatenated projection back to the original runs.
        """

        n_clusters = centers.shape[0]
        pdbs_dir = f"{self.out_dir}/centroids_{n_clusters}K_{self.clustering_model}_{self.embedding_model}"
        os.makedirs(pdbs_dir, exist_ok=True)

        # Concatenate the projection to create a single array for KDTree search
        projection_concatenated = np.concatenate(projection, axis=0)
        closest_points_indices = self.find_closest_points(projection_concatenated, centers, N)
        
        # Map indices back to their corresponding trajectories and frames
        traj_lengths = [len(traj) for traj in projection]
        cumulative_lengths = np.insert(np.cumsum(traj_lengths), 0, 0)

        for centroid_idx, centroid_indexes in enumerate(closest_points_indices):
            if isinstance(centroid_indexes, np.int64):  # If N=1, convert to list for consistency
                centroid_indexes = [centroid_indexes]

            if self.embedding_dim == 2:
                centroid_x, centroid_y = centers[centroid_idx]

            for index in centroid_indexes:
                traj_idx = np.searchsorted(cumulative_lengths, index, side='right') - 1
                frame_idx = index - cumulative_lengths[traj_idx]

                traj_file = self.traj_list[traj_idx]
                try:
                    u = mda.Universe(self.prmtop_file, traj_file)
                    u.trajectory[frame_idx]
                    
                    if self.embedding_dim == 2:
                        filename = f"{pdbs_dir}/milestone_{centroid_x:.2f}_{centroid_y:.2f}_frame{frame_idx}.pdb"
                    else:
                        filename = f"{pdbs_dir}/milestone_{centroid_idx}_frame{frame_idx}.pdb"

                    u.atoms.write(filename)
                except Exception as e:
                    logging.error(f"Error writing PDB for centroid {centroid_idx} at frame {frame_idx} in traj {traj_idx}: {e}")
                    
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
    
    def evaluate_silhouette_scores(self, projection, k_values: List[int]):

        silhouette_scores = []

        for k in k_values:
            self.n_clusters = k
            _, cluster_labels, _ = self.fit_clustering_model(projection)
            cluster_labels_concatenated = np.concatenate(cluster_labels, axis=0)
            projection_concatenated = np.concatenate(projection, axis=0)
            score = silhouette_score(projection_concatenated, cluster_labels_concatenated)
            silhouette_scores.append(score)
            logging.info(f'K={k}, Silhouette Score={score}')

        # Plot the silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.tight_layout()
        plt.savefig(f'{self.out_dir}/silhouette_scores.png')
        plt.close()

        return silhouette_scores
    
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
            data_fname:str=None,
            prmtop_file:str=None,
            traj_list:List[str]=None,
            ):
        
        data = self._get_raw_data(data_fname)
        self.prmtop_file = prmtop_file
        self.traj_list = traj_list

        if self.mosaic_top_clusters is not None:
            correlation_matrix, clusters, clustered_X = self.mosaic_clustering(data)
            clusters_sorted = self.plot_clusters(correlation_matrix, clusters, self.out_dir)
            data = self.filter_data_mosaic(data, clusters_sorted, top_clusters=self.mosaic_top_clusters)

        fitted_embedding_model, projection = self.fit_embedding_model(data)
        save_model(fitted_embedding_model, f'{self.out_dir}/{self.embedding_model}_model_{self.embedding_dim}d_{self.embedding_lagtime}lag.pkl')

        fitted_clustering_model, dtrajs, centroids = self.fit_clustering_model(projection)
        save_model(fitted_clustering_model, f'{self.out_dir}/{self.clustering_model}_model_{self.embedding_dim}d_K{self.n_clusters}.pkl')

        # self.evaluate_silhouette_scores(projection, k_values=range(2, 50, 2))

        # Plot the projections
        self.plot_projection(projection, dtrajs, centroids)
        # self.plot_individual_projections(projection)
    
        if self.write_pdbs:
            self.write_centroids_pdb(projection, centroids, N=1)

        return