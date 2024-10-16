import os
import logging
import numpy as np
import pandas as pd
from glob import glob
from typing import Union, List

import matplotlib.pyplot as plt
import seaborn as sns

from deeptime.decomposition import VAMP, KVAD
from deeptime.clustering import KMeans
from deeptime.kernels import GaussianKernel

import MDAnalysis as mda
import pyemma

from autopath.utils import save_model, load_model

class ClusterTrajectories:
    def __init__(self, 
                prmtop_file:str=None,
                traj_ref:str=None, 
                traj_list:List[str]=None,
                ligand_atoms_indexes:List[int]=None,
                pocket_atom_indexes:List[int]=None,
                out_dir:str='milestones',
                seed:int=42
                ):
        self.prmtop_file = prmtop_file
        self.traj_ref = traj_ref
        self.traj_list = traj_list
        self.ligand_atoms_indexes = ligand_atoms_indexes
        self.pocket_atom_indexes = pocket_atom_indexes

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.seed = seed
        np.random.seed(seed)

        return
    
    def get_raw_data(self):

        data_fname = f'{self.out_dir}/features_raw.npy'

        if os.path.exists(data_fname):
            logging.info(f'Loading precomputed features from {data_fname}')
            return np.load(data_fname)
        else:
            data = self.compute_features()
            np.save(data_fname, data)
            return data

    def compute_features(self):
        logging.info('Computing features..')

        featurizer = pyemma.coordinates.featurizer(self.prmtop_file)
        # featurizer.add_minrmsd_to_ref(self.traj_ref, ref_frame=0, atom_indices=self.ligand_atoms_indexes)
        # featurizer.add_group_COM([self.pocket_atom_indexes,self.ligand_atoms_indexes], mass_weighted=True)
        featurizer.add_distances(indices=self.pocket_atom_indexes)
        
        data = pyemma.coordinates.load(self.traj_list, featurizer)

        return data

    @staticmethod
    def fit_vamp_model(data, lagtime, embedding_dim):
        
        if isinstance(embedding_dim, float):
            var_cutoff = embedding_dim
            dim = None
        else:
            var_cutoff = None
            dim = embedding_dim

        vamp_estimator = VAMP(lagtime=lagtime, dim=dim, var_cutoff=var_cutoff)
        vamp_model = vamp_estimator.fit(data).fetch_model()

        return vamp_model

    @staticmethod
    def fit_kvad_model(data, lagtime, embedding_dim):

        logging.info('Fitting KVAD model..')
        kvad_estimator = KVAD(kernel=GaussianKernel(1.),
            lagtime=lagtime, epsilon=1e-5, dim=embedding_dim,
            # observable_transform=ChiRnd()
            )
        kvad_model = kvad_estimator.fit(data).fetch_model()

        return kvad_model

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

    @staticmethod
    def kmeans_clustering(projection_concatenated, n_clusters=100):

        kmeans_estimator = KMeans(
            n_clusters=n_clusters,  # place 100 cluster centers
            init_strategy='kmeans++',  # uniform initialization strategy
            # max_iter=0,  # don't actually perform the optimization, just place centers
            fixed_seed=42,
            n_jobs=None,
        )

        kmeans_model = kmeans_estimator.fit(projection_concatenated).fetch_model()

        return kmeans_model

    def plot_projection(self, projection, centroids, n_clusters):
        pyemma.plots.plot_density(*projection.T, alpha=0.2)
        # plot_density(*projection.T, contourf_kws={'norm':'logit'})
        plt.scatter(*(centroids.T), s=15, c='C1')
        plt.xlabel('comp 1')
        plt.ylabel('comp 2')
        plt.title(f'k = {n_clusters} centers')
        plt.savefig(f'{self.out_dir}/projection_{n_clusters}-clusters.png')
        plt.close()
        return

    def find_closest_points(self, X, centroids):
        closest_points = []
        for centroid in centroids:
            distances = np.linalg.norm(X - centroid, axis=1)
            closest_point_index = np.argmin(distances)
            closest_points.append(closest_point_index)
        return closest_points
  
    def write_centroids_pdb(self, closest_points_indices, projection):

        # This is a bit tricky, but we need to map the indices of the concatenated projection back to the original runs
        index_mapping = np.concatenate([np.arange(proj.shape[0]) for proj in projection])
        indexes = index_mapping[closest_points_indices]

        traj_lengths = [len(mda.Universe(self.prmtop_file, traj).trajectory) for traj in self.traj_list]
        cumulative_lengths = np.cumsum(traj_lengths)

        for index in indexes:
            try:
                traj_idx = np.searchsorted(cumulative_lengths, index, side='right')
                if traj_idx == 0:
                    frame_idx = index
                else:
                    frame_idx = index - cumulative_lengths[traj_idx - 1]

                traj = self.traj_list[traj_idx]
                u = mda.Universe(self.prmtop_file, traj)
                u.trajectory[frame_idx]
                u.atoms.write(f"{self.out_dir}/milestone_{index}.pdb")
            except Exception as e:
                logging.error(f'Error writing pdb for index {index} in traj {traj_idx}\n{e}')
        return

    def run_vamp_cv(self, 
                    dims:List[int], 
                    lags:List[int], 
                    n_splits:int=10, 
                    val_frac:float=0.5
                    ):

        data = self.get_raw_data()

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
                embedding_dim:Union[int, float]=2, 
                lagtime:int=20,
                n_clusters:int=100,
                write_pdb:bool=False,
                plot_projection:bool=False
                ):

        data = self.get_raw_data()

        # vamp_model = self.fit_vamp_model(data, lagtime, embedding_dim)
        # save_model(vamp_model, f'{self.out_dir}/vamp_model.pkl')
        # projection = vamp_model.transform(data)
        kvar_model = self.fit_kvad_model(data, lagtime, embedding_dim)
        save_model(kvar_model, f'{self.out_dir}/kvar_model.pkl')
        projection = kvar_model.transform(data)

        np.save(f'{self.out_dir}/projection_kvar.npy', projection)
        projection_concatenated = np.concatenate(projection, axis=0)

        kmeans_model = self.kmeans_clustering(projection_concatenated, n_clusters)
        
        kmeans_labels = [kmeans_model.transform(run) for run in projection] #Transform each run to cluster labels separately
        kmeans_labels_concatenated = np.concatenate(kmeans_labels, axis=0)
        kmeans_centroids = kmeans_model.cluster_centers
        
        if write_pdb:
            closest_points_indices = self.find_closest_points(projection_concatenated, kmeans_centroids)
            self.write_centroids_pdb(closest_points_indices, projection)
            
        if plot_projection:
            self.plot_projection(projection_concatenated, kmeans_centroids, n_clusters)

        return