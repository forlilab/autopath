import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import kmedoids
from dtaidistance import dtw_ndim

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from .SMDData import SMDData
from .Diagnostics import make_unbinding_paths_pml
from abc import ABC, abstractmethod

import logging
logger = logging.getLogger("autopath")

class PathModel(ABC):
    """Abstract base class for path models in sMDAnalysis."""
    def __init__(self):
        pass

    @abstractmethod
    def fit_transform(self, feature_df: pd.DataFrame) -> SMDData:
        """Fit the path model to the provided SMD data."""
        pass


class NullPathModel(PathModel):
    """Assigns all trajectories to a single path, bypassing clustering entirely.

    Useful for isolating the effect of clustering on downstream results.
    All trajectories receive path label ``'path-0'``.
    """

    def fit_transform(self, feature_df: pd.DataFrame, **kwargs) -> dict:
        trajnames = feature_df['trajname'].unique().tolist()
        logger.info(
            f"NullPathModel: assigning all {len(trajnames)} trajectories to 'path-0' "
            "(no clustering performed)."
        )
        return {name: "path-0" for name in trajnames}


class DTWPathModel(PathModel):
    
    """
    Cluster trajectories using Dynamic Time Warping (DTW) and k-medoids.
    For more information on DTW see: https://doi.org/10.1073/pnas.231354212
                                     https://dtaidistance.readthedocs.io/en/latest/index.html
    """
    def __init__(self,
                 seed: int = 42,
                 do_plots: bool = True,
                 outdir: str = 'clustering_results',
                 n_geom_pcs: int | None = 3,
                 geom_feature_prefix: str = 'dist_',
                 ):

        self.seed = seed
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.do_plots = do_plots
        self._use_silhouette = True
        self.n_geom_pcs = n_geom_pcs
        self.geom_feature_prefix = geom_feature_prefix

        return None
    
    def fit_transform(self,
                    feature_df: pd.DataFrame,
                    r_range: tuple = None,  # e.g., (0, 1.5)
                    n_paths: int = None,
                    cluster_across_speeds: bool = False,
                    reference_pdb: str = None,
                    ligand_select: str = None,
                    pocket_select: str = None,
                    trajectory_files: dict = None,
                    ) -> dict:
        """
        Fit and transform trajectory clustering.

        Parameters
        ----------
        r_range : tuple or float, optional
            If a tuple (low, high), filter trajectories to that r_coord range before clustering.
            If a float in (0, 1], use the first r_range fraction of each trajectory for
            clustering (e.g. 0.8 uses the first 80% of frames). Plots always reflect the
            full trajectory regardless.
        cluster_across_speeds : bool
            If True, cluster all trajectories together regardless of speed.
            If False (default), cluster trajectories independently per speed.
        reference_pdb : str, optional
            Path to reference PDB file for unbinding path visualization.
        ligand_select : str, optional
            MDAnalysis selection string for ligand atoms.
        trajectory_files : dict, optional
            Dictionary mapping trajectory names to (topology, trajectory) tuples
            for unbinding path visualization.
        """
        
        # r-range filtering. You may want to cluster only around the TS region
        cluster_fraction = None
        if r_range is not None:
            if isinstance(r_range, float):
                cluster_fraction = r_range
                logger.warning(
                    f'Clustering will use the first {cluster_fraction*100:.0f}% of frames per trajectory. '
                    f'Plots will reflect the full trajectory.'
                )
            else:
                low, high = map(float, r_range)
                logger.warning(
                    f'Filtering trajectories to r_target in [{low}, {high}] for clustering.'
                )
                feature_df = feature_df[
                    (feature_df['r_coord'] >= low) &
                    (feature_df['r_coord'] <= high)
                ]

        feature_cols = [
            c for c in feature_df.columns
            if c not in ['trajname', 'time', 'step', 'speed', 'path']
        ]
        
        #IDK why this happens but sometimes we get NaN values in the features.
        # Warn and drop those rows if present.
        logger.info(f'Features used for clustering: {feature_cols}')
        if feature_df[feature_cols].isnull().any().any():
            logger.warning(
            "NaN values detected in features. "
            "DTW distance matrix will be unreliable. Please check your data."
            )
            #drop rows with NaN values in feature columns
            feature_df = feature_df.dropna(subset=feature_cols)
                                 
        logger.info(f"Feature matrix shape after NaN removal: {feature_df.shape}")
              
        # ecide grouping strategy
        if cluster_across_speeds:
            grouping_iter = [(None, feature_df)]
        else:
            grouping_iter = feature_df.groupby("speed")

        all_path_mappings = {}
        all_medoid_names = set()  # Track medoids across all speeds
        medoid_to_path = {}  # Map medoid trajectory name -> path ID

        for speed_key, speed_df in grouping_iter:
            speed_df_full = speed_df

            if cluster_fraction is not None:
                speed_df_cluster = (
                    speed_df_full
                    .groupby('trajname', group_keys=False)
                    .apply(lambda df: df.sort_values('step').iloc[:max(1, int(np.floor(cluster_fraction * len(df))))])
                )
            else:
                speed_df_cluster = speed_df_full

            # Build per-trajectory arrays (required by dtw_ndim)
            data_struct = {}
            for trajname, traj_df in speed_df_cluster.groupby("trajname"):
                traj_df = traj_df.sort_values('step')
                if traj_df.shape[0] < 2:
                    logger.error(
                        f"Skipping trajectory {trajname} due to insufficient data points."
                    )
                    continue
                data_struct[trajname] = traj_df[feature_cols].to_numpy()

            if len(data_struct) < 2:
                logger.warning(
                    f"Not enough trajectories to cluster "
                    f"{'across all speeds' if speed_key is None else f'at speed={speed_key}'}."
                )
                continue

            trajnames = list(data_struct.keys())
            vectors_stacked = [data_struct[name] for name in trajnames]

            # Scale features across all frames / trajectories
            scaler = StandardScaler(with_mean=True, with_std=True)
            scaler.fit(np.vstack(vectors_stacked))
            vectors_stacked_scaled = [
                scaler.transform(arr) for arr in vectors_stacked
            ]

            # PCA reduction of geometric features to equalize contribution
            # with trace features when both are present (merged feature mode).
            geom_idx = [
                i for i, c in enumerate(feature_cols)
                if c.startswith(self.geom_feature_prefix)
            ]
            other_idx = [
                i for i, c in enumerate(feature_cols)
                if not c.startswith(self.geom_feature_prefix)
            ]
            if geom_idx and other_idx and self.n_geom_pcs is not None:
                n_components = min(self.n_geom_pcs, len(geom_idx))
                logger.info(
                    f"Applying PCA to {len(geom_idx)} geometric features "
                    f"(prefix='{self.geom_feature_prefix}') → {n_components} components."
                )
                X_geom_all = np.vstack([arr[:, geom_idx] for arr in vectors_stacked_scaled])
                pca_geom = PCA(n_components=n_components)
                pca_geom.fit(X_geom_all)
                vectors_stacked_scaled = [
                    np.hstack([
                        pca_geom.transform(arr[:, geom_idx]),
                        arr[:, other_idx],
                    ])
                    for arr in vectors_stacked_scaled
                ]
                logger.info(
                    f"Geometric PCA explained variance: "
                    f"{pca_geom.explained_variance_ratio_.sum()*100:.1f}%"
                )

            # DTW distance matrix
            distmatrix = dtw_ndim.distance_matrix_fast(
                s=vectors_stacked_scaled
            )

            #choose number of clusters
            if n_paths is None:
                K_MAX = min(6, len(trajnames))
                if K_MAX < 2:
                    raise RuntimeError(
                        "Not enough trajectories to form at least 2 clusters."
                    )

                scores = {}
                for k in range(2, K_MAX):
                    c = kmedoids.fasterpam(
                        distmatrix, k, random_state=self.seed
                    )
                    if self._use_silhouette:
                        scores[k] = silhouette_score(
                            distmatrix, c.labels, metric="precomputed"
                        )
                    else:
                        scores[k] = -c.loss

                K = max(scores, key=scores.get)
                logger.info(
                    f"{'Global' if speed_key is None else f'Speed {speed_key}'}: "
                    f"Found {K} paths with score {scores[K]:.2f}"
                )
            else:
                K = n_paths
                scores = {}
                logger.info(
                    f"{'Global' if speed_key is None else f'Speed {speed_key}'}: "
                    f"Using user-specified number of paths: K={K}"
                )

            # Final clustering
            cluster_model = kmedoids.fasterpam(
                distmatrix, K, random_state=self.seed
            )

            self.medoid_names = [trajnames[idx] for idx in cluster_model.medoids]
            all_medoid_names.update(self.medoid_names)  # Accumulate medoids across speeds

            # Record medoid → path mapping
            for medoid_idx, cluster_id in zip(cluster_model.medoids, range(K)):
                medoid_name = trajnames[medoid_idx]
                if speed_key is not None:
                    path_id = f"path-{cluster_id}_v{speed_key}"
                else:
                    path_id = f"path-{cluster_id}"
                medoid_to_path[medoid_name] = path_id

            # # Report cluster sizes
            # unique, counts = np.unique(cluster_model.labels, return_counts=True)
            # for u, c in zip(unique, counts):
            #     logger.info(
            #         f"{'Global' if speed_key is None else f'Speed {speed_key}'} "
            #         f"Path {u} has {c} trajectories"
            #     )

            # Map cluster labels with speed-aware path IDs to ensure uniqueness across speeds
            path_mapping_dic = {}
            for trajname, cluster_id in zip(trajnames, cluster_model.labels):
                # Create globally unique path ID: include speed if clustering per speed
                if speed_key is not None:
                    unique_path_id = f"path-{cluster_id}_v{speed_key}"
                else:
                    unique_path_id = cluster_id
                path_mapping_dic[trajname] = unique_path_id

            all_path_mappings.update(path_mapping_dic)
            
            # Optional plots (per clustering run)
            if self.do_plots:
                self.speed_name = speed_key if speed_key is not None else 'Global'
                self.plot_distance_matrix(distmatrix)
                self.plot_elbow(scores, K)

                if cluster_fraction is not None:
                    # Rebuild scaled vectors over the full trajectories for plotting
                    vectors_full = [
                        speed_df_full[speed_df_full['trajname'] == name]
                        .sort_values('step')[feature_cols].to_numpy()
                        for name in trajnames
                    ]
                    vectors_plot = [scaler.transform(arr) for arr in vectors_full]
                    if geom_idx and other_idx and self.n_geom_pcs is not None:
                        vectors_plot = [
                            np.hstack([
                                pca_geom.transform(arr[:, geom_idx]),
                                arr[:, other_idx],
                            ])
                            for arr in vectors_plot
                        ]
                    speed_df_for_plot = speed_df_full[speed_df_full['trajname'].isin(trajnames)].copy()
                else:
                    vectors_plot = vectors_stacked_scaled
                    speed_df_for_plot = speed_df_full

                self.plot_clusters_PCA(
                    feature_df=speed_df_for_plot,
                    path_mapping_dic=path_mapping_dic,
                    vectors_stacked_scaled=vectors_plot,
                )

        # Generate unbinding paths visualization if trajectories and reference PDB are available
        if self.do_plots and trajectory_files is not None and reference_pdb is not None and ligand_select is not None:
            # Build paths dictionary: path_id -> [(topology, trajectory), ...] 
            # Only include medoid trajectories (one representative per path)
            paths_dict = {}
            for trajname, path_id in all_path_mappings.items():
                # Only include if this trajectory is a medoid
                if trajname not in all_medoid_names:
                    continue
                if path_id not in paths_dict:
                    paths_dict[path_id] = []
                if trajname in trajectory_files:
                    paths_dict[path_id].append(trajectory_files[trajname])
            
            if paths_dict:
                try:
                    make_unbinding_paths_pml(
                        paths=paths_dict,
                        reference_pdb=reference_pdb,
                        ligand_select=ligand_select,
                        outdir=os.path.join(self.outdir, "unbinding_paths"),
                        pocket_select=pocket_select
                    )
                    logger.info(f"Unbinding paths visualization generated in {self.outdir}")
                except Exception as e:
                    logger.warning(f"Could not generate unbinding paths visualization: {e}")

        # Persist medoid information as instance attributes
        self.all_medoid_names = list(all_medoid_names)
        self.medoid_to_path = medoid_to_path

        # Save medoid info to disk for downstream use (e.g., milestone extraction)
        medoid_info = {
            "medoid_names": self.all_medoid_names,
            "medoid_to_path": self.medoid_to_path,
        }
        medoid_info_path = os.path.join(self.outdir, "medoid_info.json")
        with open(medoid_info_path, "w") as f:
            json.dump(medoid_info, f, indent=2)
        logger.info(f"Saved medoid info to {medoid_info_path}")

        return all_path_mappings
    
    def plot_distance_matrix(self, distmatrix:np.ndarray=None):
        plt.figure(figsize=(8,6))
        sns.heatmap(distmatrix, cmap='viridis')
        plt.title('DTW Distance Matrix');         plt.xlabel('Trajectories')
        plt.ylabel('Trajectories')
        plt.savefig(os.path.join(self.outdir, f'cluster_dtw_heatmap_v{self.speed_name}.png'))
        plt.tight_layout()
        plt.close()
        return
    
    def plot_elbow(self, scores: dict, K: int):
        plt.figure(figsize=(6, 5))
        K_values = [int(k) for k in scores.keys()]
        sns.lineplot(x=K_values, y=[scores[k] for k in K_values])
        plt.title(f"Optimal number of paths: {K}")
        plt.axvline(x=K, color='red', linestyle='--', label=f'Optimal K={K}')
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score" if self._use_silhouette else "Score")
        plt.xticks(K_values)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"cluster_elbowplot_v{self.speed_name}.png"))
        plt.close()
        return
        
    def plot_clusters_PCA(self, 
                          feature_df: pd.DataFrame=None,
                          path_mapping_dic: dict=None,
                          vectors_stacked_scaled: list=None,
                          ):
        

        feature_df['path'] = feature_df['trajname'].map(path_mapping_dic)
        
        # generate PCA plot of the clustered paths
        # Get the minimum number of frames across all trajectories
        min_len = min(arr.shape[0] for arr in vectors_stacked_scaled)

        # Trim all arrays to this length, required for PCA
        vectors_trimmed = [arr[:min_len, :] for arr in vectors_stacked_scaled]
        X = np.vstack(vectors_trimmed)   # shape (N_traj * min_len, d)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # figure out trajectory order matching vectors_stacked_scaled
        # groupby preserves the order of appearance of trajname in feature_df,
        # which is what cluster_time_series used when building vectors_stacked_scaled
        traj_order = [name for name, _ in feature_df.groupby('trajname')]
        n_traj = len(traj_order)
        assert n_traj == len(vectors_trimmed), "traj_order and vectors_stacked_scaled misaligned"

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                        alpha=0.75, color='lightgray', s=50, linewidth=0)

        # color palette by path label
        n_paths = len(set(path_mapping_dic.values()))
        # palette = sns.color_palette("tab10", n_colors=n_paths)

        # overlay medoid trajectories, colored by path
        used_labels = set()
        for i, trajname in enumerate(traj_order):
            if trajname not in self.medoid_names:
                continue

            start = i * min_len
            end = start + min_len
            path_label = path_mapping_dic[trajname]
            # color = palette[path_label]

            label = f'{path_label.split("_")[0]}'
            # label = f'path-{path_label}'#_{trajname}'

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

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title(f"PCA space - speed={self.speed_name} nm/ps")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"cluster_pca_v{self.speed_name}.png"))
        plt.close()
        return