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
from .Diagnostics import make_unbinding_paths_visualization
from abc import ABC, abstractmethod

import logging
logger = logging.getLogger("autopath.PathModel")

class PathModel(ABC):
    """Abstract base class for path models in pulling."""
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
                 outdir: str = 'path_analysis',
                 n_geom_pcs: int | None = 4,
                 geom_feature_prefix: str | tuple[str, ...] = ('dist_', 'geom_'),
                 pca_all_features: bool = True,
                 ):

        self.seed = seed
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.do_plots = do_plots
        self._use_silhouette = True
        self.n_geom_pcs = n_geom_pcs
        # Prefix(es) whose columns form the geometric block that is PCA-reduced
        # before DTW (pocket distances dist_* and inline geom features geom_*).
        # str.startswith accepts a tuple, so multiple prefixes work directly.
        self.geom_feature_prefix = geom_feature_prefix
        # When True (default), PCA the ENTIRE standardized feature matrix together
        # (trace + geom) into n_geom_pcs components, rather than only the
        # geom-prefixed block with trace passed through raw. This is the hybrid
        # default; set False to keep trace raw and PCA only the geom block.
        self.pca_all_features = pca_all_features
    
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
            If False, cluster trajectories independently per speed.
        reference_pdb : str, optional
            Path to reference PDB file for unbinding path visualization.
        ligand_select : str, optional
            MDAnalysis selection string for ligand atoms.
        trajectory_files : dict, optional
            Dictionary mapping trajectory names to (topology, trajectory) tuples
            for unbinding path visualization.

        Returns
        -------
        dict
            Mapping of trajectory name → path label (e.g. ``"path-0_v0.001"``).
            ``paths_`` in downstream SMDData contains only medoid trajectories
            (one representative per cluster), not all trajectories.

        Notes
        -----
        Feature handling (merged feature mode):
            When both geometric features (columns starting with
            ``geom_feature_prefix``) and trace features are present,
            a PCA is applied to the geometric features *before* DTW distance
            computation, reducing them to ``n_geom_pcs`` principal components.
            This equalises their dimensionality contribution relative to the
            trace features; without it, large numbers of pairwise-distance
            columns would dominate the DTW metric.
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
        
        # NaN values occasionally appear in DTW feature matrices, likely from
        # trajectory frames with undefined geometry (e.g., PBC artifacts or
        # extreme conformations). Drop them to avoid distance computation failures.
        logger.info(f'Features used for clustering: {feature_cols}')
        if feature_df[feature_cols].isnull().any().any():
            logger.warning(
            "NaN values detected in features. "
            "DTW distance matrix will be unreliable. Please check your data."
            )
            #drop rows with NaN values in feature columns
            feature_df = feature_df.dropna(subset=feature_cols)
                                 
        # logger.info(f"Feature matrix shape after NaN removal: {feature_df.shape}")
              
        # Decide grouping strategy
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

            # Equalize geom vs trace feature contribution (see fit_transform Notes).
            # PCA is applied whenever geometric (distance) features are present and
            # n_geom_pcs is set — both in merged mode (geom + trace) and in
            # distance-only mode (geom only, other_idx empty). An empty other_idx
            # makes arr[:, other_idx] shape (n, 0), which np.hstack absorbs cleanly.
            if self.pca_all_features:
                # PCA the entire standardized feature matrix together (trace + geom).
                geom_idx = list(range(len(feature_cols)))
                other_idx = []
            else:
                geom_idx = [
                    i for i, c in enumerate(feature_cols)
                    if c.startswith(self.geom_feature_prefix)
                ]
                other_idx = [
                    i for i, c in enumerate(feature_cols)
                    if not c.startswith(self.geom_feature_prefix)
                ]
            if geom_idx and self.n_geom_pcs is not None:
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

            speed_medoid_names = [trajnames[idx] for idx in cluster_model.medoids]
            all_medoid_names.update(speed_medoid_names)

            # Map every trajectory to its speed-aware path ID
            path_mapping_dic = {
                trajname: (
                    f"path-{cluster_id}_v{speed_key}"
                    if speed_key is not None
                    else f"path-{cluster_id}"
                )
                for trajname, cluster_id in zip(trajnames, cluster_model.labels)
            }
            all_path_mappings.update(path_mapping_dic)

            # Medoid → path is just the subset of path_mapping_dic for medoid trajectories
            medoid_to_path.update({name: path_mapping_dic[name] for name in speed_medoid_names})

            # Set per-speed attribute used by plot_clusters_PCA below
            self.medoid_names = speed_medoid_names
            
            # Optional plots (per clustering run)
            if self.do_plots:
                self.speed_name = speed_key if speed_key is not None else 'Global'
                self.plot_distance_matrix(distmatrix)
                self.plot_elbow(scores, K)

                # Build plot-space feature names — geom features are PCA-reduced
                # so their names become geom_PC1, geom_PC2, …
                if geom_idx and self.n_geom_pcs is not None:
                    n_pcs = min(self.n_geom_pcs, len(geom_idx))
                    plot_feature_names = (
                        [f"geom_PC{k+1}" for k in range(n_pcs)]
                        + [feature_cols[i] for i in other_idx]
                    )
                else:
                    plot_feature_names = feature_cols

                if cluster_fraction is not None:
                    # Rebuild scaled vectors over the full trajectories for plotting
                    vectors_full = [
                        speed_df_full[speed_df_full['trajname'] == name]
                        .sort_values('step')[feature_cols].to_numpy()
                        for name in trajnames
                    ]
                    vectors_plot = [scaler.transform(arr) for arr in vectors_full]
                    if geom_idx and self.n_geom_pcs is not None:
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

                # When the geom PCA already reduced the whole vector (pca_all),
                # the plot must NOT PCA again — pass the fitted PCA so the scatter
                # uses its scores directly and loadings map the ORIGINAL features.
                _plot_pca = (pca_geom if (self.pca_all_features and geom_idx
                                          and self.n_geom_pcs is not None) else None)
                self.plot_clusters_PCA(
                    feature_df=speed_df_for_plot,
                    path_mapping_dic=path_mapping_dic,
                    vectors_stacked_scaled=vectors_plot,
                    feature_names=plot_feature_names,
                    pca_model=_plot_pca,
                    pca_feature_names=feature_cols,
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
                    make_unbinding_paths_visualization(
                        paths=paths_dict,
                        reference_pdb=reference_pdb,
                        ligand_select=ligand_select,
                        outdir=os.path.join(self.outdir, "path_analysis"),
                        pocket_select=pocket_select
                    )
                    logger.info(f"Unbinding paths visualization generated in {self.outdir}")
                except Exception as e:
                    logger.warning(f"Could not generate unbinding paths visualization: {e}")

        # Persist medoid information as instance attributes (all speeds)
        self.all_medoid_names = list(all_medoid_names)
        self.medoid_names = self.all_medoid_names
        self.medoid_to_path = medoid_to_path

        # Save medoid info to disk for downstream use (e.g., milestone extraction)
        medoid_info = {
            "medoid_names": self.all_medoid_names,
            "medoid_to_path": self.medoid_to_path,
        }
        medoid_info_path = os.path.join(self.outdir, "path_analysis", "medoid_info.json")
        with open(medoid_info_path, "w") as f:
            json.dump(medoid_info, f, indent=2)
        logger.info(f"Saved medoid info to {medoid_info_path}")

        return all_path_mappings
    
    def plot_distance_matrix(self, distmatrix: np.ndarray = None):
        """Plot and save the pairwise DTW distance matrix as a heatmap.

        Parameters
        ----------
        distmatrix : np.ndarray
            Square (N_traj × N_traj) DTW distance matrix.
        """
        plt.figure(figsize=(8,6))
        sns.heatmap(distmatrix, cmap='viridis')
        plt.title('DTW Distance Matrix');         plt.xlabel('Trajectories')
        plt.ylabel('Trajectories')
        plt.savefig(os.path.join(self.outdir, "path_analysis", f'cluster_dtw_heatmap_v{self.speed_name}.png'))
        plt.tight_layout()
        plt.close()
        return
    
    def plot_elbow(self, scores: dict, K: int):
        """Plot and save the elbow/silhouette curve used to select the number of clusters.

        Parameters
        ----------
        scores : dict
            Mapping of k (number of clusters) → silhouette score (or negative
            loss when silhouette is disabled).
        K : int
            The selected optimal k, marked with a vertical red dashed line.
        """
        plt.figure(figsize=(6, 5))
        K_values = [int(k) for k in scores.keys()]
        sns.lineplot(x=K_values, y=[scores[k] for k in K_values])
        plt.title(f"Optimal number of paths: {K}")
        plt.axvline(x=K, color='red', linestyle='--', label=f'Optimal K={K}')
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score" if self._use_silhouette else "Score")
        plt.xticks(K_values)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, "path_analysis", f"cluster_elbowplot_v{self.speed_name}.png"))
        plt.close()
        return
        
    def plot_clusters_PCA(self,
                          feature_df: pd.DataFrame = None,
                          path_mapping_dic: dict = None,
                          vectors_stacked_scaled: list = None,
                          feature_names: list = None,
                          pca_model=None,
                          pca_feature_names: list = None,
                          ):
        """Project scaled trajectory features onto 2-D PCA and colour by cluster.

        Only medoid trajectories are coloured; all others are shown in light
        gray. Trajectory vectors are truncated to the minimum trajectory length
        before stacking, so the PCA is well-defined even when trajectories differ
        in length. A feature-loadings bar chart is added as a second panel when
        ``feature_names`` is provided.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Full (untruncated) feature DataFrame for the current speed group.
        path_mapping_dic : dict
            Mapping of trajectory name → path label.
        vectors_stacked_scaled : list of np.ndarray
            Scaled (and optionally PCA-reduced) feature arrays, one per trajectory.
        feature_names : list of str, optional
            Column names used for the loadings panel; if ``None``, the loadings
            panel is omitted.
        """
        feature_df['path'] = feature_df['trajname'].map(path_mapping_dic)

        min_len = min(arr.shape[0] for arr in vectors_stacked_scaled)
        vectors_trimmed = [arr[:min_len, :] for arr in vectors_stacked_scaled]
        X = np.vstack(vectors_trimmed)   # (N_traj * min_len, d)

        traj_order = [name for name, _ in feature_df.groupby('trajname')]
        if len(traj_order) != len(vectors_trimmed):
            raise ValueError(
                f"Trajectory order ({len(traj_order)}) and feature vectors "
                f"({len(vectors_trimmed)}) are misaligned — check that all "
                f"trajectories in feature_df are present in vectors_stacked_scaled."
            )

        # Avoid a PCA-of-PCA: when the clustering vectors are already PCA scores
        # (pca_model provided), plot the first two components directly and take the
        # loadings from that PCA over the ORIGINAL features. Otherwise reduce here.
        if pca_model is not None and X.shape[1] >= 2:
            X2 = X[:, :2]
            evr = pca_model.explained_variance_ratio_
            load1, load2 = pca_model.components_[0], pca_model.components_[1]
            load_names = pca_feature_names if pca_feature_names is not None else feature_names
        else:
            _pca = PCA(n_components=2)
            X2 = _pca.fit_transform(X)
            evr = _pca.explained_variance_ratio_
            load1, load2 = _pca.components_[0], _pca.components_[1]
            load_names = feature_names

        has_loadings = load_names is not None
        fig, axes = plt.subplots(1, 2 if has_loadings else 1,
                                 figsize=(12, 5) if has_loadings else (6, 5))
        ax_scatter = axes[0] if has_loadings else axes

        ax_scatter.scatter(X2[:, 0], X2[:, 1],
                           alpha=0.75, color='lightgray', s=50, linewidth=0)

        used_labels = set()
        for i, trajname in enumerate(traj_order):
            if trajname not in self.medoid_names:
                continue
            start = i * min_len
            end   = start + min_len
            path_label = path_mapping_dic[trajname]
            label = f'{str(path_label).split("_")[0]}' if path_label not in used_labels else None
            used_labels.add(path_label)
            ax_scatter.scatter(X2[start:end, 0], X2[start:end, 1],
                               s=20, alpha=0.9, label=label)

        if used_labels:
            ax_scatter.legend(frameon=False)
        ax_scatter.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        ax_scatter.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        ax_scatter.set_title(f"PCA space - speed={self.speed_name} nm/ps")

        # ── Feature loadings panel ─────────────────────────────────────────
        if has_loadings:
            ax_load = axes[1]
            n_feats = len(load_names)
            y = np.arange(n_feats)
            w = 0.35
            ax_load.barh(y + w / 2, load1[:n_feats], w,
                         label=f"PC1 ({evr[0]*100:.1f}%)")
            ax_load.barh(y - w / 2, load2[:n_feats], w,
                         label=f"PC2 ({evr[1]*100:.1f}%)")
            ax_load.set_yticks(y)
            ax_load.set_yticklabels(load_names)
            ax_load.axvline(0, color='black', lw=0.8)
            ax_load.set_xlabel("Loadings")
            ax_load.set_title("Feature contributions to PCs")
            ax_load.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, "path_analysis", f"cluster_pca_v{self.speed_name}.png"), dpi=150)
        plt.close()
        return