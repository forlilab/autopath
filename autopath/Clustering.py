"""Frame clustering and representative-frame selection.

Clusters frames WITHIN trajectories. Contrast pulling/PathModel.py, which
clusters whole trajectories by shape into unbinding routes.
"""
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd as _mda_rmsd

logger = logging.getLogger("autopath")


class Featurizer(ABC):
    """Per-frame feature matrix, shape (n_frames, n_features)."""

    @abstractmethod
    def compute(self, u: mda.Universe) -> np.ndarray:
        ...


class PairwiseDistances(Featurizer):
    """Flattened ligand-to-pocket pairwise distances per frame."""

    def __init__(self, ligand_sel: str, pocket_sel: str, stride: int = 1):
        self.ligand_sel, self.pocket_sel, self.stride = ligand_sel, pocket_sel, stride

    def compute(self, u):
        from autopath.utils import compute_distance_features
        return compute_distance_features(u, self.ligand_sel, self.pocket_sel,
                                         stride=self.stride)


class CoordinateRMSD(Featurizer):
    """Per-frame RMSD over a selection, against `ref` or the first frame."""

    def __init__(self, sel: str, ref: mda.Universe | None = None, stride: int = 1):
        self.sel, self.ref, self.stride = sel, ref, stride

    def compute(self, u):
        ag = u.select_atoms(self.sel)
        if self.ref is None:
            u.trajectory[0]
            ref_pos = ag.positions.copy()
        else:
            ref_pos = self.ref.select_atoms(self.sel).positions.copy()
        out = []
        for ts in u.trajectory[::self.stride]:
            out.append(_mda_rmsd(ag.positions, ref_pos, superposition=True))
        return np.asarray(out, dtype=float).reshape(-1, 1)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Clustering:
    """Feature -> reduce -> select representative frames -> write.

    fit/select/plot need no OpenMM; only write_states does.
    """

    def __init__(self, featurizer: Featurizer, n_components: int = 2,
                 scale: bool = True, seed: int = 42):
        self.featurizer = featurizer
        self.n_components = n_components
        self.scale = scale
        self.seed = seed
        self.X = None
        self.X_reduced = None
        self.explained_variance_ratio_ = None
        self.labels_ = None
        self.centers_ = None
        self._pca = None

    def fit(self, u):
        X = self.featurizer.compute(u)
        if X.ndim != 2:
            raise ValueError("featurizer must return a 2D (n_frames, n_features) array")
        self.X = X
        Xs = StandardScaler().fit_transform(X) if self.scale else X
        # A featurizer may return fewer features than n_components (CoordinateRMSD
        # returns one), and PCA refuses n_components > n_features.
        n_comp = min(self.n_components, Xs.shape[1])
        if n_comp < self.n_components:
            logger.warning("n_components=%d exceeds the %d available features; using %d",
                           self.n_components, Xs.shape[1], n_comp)
        pca = PCA(n_components=n_comp, random_state=self.seed)
        self.X_reduced = pca.fit_transform(Xs)
        self._pca = pca
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        logger.info("PCA: explained variance ratio = %s",
                    [f"{v*100:.1f}%" for v in pca.explained_variance_ratio_])
        return self

    def _require_fit(self):
        if self.X_reduced is None:
            raise RuntimeError("call fit() before select()/plot()")
