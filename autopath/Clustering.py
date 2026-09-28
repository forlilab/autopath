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
