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
import openmm as mm
from openmm import app, unit as ommunit

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
from deeptime.clustering import RegularSpace

_STRATEGIES = ("centroid", "time_spaced")


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
        self._stride = 1

    def fit(self, u):
        # Feature row i is trajectory frame i*stride.
        self._stride = max(1, int(getattr(self.featurizer, "stride", 1) or 1))
        X = self.featurizer.compute(u)
        if X.ndim != 2:
            raise ValueError("featurizer must return a 2D (n_frames, n_features) array")
        self.X = X
        Xs = StandardScaler().fit_transform(X) if self.scale else X
        # PCA refuses n_components > n_features.
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

    def select(self, n: int, strategy: str, min_frame_separation: int = 0):
        """Representative trajectory frame indices (already de-strided).

        strategy="centroid"    RegularSpace along PC1 -> nearest real frame.
                               Diverse starting points (milestones, exploration).
        strategy="time_spaced" evenly spaced in time. Population-weighted, the
                               correct choice when seeding an ensemble average.

        `min_frame_separation` counts featurized rows, i.e. strided frames.
        """
        self._require_fit()
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
        n_frames = self.X_reduced.shape[0]
        if n > n_frames:
            raise ValueError(f"requested {n} frames but only {n_frames} are available")

        # Clustering state is only meaningful for a centroid run.
        self.labels_ = None
        self.centers_ = None

        if strategy == "time_spaced":
            frames = np.unique(np.linspace(0, n_frames - 1, n).astype(int))
            spacing = int(np.diff(frames).min()) if len(frames) > 1 else n_frames
            if min_frame_separation and spacing < min_frame_separation:
                raise ValueError(
                    f"{n} evenly spaced frames over {n_frames} gives a gap of "
                    f"{spacing}, below min_frame_separation={min_frame_separation}; "
                    f"request fewer frames or lower the separation")
            return frames * self._stride

        from autopath.utils import match_cluster_centroids_unique
        pc1 = self.X_reduced[:, 0].reshape(-1, 1)
        spread = float(pc1.max() - pc1.min())
        if spread <= 0:
            raise ValueError(
                "PC1 has zero spread, so RegularSpace cannot place distinct centres "
                "(it returns one regardless of n). This is what a converged "
                "trajectory looks like: there are no basins to find. Use "
                "strategy='time_spaced' instead.")
        model = RegularSpace(dmin=spread / n, max_centers=n).fit(pc1).fetch_model()
        self.labels_ = model.transform(pc1)
        centers = model.cluster_centers
        self.centers_ = centers[np.argsort(centers[:, 0])]
        # Returned in PC1 (centroid) order, not frame order.
        frames = np.asarray(match_cluster_centroids_unique(
            pc1, self.centers_, min_frame_separation=min_frame_separation), dtype=int)
        if len(frames) < n:
            logger.warning(
                "requested %d centroid frames but RegularSpace yielded %d; PC1 spread "
                "may be too small or min_frame_separation=%d too large",
                n, len(frames), min_frame_separation)
        return frames * self._stride

    def write_states(self, u, frames, topology, system, out_dir, prefix,
                     temperature=300.0, ligand_sel=None):
        """Serialized State per frame, with velocities drawn at `temperature`.

        `ligand_sel` re-images the ligand next to the protein before serializing."""
        from autopath.utils import _reimaging_ligand
        os.makedirs(out_dir, exist_ok=True)
        # Otherwise a mismatch surfaces as an opaque setPositions error.
        n_u, n_sys = u.atoms.n_atoms, system.getNumParticles()
        n_top = topology.getNumAtoms()
        if not (n_u == n_sys == n_top):
            raise ValueError(
                f"atom-count mismatch: Universe has {n_u}, System has {n_sys}, "
                f"Topology has {n_top} — they must describe the same particles")
        integ = mm.LangevinMiddleIntegrator(
            temperature * ommunit.kelvin, 1 / ommunit.picosecond,
            0.002 * ommunit.picoseconds)
        # CPU only: this context just serializes coordinates.
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
        out = []
        try:
            for i, fi in enumerate(int(f) for f in frames):
                ts = u.trajectory[fi]
                if ligand_sel is not None:
                    _reimaging_ligand(u, ligand_sel)
                if ts.triclinic_dimensions is None:
                    raise ValueError(
                        f"frame {fi} carries no unit cell. Falling back to the System's "
                        "default box would silently restart at the wrong density, so "
                        "this is refused. Supply a trajectory with box information "
                        "(e.g. MDAnalysis transformations.boxdimensions.set_dimensions).")
                ctx.setPeriodicBoxVectors(*ts.triclinic_dimensions / 10.0
                                          * ommunit.nanometer)
                ctx.setPositions(u.atoms.positions / 10.0 * ommunit.nanometer)
                ctx.setVelocitiesToTemperature(temperature * ommunit.kelvin)
                st = ctx.getState(getPositions=True, getVelocities=True)
                fp = os.path.join(out_dir, f"{prefix}_{i+1}_frame_{fi}.xml")
                with open(fp, "w") as fh:
                    fh.write(mm.XmlSerializer.serialize(st))
                out.append(fp)
        finally:
            del ctx, integ
        logger.info("wrote %d states to %s", len(out), out_dir)
        return out

    def write_pdbs(self, u, frames, out_dir, prefix, ligand_sel=None):
        """PDB per frame, retained for back-compatibility."""
        from autopath.utils import _reimaging_ligand
        os.makedirs(out_dir, exist_ok=True)
        out = []
        for i, fi in enumerate(int(f) for f in frames):
            u.trajectory[fi]
            if ligand_sel is not None:
                _reimaging_ligand(u, ligand_sel)
            fp = os.path.join(out_dir, f"{prefix}_{i+1}_frame_{fi}.pdb")
            with mda.Writer(fp, u.atoms.n_atoms, reindex=True) as W:
                W.write(u.atoms)
            out.append(fp)
        return out

    def plot(self, out_dir, prefix, frames=None):
        """PCA scatter with cluster labels and the selected frames annotated.

        `frames` are trajectory indices, as returned by select().
        """
        from autopath.utils import _plot_pca_milestones
        self._require_fit()
        if self._pca is None:
            raise RuntimeError("no fitted PCA; call fit() first")
        if self.X_reduced.shape[1] < 2:
            raise RuntimeError(
                "plot() needs at least 2 PCA components; this fit produced "
                f"{self.X_reduced.shape[1]} (a single-feature featurizer such as "
                "CoordinateRMSD cannot give a 2D scatter)")
        os.makedirs(out_dir, exist_ok=True)
        labels = self.labels_  # None after time_spaced: drawn as one cluster
        rows = ([int(f) // self._stride for f in frames]
                if frames is not None else [])
        _plot_pca_milestones(self.X_reduced, labels, rows,
                             out_dir, prefix, self._pca)
        return os.path.join(out_dir, f"{prefix}_pca_clusters.png")


def read_positions(path: str) -> np.ndarray:
    """Positions in nm from a serialized State (.xml) or a PDB (.pdb)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xml":
        with open(path) as f:
            st = mm.XmlSerializer.deserialize(f.read())
        return np.asarray(st.getPositions(asNumpy=True)
                          .value_in_unit(ommunit.nanometer))
    if ext == ".pdb":
        return np.asarray(app.PDBFile(path).positions
                          .value_in_unit(ommunit.nanometer))
    raise ValueError(f"unsupported extension {ext!r}: expected .xml or .pdb")


def order_milestones_bound_to_unbound(milestones: list[str],
                                      pocket_atoms, ligand_atoms) -> list[str]:
    """Reverse `milestones` if they run unbound -> bound.

    PC1's sign is arbitrary, but downstream code assumes index 0 is the most
    bound; the direction is anchored on pocket-ligand centroid separation.
    """
    if len(milestones) < 2:
        return milestones

    def _sep(path):
        pos = read_positions(path)
        return float(np.linalg.norm(pos[ligand_atoms].mean(axis=0)
                                    - pos[pocket_atoms].mean(axis=0)))

    first, last = _sep(milestones[0]), _sep(milestones[-1])
    if first > last:
        logger.info(
            "Reversing milestone order: the first is %.3f nm from the pocket and "
            "the last %.3f nm, so the list runs unbound -> bound.", first, last)
        return list(reversed(milestones))
    return milestones
