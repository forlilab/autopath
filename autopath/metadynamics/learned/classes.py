"""Class labels for the HLDA/Fisher steps: r thresholds, or nearest AutoPath milestone."""
import glob
import os
import re

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.lib.mdamath import angle, norm

from autopath.clustering import order_milestones_bound_to_unbound, read_positions
from autopath.utils import compute_distance_features

from .select import labels_from_r  # noqa: F401  (r-threshold classes)

R_ORDER = ["bound", "vestibule", "bulk"]


def find_milestones(milestones):
    """XML paths from a list or ONE path's directory (milestone_*_frame_*.xml, no *_relax*), by number."""
    if isinstance(milestones, (str, os.PathLike)):
        d = str(milestones)
        subs = [x for x in glob.glob(os.path.join(d, "*", "milestone_*_frame_*.xml"))]
        if subs:
            raise ValueError(f"{d} has per-path subdirectories; pass one path's milestone directory")
        paths = [p for p in glob.glob(os.path.join(d, "milestone_*_frame_*.xml")) if "_relax" not in os.path.basename(p)]
    else:
        paths = [str(p) for p in milestones]
    nums = []
    for p in paths:
        m = re.match(r"milestone_(\d+)_frame_", os.path.basename(p))
        if not m:
            raise ValueError(f"not a milestone file name (milestone_<k>_frame_<j>.xml): {p}")
        nums.append(int(m.group(1)))
    if len(set(os.path.dirname(os.path.abspath(p)) for p in paths)) > 1:
        raise ValueError("milestones span several directories; pass one path's milestones only")
    if len(set(nums)) != len(nums):
        raise ValueError("duplicate milestone numbers: milestones from several paths are mixed")
    return [p for _, p in sorted(zip(nums, paths))]


def heavy_ligand_idx(topology, ligand_idx):
    """Ligand heavy atoms (the clustering used `not name H*`)."""
    u = topology if isinstance(topology, mda.Universe) else mda.Universe(topology)
    return [int(i) for i in u.select_atoms(f"index {' '.join(map(str, ligand_idx))} and not name H*").indices]


def _sel(idx):
    return f"index {' '.join(map(str, idx))}"


def distance_features(u, X_nm, ligand_idx, pocket_idx, dims=None):
    """compute_distance_features on in-memory coordinates X_nm (n, N, 3); dims (n, 6) in A or None."""
    u.load_new(np.asarray(X_nm, np.float32) * 10.0, format=MemoryReader, dimensions=dims)
    return compute_distance_features(u, _sel(ligand_idx), _sel(pocket_idx))


def _state_dims(path):
    """Box (A, deg) of a serialized State, or None."""
    import openmm as mm
    from openmm import unit
    with open(path) as f:
        st = mm.XmlSerializer.deserialize(f.read())
    a, b, c = (np.asarray(v.value_in_unit(unit.angstrom)) for v in st.getPeriodicBoxVectors())
    if not (norm(a) and norm(b) and norm(c)):
        return None
    return np.array([norm(a), norm(b), norm(c), np.degrees(angle(b, c)), np.degrees(angle(a, c)),
                     np.degrees(angle(a, b))], np.float32)


def milestone_features(u, paths, ligand_idx, pocket_idx):
    """(K, n_features) distance features of the milestone states, in the given order."""
    rows = []
    for p in paths:
        dims = _state_dims(p)
        rows.append(distance_features(u, read_positions(p)[None], ligand_idx, pocket_idx,
                                      None if dims is None else dims[None])[0])
    return np.array(rows)


def order_paths(paths, ligand_idx, pocket_idx):
    return order_milestones_bound_to_unbound(list(paths), list(pocket_idx), list(ligand_idx))


def nearest_milestone(D, Fm):
    """Index of the nearest milestone per frame in the space standardised on all frames D (Euclidean)."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(D)
    Z, M = sc.transform(D), sc.transform(Fm)
    d2 = (Z ** 2).sum(1)[:, None] - 2 * Z @ M.T + (M ** 2).sum(1)[None]
    return d2.argmin(1)


def milestone_labels(milestones, topology, ligand_idx, pocket_idx, coords=None, features=None, dims=None):
    """Nearest-milestone class per frame: ("m1".."mK" bound -> unbound, order list, ordered paths).

    coords: (n, N, 3) nm, or features: precomputed (n, n_features) from distance_features.
    ligand_idx is reduced to heavy atoms; pocket_idx is the atom set the clustering used.
    """
    paths = find_milestones(milestones)
    if len(paths) < 2:
        raise ValueError(f"need at least 2 milestones, found {len(paths)}")
    u = mda.Universe(topology) if isinstance(topology, str) else topology
    lig = heavy_ligand_idx(u, ligand_idx)
    paths = order_paths(paths, lig, pocket_idx)
    if features is None:
        features = distance_features(u, coords, lig, pocket_idx, dims)
    near = nearest_milestone(features, milestone_features(u, paths, lig, pocket_idx))
    order = [f"m{k + 1}" for k in range(len(paths))]
    return np.array(order, dtype=object)[near], order, paths
