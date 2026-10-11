# autopath/metadynamics/learned/imaging.py
"""Frames of aligned trajectories (one periodic cell imaged around the anchor molecules) where cn is exact."""
import numpy as np

from .features import CN_DMAX


def anchor_atoms(topology_pdb) -> list:
    """Atoms of the anchor molecules the trajectory imaging centred (mdtraj heuristic), sorted."""
    import mdtraj as md
    top = md.load_topology(topology_pdb)
    return sorted(a.index for mol in top.guess_anchor_molecules() for a in mol)


def inscribed_radius(box) -> np.ndarray:
    """Half the smallest distance between opposite faces of each (3, 3) box (rows a, b, c), nm."""
    box = np.asarray(box, dtype=float).reshape(-1, 3, 3)
    vol = np.abs(np.linalg.det(box))
    areas = np.stack([np.linalg.norm(np.cross(box[:, i], box[:, j]), axis=1) for i, j in ((1, 2), (0, 2), (0, 1))])
    return (vol / areas).min(0) / 2


def cn_centres(features) -> list:
    """Atoms on the first side of every cn term, sorted."""
    return sorted({int(i) for f in features if f.kind == "cn" for a, _, _ in f.terms for i in a})


def safe_frames(X, box, centre_idx, centres):
    """Frames whose cn centre atoms all lie within (inscribed radius - CN_DMAX) of the imaging centre.

    Parameters
    ----------
    X : (F, N, 3) aligned coordinates, nm
    box : (F, 3, 3) box per frame, nm
    centre_idx : atoms whose unweighted centroid is the imaging centre
    centres : cn centre atoms

    Returns
    -------
    valid : (F,) bool
    margin : (F,) safe radius minus the farthest centre atom's distance, nm
    """
    if box is None:
        raise ValueError("cn features need box vectors in the trajectory; frames without a box are not supported")
    X = np.asarray(X, dtype=float)
    box = np.asarray(box, dtype=float)
    if box.shape != (len(X), 3, 3) or not np.isfinite(box).all():
        raise ValueError(f"box must be ({len(X)}, 3, 3) finite nm; not supported")
    c = X[:, list(centre_idx)].mean(1)
    d = np.linalg.norm(X[:, list(centres)] - c[:, None], axis=-1).max(1)
    margin = inscribed_radius(box) - CN_DMAX - d
    return margin >= 0, margin


def open_box(X):
    """Cubic box (3, 3) nm in which the cn minimum image equals the plain distance for every pair of X."""
    X = np.asarray(X, dtype=float)
    ext = float((X.max(axis=1) - X.min(axis=1)).max())
    return np.eye(3) * (ext + 2 * CN_DMAX + 0.1)
