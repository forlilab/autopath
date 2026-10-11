# autopath/metadynamics/learned/imaging.py
"""Frames of aligned trajectories (one periodic cell imaged around the anchor molecules) where cn is exact."""
import numpy as np

from .features import CN_DMAX

IMAGED_TOL_NM = 1.0


def anchor_atoms(topology_pdb) -> list:
    """Atoms of the anchor molecules the trajectory imaging centred (mdtraj heuristic), sorted."""
    import mdtraj as md
    top = md.load_topology(topology_pdb)
    return sorted(a.index for mol in top.guess_anchor_molecules() for a in mol)


def cn_centres(features) -> list:
    """Atoms on the first side of every cn term, sorted."""
    return sorted({int(i) for f in features if f.kind == "cn" for a, _, _ in f.terms for i in a})


def safe_frames(X, box, centre_idx, centres, water_mols):
    """Frames whose cn centre atoms all lie at least CN_DMAX inside the faces of the wrapped cell.

    The cell is the brick of half widths (a_x, b_y, c_z)/2 around the unweighted centroid of centre_idx
    (mdtraj image_molecules on a reduced box). Its face distances are reduced by how far water centres
    stick out of it (alignment rotates the cell slightly).

    Parameters
    ----------
    X : (F, N, 3) aligned coordinates, nm
    box : (F, 3, 3) reduced lower-triangular box per frame, nm
    centre_idx : atoms whose unweighted centroid is the imaging centre
    centres : cn centre atoms
    water_mols : (n_waters, k) atom indices of each water

    Returns
    -------
    valid : (F,) bool
    margin : (F,) smallest face distance minus CN_DMAX minus the water excess, nm
    """
    if box is None:
        raise ValueError("cn features need box vectors in the trajectory; frames without a box are not supported")
    X = np.asarray(X, dtype=float)
    box = np.asarray(box, dtype=float)
    if box.shape != (len(X), 3, 3) or not np.isfinite(box).all():
        raise ValueError(f"box must be ({len(X)}, 3, 3) finite nm; not supported")
    if np.any(np.abs(box[:, [0, 0, 1], [1, 2, 2]]) > 1e-9):
        raise ValueError("box must be reduced lower-triangular; not supported")
    wm = np.asarray(water_mols, dtype=int)
    if wm.ndim != 2 or not len(wm):
        raise ValueError("no water molecules to verify the imaging of the cell; cn features not supported")
    half = np.diagonal(box, axis1=1, axis2=2) / 2
    c = X[:, list(centre_idx)].mean(1)
    excess = np.maximum((np.abs(X[:, wm].mean(2) - c[:, None]) - half[:, None]).max(axis=(1, 2)), 0.0)
    if excess.max() > IMAGED_TOL_NM:
        f = int(np.argmax(excess))
        raise ValueError(f"water centres lie {excess[f]:.2f} nm outside the cell around the imaging centre in frame "
                         f"{f}: trajectory not imaged into one cell; cn features not supported")
    face = (half[:, None] - np.abs(X[:, list(centres)] - c[:, None])).min(axis=(1, 2))
    margin = face - CN_DMAX - excess
    return margin >= 0, margin


def open_box(X):
    """Cubic box (3, 3) nm in which the cn minimum image equals the plain distance for every pair of X."""
    X = np.asarray(X, dtype=float)
    ext = float((X.max(axis=1) - X.min(axis=1)).max())
    return np.eye(3) * (ext + 2 * CN_DMAX + 0.1)
