"""Class labels for the HLDA/Fisher steps: r thresholds, or nearest AutoPath milestone."""
import glob
import logging
import os
import re

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.lib.mdamath import angle, norm

from autopath.clustering import order_milestones_bound_to_unbound, read_positions
from autopath.utils import compute_distance_features

from .select import labels_from_r  # noqa: F401  (r-threshold classes)

R_ORDER = ["bound", "vestibule", "bulk"]
logger = logging.getLogger("autopath.metadynamics.learned")


def find_milestones(milestones):
    """XML paths from a list or ONE path's directory (milestone_*_frame_*.xml, no *_relax*), by number."""
    if isinstance(milestones, (str, os.PathLike)):
        d = str(milestones)
        paths = [p for p in glob.glob(os.path.join(d, "milestone_*_frame_*.xml")) if "_relax" not in os.path.basename(p)]
        subs = glob.glob(os.path.join(d, "*", "milestone_*_frame_*.xml"))
        if subs and not paths:
            raise ValueError(f"{d} has per-path subdirectories; pass one path's milestone directory")
        if subs:
            logger.info("%s: using its flat milestones, ignoring per-path subdirectories", d)
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


RUPTURE_ORDER = ["B", "TS", "U"]
SMOOTH_PS = 2.0
RISE_FRAC, SLIP_FRAC, FREE_FRAC = 0.2, 0.5, 0.1
BASE_B_FRAC, BASE_U_FRAC, LATE_PEAK_FRAC = 0.02, 0.10, 0.10


def smoothed_force(t, f, window_ps=SMOOTH_PS):
    """Centred rolling mean of f over ``window_ps`` (rows from the median time step).

    Parameters
    ----------
    t, f : (n,) arrays, time (ps) and force
    window_ps : float

    Returns
    -------
    (n,) array; edge rows average the partial window.
    """
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(window_ps / dt)))
    return pd.Series(np.asarray(f, float)).rolling(w, center=True, min_periods=1).mean().to_numpy()


def rupture_labels(dat, n_frames, window_ps=SMOOTH_PS):
    """B / TS / U class per frame from one pull's force.

    Parameters
    ----------
    dat : str or DataFrame
        sMD ``.dat`` path (``#`` comment lines) or a frame with ``time`` and ``force`` columns.
    n_frames : int
        Frames to label, spread evenly over the pull time like ``pull_frames``.
    window_ps : float
        Smoothing window.

    Returns
    -------
    labels : (n_frames,) object array of "B", "TS", "U" or "" (unlabelled or excluded pull)
    info : dict
        t_rise, t_peak, t_slip, t_free, f_peak, dF, base_b, base_u, and reason ("" if labelled).
    """
    d = dat if isinstance(dat, pd.DataFrame) else pd.read_csv(dat, comment="#")
    t, raw = d["time"].to_numpy(float), d["force"].to_numpy(float)
    n = len(raw)
    nb = max(1, int(np.ceil(BASE_B_FRAC * n)))
    if nb >= n:
        raise ValueError(f"too few rows ({n}) to label a rupture")
    fs = smoothed_force(t, raw, window_ps)
    tk = t[-1] * np.arange(n_frames) / max(n_frames - 1, 1)
    lab = np.full(n_frames, "", dtype=object)
    # raw medians: smoothing would spread a start spike over the baseline window
    base_b = float(np.median(raw[:nb]))
    base_u = float(np.median(raw[-max(1, int(np.ceil(BASE_U_FRAC * n))):]))
    ip = nb + int(np.argmax(fs[nb:]))  # search past the baseline window: skips start-of-pull transients
    dF = float(fs[ip] - base_u)
    info = dict(t_rise=np.nan, t_peak=float(t[ip]), t_slip=np.nan, t_free=np.nan, f_peak=float(fs[ip]),
                dF=dF, base_b=base_b, base_u=base_u, reason="")
    if t[ip] >= t[0] + (1 - LATE_PEAK_FRAC) * (t[-1] - t[0]):
        return lab, dict(info, reason="peak in the last 10% of the pull")
    rise = nb + np.flatnonzero(fs[nb:] > base_b + RISE_FRAC * (fs[ip] - base_b))
    if not len(rise):
        return lab, dict(info, reason="no force rise")
    slip = ip + 1 + np.flatnonzero(fs[ip + 1:] < base_u + SLIP_FRAC * dF)
    if not len(slip):
        return lab, dict(info, reason="no slip after the peak")
    i_slip = int(slip[0])
    above = np.flatnonzero(fs > base_u + FREE_FRAC * dF)
    i_free = max(i_slip + 1, int(above[-1]) + 1 if len(above) else 0)
    if i_free >= n:
        return lab, dict(info, reason="force never settles to the unbound baseline")
    t_rise, t_slip, t_free = float(t[rise[0]]), float(t[i_slip]), float(t[i_free])
    lab[tk < t_rise] = "B"
    lab[(tk >= t[ip]) & (tk <= t_slip)] = "TS"
    lab[tk > t_free] = "U"
    return lab, dict(info, t_rise=t_rise, t_slip=t_slip, t_free=t_free)
