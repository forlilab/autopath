import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from autopath.clustering import read_positions


def z_where_guest_reaches_bulk(universe, guest, host_com, axis,
                               cutoff_ang: float = 10.0,
                               protein_selection: str = "protein",
                               step_ang: float = 0.5,
                               max_extra_ang: float = 80.0):
    """Smallest funnel z at which every guest atom clears `cutoff_ang` from the host.

    The cylinder holds the unbound reference, so its cap must reach where the guest
    no longer interacts with the host.
    """
    protein = universe.select_atoms(protein_selection)
    if protein.n_atoms == 0:
        raise ValueError(f"protein_selection {protein_selection!r} selected no atoms")
    g0 = guest.positions.copy()
    z0 = float(np.dot(guest.center_of_mass() - host_com, axis))
    try:
        for dz in np.arange(0.0, max_extra_ang + step_ang, step_ang):
            guest.positions = g0 + axis * float(dz)
            if distance_array(guest.positions, protein.positions).min() > cutoff_ang:
                return z0 + float(dz)
    finally:
        guest.positions = g0
    raise ValueError(
        f"guest never cleared {cutoff_ang:.1f} A of the host within "
        f"{max_extra_ang:.0f} A along the unbinding axis; the axis is probably not "
        "pointing out of the site")


def com_distance_at_z(positions_ang, guest_idx, host_idx, axis,
                     z_target_ang: float) -> float:
    """Host-guest COM separation (nm) with the guest moved along `axis` to `z_target_ang`.

    Converts the axial funnel cap into the COM-separation coordinate of the path.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    g = np.asarray(positions_ang)[guest_idx].mean(axis=0)
    h = np.asarray(positions_ang)[host_idx].mean(axis=0)
    z_now = float(np.dot(g - h, axis))
    g_moved = g + axis * (float(z_target_ang) - z_now)
    return float(np.linalg.norm(g_moved - h)) / 10.0


def z_max_for_bulk(universe, guest_selection, host_selection, axis,
                   frame_z_pairs, cutoff_ang: float = 10.0, n_probe: int = 25,
                   **kwargs) -> float:
    """Cylinder cap that reaches bulk from EVERY probed configuration.

    The required travel depends on lateral position, so the most-unbound frames
    are probed and the largest requirement is used.
    """
    pairs = sorted(frame_z_pairs, key=lambda fz: abs(fz[1]), reverse=True)
    probe = [int(f) for f, _ in pairs[:max(1, int(n_probe))]]
    guest = universe.select_atoms(guest_selection)
    host = universe.select_atoms(host_selection)
    required = []
    for idx in probe:
        universe.trajectory[min(idx, universe.trajectory.n_frames - 1)]
        required.append(z_where_guest_reaches_bulk(
            universe, guest, host.center_of_mass(), axis,
            cutoff_ang=cutoff_ang, **kwargs))
    z = float(np.max(required))
    logger.info(
        "Cylinder cap from %d probed frames: max %.2f A, median %.2f A, min %.2f A "
        "-- using the max so the wall is beyond bulk for all of them.",
        len(required), z, float(np.median(required)), float(np.min(required)))
    return z

logger = logging.getLogger("autopath")


def funnel_basins(params, bound_percentile_z: float = 10.0):
    """Bound and unbound basins on the funnel axis (Å) for counting round trips.

    Unbound is the cylinder (|z| > z_cc); bound is below halfway from the site to
    z_cc, so recrossings of the exit barrier do not count.
    """
    z_cc = float(params["z_cc"].value_in_unit(openmmunit.angstrom))
    z_site = float(np.percentile(np.abs(np.asarray(params["axial_distances"])),
                                 bound_percentile_z))
    return z_site + 0.5 * (z_cc - z_site), z_cc


# Lower wall on z, this far below the sMD bound site (Å).
FUNNEL_WALL_BELOW_SITE_ANG = 6.0
# Funnel z bias grid starts this far below the lower wall (Å).
FUNNEL_Z_GRID_BELOW_WALL_ANG = 3.0


def funnel_lower_wall_ang(params, bound_percentile_z: float = 10.0) -> float:
    """z (Å) below which the one-sided funnel pushes the guest back up the axis."""
    if "z_low" in params:
        return float(params["z_low"].value_in_unit(openmmunit.angstrom))
    if "axial_distances" not in params:
        return -float(params["z_cc"].value_in_unit(openmmunit.angstrom))
    z_site = float(np.percentile(np.abs(np.asarray(params["axial_distances"])),
                                 bound_percentile_z))
    return z_site - FUNNEL_WALL_BELOW_SITE_ANG


def funnel_z_grid_min_ang(params) -> float:
    """Lower edge (Å) of the funnel z bias grid, clear of the lower wall."""
    return funnel_lower_wall_ang(params) - FUNNEL_Z_GRID_BELOW_WALL_ANG


def funnel_z_bounds_nm(params) -> tuple:
    """Boundary-correction edges (nm) for the funnel z CV: only the clamped cap.

    The lower wall is soft; under bias the guest pushes into it, and amplified
    hills there build a false minimum, so it gets no correction.
    """
    return (-np.inf, float(params["z_max"].value_in_unit(openmmunit.nanometer)))


def funnel_z_ang(positions_nm, host_idx, guest_idx, masses, axis, frame=None, runtime=False) -> float:
    """Guest position on the funnel axis (Å): n·(COM_guest − origin), mass-weighted.

    ``frame`` (from :func:`funnel_frame`): a protein frame rebuilds n at these positions with origin
    COM_host; None uses the literal ``axis`` and COM_host. For an orientation frame (``host_idx``
    unused): ``runtime=True`` is the run-time force value, origin COM_core + v and n = ``axis``
    (lab constants); ``runtime=False`` measures in the configuration's own core orientation,
    origin COM_core + Rᵀv and n = Rᵀ``axis`` with R the unweighted Kabsch fit of the core onto
    ``orient_ref``. The two agree only at θ = 0; at residual θ the run-time origin is off by
    about |v × n|·θ.
    """
    pos = np.asarray(positions_nm, dtype=float)
    m = np.asarray(masses, dtype=float)
    guest = np.average(pos[guest_idx], axis=0, weights=m[guest_idx])
    if is_orient_frame(frame):
        origin, n = (_orient_runtime_origin_axis if runtime else _orient_origin_axis)(pos, m, frame, axis)
        return float(np.dot(guest - origin, n)) * 10.0
    host = np.average(pos[host_idx], axis=0, weights=m[host_idx])
    if frame is None:
        n = np.asarray(axis, dtype=float)
        n = n / np.linalg.norm(n)
    else:
        n = _frame_axis(pos, m, host_idx, frame)
    return float(np.dot(guest - host, n)) * 10.0


# Protein frame: CA atoms farther than this from the host COM form the two frame groups.
FRAME_MIN_DIST_NM = 1.2
FRAME_MIN_ATOMS = 20
FRAME_ANGLE_LIMITS_DEG = (30.0, 150.0)
_lab_frame_logged = False


def frame_basis(a, b, c) -> np.ndarray:
    """Orthonormal frame (rows e1, e2, e3) from centroids a (origin), b and c.

    e1 = (b − a)/|b − a|; e2 = the part of (c − a) normal to e1, normalised; e3 = e1 × e2.
    """
    a, b, c = (np.asarray(x, dtype=float) for x in (a, b, c))
    e1 = (b - a) / np.linalg.norm(b - a)
    v = c - a
    w = v - np.dot(v, e1) * e1
    e2 = w / np.linalg.norm(w)
    return np.array([e1, e2, np.cross(e1, e2)])


def _frame_axis(pos, masses, host_idx, frame) -> np.ndarray:
    cen = lambda idx: np.average(pos[list(idx)], axis=0, weights=masses[list(idx)])
    return np.asarray(frame["coeffs"], float) @ frame_basis(cen(host_idx), cen(frame["b"]), cen(frame["c"]))


def funnel_frame(params):
    """Protein frame {"b", "c", "coeffs"}, orientation frame {"orient", "offset", "ref"}, or None (lab frame).

    Lab-frame params keep the constant ``unbinding_axis``.
    """
    global _lab_frame_logged
    if funnel_mode(params) == "orient":
        return {"orient": [int(i) for i in params["orient_index"]],
                "offset": np.asarray(params["orient_offset"], dtype=float),
                "ref": np.asarray(params["orient_ref"], dtype=float)}
    if "axis_coeffs" not in params:
        if not _lab_frame_logged:
            logger.info("funnel params have no protein frame: the funnel axis is fixed in the lab frame")
            _lab_frame_logged = True
        return None
    return {"b": [int(i) for i in params["frame_b_index"]], "c": [int(i) for i in params["frame_c_index"]],
            "coeffs": np.asarray(params["axis_coeffs"], dtype=float)}


def funnel_axis_at(params, positions_nm, masses) -> np.ndarray:
    """Unit funnel axis at these positions (nm): protein frame, orientation frame, or the lab axis."""
    frame = funnel_frame(params)
    if frame is None:
        n = np.asarray(params["unbinding_axis"], dtype=float)
        return n / np.linalg.norm(n)
    pos, m = np.asarray(positions_nm, float), np.asarray(masses, float)
    if is_orient_frame(frame):
        return _orient_origin_axis(pos, m, frame, params["unbinding_axis"])[1]
    return _frame_axis(pos, m, [int(i) for i in params["host_index"]], frame)


def funnel_origin_axis_at(params, positions_nm, masses, runtime=False):
    """(origin nm, unit axis) of the funnel at these positions, as :func:`funnel_z_ang` measures them.

    ``runtime`` selects the run-time (lab-constant) orientation frame, see :func:`funnel_z_ang`.
    """
    pos, m = np.asarray(positions_nm, float), np.asarray(masses, float)
    frame = funnel_frame(params)
    if is_orient_frame(frame):
        f = _orient_runtime_origin_axis if runtime else _orient_origin_axis
        return f(pos, m, frame, params["unbinding_axis"])
    host = [int(i) for i in params["host_index"]]
    return np.average(pos[host], axis=0, weights=m[host]), funnel_axis_at(params, pos, m)


def funnel_coords_ang(params, positions_nm, masses, runtime=False):
    """(z, r_xy) of the guest in Å, the coordinates of the funnel restraint (mass-weighted centroids).

    ``runtime`` selects the run-time (lab-constant) orientation frame, see :func:`funnel_z_ang`.
    """
    pos, m = np.asarray(positions_nm, float), np.asarray(masses, float)
    origin, n = funnel_origin_axis_at(params, pos, m, runtime=runtime)
    g = [int(i) for i in params["guest_index"]]
    r = np.average(pos[g], axis=0, weights=m[g]) - origin
    return float(n @ r) * 10.0, float(np.linalg.norm(np.cross(n, r))) * 10.0


# ---------------------------------------------------------------------------
# Orientation-restrained funnel: OrientationRestraintForce pins the core, axis and offset are lab constants
# ---------------------------------------------------------------------------

ORIENT_KEYS = ("orient_index", "orient_k", "orient_ref", "orient_offset")
PROTEIN_FRAME_KEYS = ("axis_coeffs", "frame_b_index", "frame_c_index")
ORIENT_K_DEFAULT = 1.0e6                  # kJ/mol
ORIENT_CORE_GUEST_CUTOFF_NM = 1.0
ORIENT_MAX_START_DEG = 2.0
# a whole core fits its reference far below this; a core split across the box does not
ORIENT_MAX_FIT_RMSD_NM = 0.4
ORIENT_FORCE_GROUP = 11
ORIENT_FORCE_NAME = "orient_restraint"
ORIENT_OFFSET_PARAMS = ("fo_vx", "fo_vy", "fo_vz")


def funnel_mode(params) -> str:
    """"lab", "protein" or "orient"; raises ValueError on mixed or partial frame keys."""
    has = [k for k in ORIENT_KEYS if k in params]
    if has:
        if len(has) != len(ORIENT_KEYS):
            raise ValueError(f"funnel params with only {has} of {list(ORIENT_KEYS)}: not supported")
        mixed = [k for k in PROTEIN_FRAME_KEYS + ("origin_index",) if k in params]
        if mixed:
            raise ValueError(f"orientation-restrained funnel params with {mixed}: not supported")
        return "orient"
    return "protein" if "axis_coeffs" in params else "lab"


def is_orient_frame(frame) -> bool:
    """True for an orientation frame from :func:`funnel_frame`."""
    return frame is not None and "orient" in frame


def kabsch_rotation(x, ref) -> np.ndarray:
    """Unweighted Kabsch rotation R minimising |R(x − x̄) − (ref − r̄)|, as OrientationRestraintForce."""
    xc = np.asarray(x, float) - np.mean(x, axis=0)
    yc = np.asarray(ref, float) - np.mean(ref, axis=0)
    U, _, Vt = np.linalg.svd(xc.T @ yc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def rotation_angle_deg(R) -> float:
    """Rotation angle (deg) of a rotation matrix."""
    from scipy.spatial.transform import Rotation
    # quaternion magnitude: arccos of the trace loses precision near 0
    return float(np.degrees(Rotation.from_matrix(np.asarray(R, float)).magnitude()))


def _orient_fit(pos, frame):
    """(R, fit RMSD nm) of the core at ``pos`` onto the frame reference; refuses a split core."""
    x = pos[frame["orient"]]
    R = kabsch_rotation(x, frame["ref"])
    xc = (x - x.mean(0)) @ R.T
    rmsd = float(np.sqrt(((xc - (frame["ref"] - frame["ref"].mean(0))) ** 2).sum(1).mean()))
    if rmsd > ORIENT_MAX_FIT_RMSD_NM:
        raise ValueError(f"orientation core fits its reference at {rmsd:.2f} nm RMSD (> {ORIENT_MAX_FIT_RMSD_NM} nm): "
                         "core split across the periodic box or a different structure: not supported")
    return R, rmsd


def _orient_origin_axis(pos, m, frame, axis):
    R, _ = _orient_fit(pos, frame)
    core = frame["orient"]
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return np.average(pos[core], axis=0, weights=m[core]) + R.T @ frame["offset"], R.T @ n


def _orient_runtime_origin_axis(pos, m, frame, axis):
    core = frame["orient"]
    n = np.asarray(axis, dtype=float)
    return np.average(pos[core], axis=0, weights=m[core]) + frame["offset"], n / np.linalg.norm(n)


def orient_theta_deg(params, positions_nm) -> float:
    """Rotation angle (deg) of the orientation core at these positions relative to ``orient_ref``."""
    return rotation_angle_deg(_orient_fit(np.asarray(positions_nm, float), funnel_frame(params))[0])


def check_orient_start(params, positions_nm) -> float:
    """Refuse a start whose orientation core is far from ``orient_ref``.

    Parameters
    ----------
    params : orientation-restrained funnel params
    positions_nm : (N, 3) start positions (nm)

    Returns
    -------
    float
        Core rotation θ (deg) relative to ``orient_ref``.

    Raises
    ------
    ValueError
        θ above ``ORIENT_MAX_START_DEG``, or a core split across the periodic box.
    """
    th = orient_theta_deg(params, positions_nm)
    if th > ORIENT_MAX_START_DEG:
        raise ValueError(f"orientation core is {th:.2f} deg from orient_ref (> {ORIENT_MAX_START_DEG} deg): "
                         "not supported: re-target with orient_params_for_state")
    return th


def _default_core(pos, topology, host, guest):
    """CA atoms of DSSP H/E residues of the host's molecule, beyond the guest cutoff, minus host atoms."""
    from MDAnalysis.analysis.dssp.pydssp_numpy import assign
    mol = _molecules(topology)
    if len(set(mol[host])) != 1:
        raise FunnelFrameError("orientation-restrained funnel: host atoms span more than one molecule: "
                               "not supported")
    bb, ca = [], []
    for r in topology.residues():
        d = {a.name: a.index for a in r.atoms()}
        if all(k in d for k in ("N", "CA", "C", "O")) and mol[d["CA"]] == mol[host[0]]:
            bb.append([d[k] for k in ("N", "CA", "C", "O")])
            ca.append(d["CA"])
    if not ca:
        return []
    ss = np.asarray(assign(pos[np.array(bb)] * 10.0))     # pydssp works in Å
    ss = np.array(["-", "H", "E"])[ss.argmax(1)] if ss.ndim == 2 else ss
    ca = np.array(ca)
    dmin = np.linalg.norm(pos[ca][:, None] - pos[guest][None], axis=-1).min(1)
    keep = (ss != "-") & (dmin > ORIENT_CORE_GUEST_CUTOFF_NM) & ~np.isin(ca, host)
    return [int(i) for i in ca[keep]]


def orientation_restrained_params(params, reference_positions, topology, masses, k=ORIENT_K_DEFAULT,
                                  core_index=None) -> Dict:
    """Orientation-restrained copy of lab- or protein-frame funnel params.

    Parameters
    ----------
    params : funnel_params dict (lab or protein frame)
    reference_positions : (N, 3) nm; the core orientation the restraint holds
    topology : openmm.app.Topology of the system
    masses : (N,) particle masses (Da)
    k : orientation restraint force constant (kJ/mol)
    core_index : core atom indices; None selects the CA atoms of DSSP H/E residues of the host's molecule
        farther than ``ORIENT_CORE_GUEST_CUTOFF_NM`` from every guest atom, host atoms excluded.
        The core must be one molecule (centroids and the fit use no periodic images).

    Returns
    -------
    dict
        ``params`` without the protein-frame keys, with ``unbinding_axis`` = the axis at the reference,
        ``orient_index``, ``orient_k``, ``orient_ref`` (nm), ``orient_offset`` v = COM_host − COM_core (nm),
        and the provenance ``master_axis``, ``master_offset``, ``orient_rotation`` (identity).
    """
    if funnel_mode(params) == "orient":
        raise ValueError("funnel params are already orientation-restrained: not supported")
    pos = np.asarray(reference_positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    host = [int(i) for i in params["host_index"]]
    guest = [int(i) for i in params["guest_index"]]
    core = _default_core(pos, topology, host, guest) if core_index is None else [int(i) for i in core_index]
    if len(core) < FRAME_MIN_ATOMS:
        raise FunnelFrameError(f"orientation-restrained funnel: {len(core)} core atoms, fewer than "
                               f"{FRAME_MIN_ATOMS}: not supported")
    mol = _molecules(topology)
    if len(set(mol[core])) != 1:
        raise FunnelFrameError("orientation-restrained funnel: core atoms span more than one molecule: "
                               "not supported")
    n = funnel_axis_at(params, pos, m)
    com = lambda idx: np.average(pos[idx], axis=0, weights=m[idx])
    v = com(host) - com(core)
    out = {key: val for key, val in params.items() if key not in PROTEIN_FRAME_KEYS}
    out.update(unbinding_axis=n, orient_index=core, orient_k=float(k), orient_ref=pos[core].copy(),
               orient_offset=v, master_axis=n.copy(), master_offset=v.copy(), orient_rotation=np.eye(3))
    z0, r0 = funnel_coords_ang(params, pos, m)
    z1, r1 = funnel_coords_ang(out, pos, m)
    if abs(z1 - z0) > 1e-5 or abs(r1 - r0) > 1e-5:                 # 1e-6 nm
        raise ValueError(f"not supported: the orientation frame moves the reference (z {z0} -> {z1} A, "
                         f"r_xy {r0} -> {r1} A)")
    logger.info("Orientation-restrained funnel: %d core atoms, k %.3g kJ/mol, |v| %.3f nm",
                len(core), k, float(np.linalg.norm(v)))
    return out


def orient_params_for_state(params, positions) -> Dict:
    """Orientation-restrained params carried rigidly into the core orientation of a state.

    Parameters
    ----------
    params : orientation-restrained funnel params
    positions : (N, 3) state positions (nm)

    Returns
    -------
    dict
        ``params`` with, for R the unweighted Kabsch fit of the state's core onto ``orient_ref``,
        ``orient_ref`` = Rᵀ(ref − r̄) + x̄, ``unbinding_axis`` = Rᵀn and ``orient_offset`` = Rᵀv, so
        the state's core is at θ = 0; ``master_axis``/``master_offset`` kept; ``orient_rotation`` Q
        (``unbinding_axis`` = Q ``master_axis``) and ``orient_theta_deg`` (angle of Q).

    Raises
    ------
    ValueError
        Params not orientation-restrained, or a core split across the periodic box.
    """
    if funnel_mode(params) != "orient":
        raise ValueError("orient_params_for_state needs orientation-restrained params: not supported")
    pos = np.asarray(positions, dtype=float)
    frame = funnel_frame(params)
    R, _ = _orient_fit(pos, frame)
    x = pos[frame["orient"]]
    ref = frame["ref"]
    Q = R.T @ np.asarray(params.get("orient_rotation", np.eye(3)), dtype=float)
    out = dict(params)
    out.update(orient_ref=(ref - ref.mean(0)) @ R + x.mean(0),
               unbinding_axis=R.T @ np.asarray(params["unbinding_axis"], float),
               orient_offset=R.T @ frame["offset"],
               master_axis=np.asarray(params.get("master_axis", params["unbinding_axis"]), float),
               master_offset=np.asarray(params.get("master_offset", frame["offset"]), float),
               orient_rotation=Q, orient_theta_deg=rotation_angle_deg(Q))
    logger.info("Orientation frame re-targeted by %.2f deg (%.2f deg from the master)",
                rotation_angle_deg(R), rotation_angle_deg(Q))
    return out


def orientation_restraint_force(params, n_particles: int):
    """OrientationRestraintForce of orientation-restrained params, in ``ORIENT_FORCE_GROUP``."""
    frame = funnel_frame(params)
    if not is_orient_frame(frame):
        raise ValueError("orientation_restraint_force needs orientation-restrained params: not supported")
    ref = np.zeros((int(n_particles), 3))
    ref[frame["orient"]] = frame["ref"]
    f = OrientationRestraintForce(float(params["orient_k"]), [Vec3(*map(float, p)) for p in ref],
                                  frame["orient"])
    f.setForceGroup(ORIENT_FORCE_GROUP)
    f.setName(ORIENT_FORCE_NAME)
    return f


def _molecules(topology):
    """Atom index -> connected-component id over the topology bonds."""
    parent = list(range(topology.getNumAtoms()))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a1, a2 in topology.bonds():
        parent[root(a1.index)] = root(a2.index)
    return np.array([root(i) for i in range(len(parent))])


class FunnelFrameError(ValueError):
    """The protein frame cannot be built for this system."""


def select_frame_groups(positions_nm, topology, host_index, masses):
    """Frame groups (B, C): protein CA atoms beyond ``FRAME_MIN_DIST_NM`` of the host COM, split by 2-means.

    Candidates are the CAs (atoms named CA in residues with N and C) of the host's molecule.
    B is the cluster holding the lowest atom index.

    Raises
    ------
    ValueError
        Host atoms in more than one molecule; a group under ``FRAME_MIN_ATOMS`` atoms; or an angle
        between (B − A) and (C − A) outside ``FRAME_ANGLE_LIMITS_DEG``.
    """
    return _frame_groups_and_angle(positions_nm, topology, host_index, masses)[:2]


def _frame_groups_and_angle(positions_nm, topology, host_index, masses):
    """(B, C, angle in deg between B − A and C − A) for :func:`select_frame_groups`."""
    from sklearn.cluster import KMeans
    pos = np.asarray(positions_nm, dtype=float)
    m = np.asarray(masses, dtype=float)
    host = [int(i) for i in host_index]
    mol = _molecules(topology)
    if len(set(mol[host])) != 1:
        raise FunnelFrameError("protein-frame funnel: host atoms span more than one molecule: not supported")
    ca = [a.index for r in topology.residues() if {"N", "CA", "C"} <= {x.name for x in r.atoms()}
          for a in r.atoms() if a.name == "CA" and mol[a.index] == mol[host[0]]]
    hc = np.average(pos[host], axis=0, weights=m[host])
    ca = np.array([i for i in ca if np.linalg.norm(pos[i] - hc) > FRAME_MIN_DIST_NM], dtype=int)
    if len(ca) < 2 * FRAME_MIN_ATOMS:
        raise FunnelFrameError(f"protein-frame funnel: {len(ca)} CA atoms beyond {FRAME_MIN_DIST_NM} nm of the host, "
                         f"fewer than {FRAME_MIN_ATOMS} per frame group")
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(pos[ca])
    groups = sorted(([int(i) for i in ca[lab == k]] for k in (0, 1)), key=min)
    b, c = groups
    if min(len(b), len(c)) < FRAME_MIN_ATOMS:
        raise FunnelFrameError(f"protein-frame funnel: frame groups of {len(b)} and {len(c)} atoms, "
                         f"fewer than {FRAME_MIN_ATOMS} in one")
    u = np.average(pos[b], axis=0, weights=m[b]) - hc
    v = np.average(pos[c], axis=0, weights=m[c]) - hc
    ang = np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0)))
    lo, hi = FRAME_ANGLE_LIMITS_DEG
    if not lo <= ang <= hi:
        raise FunnelFrameError(f"protein-frame funnel: angle {ang:.1f} deg between the frame groups seen from the "
                         f"host is outside [{lo:.0f}, {hi:.0f}] deg (ill-conditioned frame)")
    return b, c, ang


def upgrade_funnel_params(params, reference_positions, topology, masses) -> Dict:
    """Copy of lab-frame funnel params with a protein frame (``frame_b_index``, ``frame_c_index``,
    ``axis_coeffs``).

    Parameters
    ----------
    params : funnel_params dict without a frame
    reference_positions : (N, 3) nm, in the orientation ``unbinding_axis`` was computed in
    topology : openmm.app.Topology of the system
    masses : (N,) particle masses (Da), the system's

    Returns
    -------
    dict
        ``params`` plus the frame groups and the axis coefficients (n·e1, n·e2, n·e3) at the reference.
    """
    if "axis_coeffs" in params:
        raise FunnelFrameError("funnel params already have a protein frame")
    pos = np.asarray(reference_positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    host = [int(i) for i in params["host_index"]]
    b, c, ang = _frame_groups_and_angle(pos, topology, host, m)
    cen = lambda idx: np.average(pos[idx], axis=0, weights=m[idx])
    n = np.asarray(params["unbinding_axis"], dtype=float)
    n = n / np.linalg.norm(n)
    e = frame_basis(cen(host), cen(b), cen(c))
    out = dict(params, frame_b_index=b, frame_c_index=c, axis_coeffs=e @ n)
    logger.info("Protein-frame funnel: groups of %d and %d CA atoms, %.1f deg apart seen from the host; "
                "axis coefficients %s", len(b), len(c), ang, np.round(e @ n, 4).tolist())
    return out


def _frame_definitions(h: int, g: int, o: int, b: int, c: int, coeffs, radial: bool = True) -> str:
    """Expression definitions of r_z (and cx, cy, cz = n × r if ``radial``) in the protein frame.

    r = centroid g − centroid h; the frame is built from centroid groups o (origin), b and c,
    and r is expanded in it (s1, s2, s3), so n is never formed: r_z = a·s, n × r = a × s.
    """
    a1, a2, a3 = (f"({float(x)!r})" for x in coeffs)
    d = [f"r_z = {a1}*pf_s1 + {a2}*pf_s2 + {a3}*pf_s3"]
    if radial:
        d += [f"cx = {a2}*pf_s3 - {a3}*pf_s2", f"cy = {a3}*pf_s1 - {a1}*pf_s3", f"cz = {a1}*pf_s2 - {a2}*pf_s1"]
    d += ["pf_s1 = pf_ur*pf_iu", "pf_s2 = (pf_vr - pf_p*pf_ur)*pf_iw", "pf_s3 = pf_t*pf_iu*pf_iw",
          "pf_iw = 1/sqrt(pf_vv - pf_p*pf_uv)", "pf_p = pf_uv/pf_uu", "pf_iu = 1/sqrt(pf_uu)",
          "pf_t = pf_rx*(pf_uy*pf_vz - pf_uz*pf_vy) + pf_ry*(pf_uz*pf_vx - pf_ux*pf_vz)"
          " + pf_rz*(pf_ux*pf_vy - pf_uy*pf_vx)",
          "pf_ur = pf_ux*pf_rx + pf_uy*pf_ry + pf_uz*pf_rz", "pf_vr = pf_vx*pf_rx + pf_vy*pf_ry + pf_vz*pf_rz",
          "pf_uu = pf_ux^2 + pf_uy^2 + pf_uz^2", "pf_vv = pf_vx^2 + pf_vy^2 + pf_vz^2",
          "pf_uv = pf_ux*pf_vx + pf_uy*pf_vy + pf_uz*pf_vz"]
    d += [f"pf_r{k} = {k}{g} - {k}{h}" for k in "xyz"]
    d += [f"pf_u{k} = {k}{b} - {k}{o}" for k in "xyz"]
    d += [f"pf_v{k} = {k}{c} - {k}{o}" for k in "xyz"]
    return ";".join(d)


def add_funnel_restraints(
    system: System,
    host_index: List[int],
    guest_index: List[int],
    k_xy: Optional[openmmunit.Quantity] = 10.0
    * openmmunit.kilocalorie_per_mole
    / openmmunit.angstrom**2,
    z_cc: Optional[openmmunit.Quantity] = 11.0 * openmmunit.angstrom,
    alpha: Optional[openmmunit.Quantity] = 35.0 * openmmunit.degrees,
    R_cylinder: Optional[openmmunit.Quantity] = 1.0 * openmmunit.angstrom,
    force_name: str = "k_funnel",
    force_group: Optional[int] = 10,
):
    """
    Applies a Z-axis-aligned funnel potential restraint to a guest molecule.
    Limongelli, V., Bonomi, M., & Parrinello, M. (2013). Funnel metadynamics as accurate binding free-energy method. Proceedings of the National Academy of Sciences, 110(16), 6358-6363.
    https://github.com/jeff231li/funnel_potential

    For a trajectory-derived axis with a z_max cap use
    :func:`create_funnel_force_from_trajectory_analysis`.
    """

    funnel = CustomCentroidBondForce(
        2,
        "U_funnel + U_cylinder;"
        "U_funnel = step(z_cc - abs(r_z))*step(r_xy - R_funnel)*Wall_funnel;"
        "U_cylinder = step(abs(r_z) - z_cc)*step(r_xy - R_cylinder)*Wall_cylinder;"
        "Wall_funnel = 0.5 * k_xy * (r_xy - R_funnel)^2;"
        "Wall_cylinder = 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        "R_funnel = (z_cc-abs(r_z))*tan(alpha) + R_cylinder;"
        "r_xy = sqrt((x2 - x1)^2 + (y2 - y1)^2 + 1e-12);"
        "r_z = z2 - z1;",
    )
    funnel.setUsesPeriodicBoundaryConditions(False)
    funnel.setForceGroup(force_group)

    funnel.addGlobalParameter("k_xy", k_xy)
    funnel.addGlobalParameter("z_cc", z_cc)
    funnel.addGlobalParameter("alpha", alpha)
    funnel.addGlobalParameter("R_cylinder", R_cylinder)

    # empty weights: mass-weighted centroids, as funnel z and the funnel geometry
    g1 = funnel.addGroup([int(i) for i in host_index])
    g2 = funnel.addGroup([int(i) for i in guest_index])

    funnel.addBond([g1, g2], [])
    funnel.setName(force_name)
    system.addForce(funnel)

    return


def generate_funnel_parameters_from_trajectory(
    universe,
    host_selection: str = "protein",
    guest_selection: str = "resname UNK",
    k_xy: Optional[openmmunit.Quantity] = 10.0
    * openmmunit.kilocalorie_per_mole
    / openmmunit.angstrom**2,
    k_z: Optional[openmmunit.Quantity] = 500.0
    * openmmunit.kilocalorie_per_mole
    / openmmunit.angstrom**2,
    use_pca: bool = True,
    percentile_z: float = 95.0,
    z_cc_ang: Optional[float] = None,
    bulk_cutoff_ang: float = 10.0,
    R_cylinder_ang: Optional[float] = None,
    percentile_r_funnel: float = 95.0,
    bound_percentile_z: float = 10.0,
    alpha_cone_degrees: float = 25.0,
    stride: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    Derive funnel parameters (axis, z_cc, z_max, alpha, R_cylinder) from an unbinding trajectory.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe containing the trajectory to analyze
    host_selection : str
        MDAnalysis selection string for host/protein atoms (default: "protein")
    guest_selection : str
        MDAnalysis selection string for guest/ligand atoms (default: "resname UNK")
    k_xy : openmmunit.Quantity
        Force constant for XY plane restraint
    k_z : openmmunit.Quantity
        Force constant of the lower and z_max walls (default 500 kcal/mol/Å²);
        the bias presses the guest into a soft wall.
    use_pca : bool
        If True, use PCA to determine the unbinding axis. If False, use the axis
        with largest COM displacement (default: True)
    percentile_z : float
        Percentile of Z-displacement to use for z_cc parameter (0-100, default: 95).
        Ignored when z_cc_ang is provided.
    z_cc_ang : float, optional
        Explicit cone-to-cylinder transition distance in Å; overrides percentile_z.
    bulk_cutoff_ang : float
        z_max is placed where every guest atom is farther than this from the
        protein, across the most-unbound frames (default: 10 Å).
    R_cylinder_ang : float
        Cylinder radius in Å. If None, derived from the radial spread beyond z_cc.
        Limongelli 2013 recommends 1–3 Å.
    percentile_r_funnel : float
        Percentile of radial displacement used to derive R_cylinder and the
        bound-state spread (default: 95)
    alpha_cone_degrees : float
        Cone half-angle in degrees (default: 25.0)
    verbose : bool
        Print diagnostic information (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - "host_index": List of host atom indices
        - "guest_index": List of guest atom indices
        - "k_xy": Force constant
        - "k_z": Axial wall force constant
        - "z_cc": Z-crossing point parameter
        - "alpha": Cone angle parameter
        - "R_cylinder": Inner cylinder radius
        - "force_group": Suggested force group (10)
        - "force_name": Suggested force name ("k_funnel_trajectory")
        - "com_trajectory": Array of COM positions over trajectory
        - "host_com_trajectory": Array of host COMs over trajectory
        - "guest_com_trajectory": Array of guest COMs over trajectory
        - "radial_distances": Array of radial distances over trajectory
        - "axial_distances": Array of axial distances over trajectory
        - "unbinding_axis": Unit vector along unbinding direction
        - "trajectory_length": Length of analyzed trajectory
    """

    host_atoms = universe.select_atoms(host_selection)
    guest_atoms = universe.select_atoms(guest_selection)

    host_indices = list(host_atoms.indices)
    guest_indices = list(guest_atoms.indices)

    if len(host_indices) == 0:
        raise ValueError(f"No atoms found for host selection: {host_selection}")
    if len(guest_indices) == 0:
        raise ValueError(f"No atoms found for guest selection: {guest_selection}")

    if verbose:
        logger.info(f"Host atoms: {len(host_indices)} ({host_selection})")
        logger.info(f"Guest atoms: {len(guest_indices)} ({guest_selection})")

    host_coms = []
    guest_coms = []
    radial_distances = []
    axial_distances = []

    for frame in universe.trajectory[::stride]:
        host_com = host_atoms.center_of_mass()
        guest_com = guest_atoms.center_of_mass()
        host_coms.append(host_com)
        guest_coms.append(guest_com)

    host_coms = np.array(host_coms)
    guest_coms = np.array(guest_coms)

    trajectory_length = len(host_coms)

    if trajectory_length < 2:
        raise ValueError("Trajectory must have at least 2 frames")

    if verbose:
        logger.info(f"Analyzed {trajectory_length} frames")

    if use_pca:
        guest_displacement = guest_coms - guest_coms[0]
        displacement_centered = guest_displacement - guest_displacement.mean(axis=0)
        cov_matrix = np.cov(displacement_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        unbinding_axis = eigenvectors[:, -1]
    else:
        guest_displacement = guest_coms[-1] - guest_coms[0]
        unbinding_axis = guest_displacement / np.linalg.norm(guest_displacement)

    # eigh's eigenvector sign is arbitrary; orient the axis bound -> unbound.
    _net = guest_coms[-1] - guest_coms[0]
    if np.dot(unbinding_axis, _net) < 0:
        unbinding_axis = -unbinding_axis

    relative_positions = guest_coms - host_coms
    z_distances = np.dot(relative_positions, unbinding_axis)

    for rel_pos in relative_positions:
        z_component = np.dot(rel_pos, unbinding_axis) * unbinding_axis
        radial_component = rel_pos - z_component
        radial_dist = np.linalg.norm(radial_component)
        radial_distances.append(radial_dist)

    radial_distances = np.array(radial_distances)
    z_distances = np.array(z_distances)

    if z_cc_ang is not None:
        z_cc = float(z_cc_ang)
        if verbose:
            logger.info(f"z_cc overridden by user: {z_cc:.2f} Å (percentile_z ignored)")
    else:
        z_cc = np.percentile(z_distances, percentile_z)

    # Bound state = low-|z| tail, not leading frames (arrays pool many trajectories).
    _absz = np.abs(z_distances)
    z_site = float(np.percentile(_absz, bound_percentile_z))
    R_funnel_bound = float(np.percentile(
        radial_distances[_absz <= z_site], percentile_r_funnel))

    # The cone must contain the binding site so the restraint does not bias it.
    R_cone_at_site = R_cylinder_ang if R_cylinder_ang is not None else 0.0
    R_cone_at_site += (z_cc - z_site) * np.tan(np.radians(alpha_cone_degrees))
    if z_cc <= z_site:
        logger.warning(
            "z_cc=%.2f A does not reach the bound state (|z| p%.0f = %.2f A): the "
            "binding site falls in the cylinder, where the restraint is at its "
            "tightest. Raise percentile_z or set z_cc_ang.",
            z_cc, bound_percentile_z, z_site)
    elif R_cone_at_site < R_funnel_bound:
        logger.warning(
            "cone radius at the bound state is %.2f A but the ligand's radial "
            "spread there is %.2f A, so the restraint acts inside the binding "
            "site. Raise percentile_z or alpha.",
            R_cone_at_site, R_funnel_bound)

    # Size the cylinder from the radial spread where it applies (|z| >= z_cc).
    if R_cylinder_ang is None:
        _cyl = radial_distances[np.abs(z_distances) >= z_cc]
        if _cyl.size == 0:
            raise ValueError(
                f"no trajectory frames beyond z_cc={z_cc:.2f} A, so the cylinder "
                "radius cannot be derived; pass R_cylinder_ang explicitly.")
        R_cylinder = float(np.percentile(_cyl, percentile_r_funnel))
        if verbose:
            logger.info(
                "Derived R_cylinder=%.2f A from the p%.0f radial spread of the "
                "%d frames beyond z_cc (%.1f%% of them fall outside it).",
                R_cylinder, percentile_r_funnel, _cyl.size,
                100.0 * float((_cyl > R_cylinder).mean()))
    else:
        R_cylinder = float(R_cylinder_ang)

    # Cylinder cap must reach bulk.
    _pairs = list(zip(np.arange(z_distances.size) * int(stride),
                      np.asarray(z_distances, dtype=float)))
    z_max = z_max_for_bulk(
        universe, guest_selection, host_selection, unbinding_axis,
        frame_z_pairs=_pairs, cutoff_ang=bulk_cutoff_ang)
    _z_max_why = (f"every guest atom > {bulk_cutoff_ang:.1f} A from the host "
                  "across the most-unbound probed frames")
    if z_max <= z_cc:
        raise ValueError(
            f"z_max ({z_max:.2f} A) is not beyond z_cc ({z_cc:.2f} A); the cylinder "
            "would have zero or negative length")

    z_cc_quantity = z_cc * openmmunit.angstrom
    z_max_quantity = z_max * openmmunit.angstrom
    R_cylinder_quantity = R_cylinder * openmmunit.angstrom
    alpha_quantity = alpha_cone_degrees * openmmunit.degrees

    R_funnel_per_frame = np.where(
        np.abs(z_distances) < z_cc,
        (z_cc - np.abs(z_distances)) * np.tan(np.radians(alpha_cone_degrees)) + R_cylinder,
        R_cylinder,
    )
    frac_inside = float(np.mean(radial_distances <= R_funnel_per_frame))
    if frac_inside < 0.7:
        logger.warning(
            f"Only {100 * frac_inside:.1f}% of sMD frames are inside the funnel cone "
            f"(threshold: 70%). The funnel is likely misaligned with the unbinding path. "
            "Recommendation: pass pocket atoms as host_selection instead of 'protein' so "
            "the funnel axis passes through the binding site."
        )
    else:
        logger.info(f"Funnel coverage: {100 * frac_inside:.1f}% of sMD frames inside cone.")

    if verbose:
        logger.info(f"Unbinding axis: {unbinding_axis}")
        logger.info(f"Z-crossing point (z_cc): {z_cc:.2f} Å")
        logger.info(f"Cylinder cap (z_max): {z_max:.2f} Å  ({_z_max_why})")
        _cyl_len = z_max - z_cc
        logger.info(
            "Cylinder length %.2f Å vs cone %.2f Å (ratio %.2f). The cylinder holds "
            "the unbound reference, so a short one leaves no flat region to "
            "normalise against.", _cyl_len, z_cc, _cyl_len / z_cc)
        _ref = "" if R_cylinder <= 3.0 else (
            "  NOTE: wider than the 1-3 A of Limongelli 2013 -- the observed "
            "channel is broad, so the funnel confines weakly")
        logger.info("Cylinder radius (R_cylinder): %.2f Å%s", R_cylinder, _ref)
        logger.info(f"Bound-state radial spread (informational): {R_funnel_bound:.2f} Å")
        logger.info(f"Cone angle (alpha): {alpha_cone_degrees}°")
        logger.info("Axial wall force constant (k_z): %.0f kcal/mol/Å²",
                    k_z.value_in_unit(openmmunit.kilocalorie_per_mole / openmmunit.angstrom**2))
        logger.info(f"Radial distances - min: {radial_distances.min():.2f}, max: {radial_distances.max():.2f}, mean: {radial_distances.mean():.2f} Å")
        logger.info(f"Axial distances - min: {z_distances.min():.2f}, max: {z_distances.max():.2f}, mean: {z_distances.mean():.2f} Å")

    results = {
        "host_index": host_indices,
        "guest_index": guest_indices,
        "k_xy": k_xy,
        "k_z": k_z,
        "z_cc": z_cc_quantity,
        "z_max": z_max_quantity,
        "alpha": alpha_quantity,
        "R_cylinder": R_cylinder_quantity,
        "force_group": 10,
        "force_name": "k_funnel_trajectory",
        "com_trajectory": guest_coms - host_coms,
        "host_com_trajectory": host_coms,
        "guest_com_trajectory": guest_coms,
        "radial_distances": radial_distances,
        "axial_distances": z_distances,
        "unbinding_axis": unbinding_axis,
        "trajectory_length": trajectory_length,
    }

    return results


def save_funnel_params(params_dict: Dict, filepath: str) -> None:
    """Serialise a funnel_params dict returned by generate_funnel_parameters_from_trajectory.

    Saves a single .npz file that can be reloaded with load_funnel_params() for
    post-hoc PMF correction or funnel visualisation.

    Units: z_cc, z_max, z_low, R_cylinder and trajectory arrays in Å; alpha in degrees;
    k_xy and k_z in kcal/mol/Å².
    """
    extra = {}
    if "z_low" in params_dict:
        extra["z_low_ang"] = np.array(params_dict["z_low"].value_in_unit(openmmunit.angstrom))
    if "k_z" in params_dict:
        extra["k_z_kcal_per_mol_per_ang2"] = np.array(params_dict["k_z"].value_in_unit(
            openmmunit.kilocalorie_per_mole / openmmunit.angstrom**2))
    np.savez(
        filepath,
        **extra,
        z_cc_ang=np.array(
            params_dict["z_cc"].value_in_unit(openmmunit.angstrom)
        ),
        z_max_ang=np.array(
            params_dict["z_max"].value_in_unit(openmmunit.angstrom)
        ),
        R_cylinder_ang=np.array(
            params_dict["R_cylinder"].value_in_unit(openmmunit.angstrom)
        ),
        alpha_deg=np.array(
            params_dict["alpha"].value_in_unit(openmmunit.degrees)
        ),
        k_xy_kcal_per_mol_per_ang2=np.array(
            params_dict["k_xy"].value_in_unit(
                openmmunit.kilocalorie_per_mole / openmmunit.angstrom**2
            )
        ),
        unbinding_axis=np.array(params_dict["unbinding_axis"]),
        host_index=np.array(params_dict["host_index"]),
        guest_index=np.array(params_dict["guest_index"]),
        com_trajectory=np.array(params_dict["com_trajectory"]),
        host_com_trajectory=np.array(params_dict["host_com_trajectory"]),
        guest_com_trajectory=np.array(params_dict["guest_com_trajectory"]),
        radial_distances=np.array(params_dict["radial_distances"]),
        axial_distances=np.array(params_dict["axial_distances"]),
        force_group=np.array(params_dict["force_group"]),
        trajectory_length=np.array(params_dict["trajectory_length"]),
        force_name=np.array(params_dict.get("force_name", "k_funnel_trajectory")),
        **({} if "axis_coeffs" not in params_dict else dict(
            frame_b_index=np.array(params_dict["frame_b_index"]),
            frame_c_index=np.array(params_dict["frame_c_index"]),
            axis_coeffs=np.array(params_dict["axis_coeffs"], dtype=float))),
        **({} if funnel_mode(params_dict) != "orient" else _orient_save_dict(params_dict)),
    )
    logger.info(f"Funnel parameters saved to {filepath}.npz")


_ORIENT_SAVED = {"orient_index": "orient_index", "orient_k": "orient_k_kj_per_mol", "orient_ref": "orient_ref_nm",
                 "orient_offset": "orient_offset_nm", "master_axis": "master_axis",
                 "master_offset": "master_offset_nm", "orient_rotation": "orient_rotation",
                 "orient_theta_deg": "orient_theta_deg"}


def _orient_save_dict(params):
    return {f: np.array(params[k], dtype=int if k == "orient_index" else float)
            for k, f in _ORIENT_SAVED.items() if k in params}


def load_funnel_params(filepath: str) -> Dict:
    """Load a funnel_params dict previously saved by save_funnel_params().

    Same structure as generate_funnel_parameters_from_trajectory() output,
    with OpenMM Quantities for the geometric parameters.

    Parameters
    ----------
    filepath : str
        Path to the .npz file (with or without the .npz extension).
    """
    data = np.load(filepath if filepath.endswith(".npz") else filepath + ".npz")
    kcal_a2 = openmmunit.kilocalorie_per_mole / openmmunit.angstrom**2
    # older files have no k_z; the force then falls back to k_xy
    extra = ({"k_z": float(data["k_z_kcal_per_mol_per_ang2"]) * kcal_a2}
             if "k_z_kcal_per_mol_per_ang2" in data.files else {})
    if "z_low_ang" in data.files:
        extra["z_low"] = float(data["z_low_ang"]) * openmmunit.angstrom
    if "axis_coeffs" in data.files:
        extra.update(frame_b_index=[int(i) for i in data["frame_b_index"]],
                     frame_c_index=[int(i) for i in data["frame_c_index"]],
                     axis_coeffs=np.array(data["axis_coeffs"], dtype=float))
    for k, f in _ORIENT_SAVED.items():
        if f in data.files:
            extra[k] = ([int(i) for i in data[f]] if k == "orient_index"
                        else float(data[f]) if data[f].ndim == 0 else np.array(data[f], dtype=float))
    return {
        **extra,
        "z_cc": float(data["z_cc_ang"]) * openmmunit.angstrom,
        "z_max": float(data["z_max_ang"]) * openmmunit.angstrom,
        "R_cylinder": float(data["R_cylinder_ang"]) * openmmunit.angstrom,
        "alpha": float(data["alpha_deg"]) * openmmunit.degrees,
        "k_xy": (
            float(data["k_xy_kcal_per_mol_per_ang2"])
            * openmmunit.kilocalorie_per_mole
            / openmmunit.angstrom**2
        ),
        "unbinding_axis": data["unbinding_axis"],
        "host_index": list(data["host_index"]),
        "guest_index": list(data["guest_index"]),
        "com_trajectory": data["com_trajectory"],
        "host_com_trajectory": data["host_com_trajectory"],
        "guest_com_trajectory": data["guest_com_trajectory"],
        "radial_distances": data["radial_distances"],
        "axial_distances": data["axial_distances"],
        "force_group": int(data["force_group"]),
        "force_name": str(data["force_name"]),
        "trajectory_length": int(data["trajectory_length"]),
    }


def create_funnel_force_from_trajectory_analysis(
    params_dict: Dict,
    system: System = None,
    axial_walls: bool = True,
) -> CustomCentroidBondForce:
    """
    Create a funnel force from trajectory analysis parameters.

    Parameters
    ----------
    params_dict : Dict
        Dictionary returned by `generate_funnel_parameters_from_trajectory`
    system : System, optional
        If provided, the force is also added to it.
    axial_walls : bool, optional
        False keeps only the lateral cone/cylinder walls (no ``z_max`` and ``z_low`` walls).

    Returns
    -------
    CustomCentroidBondForce
        Configured funnel force ready to use in metadynamics

    Funnel along the trajectory-derived axis, one-sided: a cone up to ``z_cc``,
    a cylinder up to a ``z_max`` wall, and a lower wall at ``z_low`` (default
    from :func:`funnel_lower_wall_ang`) so the guest cannot leave through z < 0.
    Params with a protein frame (:func:`upgrade_funnel_params`) rebuild the axis each step
    from the centroids of the host and the frame groups; the protein must stay whole
    (one molecule), as the centroids are taken without periodic images.
    Orientation-restrained params (:func:`orientation_restrained_params`) measure from
    COM(core) + v along the lab axis. ``system`` gets only the funnel force: the
    OrientationRestraintForce is added by ``MetadynamicsMD.add_orientation_restraint``,
    which checks the start orientation first.
    """

    host_index = params_dict["host_index"]
    guest_index = params_dict["guest_index"]
    k_xy = params_dict["k_xy"]
    z_cc = params_dict["z_cc"]
    z_max = params_dict["z_max"]
    alpha = params_dict["alpha"]
    R_cylinder = params_dict["R_cylinder"]
    k_z = params_dict.get("k_z", k_xy)
    force_group = params_dict.get("force_group", 10)
    force_name = params_dict.get("force_name", "k_funnel_trajectory")
    unbinding_axis = params_dict["unbinding_axis"]  # unit vector (3,)
    z_low = funnel_lower_wall_ang(params_dict) * openmmunit.angstrom
    frame = funnel_frame(params_dict)
    orient = is_orient_frame(frame)

    # r_z, r_xy are relative to the unbinding axis (nx, ny, nz).
    # z_max is a harmonic wall (U_axial), not a switch: gating the cylinder off
    # beyond z_max would free the guest and make the energy discontinuous.
    axial = ("U_axial = step(r_z - z_max)*0.5*k_z*(r_z - z_max)^2;"
             "U_lower = step(z_low - r_z)*0.5*k_z*(z_low - r_z)^2;")
    expression = (
        ("U_funnel + U_cylinder + U_axial + U_lower;" if axial_walls else "U_funnel + U_cylinder;")
        + "U_funnel = step(z_cc - r_z)*step(r_xy - R_funnel)*Wall_funnel;"
        "U_cylinder = step(r_z - z_cc)*step(r_xy - R_cylinder)*Wall_cylinder;"
        + (axial if axial_walls else "")
        + "Wall_funnel = 0.5 * k_xy * (r_xy - R_funnel)^2;"
        "Wall_cylinder = 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        # the cone keeps widening below z=0, down to the lower wall (as in FMAP)
        "R_funnel = max(z_cc-r_z, 0)*tan(alpha) + R_cylinder;"
        # r_xy = |n x r|: sqrt(r2 - r_z^2) cancels in single precision and has an
        # infinite derivative at 0, which the zero step() turns into NaN.
        "r_xy = sqrt(cx^2 + cy^2 + cz^2 + 1e-12);"
    )
    if orient:
        funnel = CustomCentroidBondForce(2, expression + (
            "cx = ny*rz - nz*ry;"
            "cy = nz*rx - nx*rz;"
            "cz = nx*ry - ny*rx;"
            "r_z = nx*rx + ny*ry + nz*rz;"
            "rx = x2-x1-fo_vx; ry = y2-y1-fo_vy; rz = z2-z1-fo_vz;"))
    elif frame is None:
        funnel = CustomCentroidBondForce(2, expression + (
            "cx = ny*(z2-z1) - nz*(y2-y1);"
            "cy = nz*(x2-x1) - nx*(z2-z1);"
            "cz = nx*(y2-y1) - ny*(x2-x1);"
            "r_z = nx*(x2-x1) + ny*(y2-y1) + nz*(z2-z1);"))
    else:
        # groups 3-5: frame B, C and the host again, mass-weighted like funnel z
        funnel = CustomCentroidBondForce(5, expression + _frame_definitions(1, 2, 5, 3, 4, frame["coeffs"]))

    funnel.setUsesPeriodicBoundaryConditions(False)
    funnel.setForceGroup(force_group)

    funnel.addGlobalParameter("k_xy", k_xy)
    if axial_walls:
        funnel.addGlobalParameter("k_z", k_z)
    funnel.addGlobalParameter("z_cc", z_cc)
    if axial_walls:
        funnel.addGlobalParameter("z_max", z_max)
        funnel.addGlobalParameter("z_low", z_low)
    funnel.addGlobalParameter("alpha", alpha)
    funnel.addGlobalParameter("R_cylinder", R_cylinder)
    # empty weights: mass-weighted centroids, as funnel z and the funnel geometry
    # orientation frame: group 1 is the core, the origin is COM(core) + v
    g1 = funnel.addGroup(frame["orient"] if orient else [int(i) for i in host_index])
    g2 = funnel.addGroup([int(i) for i in guest_index])
    if orient:
        n = np.asarray(unbinding_axis, dtype=float)
        n = n / np.linalg.norm(n)
        for name, val in zip(("nx", "ny", "nz") + ORIENT_OFFSET_PARAMS, (*n, *frame["offset"])):
            funnel.addGlobalParameter(name, float(val))
        funnel.addBond([g1, g2], [])
    elif frame is None:
        funnel.addGlobalParameter("nx", float(unbinding_axis[0]))
        funnel.addGlobalParameter("ny", float(unbinding_axis[1]))
        funnel.addGlobalParameter("nz", float(unbinding_axis[2]))
        funnel.addBond([g1, g2], [])
    else:
        # empty weights: mass-weighted centroids
        groups = [funnel.addGroup([int(i) for i in idx]) for idx in (frame["b"], frame["c"], host_index)]
        funnel.addBond([g1, g2, *groups], [])
    funnel.setName(force_name)

    if system is not None:
        system.addForce(funnel)
        logger.info(f"Added funnel force '{force_name}' to system")

    return funnel


# ---------------------------------------------------------------------------
# Funnel visualization helpers
# ---------------------------------------------------------------------------

def _perp_basis(axis: np.ndarray):
    """Return two unit vectors spanning the plane perpendicular to *axis*."""
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def _funnel_radius(z, z_cc: float, R_cylinder: float, alpha_deg: float):
    """Funnel radius at axial z (Å), as in the force expression."""
    return np.maximum(z_cc - np.asarray(z, float), 0.0) * np.tan(np.radians(alpha_deg)) + R_cylinder


def _funnel_cgo(
    host_com: np.ndarray,
    axis: np.ndarray,
    z_low: float,
    z_cc: float,
    z_max: float,
    R_cylinder: float,
    alpha_deg: float,
    n_pts: int = 24,
    dz: float = 1.0,
) -> list:
    """PyMOL CGO wireframe of the funnel from the lower wall to the z_max cap (Å).

    z is measured from ``host_com`` along ``axis``, as in the force expression.
    """
    from pymol.cgo import BEGIN, LINES, END, VERTEX, COLOR, LINEWIDTH

    host_com = np.asarray(host_com, dtype=float)
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    e1, e2 = _perp_basis(axis)
    z_all = np.unique(np.r_[np.arange(z_low, z_max, dz), z_cc, z_max])
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    rings = [[host_com + z * axis + _funnel_radius(z, z_cc, R_cylinder, alpha_deg)
              * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles] for z in z_all]

    cgo = [LINEWIDTH, 1.5, COLOR, 0.35, 0.35, 0.35, BEGIN, LINES]
    for pts in rings:
        for i in range(n_pts):
            cgo += [VERTEX, *map(float, pts[i]), VERTEX, *map(float, pts[(i + 1) % n_pts])]
    for k in range(0, n_pts, max(1, n_pts // 8)):
        for i in range(len(rings) - 1):
            cgo += [VERTEX, *map(float, rings[i][k]), VERTEX, *map(float, rings[i + 1][k])]
    cgo.append(END)
    return cgo


def _wall_disk_cgo(host_com, axis, z: float, radius: float, rgb, thickness: float = 0.3) -> list:
    """Flat disk (thin CGO cylinder) for a wall at axial z (Å)."""
    from pymol.cgo import CYLINDER

    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    p1 = np.asarray(host_com, dtype=float) + (z - thickness / 2) * axis
    p2 = p1 + thickness * axis
    return [CYLINDER, *map(float, p1), *map(float, p2), float(radius), *rgb, *rgb]



def write_funnel_pymol(
    funnel_params: dict,
    reference_pdb: str,
    milestone_files: list,
    outdir: str,
    protein_selection: str = "protein",
    ligand_selection: str = "resname UNK",
    output_format: str = "pse",
) -> str:
    """Create a standalone PyMOL PSE visualising the funnel potential geometry.

    Shows the protein (cartoon), the funnel boundary (CGO wireframe) from the lower
    wall to the z_max cap, the two walls as disks (blue lower, red upper), an arrow
    for the axis, one sphere per milestone at the
    ligand COM, and the sMD COM trajectory as dots coloured blue→red by progress.

    Args:
        funnel_params: Dict returned by
            :func:`~autopath.metadynamics.Funnel.generate_funnel_parameters_from_trajectory`.
        reference_pdb: Path to the solvated system PDB (for protein context).
        milestone_files: Ordered list of milestone PDB paths.
        outdir: Directory where ``funnel_visualization.pse`` is written.
        protein_selection: PyMOL/MDAnalysis selection string for the protein.
        ligand_selection: MDAnalysis selection string for the ligand.
        output_format: Only ``"pse"`` is supported; kept for API consistency.

    Returns:
        Absolute path of the written PSE file.
    """
    try:
        import pymol2
    except ImportError:
        raise ImportError("pymol2 is required for funnel visualization.")

    from pymol.cgo import COLOR, SPHERE, CYLINDER, CONE
    import colorsys
    import tempfile as _tempfile

    os.makedirs(outdir, exist_ok=True)
    outdir = Path(outdir)

    z_cc = funnel_params["z_cc"].value_in_unit(openmmunit.angstrom)
    R_cylinder = funnel_params["R_cylinder"].value_in_unit(openmmunit.angstrom)
    alpha_deg = funnel_params["alpha"].value_in_unit(openmmunit.degrees)
    unbinding_axis = np.array(funnel_params["unbinding_axis"], dtype=float)
    unbinding_axis /= np.linalg.norm(unbinding_axis)
    com_traj = funnel_params["com_trajectory"].astype(float)
    if "orient_rotation" in funnel_params:
        # sMD COM vectors are in the master orientation
        com_traj = com_traj @ np.asarray(funnel_params["orient_rotation"], float).T

    z_max = funnel_params["z_max"].value_in_unit(openmmunit.angstrom)
    z_low = funnel_lower_wall_ang(funnel_params)

    # com_traj is guest_com - host_com; add the reference PDB's host COM to place it.
    u_ref_anchor = mda.Universe(str(Path(reference_pdb).resolve()))
    prot_anchor = u_ref_anchor.select_atoms(protein_selection)
    prot_com_pdb = prot_anchor.center_of_mass()

    host_indices = list(funnel_params.get("host_index", []))
    if funnel_mode(funnel_params) == "orient":
        # origin COM(core) + v and the axis carried into the reference PDB's core orientation
        _o, unbinding_axis = funnel_origin_axis_at(funnel_params, u_ref_anchor.atoms.positions / 10.0,
                                                   u_ref_anchor.atoms.masses)
        host_com_ref = _o * 10.0
    elif host_indices:
        host_anchor = u_ref_anchor.select_atoms(
            f"index {' '.join(map(str, host_indices))}"
        )
        host_com_ref = (
            host_anchor.center_of_mass()
            if len(host_anchor) > 0
            else prot_com_pdb
        )
    else:
        host_com_ref = prot_com_pdb

    funnel_cgo = _funnel_cgo(host_com_ref, unbinding_axis, z_low, z_cc, z_max,
                             R_cylinder, alpha_deg)
    walls_cgo = (_wall_disk_cgo(host_com_ref, unbinding_axis, z_low,
                                _funnel_radius(z_low, z_cc, R_cylinder, alpha_deg), (0.2, 0.4, 0.9))
                 + _wall_disk_cgo(host_com_ref, unbinding_axis, z_max, R_cylinder, (0.9, 0.3, 0.2)))

    arrow_mid = (host_com_ref + z_cc * unbinding_axis).tolist()
    arrow_end = (host_com_ref + (z_max + 2.0) * unbinding_axis).tolist()
    axis_cgo = [
        CYLINDER,
        *(host_com_ref + z_low * unbinding_axis).tolist(), *arrow_mid,
        0.25, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0,
        CONE,
        *arrow_mid, *arrow_end,
        0.6, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 1.0, 1.0,
    ]

    n_ms = len(milestone_files)
    ms_data = []
    for i, ms_file in enumerate(milestone_files):
        try:
            u_ms = mda.Universe(reference_pdb)
            u_ms.atoms.positions = read_positions(ms_file) * 10.0
            lig = u_ms.select_atoms(f"({ligand_selection}) and not name H*")
            if len(lig) == 0:
                lig = u_ms.select_atoms(
                    "not (protein or resname HOH SOL WAT or name NA CL K MG)")
            if len(lig) == 0:
                continue
            if host_indices:
                host_ms = u_ms.select_atoms(
                    f"index {' '.join(map(str, host_indices))}"
                )
            if not host_indices or len(host_ms) == 0:
                host_ms = u_ms.select_atoms(protein_selection)
            if len(host_ms) == 0:
                continue
            lig_pos_ref = lig.positions - host_ms.center_of_mass() + host_com_ref
            t = i / max(n_ms - 1, 1)
            rgb = colorsys.hsv_to_rgb(2 / 3 * (1.0 - t), 1.0, 1.0)
            ms_data.append((f"milestone_{i}", lig, lig_pos_ref, rgb))
        except Exception as exc:
            logger.warning(f"Could not load milestone {ms_file}: {exc}")
            continue

    abs_positions = com_traj + host_com_ref
    sampled = abs_positions[::10]
    n_samp = max(1, len(sampled) - 1)
    traj_cgo = []
    for k, pos in enumerate(sampled):
        t = k / n_samp
        traj_cgo += [COLOR, float(t), 0.0, float(1.0 - t),
                     SPHERE, float(pos[0]), float(pos[1]), float(pos[2]), 0.25]

    pse_path = str(outdir / "funnel_visualization.pse")

    u_ref = mda.Universe(str(Path(reference_pdb).resolve()))
    prot_atoms = u_ref.select_atoms(protein_selection)

    with _tempfile.TemporaryDirectory() as tmpdir:
        prot_pdb = os.path.join(tmpdir, "protein.pdb")
        with mda.Writer(prot_pdb, prot_atoms.n_atoms) as w:
            w.write(prot_atoms)

        ms_pdb_data = []
        for obj_name, lig_ag, lig_pos_ref, rgb in ms_data:
            ms_pdb = os.path.join(tmpdir, f"{obj_name}.pdb")
            lig_ag.positions = lig_pos_ref
            with mda.Writer(ms_pdb, lig_ag.n_atoms) as w:
                w.write(lig_ag)
            ms_pdb_data.append((obj_name, ms_pdb, rgb))

        with pymol2.PyMOL() as pymol:
            c = pymol.cmd
            c.bg_color("white")
            c.set("antialias", 2)

            c.load(prot_pdb, "protein")
            c.hide("everything", "protein")
            c.show("cartoon", "protein")
            c.color("grey90", "protein")
            c.set("cartoon_transparency", 0.25, "protein")

            c.load_cgo(funnel_cgo, "funnel_boundary")
            c.load_cgo(walls_cgo, "funnel_walls")
            c.set("cgo_transparency", 0.5, "funnel_walls")
            c.load_cgo(axis_cgo, "unbinding_axis")

            for obj_name, ms_pdb, (r, g, b) in ms_pdb_data:
                c.load(ms_pdb, obj_name)
                c.show("sticks", obj_name)
                c.hide("lines", obj_name)
                color_name = f"ms_col_{obj_name}"
                c.set_color(color_name, [float(r), float(g), float(b)])
                c.color(color_name, obj_name)
                c.set("stick_radius", 0.15, obj_name)

            if traj_cgo:
                c.load_cgo(traj_cgo, "smd_com_traj")

            c.zoom("all", 5)
            c.save(pse_path)

    logger.info(f"Funnel visualization saved: {pse_path}")
    return pse_path
