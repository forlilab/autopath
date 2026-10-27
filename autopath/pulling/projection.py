"""Protein-frame projection CV, spring and parameters for projection pulls."""
import hashlib
import itertools

import cvpack
import numpy as np
import openmm.unit as openmmunit

from autopath.metadynamics.Funnel import (PROTEIN_FRAME_KEYS, _frame_definitions, _molecules, funnel_coords_ang,
                                          funnel_frame, funnel_mode, pca_unbinding_axis, upgrade_funnel_params,
                                          z_where_guest_reaches_bulk)

MAX_BOND_NM = 0.3


def triclinic_minimum_image(d, box_nm) -> np.ndarray:
    """Periodic image of ``d`` as OpenMM reduces it: by box vector c, then b, then a.

    Parameters
    ----------
    d : array_like, shape (3,) or (M, 3)
        Displacement(s) (nm).
    box_nm : array_like, shape (3, 3)
        Box vectors a, b, c as rows (nm), in OpenMM's reduced form.

    Returns
    -------
    numpy.ndarray
        Same shape as ``d``.
    """
    box = np.asarray(box_nm, dtype=float)
    d = np.array(d, dtype=float)
    for k in (2, 1, 0):
        d = d - np.round(d[..., k] / box[k, k])[..., None] * box[k]
    return d


def projection_pbc_limit(n, box_nm):
    """Largest z along ``n`` before the ligand–pocket vector leaves the pocket's periodic cell.

    min over the lattice vectors T = i·a + j·b + k·c (i, j, k in {-1, 0, 1}, not all 0, n·T ≠ 0)
    of |T|² / (2 |n·T|).

    Parameters
    ----------
    n : array_like, shape (3,)
        Unit pull axis.
    box_nm : array_like, shape (3, 3)
        Box vectors as rows (nm).

    Returns
    -------
    (float, tuple of int)
        The limit (nm) and the (i, j, k) of the limiting T.
    """
    n = np.asarray(n, dtype=float)
    box = np.asarray(box_nm, dtype=float)
    best, best_ijk = np.inf, None
    for ijk in itertools.product((-1, 0, 1), repeat=3):
        if ijk == (0, 0, 0):
            continue
        T = np.asarray(ijk, dtype=float) @ box
        nT = abs(float(n @ T))
        if nT < 1e-12:
            continue
        val = float(T @ T) / (2.0 * nT)
        if val < best:
            best, best_ijk = val, ijk
    return best, best_ijk


def frame_index_sha1(params) -> str:
    """sha1 hex digest of ``"b=<sorted frame_b_index>;c=<sorted frame_c_index>"`` (comma-separated)."""
    b = ",".join(str(i) for i in sorted(int(i) for i in params["frame_b_index"]))
    c = ",".join(str(i) for i in sorted(int(i) for i in params["frame_c_index"]))
    return hashlib.sha1(f"b={b};c={c}".encode()).hexdigest()


def check_projection_groups_whole(params, positions_nm, masses, box_nm, topology, max_bond_nm=MAX_BOND_NM,
                                  tol_nm=0.05) -> None:
    """Raise RuntimeError if a projection group is split across periodic images.

    The projection centroids use no periodic images. A group is whole when every topology bond of the
    molecules holding the pocket, ligand, frame b and frame c atoms is shorter than ``max_bond_nm`` as
    stored, and each COM−COM vector (ligand, frame b, frame c to the pocket) equals its
    :func:`triclinic_minimum_image` within ``tol_nm``.

    Parameters
    ----------
    params : dict
        Protein-frame funnel_params.
    positions_nm : array_like, shape (N, 3)
        Positions as stored by the context (nm, not wrapped).
    masses : array_like, shape (N,)
        Particle masses (Da).
    box_nm : array_like, shape (3, 3)
        Box vectors as rows (nm).
    topology : openmm.app.Topology
    max_bond_nm, tol_nm : float
    """
    pos = np.asarray(positions_nm, dtype=float)
    m = np.asarray(masses, dtype=float)
    groups = {"pocket": params["host_index"], "ligand": params["guest_index"],
              "frame_b": params["frame_b_index"], "frame_c": params["frame_c_index"]}
    groups = {k: np.array([int(i) for i in v], dtype=int) for k, v in groups.items()}
    mol = _molecules(topology)
    bonds = np.array([(a.index, b.index) for a, b in topology.bonds()], dtype=int).reshape(-1, 2)
    if len(bonds):
        length = np.linalg.norm(pos[bonds[:, 0]] - pos[bonds[:, 1]], axis=1)
        mols = np.unique(np.concatenate([mol[idx] for idx in groups.values()]))
        bad = np.isin(mol[bonds[:, 0]], mols) & (length >= max_bond_nm)
        if bad.any():
            worst = np.flatnonzero(bad)[np.argmax(length[bad])]
            i, j = bonds[worst]
            names = [k for k, idx in groups.items() if mol[i] in mol[idx]]
            raise RuntimeError(f"projection pull: {'/'.join(names)} group molecule has bond {i}-{j} of "
                               f"{length[worst]:.3f} nm (>= {max_bond_nm} nm): split across periodic images")
    coms = {k: np.average(pos[idx], axis=0, weights=m[idx]) for k, idx in groups.items()}
    for name in ("ligand", "frame_b", "frame_c"):
        d = coms[name] - coms["pocket"]
        d_min = triclinic_minimum_image(d, box_nm)
        if np.linalg.norm(d - d_min) > tol_nm:
            raise RuntimeError(f"projection pull: {name}-pocket COM vector {np.round(d, 3)} nm is not "
                               f"its minimum image {np.round(d_min, 3)} nm: groups in different images")


def projection_params(axis, ligand_idx, pocket_idx, reference_positions_nm, topology, masses,
                      z_cc_ang, z_max_ang, R_cylinder_ang=2.0, alpha_deg=45.0,
                      k_xy_kcal_per_mol_per_ang2=10.0) -> dict:
    """Funnel params for a projection pull, in the protein frame.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Unbinding direction in the lab frame of ``reference_positions_nm``.
    ligand_idx, pocket_idx : list of int
        Atom indices of the sMD pull groups (ligand = guest, pocket = host).
    reference_positions_nm : array_like, shape (N, 3)
        Positions (nm).
    topology : openmm.app.Topology
    masses : array_like, shape (N,)
        Particle masses (Da).
    z_cc_ang, z_max_ang : float
        End of the cone and of the cylinder along the axis (Å).
    R_cylinder_ang : float
        Cylinder radius (Å).
    alpha_deg : float
        Cone half-angle (degrees).
    k_xy_kcal_per_mol_per_ang2 : float
        Lateral wall force constant (kcal/mol/Å²).

    Returns
    -------
    dict
        funnel_params for ``create_funnel_force_from_trajectory_analysis``, with
        ``frame_b_index``, ``frame_c_index`` and ``axis_coeffs`` from ``upgrade_funnel_params``.
    """
    n = np.asarray(axis, dtype=float)
    params = {
        "unbinding_axis": n / np.linalg.norm(n),
        "guest_index": [int(i) for i in ligand_idx],
        "host_index": [int(i) for i in pocket_idx],
        "z_cc": z_cc_ang * openmmunit.angstrom,
        "z_max": z_max_ang * openmmunit.angstrom,
        "R_cylinder": R_cylinder_ang * openmmunit.angstrom,
        "alpha": alpha_deg * openmmunit.degrees,
        "k_xy": k_xy_kcal_per_mol_per_ang2 * openmmunit.kilocalorie_per_mole / openmmunit.angstrom**2,
    }
    return upgrade_funnel_params(params, reference_positions_nm, topology, masses)


def check_projection_params(params, ligand_idx, pocket_idx) -> None:
    """Raise ValueError("not supported: ...") unless ``params`` fit a projection pull on these groups.

    Parameters
    ----------
    params : dict
        funnel_params.
    ligand_idx, pocket_idx : list of int
        Atom indices of the sMD pull groups.
    """
    _require_protein_frame(params)
    if sorted(int(i) for i in params["guest_index"]) != sorted(int(i) for i in ligand_idx):
        raise ValueError("not supported: funnel guest_index differs from the pulled ligand group")
    if sorted(int(i) for i in params["host_index"]) != sorted(int(i) for i in pocket_idx):
        raise ValueError("not supported: funnel host_index differs from the sMD pocket group")


def _require_protein_frame(params):
    try:
        mode = funnel_mode(params)
    except ValueError as e:
        raise ValueError(f"not supported: {e}") from e
    missing = [k for k in PROTEIN_FRAME_KEYS if k not in params]
    if mode != "protein" or missing:
        raise ValueError(f"not supported: projection pulls need protein-frame funnel params; missing {missing}")


def _groups_and_definitions(params):
    _require_protein_frame(params)
    frame = funnel_frame(params)
    groups = [[int(i) for i in params["host_index"]], [int(i) for i in params["guest_index"]],
              frame["b"], frame["c"]]
    return groups, _frame_definitions(1, 2, 1, 3, 4, frame["coeffs"], radial=False)


def projection_cv(params):
    """Unclamped z = n·(COM_ligand − COM_pocket) (nm), n rebuilt from the protein frame.

    Parameters
    ----------
    params : protein-frame funnel_params dict

    Returns
    -------
    cvpack.CentroidFunction
    """
    groups, defs = _groups_and_definitions(params)
    return cvpack.CentroidFunction("r_z;" + defs, openmmunit.nanometers, groups,
                                   weighByMass=True, pbc=False)


def projection_spring(params, k_kj_per_mol_nm2):
    """Harmonic spring 0.5*fc_pull*(z − r0_smd)^2 on the projection z.

    Parameters
    ----------
    params : protein-frame funnel_params dict
    k_kj_per_mol_nm2 : float
        Initial ``fc_pull`` (kJ/mol/nm²); ``r0_smd`` starts at 0.

    Returns
    -------
    cvpack.CentroidFunction
        With global parameters ``r0_smd`` and ``fc_pull``.
    """
    groups, defs = _groups_and_definitions(params)
    spring = cvpack.CentroidFunction("0.5*fc_pull*(r_z - r0_smd)^2;" + defs, openmmunit.kilojoules_per_mole,
                                     groups, weighByMass=True, pbc=False)
    spring.addGlobalParameter("r0_smd", 0.0)
    spring.addGlobalParameter("fc_pull", float(k_kj_per_mol_nm2))
    return spring


def projection_z_nm(params, positions_nm, masses) -> float:
    """Projection z (nm) from numpy positions, as :func:`projection_cv` measures it.

    Parameters
    ----------
    params : dict
        Protein-frame funnel_params.
    positions_nm : array_like, shape (N, 3)
        Positions (nm), not wrapped into the box.
    masses : array_like, shape (N,)
        Particle masses (Da).

    Returns
    -------
    float
    """
    return funnel_coords_ang(params, positions_nm, masses)[0] / 10.0


def axis_from_pulls(universe, ligand_idx, pocket_idx, ligand_bulk_idx, frames=None, z_cc_percentile=95.0,
                    bulk_cutoff_ang=10.0, protein_idx=None, topology=None, R_cylinder_ang=2.0, alpha_deg=45.0,
                    k_xy_kcal_per_mol_per_ang2=10.0) -> dict:
    """Pull axis, z_cc, z_max and protein-frame params from existing pull trajectories.

    The axis is the leading principal axis of the ligand COM relative to the pocket COM, oriented
    bound -> unbound, so protein drift or tumbling does not enter it. z is measured with the pulled
    groups (mass-weighted COMs, origin at the pocket COM). z_max is the
    pulled-group z once the whole ligand, moved rigidly along the axis from the first frame, clears
    ``bulk_cutoff_ang`` from the protein atoms. Trajectories must be imaged and whole: a pulled-group COM
    step above half the smallest box length raises (checked only when the universe has box dimensions).

    Parameters
    ----------
    universe : MDAnalysis.Universe
    ligand_idx, pocket_idx : list of int
        Atom indices of the pulled ligand and pocket groups.
    ligand_bulk_idx : list of int
        Atom indices of the whole ligand; ``ligand_idx`` must be a subset.
    frames : sequence of int, optional
        Frames of one path; all frames if None.
    z_cc_percentile : float
        Percentile of the pulled-group z that sets z_cc.
    bulk_cutoff_ang : float
        Clearance from the protein (Å).
    protein_idx : list of int, optional
        Atoms the ligand must clear; default ``universe.select_atoms("protein")``. Ligand atoms are removed.
    topology : openmm.app.Topology, optional
        Topology of the universe's atoms (same order), for the protein-frame groups, which need the
        bonds and backbone names an MDAnalysis universe often lacks. Without it ``params`` is None.
    R_cylinder_ang, alpha_deg, k_xy_kcal_per_mol_per_ang2 : float
        Restraint geometry of ``params``, see :func:`projection_params`.

    Returns
    -------
    dict
        ``unbinding_axis`` (bound -> unbound, lab frame of the first frame), ``z_cc_ang``, ``z_max_ang``
        and ``params``: :func:`projection_params` built from the first frame (positions, universe
        masses), or None without ``topology``.

    Raises
    ------
    ValueError
        "not supported: ..." for <2 frames, no protein atoms, an unimaged jump, ``z_max_ang <= z_cc_ang``,
        or a topology whose atom count differs from the universe's.
    """
    frames = list(range(universe.trajectory.n_frames)) if frames is None else [int(f) for f in frames]
    if len(frames) < 2:
        raise ValueError("not supported: axis_from_pulls needs at least 2 frames")
    if topology is not None and topology.getNumAtoms() != universe.atoms.n_atoms:
        raise ValueError(f"not supported: topology has {topology.getNumAtoms()} atoms, the universe "
                         f"{universe.atoms.n_atoms}")
    lig = universe.atoms[list(ligand_idx)]
    pocket = universe.atoms[list(pocket_idx)]
    bulk = universe.atoms[list(ligand_bulk_idx)]
    g, h = [], []
    for f in frames:
        universe.trajectory[f]
        g.append(lig.center_of_mass())
        h.append(pocket.center_of_mass())
    g, h = np.array(g), np.array(h)
    box = universe.trajectory[frames[0]].dimensions
    if box is not None and box[:3].min() > 0:
        limit = 0.5 * float(box[:3].min())
        step = max(np.linalg.norm(np.diff(g, axis=0), axis=1).max(), np.linalg.norm(np.diff(h, axis=0), axis=1).max())
        if step > limit:
            raise ValueError(f"not supported: pulled-group COM step {step:.1f} A exceeds half the box "
                             f"({limit:.1f} A); the trajectory must be imaged and whole")
    axis = pca_unbinding_axis(g - h, np.zeros_like(h))
    z = (g - h) @ axis
    z_cc = float(np.percentile(z, z_cc_percentile))

    if protein_idx is None:
        protein = universe.select_atoms("protein")
    else:
        protein = universe.atoms[list(protein_idx)]
    protein = protein - bulk
    if protein.n_atoms == 0:
        raise ValueError("not supported: no protein atoms to clear for z_max")
    universe.trajectory[frames[0]]
    host_com = h[0]
    sel = "index " + " ".join(str(int(i)) for i in protein.indices)
    z_bulk = z_where_guest_reaches_bulk(universe, bulk, host_com, axis, cutoff_ang=bulk_cutoff_ang,
                                        protein_selection=sel)
    dz = z_bulk - float(np.dot(bulk.center_of_mass() - host_com, axis))
    z_max = float(z[0] + dz)
    if z_max <= z_cc:
        raise ValueError(f"not supported: z_max ({z_max:.2f} A) is not beyond z_cc ({z_cc:.2f} A)")
    params = None
    if topology is not None:
        universe.trajectory[frames[0]]
        params = projection_params(axis, ligand_idx, pocket_idx, universe.atoms.positions / 10.0, topology,
                                   universe.atoms.masses, z_cc, z_max, R_cylinder_ang=R_cylinder_ang,
                                   alpha_deg=alpha_deg, k_xy_kcal_per_mol_per_ang2=k_xy_kcal_per_mol_per_ang2)
    return {"unbinding_axis": axis, "z_cc_ang": z_cc, "z_max_ang": z_max, "params": params}
