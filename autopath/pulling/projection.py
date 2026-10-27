"""Protein-frame projection CV, spring and parameters for projection pulls."""
import cvpack
import numpy as np
import openmm.unit as openmmunit

from autopath.metadynamics.Funnel import (PROTEIN_FRAME_KEYS, _frame_definitions, funnel_coords_ang, funnel_frame,
                                          funnel_mode, upgrade_funnel_params)


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
