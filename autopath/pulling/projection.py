"""Protein-frame projection CV, spring and parameters for projection pulls."""
import cvpack
import numpy as np
import openmm.unit as openmmunit

from autopath.metadynamics.Funnel import (_frame_definitions, funnel_coords_ang, funnel_frame, funnel_mode,
                                          upgrade_funnel_params)


def projection_params(axis, ligand_idx, pocket_idx, reference_positions_nm, topology, masses,
                      z_cc_ang, z_max_ang, R_cylinder_ang=2.0, alpha_deg=45.0,
                      k_xy_kcal_per_mol_per_ang2=10.0) -> dict:
    """Funnel params for a projection pull, in the protein frame.

    Parameters
    ----------
    axis : (3,) unbinding direction in the lab frame of ``reference_positions_nm``
    ligand_idx, pocket_idx : atom indices of the sMD pull groups (ligand = guest, pocket = host)
    reference_positions_nm : (N, 3) positions (nm)
    topology : openmm.app.Topology
    masses : (N,) particle masses (Da)
    z_cc_ang, z_max_ang : end of the cone and of the cylinder along the axis (Å)
    R_cylinder_ang : cylinder radius (Å)
    alpha_deg : cone half-angle (degrees)
    k_xy_kcal_per_mol_per_ang2 : lateral wall force constant

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
    params : funnel_params dict
    ligand_idx, pocket_idx : atom indices of the sMD pull groups
    """
    if funnel_mode(params) != "protein":
        raise ValueError("not supported: projection pulls need protein-frame funnel params "
                         "(axis_coeffs, frame_b_index, frame_c_index)")
    if sorted(int(i) for i in params["guest_index"]) != sorted(int(i) for i in ligand_idx):
        raise ValueError("not supported: funnel guest_index differs from the pulled ligand group")
    if sorted(int(i) for i in params["host_index"]) != sorted(int(i) for i in pocket_idx):
        raise ValueError("not supported: funnel host_index differs from the sMD pocket group")


def _groups_and_definitions(params):
    if funnel_mode(params) != "protein":
        raise ValueError("not supported: projection CV needs protein-frame funnel params")
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
    """Projection z (nm) from numpy positions, as :func:`projection_cv` measures it."""
    return funnel_coords_ang(params, positions_nm, masses)[0] / 10.0
