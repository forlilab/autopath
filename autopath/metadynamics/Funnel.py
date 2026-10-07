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


def funnel_z_ang(positions_nm, host_idx, guest_idx, masses, axis) -> float:
    """Guest position on the funnel axis (Å): n·(COM_guest − COM_host), mass-weighted."""
    pos = np.asarray(positions_nm, dtype=float)
    m = np.asarray(masses, dtype=float)
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    host = np.average(pos[host_idx], axis=0, weights=m[host_idx])
    guest = np.average(pos[guest_idx], axis=0, weights=m[guest_idx])
    return float(np.dot(guest - host, n)) * 10.0




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

    g1 = funnel.addGroup(host_index, [1.0 for i in range(len(host_index))])
    g2 = funnel.addGroup(guest_index, [1.0 for i in range(len(guest_index))])

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

    Units: z_cc, z_max, R_cylinder and trajectory arrays in Å; alpha in degrees;
    k_xy and k_z in kcal/mol/Å².
    """
    extra = {}
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
    )
    logger.info(f"Funnel parameters saved to {filepath}.npz")


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
) -> CustomCentroidBondForce:
    """
    Create a funnel force from trajectory analysis parameters.

    Parameters
    ----------
    params_dict : Dict
        Dictionary returned by `generate_funnel_parameters_from_trajectory`
    system : System, optional
        If provided, the force is also added to it.

    Returns
    -------
    CustomCentroidBondForce
        Configured funnel force ready to use in metadynamics

    Funnel along the trajectory-derived axis, one-sided: a cone up to ``z_cc``,
    a cylinder up to a ``z_max`` wall, and a lower wall at ``z_low`` (default
    from :func:`funnel_lower_wall_ang`) so the guest cannot leave through z < 0.
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

    # r_z, r_xy are relative to the unbinding axis (nx, ny, nz).
    funnel = CustomCentroidBondForce(
        2,
        # z_max is a harmonic wall (U_axial), not a switch: gating the cylinder off
        # beyond z_max would free the guest and make the energy discontinuous.
        "U_funnel + U_cylinder + U_axial + U_lower;"
        "U_funnel = step(z_cc - r_z)*step(r_xy - R_funnel)*Wall_funnel;"
        "U_cylinder = step(r_z - z_cc)*step(r_xy - R_cylinder)*Wall_cylinder;"
        "U_axial = step(r_z - z_max)*0.5*k_z*(r_z - z_max)^2;"
        "U_lower = step(z_low - r_z)*0.5*k_z*(z_low - r_z)^2;"
        "Wall_funnel = 0.5 * k_xy * (r_xy - R_funnel)^2;"
        "Wall_cylinder = 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        # the cone keeps widening below z=0, down to the lower wall (as in FMAP)
        "R_funnel = max(z_cc-r_z, 0)*tan(alpha) + R_cylinder;"
        # r_xy = |n x r|: sqrt(r2 - r_z^2) cancels in single precision and has an
        # infinite derivative at 0, which the zero step() turns into NaN.
        "r_xy = sqrt(cx^2 + cy^2 + cz^2 + 1e-12);"
        "cx = ny*(z2-z1) - nz*(y2-y1);"
        "cy = nz*(x2-x1) - nx*(z2-z1);"
        "cz = nx*(y2-y1) - ny*(x2-x1);"
        "r_z = nx*(x2-x1) + ny*(y2-y1) + nz*(z2-z1);",
    )

    funnel.setUsesPeriodicBoundaryConditions(False)
    funnel.setForceGroup(force_group)

    funnel.addGlobalParameter("k_xy", k_xy)
    funnel.addGlobalParameter("k_z", k_z)
    funnel.addGlobalParameter("z_cc", z_cc)
    funnel.addGlobalParameter("z_max", z_max)
    funnel.addGlobalParameter("z_low", z_low)
    funnel.addGlobalParameter("alpha", alpha)
    funnel.addGlobalParameter("R_cylinder", R_cylinder)
    funnel.addGlobalParameter("nx", float(unbinding_axis[0]))
    funnel.addGlobalParameter("ny", float(unbinding_axis[1]))
    funnel.addGlobalParameter("nz", float(unbinding_axis[2]))

    g1 = funnel.addGroup(host_index, [1.0 for i in range(len(host_index))])
    g2 = funnel.addGroup(guest_index, [1.0 for i in range(len(guest_index))])

    funnel.addBond([g1, g2], [])
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

    z_max = funnel_params["z_max"].value_in_unit(openmmunit.angstrom)
    z_low = funnel_lower_wall_ang(funnel_params)

    # com_traj is guest_com - host_com; add the reference PDB's host COM to place it.
    u_ref_anchor = mda.Universe(str(Path(reference_pdb).resolve()))
    prot_anchor = u_ref_anchor.select_atoms(protein_selection)
    prot_com_pdb = prot_anchor.center_of_mass()

    host_indices = list(funnel_params.get("host_index", []))
    if host_indices:
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
