import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from openmm import *
from openmm.app import *
import openmm.unit as openmmunit

import MDAnalysis as mda

logger = logging.getLogger("autopath")


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
    Applies a funnel potential restraint to a guest molecule.
    Limongelli, V., Bonomi, M., & Parrinello, M. (2013). Funnel metadynamics as accurate binding free-energy method. Proceedings of the National Academy of Sciences, 110(16), 6358-6363.
    https://github.com/jeff231li/funnel_potential

    Note
    ----
    This is the simplified Z-axis-aligned funnel variant.
    For a general PCA-derived unbinding-axis funnel with a z_max cap,
    use :func:`create_funnel_force_from_trajectory_analysis` instead.
    """

    funnel = CustomCentroidBondForce(
        2,
        "U_funnel + U_cylinder;"
        "U_funnel = step(z_cc - abs(r_z))*step(r_xy - R_funnel)*Wall_funnel;"
        "U_cylinder = step(abs(r_z) - z_cc)*step(r_xy - R_cylinder)*Wall_cylinder;"
        "Wall_funnel = 0.5 * k_xy * (r_xy - R_funnel)^2;"
        "Wall_cylinder = 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        "R_funnel = (z_cc-abs(r_z))*tan(alpha) + R_cylinder;"
        "r_xy = sqrt((x2 - x1)^2 + (y2 - y1)^2);"
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
    use_pca: bool = True,
    percentile_z: float = 95.0,
    z_cc_ang: Optional[float] = None,
    z_max_buffer_ang: float = 3.0,
    R_cylinder_ang: float = 2.0,
    percentile_r_funnel: float = 85.0,
    alpha_cone_degrees: float = 25.0,
    stride: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    Analyze MDAnalysis trajectory to generate optimal funnel potential parameters
    for guiding unbinding along the observed pathway.

    This function analyzes the conformational changes in a trajectory to determine:
    - The unbinding axis and pathway geometry
    - Appropriate funnel cone parameters (z_cc, alpha, R_cylinder)
    - Center of mass trajectories for host and guest molecules

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
    use_pca : bool
        If True, use PCA to determine the unbinding axis. If False, use the axis
        with largest COM displacement (default: True)
    percentile_z : float
        Percentile of Z-displacement to use for z_cc parameter (0-100, default: 95).
        Ignored when z_cc_ang is provided.
    z_cc_ang : float, optional
        Explicit cone-to-cylinder transition distance in Å. When set, overrides the
        value derived from percentile_z. Useful to shorten the cone and extend the
        cylinder region (smaller value → shorter cone, longer cylinder).
    z_max_buffer_ang : float
        Extra length in Å added beyond the farthest axial distance in the sMD
        trajectory to set z_max (the hard cap of the cylinder). Increase this if
        the sMD stopped just after the transition state and you want more solvent
        sampling in metadynamics (default: 3.0 Å).
    R_cylinder_ang : float
        Cylinder radius in Å for the unbound-state restraint (default: 2.0).
        Limongelli 2013 recommends 1–3 Å.
    percentile_r_funnel : float
        Percentile of radial displacement at bound state, used only for logging (default: 85)
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

    # Ensure the axis points from the bound state (frame 0) toward the unbound state
    # (last frame). np.linalg.eigh returns eigenvectors with arbitrary sign, so the
    # PCA branch may produce an inward-pointing axis, inverting z_cc and the visualization.
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

    R_cylinder = float(R_cylinder_ang)

    n_bound = max(10, int(0.1 * trajectory_length))
    R_funnel_bound = np.percentile(radial_distances[:n_bound], percentile_r_funnel)

    z_max = float(np.max(z_distances)) + float(z_max_buffer_ang)

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
        logger.info(f"Cylinder cap (z_max): {z_max:.2f} Å  (sMD max {np.max(z_distances):.2f} Å + {z_max_buffer_ang:.1f} Å buffer)")
        logger.info(f"Cylinder radius (R_cylinder): {R_cylinder:.2f} Å  (design parameter, Limongelli 2013: 1–3 Å)")
        logger.info(f"Bound-state radial spread (informational): {R_funnel_bound:.2f} Å")
        logger.info(f"Cone angle (alpha): {alpha_cone_degrees}°")
        logger.info(f"Radial distances - min: {radial_distances.min():.2f}, max: {radial_distances.max():.2f}, mean: {radial_distances.mean():.2f} Å")
        logger.info(f"Axial distances - min: {z_distances.min():.2f}, max: {z_distances.max():.2f}, mean: {z_distances.mean():.2f} Å")

    results = {
        "host_index": host_indices,
        "guest_index": guest_indices,
        "k_xy": k_xy,
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

    Geometric scalars are stored in their natural units:
        z_cc         → angstroms
        R_cylinder   → angstroms
        alpha        → degrees
        k_xy         → kcal/mol/Å²

    Trajectory arrays (angstroms) and metadata are stored as-is.

    Note
    ----
    ``force_name`` is now saved and restored faithfully.
    """
    np.savez(
        filepath,
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

    The returned dict has the same structure as the one produced by
    generate_funnel_parameters_from_trajectory(), including OpenMM Quantities
    for the geometric parameters, so it can be passed directly to
    create_funnel_force_from_trajectory_analysis() or correct_fe_for_funnel().

    Parameters
    ----------
    filepath : str
        Path to the .npz file (with or without the .npz extension).
    """
    data = np.load(filepath if filepath.endswith(".npz") else filepath + ".npz")
    return {
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

    Uses the output dictionary from `generate_funnel_parameters_from_trajectory`
    to create a CustomCentroidBondForce configured for the observed unbinding pathway.

    Parameters
    ----------
    params_dict : Dict
        Dictionary returned by `generate_funnel_parameters_from_trajectory`
    system : System, optional
        OpenMM System object. If provided, the force will be added to the system
        and the function returns the force. If None, only the force is created.

    Returns
    -------
    CustomCentroidBondForce
        Configured funnel force ready to use in metadynamics

    Note
    ----
    This is the general PCA-axis funnel with a ``z_max`` hard cap.
    For a simpler Z-axis-aligned funnel (no trajectory analysis required),
    use :func:`add_funnel_restraints` instead.
    """

    host_index = params_dict["host_index"]
    guest_index = params_dict["guest_index"]
    k_xy = params_dict["k_xy"]
    z_cc = params_dict["z_cc"]
    z_max = params_dict["z_max"]
    alpha = params_dict["alpha"]
    R_cylinder = params_dict["R_cylinder"]
    force_group = params_dict.get("force_group", 10)
    force_name = params_dict.get("force_name", "k_funnel_trajectory")
    unbinding_axis = params_dict["unbinding_axis"]  # unit vector (3,)

    # Create the funnel force.
    # r_z and r_xy are computed relative to the PCA-derived unbinding axis so that the
    # funnel geometry matches the parameters (z_cc, R_cylinder) which were derived in that
    # frame.  nx,ny,nz are the components of the unit vector along the unbinding direction.
    funnel = CustomCentroidBondForce(
        2,
        "U_funnel + U_cylinder;"
        "U_funnel = step(z_cc - abs(r_z))*step(r_xy - R_funnel)*Wall_funnel;"
        "U_cylinder = step(abs(r_z) - z_cc)*step(z_max - abs(r_z))*step(r_xy - R_cylinder)*Wall_cylinder;"
        "Wall_funnel = 0.5 * k_xy * (r_xy - R_funnel)^2;"
        "Wall_cylinder = 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        "R_funnel = (z_cc-abs(r_z))*tan(alpha) + R_cylinder;"
        "r_xy = sqrt(max(0, r2 - r_z*r_z));"
        "r_z = nx*(x2-x1) + ny*(y2-y1) + nz*(z2-z1);"
        "r2 = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2;",
    )

    funnel.setUsesPeriodicBoundaryConditions(False)
    funnel.setForceGroup(force_group)

    funnel.addGlobalParameter("k_xy", k_xy)
    funnel.addGlobalParameter("z_cc", z_cc)
    funnel.addGlobalParameter("z_max", z_max)
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


def _funnel_cgo(
    center: np.ndarray,
    axis: np.ndarray,
    z_cc: float,
    R_cylinder: float,
    alpha_deg: float,
    n_z: int = 15,
    n_pts: int = 24,
    extension: float = 5.0,
    z_offset: float = 0.0,
) -> list:
    """Build a PyMOL CGO wireframe for the funnel boundary (all lengths in Å).

    ``center`` is the visual anchor (typically the bound-state ligand COM).
    ``z_offset`` is the axial distance from the force anchor (protein COM) to
    ``center``, so that ring radii match the force expression exactly:
    R = R_cylinder + max(0, z_cc - (z_offset + z_visual)) * tan(alpha_deg).
    Rings are drawn from z=0 (at ``center``) outward to the cylinder end.
    """
    from pymol.cgo import BEGIN, LINES, END, VERTEX, COLOR, LINEWIDTH

    center = np.array(center, dtype=float)
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    alpha_rad = np.deg2rad(alpha_deg)
    e1, e2 = _perp_basis(axis)

    z_cc_visual = z_cc - z_offset

    def make_ring(z_level: float, radius: float) -> list:
        rc = center + z_level * axis
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        return [rc + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles]

    z_cone = np.linspace(0.0, max(z_cc_visual, 0.0), n_z)
    n_cyl_rings = max(3, int(extension / 5.0) + 1)
    z_cyl = [z_cc_visual + extension * t for t in np.linspace(0, 1, n_cyl_rings, endpoint=True)]
    z_all = list(z_cone) + z_cyl

    rings = []
    for z in z_all:
        r = R_cylinder + max(0.0, (z_cc_visual - z)) * np.tan(alpha_rad)
        rings.append(make_ring(z, r))

    cgo = [LINEWIDTH, 1.5, COLOR, 0.35, 0.35, 0.35, BEGIN, LINES]
    for pts in rings:
        for i in range(n_pts):
            p1, p2 = pts[i], pts[(i + 1) % n_pts]
            cgo += [VERTEX, float(p1[0]), float(p1[1]), float(p1[2]),
                    VERTEX, float(p2[0]), float(p2[1]), float(p2[2])]
    cgo.append(END)

    step = max(1, n_pts // 8)
    cgo += [BEGIN, LINES]
    for k in range(0, n_pts, step):
        for i in range(len(rings) - 1):
            p1, p2 = rings[i][k], rings[i + 1][k]
            cgo += [VERTEX, float(p1[0]), float(p1[1]), float(p1[2]),
                    VERTEX, float(p2[0]), float(p2[1]), float(p2[2])]
    cgo.append(END)

    return cgo


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

    Shows the protein (cartoon), the funnel boundary (CGO wireframe) oriented along the
    PCA-derived unbinding axis, an arrow for the axis, one sphere per milestone at the
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
    extension = max(5.0, z_max - z_cc)

    # Re-anchor to the reference PDB coordinate frame.
    # com_traj is a relative quantity (guest_com − host_com) computed during
    # generate_funnel_parameters_from_trajectory.  To place it in the reference
    # PDB frame we must add the HOST COM from that PDB — the same atoms used to
    # define the funnel force, stored in funnel_params["host_index"].
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

    funnel_center = host_com_ref + com_traj[0]
    z_offset = float(np.dot(com_traj[0], unbinding_axis))
    z_offset = max(z_offset, 0.0)

    funnel_cgo = _funnel_cgo(
        funnel_center, unbinding_axis, z_cc, R_cylinder, alpha_deg,
        extension=extension, z_offset=z_offset,
    )

    z_cc_visual = z_cc - z_offset
    arrow_mid = (funnel_center + z_cc_visual * unbinding_axis).tolist()
    arrow_end = (funnel_center + (z_cc_visual + extension + 2.0) * unbinding_axis).tolist()
    axis_cgo = [
        CYLINDER,
        *funnel_center.tolist(), *arrow_mid,
        0.25, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0,
        CONE,
        *arrow_mid, *arrow_end,
        0.6, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 1.0, 1.0,
    ]

    n_ms = len(milestone_files)
    ms_data = []
    for i, ms_file in enumerate(milestone_files):
        try:
            u_ms = mda.Universe(ms_file)
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
