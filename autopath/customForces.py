from typing import Union, Tuple, Optional, List, Dict
from openmm import *
from openmm.app import *
import openmm.app as app
import openmm.unit as openmmunit
import numpy as np

import logging
logger = logging.getLogger("autopath")

def print_current_forces(system: System = None) -> None:
    for index, fc in enumerate(system.getForces()):
        logger.info(f"Force Index:{index} | Name: {fc.getName()} | Group: {fc.getForceGroup()}")
        print(f"Force Index:{index} | Name: {fc.getName()} | Group: {fc.getForceGroup()}")
    return

def remove_openmm_force(system: System = None, fname: str = None) -> System:
    """Remove a force from the system by name.
    It will remove all forces that start with the given name."""
    forces_to_remove = []
    for f_idx in range(system.getNumForces()):
        force = system.getForce(f_idx)
        if force.getName().startswith(fname):
            # logger.warning(f"Removing force {force.getName()} at index {f_idx}.")
            forces_to_remove.append(f_idx)

    for f_idx in sorted(forces_to_remove, reverse=True):
        system.removeForce(f_idx)

    return system

def add_COM_force(
    system: System = None,
    group_A: list = None,
    group_B: list = None,
    fc_pull: float = None,
    r0: float = None,
    force_group: int = 15,
):

    force = CustomCentroidBondForce(2, "0.5 * fc_pull * (distance(g1,g2)-r0)^2")
    force.addGlobalParameter("r0", r0)
    # force.addGlobalParameter('fc_pull', fc_pull)
    force.addPerBondParameter("fc_pull")
    force.addGroup(group_A)
    force.addGroup(group_B)
    # force.addBond([0, 1], [])
    force.addBond([0, 1], [fc_pull])
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(force_group)
    system.addForce(force)

    return


def add_harmonic_restraints(
    system: System = None,
    positions: list = None,
    topology: app.Topology = None,
    atom_idx_list: list[int] = None,
    restraint_force: int = 5,
    force_name: str = "k",
    force_group: int = 12,
):
    """
    Function to add positional harmonic restraints to a set of atoms
    """

    atoms = topology.atoms()

    force = CustomExternalForce(f"{force_name}*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_amount = (
        restraint_force * openmmunit.kilocalories_per_mole / openmmunit.angstroms**2
    )
    force.addGlobalParameter(force_name, force_amount)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    counter = 0
    for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
        if atom.index in atom_idx_list:
            force.addParticle(i, atom_crd.value_in_unit(openmmunit.nanometers))
            counter += 1
    # logger.info(f"{counter} atoms will be restrained")

    force.setName(force_name)
    force.setForceGroup(force_group)

    system.addForce(force)

    return

def add_flatbottom_COM_restraints(
    system: System = None,
    groupA: list = None,
    groupB: list = None,
    r0: float = None,
    upper_wall: int = 0.1,
    K_flat: float = 200,
    force_name: str = "k_flat_com",
    force_group: int = 30,
):

    fb_eq = "(k_flat/2)*max(distance(g1,g2) - upper_wall, r0)^2"
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(groupA)
    upper_wall_rest.addGroup(groupB)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter("k_flat", K_flat * openmmunit.kilojoules_per_mole)
    upper_wall_rest.addGlobalParameter("upper_wall", upper_wall * openmmunit.nanometer)
    upper_wall_rest.addGlobalParameter("r0", r0 * openmmunit.nanometer)

    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)

    upper_wall_rest.setForceGroup(force_group)
    upper_wall_rest.setName(force_name)

    system.addForce(upper_wall_rest)

    return

def add_flatbottom_XY_restraints(
    system: System = None,
    simulation: app.Simulation = None,
    restrain_indexes: List[int] = None,
    r0: float = None,
    upper_wall: float = 0.1,
    K_flat: float = 200,
    force_name: str = "k_flat_xy",
    force_group: Optional[int] = 31,
):
    
    initial_positions = simulation.context.getState(getPositions=True).getPositions()

    # Define the flat-bottom restraint potential
    fb_eq = """
    k_flat/2 * max(sqrt((x - x0)^2 + (y - y0)^2) - upper_wall, r0)^2
    """
    # fb_eq = """
    # (k_flat/2)*max(periodicdistance(x, y, x0, y0) - upper_wall, r0)^2
    # """

    # Create the CustomExternalForce object
    upper_wall_rest = CustomExternalForce(fb_eq)

    # Get the initial positions of the ligand atoms
    ligand_positions = [initial_positions[index] for index in restrain_indexes]

    # Add per-particle parameters for the reference position
    upper_wall_rest.addPerParticleParameter("x0")
    upper_wall_rest.addPerParticleParameter("y0")

    # Add global parameters
    upper_wall_rest.addGlobalParameter("upper_wall", upper_wall * openmmunit.nanometer)
    upper_wall_rest.addGlobalParameter("r0", r0 * openmmunit.nanometer)
    upper_wall_rest.addGlobalParameter("k_flat", K_flat * openmmunit.kilojoules_per_mole)

    # Assign the reference coordinates to each particle
    for particle, positions in zip(restrain_indexes, ligand_positions):
        upper_wall_rest.addParticle(particle, [positions.x, positions.y])

    # Set the force group
    upper_wall_rest.setForceGroup(force_group)
    upper_wall_rest.setName(force_name)

    # Add the force to the system
    system.addForce(upper_wall_rest)

    return None

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
    """

    # Funnel potential string expression
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

    # Funnel parameters
    funnel.addGlobalParameter("k_xy", k_xy)
    funnel.addGlobalParameter("z_cc", z_cc)
    funnel.addGlobalParameter("alpha", alpha)
    funnel.addGlobalParameter("R_cylinder", R_cylinder)

    # Add host and guest indices
    g1 = funnel.addGroup(host_index, [1.0 for i in range(len(host_index))])
    g2 = funnel.addGroup(guest_index, [1.0 for i in range(len(guest_index))])

    # Add bond
    funnel.addBond([g1, g2], [])

    funnel.setName(force_name)

    # Add force to system
    system.addForce(funnel)

    return

def add_cylindrical_restraints(
    system: System,
    host_index: List[int],
    guest_index: List[int],
    k_xy: Optional[openmmunit.Quantity] = 10.0
    * openmmunit.kilocalorie_per_mole
    / openmmunit.angstrom**2,
    R_cylinder: Optional[openmmunit.Quantity] = 10.0 * openmmunit.angstrom,
    r0: Optional[openmmunit.Quantity] = 5 * openmmunit.angstrom,
    force_group: Optional[int] = 10,
):
    """
    Applies a cylindrical restraint to a guest molecule, allowing it to move freely in the Z direction
    but restricting its motion in the XY plane.
    Inpired by https://github.com/jeff231li/funnel_potential
    """

    # Cylindrical restraint potential string expression
    cylindrical_restraint = CustomCentroidBondForce(
        2,
        "U_cylinder;"
        "U_cylinder = step(r_xy - R_cylinder) * 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        "r_xy = pointdistance(x1, y1, 0, x2, y2, 0);"
    )
    cylindrical_restraint.setUsesPeriodicBoundaryConditions(False)
    cylindrical_restraint.setForceGroup(force_group)

    # Cylindrical restraint parameters
    cylindrical_restraint.addGlobalParameter("k_xy", k_xy)
    cylindrical_restraint.addGlobalParameter("R_cylinder", R_cylinder)
    cylindrical_restraint.addGlobalParameter("r0", r0)

    # Add host and guest indices
    g1 = cylindrical_restraint.addGroup(host_index, [1.0 for i in range(len(host_index))])
    g2 = cylindrical_restraint.addGroup(guest_index, [1.0 for i in range(len(guest_index))])

    cylindrical_restraint.addBond([g1, g2], [])

    # Add force to system
    system.addForce(cylindrical_restraint)

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
        Percentile of Z-displacement to use for z_cc parameter (0-100, default: 95)
    R_cylinder_ang : float
        Cylinder radius in Å for the unbound-state restraint (default: 2.0).
        Limongelli 2013 recommends 1–3 Å.  This is a design parameter and must
        NOT be derived from sMD radial distances, which reflect bound-state lateral
        displacement (typically 7–10 Å) and have nothing to do with the unbound
        cylindrical constraint.
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
        
    # Get atom selections
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
    
    # Initialize arrays to store trajectory data
    host_coms = []
    guest_coms = []
    radial_distances = []
    axial_distances = []
    
    # Analyze trajectory
    for frame in universe.trajectory[::stride]:
        # Calculate centers of mass
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
    
    # Determine unbinding axis
    if use_pca:
        # Use PCA on guest COM displacement
        guest_displacement = guest_coms - guest_coms[0]
        displacement_centered = guest_displacement - guest_displacement.mean(axis=0)
        # Compute covariance matrix
        cov_matrix = np.cov(displacement_centered.T)
        # Get eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Principal component (largest variance)
        unbinding_axis = eigenvectors[:, -1]
    else:
        # Use axis with largest displacement
        guest_displacement = guest_coms[-1] - guest_coms[0]
        unbinding_axis = guest_displacement / np.linalg.norm(guest_displacement)
    
    # Project displacements onto unbinding axis
    relative_positions = guest_coms - host_coms
    z_distances = np.dot(relative_positions, unbinding_axis)
    
    # Calculate radial distances in the plane perpendicular to unbinding axis
    for rel_pos in relative_positions:
        z_component = np.dot(rel_pos, unbinding_axis) * unbinding_axis
        radial_component = rel_pos - z_component
        radial_dist = np.linalg.norm(radial_component)
        radial_distances.append(radial_dist)
    
    radial_distances = np.array(radial_distances)
    z_distances = np.array(z_distances)
    
    # z_cc: percentile of axial distances — how far the ligand travels along the axis
    z_cc = np.percentile(z_distances, percentile_z)
    z_cc = max(z_cc, 5.0)  # minimum 5 Å

    # R_cylinder is a design parameter, not derived from radial distances.
    # The sMD radial distances (typically 7–10 Å) reflect the lateral displacement
    # of the bound ligand from the unbinding axis and are NOT a suitable source for
    # R_cylinder.  R_cylinder confines the *unbound* ligand near the axis; Limongelli
    # 2013 recommends 1–3 Å.
    R_cylinder = float(R_cylinder_ang)

    # R_funnel_bound is logged for context only; it is not stored in the returned dict
    n_bound = max(10, int(0.1 * trajectory_length))
    R_funnel_bound = np.percentile(radial_distances[:n_bound], percentile_r_funnel)

    # Convert to OpenMM units
    z_cc_quantity = z_cc * openmmunit.angstrom
    R_cylinder_quantity = R_cylinder * openmmunit.angstrom
    alpha_quantity = alpha_cone_degrees * openmmunit.degrees

    if verbose:
        logger.info(f"Unbinding axis: {unbinding_axis}")
        logger.info(f"Z-crossing point (z_cc): {z_cc:.2f} Å")
        logger.info(f"Cylinder radius (R_cylinder): {R_cylinder:.2f} Å  (design parameter, Limongelli 2013: 1–3 Å)")
        logger.info(f"Bound-state radial spread (informational): {R_funnel_bound:.2f} Å")
        logger.info(f"Cone angle (alpha): {alpha_cone_degrees}°")
        logger.info(f"Radial distances - min: {radial_distances.min():.2f}, max: {radial_distances.max():.2f}, mean: {radial_distances.mean():.2f} Å")
        logger.info(f"Axial distances - min: {z_distances.min():.2f}, max: {z_distances.max():.2f}, mean: {z_distances.mean():.2f} Å")
    
    # Compile results
    results = {
        "host_index": host_indices,
        "guest_index": guest_indices,
        "k_xy": k_xy,
        "z_cc": z_cc_quantity,
        "alpha": alpha_quantity,
        "R_cylinder": R_cylinder_quantity,
        "force_group": 10,
        "force_name": "k_funnel_trajectory",
        "com_trajectory": guest_coms - host_coms,  # Relative positions
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
    """
    np.savez(
        filepath,
        # geometric parameters
        z_cc_ang=np.array(
            params_dict["z_cc"].value_in_unit(openmmunit.angstrom)
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
        # unbinding axis (unit vector)
        unbinding_axis=np.array(params_dict["unbinding_axis"]),
        # atom index groups
        host_index=np.array(params_dict["host_index"]),
        guest_index=np.array(params_dict["guest_index"]),
        # trajectory arrays (Å)
        com_trajectory=np.array(params_dict["com_trajectory"]),
        host_com_trajectory=np.array(params_dict["host_com_trajectory"]),
        guest_com_trajectory=np.array(params_dict["guest_com_trajectory"]),
        radial_distances=np.array(params_dict["radial_distances"]),
        axial_distances=np.array(params_dict["axial_distances"]),
        # metadata
        force_group=np.array(params_dict["force_group"]),
        trajectory_length=np.array(params_dict["trajectory_length"]),
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
        "force_name": "k_funnel_trajectory",
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
    """
    
    # Extract parameters
    host_index = params_dict["host_index"]
    guest_index = params_dict["guest_index"]
    k_xy = params_dict["k_xy"]
    z_cc = params_dict["z_cc"]
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
        "U_cylinder = step(abs(r_z) - z_cc)*step(r_xy - R_cylinder)*Wall_cylinder;"
        "Wall_funnel = 0.5 * k_xy * (r_xy - R_funnel)^2;"
        "Wall_cylinder = 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        "R_funnel = (z_cc-abs(r_z))*tan(alpha) + R_cylinder;"
        "r_xy = sqrt(max(0, r2 - r_z*r_z));"
        "r_z = nx*(x2-x1) + ny*(y2-y1) + nz*(z2-z1);"
        "r2 = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2;",
    )

    funnel.setUsesPeriodicBoundaryConditions(False)
    funnel.setForceGroup(force_group)

    # Add parameters
    funnel.addGlobalParameter("k_xy", k_xy)
    funnel.addGlobalParameter("z_cc", z_cc)
    funnel.addGlobalParameter("alpha", alpha)
    funnel.addGlobalParameter("R_cylinder", R_cylinder)
    funnel.addGlobalParameter("nx", float(unbinding_axis[0]))
    funnel.addGlobalParameter("ny", float(unbinding_axis[1]))
    funnel.addGlobalParameter("nz", float(unbinding_axis[2]))
    
    # Add host and guest indices
    g1 = funnel.addGroup(host_index, [1.0 for i in range(len(host_index))])
    g2 = funnel.addGroup(guest_index, [1.0 for i in range(len(guest_index))])
    
    # Add bond
    funnel.addBond([g1, g2], [])
    
    funnel.setName(force_name)
    
    # Add to system if provided
    if system is not None:
        system.addForce(funnel)
        logger.info(f"Added funnel force '{force_name}' to system")
    
    return funnel
