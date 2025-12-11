from typing import Union, Tuple, Optional, List
from openmm import *
from openmm.app import *
import openmm.app as app
import openmm.unit as openmmunit

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
