from typing import Union, Tuple, Optional, List, Dict
from openmm import *
from openmm.app import *
import openmm.app as app
import openmm.unit as openmmunit
import numpy as np

import logging
logger = logging.getLogger("autopath")

# Force group assignments (OpenMM supports groups 0–31; must not overlap)
_FG_CUSTOM_NB   = 14  # custom nonbonded (sterics/electrostatics)
_FG_FUNNEL      = 15  # funnel metadynamics restraint
_FG_FLAT_BOTTOM = 19  # flat-bottom position restraints
_FG_BAROSTAT    = 30  # barostat (Monte Carlo)
_FG_RESTRAINTS  = 31  # harmonic positional restraints

def print_current_forces(system: System = None) -> None:
    """Log all forces currently registered in *system* at INFO level.

    Parameters
    ----------
    system : openmm.System
        The system whose force list is inspected.
    """
    for index, fc in enumerate(system.getForces()):
        logger.info(f"Force Index:{index} | Name: {fc.getName()} | Group: {fc.getForceGroup()}")
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
    force_group: int = _FG_FUNNEL,
):
    """Add a harmonic COM–COM distance restraint between two atom groups.

    Applies the potential ``U = 0.5 * fc_pull * (d(g1, g2) - r0)^2``, where
    ``d(g1, g2)`` is the distance between the centroids of the two groups.
    ``fc_pull`` is a per-bond parameter so it can be updated at runtime via
    the context without rebuilding the force object.

    Parameters
    ----------
    system : openmm.System
        The system to which the force is added.
    group_A : list of int
        Atom indices for the first centroid group (e.g. pocket CA atoms).
    group_B : list of int
        Atom indices for the second centroid group (e.g. ligand heavy atoms).
    fc_pull : float
        Force constant in kJ/mol/nm² for the harmonic restraint.
    r0 : float
        Equilibrium COM–COM distance in nanometers.
    force_group : int
        OpenMM force group index (default ``_FG_FUNNEL = 15``).

    Returns
    -------
    None
        The force is added to *system* in-place.

    Note
    ----
    ``r0`` is a global parameter (updated via ``context.setParameter``), while
    ``fc_pull`` is a per-bond parameter so the force constant can be changed
    independently for each bond without touching the global namespace.
    """
    force = CustomCentroidBondForce(2, "0.5 * fc_pull * (distance(g1,g2)-r0)^2")
    force.addGlobalParameter("r0", r0)
    force.addPerBondParameter("fc_pull")
    force.addGroup(group_A)
    force.addGroup(group_B)
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
    """Add positional harmonic restraints to a set of atoms.

    Each restrained atom is held to its reference position in *positions* by a
    harmonic spring: ``U = k * periodicdistance(x, y, z, x0, y0, z0)^2``.

    Parameters
    ----------
    system : openmm.System
        The system to which the restraint force is added.
    positions : list
        Reference positions (with OpenMM units) for *all* atoms in the topology,
        indexed consistently with ``topology.atoms()``.
    topology : app.Topology
        OpenMM topology of the full system.
    atom_idx_list : list of int
        Atom indices to restrain; atoms not in this list are skipped.
    restraint_force : int or float
        Force constant in kcal/mol/Å².
    force_name : str
        Name assigned to the force and used as the global parameter key
        (accessible via ``context.setParameter(force_name, value)``).
    force_group : int
        OpenMM force group index.

    Returns
    -------
    None
        The restraint force is added to *system* in-place.
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

    for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
        if atom.index in atom_idx_list:
            force.addParticle(i, atom_crd.value_in_unit(openmmunit.nanometers))

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
    force_group: int = _FG_BAROSTAT,
):
    """Add a flat-bottom harmonic restraint on the COM–COM distance between two groups.

    The potential is zero when ``distance(g1, g2) <= upper_wall`` and grows
    quadratically for larger separations:
    ``U = (k_flat/2) * max(d(g1,g2) - upper_wall, r0)^2``.
    ``r0`` should be ≤ 0 (typically 0) so the force activates only above
    *upper_wall*.

    Parameters
    ----------
    system : openmm.System
        The system to modify in-place.
    groupA : list of int
        Atom indices for the first centroid group.
    groupB : list of int
        Atom indices for the second centroid group.
    r0 : float
        Floor value for the ``max()`` expression in nanometers (usually 0).
    upper_wall : float
        Flat-bottom radius in nanometers; restraint is inactive below this distance.
    K_flat : float
        Force constant in kJ/mol.
    force_name : str
        Name assigned to the force object.
    force_group : int
        OpenMM force group index (default ``_FG_BAROSTAT = 30``).

    Returns
    -------
    None
        The force is added to *system* in-place.
    """
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
    force_group: Optional[int] = _FG_RESTRAINTS,
):
    """Add a flat-bottom restraint in the XY plane for a set of atoms.

    Atoms can move freely in Z but are harmonically penalized for XY
    displacements exceeding *upper_wall* from their initial positions:
    ``U = (k_flat/2) * max(sqrt((x-x0)^2 + (y-y0)^2) - upper_wall, r0)^2``.

    The reference XY coordinates are read from the current simulation context
    at the time this function is called.

    Parameters
    ----------
    system : openmm.System
        The system to modify in-place.
    simulation : app.Simulation
        Active simulation; current positions are used as the XY reference.
    restrain_indexes : list of int
        Atom indices to apply the restraint to.
    r0 : float
        Floor value for the ``max()`` expression in nanometers (usually 0).
    upper_wall : float
        Flat-bottom radius in the XY plane in nanometers.
    K_flat : float
        Force constant in kJ/mol.
    force_name : str
        Name assigned to the force object.
    force_group : int
        OpenMM force group index (default ``_FG_RESTRAINTS = 31``).

    Returns
    -------
    None
        The force is added to *system* in-place.
    """
    initial_positions = simulation.context.getState(getPositions=True).getPositions()

    fb_eq = """
    k_flat/2 * max(sqrt((x - x0)^2 + (y - y0)^2) - upper_wall, r0)^2
    """

    upper_wall_rest = CustomExternalForce(fb_eq)

    ligand_positions = [initial_positions[index] for index in restrain_indexes]

    upper_wall_rest.addPerParticleParameter("x0")
    upper_wall_rest.addPerParticleParameter("y0")

    upper_wall_rest.addGlobalParameter("upper_wall", upper_wall * openmmunit.nanometer)
    upper_wall_rest.addGlobalParameter("r0", r0 * openmmunit.nanometer)
    upper_wall_rest.addGlobalParameter("k_flat", K_flat * openmmunit.kilojoules_per_mole)

    for particle, positions in zip(restrain_indexes, ligand_positions):
        upper_wall_rest.addParticle(particle, [positions.x, positions.y])

    upper_wall_rest.setForceGroup(force_group)
    upper_wall_rest.setName(force_name)

    system.addForce(upper_wall_rest)

    return None

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
    """Apply a cylindrical restraint to a guest molecule.

    The guest is free to move along Z but is penalized for XY displacements of
    its centroid beyond *R_cylinder* from the host centroid:
    ``U = step(r_xy - R_cylinder) * 0.5 * k_xy * (r_xy - R_cylinder)^2``.

    Parameters
    ----------
    system : openmm.System
        The system to which the cylindrical restraint is added.
    host_index : list of int
        Atom indices for the host (reference axis) centroid group.
    guest_index : list of int
        Atom indices for the guest (restrained) centroid group.
    k_xy : openmm.unit.Quantity
        Spring constant for the XY restraint (default 10 kcal/mol/Å²).
    R_cylinder : openmm.unit.Quantity
        Cylinder radius; the restraint is inactive below this XY distance
        (default 10 Å).
    r0 : openmm.unit.Quantity
        Additional offset parameter available in the energy expression
        (default 5 Å).
    force_group : int
        OpenMM force group index (default 10).

    Returns
    -------
    None
        The force is added to *system* in-place.

    Note
    ----
    Inspired by https://github.com/jeff231li/funnel_potential
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

