import os
import shutil
import logging
import numpy as np
import pandas as pd
from sys import stdout, exit

from typing import Union, Tuple, Optional, List
from collections import defaultdict

from openmm import *
from openmm.app import *
import openmm.app as app
import openmm.unit as openmmunit
from pdbfixer.pdbfixer import PDBFixer
from openmmtools.utils import get_fastest_platform

import parmed
import pickle

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations import wrap
from MDAnalysis.core.universe import Universe
from MDAnalysis.analysis.rms import RMSD, RMSF

import pytraj as pt

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return None

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def align_trajectory(
    prmtop_file: str = None,
    traj_file: Union[str, list] = None,
    out_fname: str = None,
    strip_mask: str = None,  #':HOH,NA,CL,K,POP'
) -> None:

    ptraj = pt.iterload(traj_file, prmtop_file)
    ptraj = ptraj.autoimage()
    ptraj = ptraj.center()
    ptraj = ptraj.superpose(ref=0, mask="@CA,C,N")

    if strip_mask is not None:
        ptraj = ptraj.strip(strip_mask)
        pt.save(f"{out_fname}_dry.prmtop", ptraj.top, overwrite=True)
    ptraj.save(f"{out_fname}.dcd")

    return


def fix_pdb(
    pdbfile: str,
    keep_heterogens: bool = False,
    ignore_terminal_missing_residues: bool = True,
    pH: float = 7.4,
) -> PDBFixer:
    """Fixes common problems in PDB such as:
            - missing atoms
            - missing residues
            - missing hydrogens
            - remove nonstandard residues

    Args:
        pdbfile (str): pdb string old format
        pdbxfile (str): pdb string new format
        keep_heterogens (bool): if False all the heterogen atoms but waters are deleted.
        ignore_terminal_missing_residues (bool): If missing residues at the beginning and the end of a chain should be ignored or built.
        pH (float):  pH value used to determine protonation state of residues
    """

    fixer = PDBFixer(str(pdbfile))
    fixer.findMissingResidues()

    if ignore_terminal_missing_residues:
        # if missing terminal residues shall be ignored, remove them from the dictionary
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=True)

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)

    return fixer


def save_pdb(topology: app.Topology, positions: list, file_path: str) -> None:
    """Saves the specified topology and position to the out_path file.

    Args:
        topology (app.Topology): topology used
        positions (list): list of 3D coords
        out_path (str): path to where to save the file
    """
    app.PDBFile.writeFile(topology, positions, file_path, keepIds=True)

    return


def save_system(system: System, out_file: str) -> None:
    """Saves the openmm system to the desired out path.

    Args:
        out_path (str): path where to save the System
        system (System): system to be saved
    """

    with open(out_file, "w") as fo:
        fo.write(XmlSerializer.serialize(system))
    return


def save_simulation(simulation, out_file: str) -> None:

    simulation.saveCheckpoint(f"{out_file}.chk")
    simulation.saveState(f"{out_file}.xml")

    return


def load_system(system_path: str) -> System:
    """Loads the desired system.

    Args:
        system_path (str): path to the system file

    Returns:
        System: system
    """
    try:
        with open(system_path) as fi:
            system = XmlSerializer.deserialize(fi.read())
    except Exception as e:
        logging.error(f"Something went wrong while opening {system_path}\n {e}")
        exit(1)
    return system


def save_amber_topology(
    topology: app.Topology = None,
    positions: list = None,
    forcefield: app.ForceField = None,
    out_path: str = None,
) -> None:

    os.makedirs(out_path, exist_ok=True)
    new_system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=10 * openmmunit.angstrom,
        removeCMMotion=False,
        rigidWater=False,
        hydrogenMass=3.0 * openmmunit.amu,
    )

    parmed_structure = parmed.openmm.topsystem.load_topology(
        topology, new_system, positions
    )

    parmed_structure.save(f"{out_path}/system.prmtop", overwrite=True, format="amber")
    parmed_structure.save(f"{out_path}/system.rst7", overwrite=True, format="rst7")

    return


def select_platform(platform_name: str = None, device_index: str = "0"):

    if platform_name == None or platform_name == "fastest":
        platform_name = get_fastest_platform().getName()

    try:
        platform = Platform.getPlatformByName(platform_name)
        logging.debug(f"Using {platform_name} platform.")

        if platform_name in ["OpenCL"]:
            platform.setPropertyDefaultValue("Precision", "mixed")
            platform.setPropertyDefaultValue("DeviceIndex", device_index)
        if platform_name in ["CUDA"]:
            platform.setPropertyDefaultValue("DeterministicForces", "false")
            platform.setPropertyDefaultValue("CudaPrecision", "mixed")
            platform.setPropertyDefaultValue("CudaDeviceIndex", device_index)
    except:
        logging.error(f"Something went wrong trying to get {platform_name} platform.")

    return platform


def add_reporters(
    simulation,
    out_dir: str = None,
    suffix: str = None,
    total_steps: int = 250000,
    logperiod: int = 2500,
) -> None:
    """Set up the reporters"""

    simulation.reporters = []  # Delete all current reporters

    simulation.reporters.append(
        StateDataReporter(
            stdout,
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            remainingTime=True,
            speed=True,
            totalSteps=total_steps,
            separator="\t",
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            f"{out_dir}/statistics_{suffix}.csv",
            logperiod,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=True,
            volume=True,
            density=True,
            remainingTime=True,
            speed=True,
            totalSteps=total_steps,
        )
    )

    # Save coordinates every N logperiods
    simulation.reporters.append(
        DCDReporter(
            f"{out_dir}/trajectory_{suffix}.dcd",
            reportInterval=logperiod,
            enforcePeriodicBox=False,  # WARNING this compromises autoimaging afterwards in some cases
        )
    )
    return


def add_barostat(system: System=None, temp: float=298.15, is_membrane: bool=False) -> System:
    """Add an appropriate barostat to the system.
    Simulation for membrane proteins are run at 0 surface tension and semiisotropic pressure
    """

    if is_membrane:
        logging.debug(f"Adding a Membrane Montecarlo Barostat to the system")
        barostat = MonteCarloMembraneBarostat(
            1 * openmmunit.atmosphere,
            0 * openmmunit.bar * openmmunit.nanometers,
            temp,
            MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFree,
            15,
        )
    else:
        logging.debug(f"Adding a Montecarlo Barostat to the system")
        barostat = MonteCarloBarostat(1 * openmmunit.atmosphere, temp)

    system.addForce(barostat)

    return system

def add_variants(modeller: Modeller, variants_dict: dict = None) -> Modeller:
    """Adds variants for specific protonation states.

    :param modeller: OpenMM Modeller
    :type Modeller: Modeller
    :param variants_dict: dict of variants to apply for the protonation states
    :type variants: dict
    :return: Modeller object with added protonation states
    :rtype: Modeller
    """

    variants = list()
    residues = list(modeller.topology.residues())
    mapping = defaultdict(list)
    for r in residues:
        mapping[r.chain.id].append(int(r.id))

    for chain in mapping:
        for res_number in mapping[chain]:
            key = f"{chain}:{res_number}"
            if key in variants_dict:
                variants.append(variants_dict[key])
            else:
                variants.append(None)

    modeller.addHydrogens(variants=variants)

    return modeller 

def _print_current_forces(system: System = None) -> None:
    for index, fc in enumerate(system.getForces()):
        logging.info(
            f"Force Index:{index} | Name: {fc.getName()} | Group: {fc.getForceGroup()}"
        )
    return


def get_protein_ha(topology: app.Topology, lig_name: str = "UNK") -> Tuple[list, list]:

    ATOMSET = set(("HOH", "WAT", "POP", "K", "CL", "NA", lig_name))

    # # Restraint heavy atoms only: C, O, N, S, P, CA and MG
    # elements = set((element.carbon, element.oxygen, element.magnesium, element.calcium,
    #                     element.nitrogen, element.sulfur, element.phosphorus))

    # protein_ha = []
    # for atom in topology.atoms():
    #     if atom.residue.name not in ATOMSET and atom.element in elements:
    #         protein_ha.append(atom.index)

    protein_ha_idx = []
    protein_ha_name = []

    for atom in topology.atoms():
        if atom.residue.name not in ATOMSET:
            if not atom.name.startswith("H"):
                protein_ha_idx.append(atom.index)
                protein_ha_name.append(atom.name)

    return protein_ha_idx, protein_ha_name


def get_ligand_ha(topology: app.Topology, lig_name: str = "UNK") -> Tuple[list, list]:
    """get names for all non-hydrogen ligand atoms"""

    residues = topology.residues()
    lig_ha_idx = []
    lig_ha_names = []
    for r in residues:
        if r.name == lig_name:
            lig_ha_names = [a.name for a in r.atoms() if not a.name.startswith("H")]
            lig_ha_idx = [a.index for a in r.atoms() if not a.name.startswith("H")]

    # mg_names = [a.name for a in topology.atoms() if a.name == "MG"]
    # mg_idx = [a.index for a in topology.atoms() if a.name == "MG"]
    # lig_ha_idx.extend(mg_idx)
    # lig_ha_names.extend(mg_names)

    return lig_ha_idx, lig_ha_names


def get_pocket_ha(topology: app.Topology, pocket_resid: list[int] = None) -> list:
    """get names for all non-hydrogen ligand atoms"""

    residues = topology.residues()
    pocket_ha_idx = []

    for r in residues:
        if r.index in pocket_resid:
            print(f"match for {r.index} {r.name} {r.id}")
            res_ha_idx = [a.index for a in r.atoms() if not a.name.startswith("H")]
            pocket_ha_idx.extend(res_ha_idx)

    return pocket_ha_idx

def get_COM_dist(simulation, groupA:list[int]=None, groupB:list[int]=None, weighByMass:bool=True) -> float:
    
    # Get positions
    state = simulation.context.getState(getPositions=True, getVelocities=False)
    positions = state.getPositions(asNumpy=True) / openmmunit.nanometers
    atoms = [atom for atom in simulation.topology.atoms()]

    # Function to calculate center (COM or COG)
    def _get_center(group, weighByMass):
        group_positions = positions[group]  # Get positions for the group

        if weighByMass:
            masses = np.array([atom.element.mass.value_in_unit(openmmunit.dalton) for atom in atoms if atom.index in group])
            center = np.average(group_positions, axis=0, weights=masses)  # Weighted average for COM
        else:
            center = np.mean(group_positions, axis=0)  # Simple mean for COG
        return center

    # Calculate centers for both groups and their distance
    centerA = _get_center(groupA, weighByMass)
    centerB = _get_center(groupB, weighByMass)
    dist = np.linalg.norm(centerA - centerB)

    return dist  # Unitless, but effectively in nanometers because.... openMM


def calculate_com_distance(
    u, ligand_atoms=None, pocket_atoms=None, weighByMass: bool = False
) -> pd.DataFrame:

    distances = []
    for ts in u.trajectory:
        if weighByMass:
            lig_com = ligand_atoms.center_of_mass(wrap=True)
            prot_com = pocket_atoms.center_of_mass(wrap=True)
        else:
            lig_com = ligand_atoms.center_of_geometry(wrap=True)
            prot_com = pocket_atoms.center_of_geometry(wrap=True)

        distances.append(np.linalg.norm(prot_com - lig_com))

    return pd.DataFrame(distances, columns=["com_d"], index=range(len(distances)))


def get_ligand_rmsd(
    u: Universe = None,
    u_ref: Universe = None,
    lig_resname: str = "UNK",
    alig_select: str = "ligand",
):
    """A function to calculate the ligand RMSD from a trajectory.

    Parameters
    ----------
    'u : Universe
        MDAnalysis Universe
    lig_resname : str
        Residue name of the ligand that was biased.
    alig_select : str
        Selection to be considered in the alignment.
    Returns
    -------
    rmsds : np.array
        ligand rmsd for every frame of the trajectory.
    """
    if alig_select == "ligand":
        alig_select = f"resname {lig_resname} and not name H*"

    # Make sure molecules are whole before rmsd calculation
    # transform = wrap(u.atoms)
    # u.trajectory.add_transformations(transform)

    # Align each frame using the backbone as reference
    # Calculate the RMSD of ligand heavy atoms

    r = RMSD(
        atomgroup=u,
        reference=u_ref,
        select=alig_select,
        groupselections=[f"resname {lig_resname} and not name H*"],
        ref_frame=0,
    ).run()

    rmsds = r.results.rmsd[1:, -1]
    rmsds = rmsds / 10  # angstroms to nm

    return pd.DataFrame(rmsds, columns=["rmsd"], index=range(len(rmsds)))


def _remove_force(force_name: str = None, system: System = None, simulation=None):
    """Remove a force from an OpenMM system based on its name."""
    counter = 0
    for index, fc in enumerate(system.getForces()):
        if fc.getName() == force_name:
            simulation.context.getSystem().removeForce(index)
            logging.info(f"Removing existing {force_name} force")
            counter += 1
    if counter == 0:
        logging.warning(f"No force was removed, check that {force_name} exist")
        _print_current_forces(system)

    return


def add_COM_force(
    system: System = None,
    group_A: list = None,
    group_B: list = None,
    fc_pull=None,
    r0=None,
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
    logging.info(f"{counter} atoms will be restrained")

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
    force_group: int = 30,
):

    fb_eq = "(k_flat/2)*max(distance(g1,g2) - upper_wall, r0)^2"
    upper_wall_rest = CustomCentroidBondForce(2, fb_eq)
    upper_wall_rest.addGroup(groupA)
    upper_wall_rest.addGroup(groupB)
    upper_wall_rest.addBond([0, 1])
    upper_wall_rest.addGlobalParameter(
        "k_flat", K_flat * openmmunit.kilojoules_per_mole
    )
    upper_wall_rest.addGlobalParameter("upper_wall", upper_wall * openmmunit.nanometer)
    upper_wall_rest.addGlobalParameter("r0", r0 * openmmunit.nanometer)

    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)

    upper_wall_rest.setForceGroup(force_group)

    system.addForce(upper_wall_rest)

    return

def add_flatbottom_XY_restraints(
    system: System = None,
    simulation: app.Simulation = None,
    restrain_indexes: List[int] = None,
    r0: float = None,
    upper_wall: float = 0.1,
    K_flat: float = 200,
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
        "r_xy = sqrt((x2 - x1)^2 + (y2 - y1)^2);"
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

    # Add bond
    cylindrical_restraint.addBond([g1, g2], [])

    # Add force to system
    system.addForce(cylindrical_restraint)

    return
