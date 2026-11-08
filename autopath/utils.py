import os
import shutil
import logging
import tempfile
import warnings

logger = logging.getLogger("autopath")
import numpy as np
import pandas as pd
from sys import stdout, exit
from pathlib import Path

from typing import Optional, List
from collections import defaultdict

from openmm import *
from openmm.app import *
import openmm.app as app
import openmm.unit as openmmunit
from pdbfixer.pdbfixer import PDBFixer
from openmmtools.utils import get_fastest_platform

import parmed

import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD, RMSF
from MDAnalysis.analysis.distances import distance_array

from pymol import cmd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("fivethirtyeight")

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Scaffolds import MurckoScaffold

def setup_logging(logfile: str = 'autopath.log',
                  logname: str = "autopath",
                  log_level: str = "INFO", 
                  ) -> logging.Logger:
    """Set up logging for the application at the entry point, i.e. cli scripts."""

    allowed_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level.upper() not in allowed_log_levels:
        raise ValueError(f"Invalid log level: {log_level}. It should be one of {allowed_log_levels}")
    
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    logger = logging.getLogger(logname)
    logger.setLevel(log_level.upper())

    # Prevent duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger  

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(logfile, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

def wrap_align_save_traj(traj_files, topology, remove_original=True, is_membrane=False,
                         reference=None, align_select="backbone", out_dir=None):
    """Centre, image and superpose trajectories, writing ``*_aligned.dcd``.

    Parameters
    ----------
    traj_files : str or list of str
        Trajectory path(s). Each is processed independently.
    topology : str
        Topology for all of *traj_files*.
    remove_original : bool
        Delete each input after writing its aligned copy (default True). Set False
        when the inputs are read-only or shared.
    is_membrane : bool
        Skip imaging: ``image_molecules`` is prohibitively slow for membrane systems
        (hundreds of lipids), and centering is sufficient for sMD.
    reference : str or mdtraj.Trajectory, optional
        Structure whose first frame is the superposition target. Default (None)
        superposes each trajectory onto its OWN first frame, which puts every
        trajectory in a *different* frame — fine for single-trajectory analysis, but
        an external reference is required when several trajectories must share one
        Cartesian frame (e.g. pooling cosolvent density maps across replicas or
        probes). Must have a topology compatible with *align_select*.
    align_select : str
        MDTraj selection used for the superposition (default ``"backbone"``).
    out_dir : str, optional
        Write aligned trajectories here instead of next to the inputs.

    Returns
    -------
    list of str
        Paths of the aligned trajectories.

    Notes
    -----
    Superposition rotates the coordinates while the stored box lengths and angles
    are rotation-invariant, so the saved box no longer records the periodic frame
    of the coordinates. Treat aligned trajectories as geometry for superposition
    and visualisation; do not run PBC-dependent calculations (minimum image,
    re-wrapping, re-imaging) on them.
    """
    import mdtraj as md
    if isinstance(traj_files, str):
        traj_files = [traj_files]

    ref_traj = None
    if reference is not None:
        ref_traj = reference if hasattr(reference, "xyz") else md.load(reference)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    aligned_paths = []
    for traj_file in traj_files:
        traj = md.load(traj_file, top=topology)
        traj = traj.center_coordinates()
        if not is_membrane:
            traj = traj.image_molecules(make_whole=True)
        sel = traj.topology.select(align_select)
        if len(sel) > 0:
            try:
                if ref_traj is None:
                    # Image BEFORE superposing only (done above). Superposition
                    # rotates the coordinates but not unitcell_vectors, so imaging
                    # afterwards wraps against a box that no longer matches them and
                    # translates whole molecules by vectors that are wrong in the
                    # rotated frame -- dropping solvent on top of solvent. A uniform
                    # rigid transform preserves every interatomic vector, so nothing
                    # needs re-imaging after it.
                    traj = traj.superpose(traj[0], atom_indices=sel)
                else:
                    ref_sel = ref_traj.topology.select(align_select)
                    if len(ref_sel) != len(sel):
                        raise ValueError(
                            f"reference selection {align_select!r} has {len(ref_sel)} "
                            f"atoms but the trajectory has {len(sel)}")
                    # Aligning to an EXTERNAL frame: a uniform rigid transform cannot
                    # move any molecule into a different image relative to the solute,
                    # so no second imaging pass is needed (and re-imaging here would
                    # re-wrap against a box that no longer matches the coordinates).
                    traj = traj.superpose(ref_traj, frame=0, atom_indices=sel,
                                          ref_atom_indices=ref_sel)
            except Exception as e:
                logging.warning(f"Superposition failed for {traj_file}: {e}. Proceeding without superposition.")
        base = os.path.basename(traj_file) if out_dir is not None else traj_file
        out_path = os.path.join(out_dir, base) if out_dir is not None else base
        out_path = out_path.replace(".dcd", "_aligned.dcd")
        traj.save(out_path)
        if remove_original:
            os.remove(traj_file)
        aligned_paths.append(out_path)
    return aligned_paths


from autopath.pdb_preprocessor import (
    fetch_smiles,
    get_scrubbed_smile,
    fetch_pdb,
    save_receptor_and_ligand_from_pdb,
    save_receptor_and_ligand_from_openmm,
    assign_bondOrders,
)


def save_pdb(topology: app.Topology, positions: list, out_path: str) -> None:
    """Saves the specified topology and position to the out_path file.

    Args:
        topology (app.Topology): topology used
        positions (list): list of 3D coords
        out_path (str): path to where to save the file
    """
    app.PDBFile.writeFile(topology, positions, out_path, keepIds=True)

    return

def save_receptor_w_colig(receptor_path: str, colig_path: str, colig_smiles: str, out_path: str, colig_name: str = None) -> None:
    """Saves the receptor and co-ligand to the out_path file.

    Args:
        receptor_path (str): path to receptor (pdb)
        colig_path (str): path to co-ligand (sdf)
        colig_smiles (str): smiles of co-ligand
        out_path (str): path to where to save the file
        colig_name (str): new name to assign co-ligand
    """
    protein = PDBFile(receptor_path)
    mol = Chem.SDMolSupplier(colig_path, sanitize=False, removeHs=True)[0]
    mol = assign_bondOrders(mol, colig_smiles)
    save_dir = os.path.dirname(out_path) 
    Chem.MolToPDBFile(mol, f"{save_dir}/temp_colig.pdb")
    colig_pdb = PDBFile(f"{save_dir}/temp_colig.pdb")

    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
                   'HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN', 'HSD', 'HSE', 'HSP', 'UNK']

    #get residue and chain id of ligand based on last amino acid info 
    max_chain_id = 'A' 
    for res in protein.topology.residues():
        if res.name in amino_acids:
            try:
                curr_chain_id = res.chain.id
                if ord(curr_chain_id) > ord(max_chain_id):
                    max_chain_id = curr_chain_id
            except ValueError:
                pass
    colig_chain_id = chr(ord(max_chain_id) + 1)
    colig_res_id = 0

    for res in colig_pdb.topology.residues():
        if colig_name is not None:
            res.name = colig_name 
        res.id = str(colig_res_id)
    for chain in colig_pdb.topology.chains():
        chain.id = colig_chain_id

    modeller = Modeller(protein.topology, protein.positions)
    modeller.add(colig_pdb.topology, colig_pdb.positions)
    app.PDBFile.writeFile(modeller.topology, modeller.positions, out_path, keepIds=True)
    os.remove(f"{save_dir}/temp_colig.pdb")

    return 



def save_system(system: System, out_path: str) -> None:
    """Saves the openmm system to the desired out path.

    Args:
        out_path (str): path where to save the System
        system (System): system to be saved
    """

    with open(out_path, "w") as fo:
        fo.write(XmlSerializer.serialize(system))
    return


def save_simulation(simulation, out_path: str) -> None:
    """Save an OpenMM simulation checkpoint and XML state to disk.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The running simulation object.
    out_path : str
        Base path (without extension); writes ``<out_path>.chk`` and
        ``<out_path>.xml``.
    """
    simulation.saveCheckpoint(f"{out_path}.chk")
    simulation.saveState(f"{out_path}.xml")

    return


def rebuild_simulation(simulation, system: System):
    """Rebuild a simulation on a modified system, carrying the current state over.

    ``Context.reinitialize(preserveState=True)`` restores global parameters by name, so
    the parameters of a removed force stay in the context and leak into saved states.
    A fresh context only knows the parameters the given system still defines.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Simulation holding the state to carry over; its context is left untouched.
    system : System
        System the new simulation is built on.

    Returns
    -------
    openmm.app.Simulation
        New simulation with the positions, velocities, box vectors and time of the old one.
    """
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    # An integrator can only drive one context, so hand the new one a copy.
    integrator = XmlSerializer.deserialize(XmlSerializer.serialize(simulation.integrator))

    new_simulation = app.Simulation(
        simulation.topology, system, integrator, simulation.context.getPlatform()
    )
    new_simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    new_simulation.context.setPositions(state.getPositions())
    new_simulation.context.setVelocities(state.getVelocities())
    new_simulation.context.setTime(state.getTime())
    new_simulation.currentStep = simulation.currentStep

    return new_simulation


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
        raise RuntimeError(f"Failed to load OpenMM system from '{system_path}'") from e
    return system

def save_amber_files(
    topology: app.Topology = None,
    positions: list = None,
    system: System = None,
    out_path: str = None,
) -> None:

    """Save an OpenMM topology and system to AMBER-format files via ParmEd.

    Writes ``<out_path>/system.prmtop`` always. Writes ``<out_path>/system.rst7``
    only when *positions* is not None.

    Parameters
    ----------
    topology : app.Topology
        OpenMM topology of the system.
    positions : list or None
        Atomic positions; if None the rst7 coordinate file is skipped.
    system : openmm.System or None
        OpenMM system; if None the prmtop will contain topology information
        only (no force-field parameters).
    out_path : str
        Directory where output files are written (created if absent).

    Note
    ----
    Uses ``parmed.openmm.topsystem.load_topology``.
    See https://parmed.github.io/ParmEd/html/openmm.html for details.
    """
    os.makedirs(out_path, exist_ok=True)

    parmed_structure = parmed.openmm.topsystem.load_topology(
        topology, system, positions
    )

    parmed_structure.save(f"{out_path}/system.prmtop", overwrite=True, format="amber")
    if positions is not None:
        parmed_structure.save(f"{out_path}/system.rst7", overwrite=True, format="rst7")

    return

def select_platform(platform_name: str = None, device_index: str = "0") -> Platform:
    """Return a configured OpenMM Platform object.

    Parameters
    ----------
    platform_name : str, optional
        Platform name: ``"CUDA"``, ``"OpenCL"``, ``"CPU"``, ``"Reference"``, or
        ``"fastest"``/``None`` to auto-select the fastest available platform.
    device_index : str
        GPU device index string (e.g. ``"0"``), used for CUDA and OpenCL.

    Returns
    -------
    openmm.Platform
        Configured platform with mixed precision set for GPU backends.
    """
    platform_name = platform_name.upper() if platform_name is not None else None
    if platform_name is None or platform_name == "FASTEST":
        platform_name = get_fastest_platform().getName().upper()

    logging.info(f"Using {platform_name} platform.")

    if platform_name == "OPENCL":
        platform = Platform.getPlatformByName('OpenCL')
        platform.setPropertyDefaultValue("Precision", "mixed")
        platform.setPropertyDefaultValue("DeviceIndex", device_index)
    elif platform_name == "CUDA":
        platform = Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue("DeterministicForces", "false")
        platform.setPropertyDefaultValue("CudaPrecision", "mixed")
        platform.setPropertyDefaultValue("CudaDeviceIndex", device_index)
    else:
        # CPU, Reference, or any other valid OpenMM platform name
        _name_map = {"CPU": "CPU", "REFERENCE": "Reference"}
        platform = Platform.getPlatformByName(_name_map.get(platform_name, platform_name))

    return platform


def add_reporters(
    simulation,
    out_dir: str = None,
    suffix: str = None,
    total_steps: int = 250000,
    logperiod: int = 2500,
    verbose: int = 2,
    atom_subset: list[int] | None = None,
) -> None:
    """Attach DCD trajectory and CSV state-data reporters to a simulation.

    Replaces any existing reporters on the simulation object. A progress
    reporter is always written to stdout. A DCD file is always written.
    A CSV with thermodynamic state data is written when ``verbose > 0``;
    full energies, temperature, volume, and density are included when
    ``verbose > 1``.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The simulation to which reporters are added.
    out_dir : str
        Directory for output files.
    suffix : str
        Filename stem; output files are ``<out_dir>/<suffix>.dcd`` and
        ``<out_dir>/<suffix>.csv``.
    total_steps : int
        Total number of integration steps (used for progress reporting).
    logperiod : int
        Reporting interval in steps.
    verbose : int
        Verbosity level: 0 = no CSV, 1 = CSV with progress only,
        2 = CSV with full thermodynamic data.
    atom_subset : list of int, optional
        Atom indices written to the DCD (None = all atoms).
    """

    logging.debug(f"Adding reporters to the simulation")
    simulation.reporters = []  # Delete all current reporters
    
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            logperiod,
            step=True,
            time=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=total_steps,
            separator="\t",
        )
    )

    simulation.reporters.append(
        DCDReporter(
            f"{out_dir}/{suffix}.dcd",
            reportInterval=logperiod,
            enforcePeriodicBox=False,  # WARNING this compromises autoimaging afterwards in some cases
            atomSubset=atom_subset,
        )
    )

    if verbose > 0:

        simulation.reporters.append(
            StateDataReporter(
                f"{out_dir}/{suffix}.csv",
                logperiod,
                step=True,
                time=True,
                potentialEnergy=True if verbose > 1 else False,
                kineticEnergy=True if verbose > 1 else False,
                totalEnergy=True if verbose > 1 else False, 
                temperature=True if verbose > 1 else False,
                progress=True,
                volume=True if verbose > 1 else False,
                density=True if verbose > 1 else False,
                remainingTime=True,
                speed=True,
                totalSteps=total_steps,
            )
        )

    return

def add_barostat(system: System = None, temp: float = 300, is_membrane: bool = False) -> System:
    """Add an appropriate Monte Carlo barostat to the OpenMM system.

    Parameters
    ----------
    system : openmm.System
        The system to modify in-place.
    temp : float
        Target temperature in Kelvin.
    is_membrane : bool
        If True, use a ``MonteCarloMembraneBarostat`` with zero surface tension
        and semi-isotropic pressure coupling (XY isotropic, Z free), which is
        appropriate for membrane protein simulations. If False, use an isotropic
        ``MonteCarloBarostat`` at 1 atm.

    Returns
    -------
    openmm.System
        The system with the barostat force added.
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
    """Apply user-specified protonation-state variants and add hydrogens.

    Parameters
    ----------
    modeller : openmm.app.Modeller
        OpenMM Modeller to modify.
    variants_dict : dict
        Mapping of ``"<chain_id>:<residue_number>"`` → variant string (e.g.
        ``"HIE"``, ``"HID"``). Residues not present in the dict receive the
        default protonation state.

    Returns
    -------
    openmm.app.Modeller
        The same Modeller object after ``addHydrogens`` has been called with
        the specified variants list.
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

def build_residue_mapping(original_pdb: str, system_pdb: str, chain: str = None) -> dict:
    """Map original PDB residue IDs to system.pdb sequential IDs via CA sequence-position alignment.

    Returns {orig_resid (int): system_resid (int)} for every protein CA in the original PDB.
    Warns when counts differ (PDBFixer modelled in missing residues) or resnames disagree
    (non-standard residue replacement).
    """
    import logging
    _log = logging.getLogger('autopath.utils')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_orig = mda.Universe(original_pdb)
        u_sys  = mda.Universe(system_pdb)

    if chain:
        orig_sel = f'(segid {chain} or chainID {chain}) and name CA'
    else:
        orig_sel = 'protein and name CA'

    orig_cas = u_orig.select_atoms(orig_sel)
    sys_cas  = u_sys.select_atoms('protein and name CA')

    if len(orig_cas) != len(sys_cas):
        _log.warning(
            f"CA count mismatch: original PDB has {len(orig_cas)} residues, "
            f"system.pdb has {len(sys_cas)}. PDBFixer may have modelled in missing residues. "
            "Mapping by sequence position up to the shorter chain — verify residue_mapping.json."
        )

    n = min(len(orig_cas), len(sys_cas))
    mapping = {}
    renamed = []
    for i in range(n):
        o, s = orig_cas[i], sys_cas[i]
        if o.resname != s.resname:
            renamed.append(f"{o.resname}{o.resid}→{s.resname}{s.resid}")
        mapping[int(o.resid)] = int(s.resid)

    if renamed:
        _log.warning(
            f"Residue name mismatches (non-standard→standard replacement?): "
            + ", ".join(renamed)
        )

    return mapping


def get_pocket_atoms_idxs(u: mda.Universe = None,
                          pocket_selection: str = None,
                          ligand_selection: str = None,
                          cutoff: float = 8.0
                          ) -> List[int]:
    """Return atom indices defining the binding pocket.

    Exactly one of *pocket_selection* or *ligand_selection* must be provided.
    When *pocket_selection* is given it is used directly. When *ligand_selection*
    is given, the pocket is the protein CA atoms within *cutoff* Å of any ligand
    heavy atom (evaluated at the last trajectory frame).

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe containing the system.
    pocket_selection : str, optional
        Arbitrary MDAnalysis selection string for the pocket atoms.
    ligand_selection : str, optional
        MDAnalysis selection string for the ligand; used to identify pocket
        residues by proximity.
    cutoff : float
        Distance cutoff in Angstroms for proximity-based pocket detection.

    Returns
    -------
    list of int
        Atom indices of the pocket atoms.
    """

    if u is None:
        raise ValueError("No MDAnalysis Universe provided.")

    u.trajectory[-1]  # set pointer to last frame if its a trajectory

    if pocket_selection is None and ligand_selection is None:
        raise ValueError("No pocket selection or ligand residue name provided.")
    # If a custom pocket selection is provided, use it directly
    elif pocket_selection is not None and ligand_selection is None:
        pocket_atoms = u.select_atoms(pocket_selection)
        pocket_atoms_indices = [atom.index for atom in pocket_atoms]
    # If no custom selection, use the ligand residue name to define the pocket
    elif ligand_selection is not None and pocket_selection is None:
        # CA within cutoff of ligand heavy atoms: any-atom contact pulls in CAs ~12 A away
        ligand = u.select_atoms(f"({ligand_selection}) and not name H*")
        pocket_atoms_indices = [int(i) for i in u.select_atoms(
            f"protein and name CA and around {cutoff} group ligand", ligand=ligand).indices]
    else:
        raise ValueError("Please provide either a pocket selection or a ligand residue name, not both.")

    if len(pocket_atoms_indices) == 0:
        raise ValueError("No atoms found for the provided pocket selection.")
        
    return pocket_atoms_indices


def reduce_to_murcko_scaffold(u, lig_selection: str, img_name: str = None):
    """
    Reduce ligand atoms to their Murcko scaffold representation.

    Returns
    -------
    reduced_ligand : MDAnalysis.AtomGroup
    highlight_rdk_indices : list of int (for RDKit visualization)
    mol : RDKit Mol object (with Hs removed and 2D coords)
    """
    if img_name is None:
        img_name = f"ligand_murcko.png"

    ligand_all = u.select_atoms(lig_selection)
    mol = ligand_all.convert_to('RDKIT')
    sel_atoms = mol.GetAtoms()

    try:
        murcko = MurckoScaffold.GetScaffoldForMol(mol)
        murcko_match = mol.GetSubstructMatch(murcko)
        murcko_atom_names = [sel_atoms[i].GetProp('_MDAnalysis_name') for i in murcko_match]
        reduced_ligand = u.select_atoms(f'({lig_selection}) and name {" ".join(murcko_atom_names)}')

        return reduced_ligand, murcko_match, mol

    except Exception as e:
        logging.warning(f"Could not extract Murcko scaffold: {e}")
        return u.select_atoms(f'({lig_selection}) and not name H*'), [], mol

def _count_pocket_contacts(u, ligand, pocket, frames: int, cutoff: float = 3.5) -> np.ndarray:
    """Per-atom count of frames with a pocket atom within *cutoff* Å.

    Samples up to *frames* evenly spaced frames from the second half of the
    trajectory, using minimum-image distances. Leaves the trajectory at the
    last frame.
    """
    n_frames = u.trajectory.n_frames
    start = n_frames // 2
    n_sample = min(frames, n_frames - start)
    sample = np.unique(np.linspace(start, n_frames - 1, n_sample).astype(int))
    counts = np.zeros(len(ligand))
    for ts in u.trajectory[sample]:
        dmat = distance_array(ligand.positions, pocket.positions, box=ts.dimensions)
        counts += (dmat < cutoff).any(axis=1)
    return counts


def get_ligand_anchor_atoms(
    u,
    lig_selection: str,
    pocket_sel: str = None,
    mode: str = "murcko",
    frames: int = 100,
    n_atoms: int = 5,
    reduce_before: bool = False,
    expand_rings: bool = True,
    out_dir: str = None,
    verbose: bool = True,
    ref_mol=None,
):
    """Select anchor atoms on the ligand for COM-based pulling and optionally visualize them.

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe (trajectory should be loaded).
    lig_selection : str
        MDAnalysis selection string identifying the ligand.
    pocket_sel : str, optional
        Selection string for the binding pocket; defaults to protein atoms within
        5 Å of the ligand (heavy atoms only).
    mode : str
        Strategy for picking anchor atoms. Options:

        * ``"lig_ha"``      — all ligand heavy atoms.
        * ``"murcko"``      — Murcko scaffold heavy atoms.
        * ``"ca"``          — ligand CA atoms (useful for peptides/cofactors).
        * ``"lig_com"``     — heavy atoms closest to the ligand COM.
        * ``"pocket_com"``  — heavy atoms closest to the pocket COM.
        * ``"contacts"``    — atoms with the most pocket contacts over the trajectory.
        * ``"inertia"``     — atoms along the principal inertia axis.
        * ``"weighted_com"``— COM of the top-contact atoms.
    frames : int
        Number of frames to sample, evenly spaced over the second half of the
        trajectory, for contact-based modes.
    n_atoms : int
        Maximum number of anchor atoms for COM/contact/inertia modes.
    reduce_before : bool
        If True, reduce the ligand to its Murcko scaffold *before* applying
        the chosen mode.
    expand_rings : bool
        If True, expand the anchor selection to include all atoms in any ring
        that contains at least one anchor atom (heavy atoms only).
    out_dir : str, optional
        Directory where the 2D highlight image is saved.
    verbose : bool
        Log ring-expansion details at DEBUG level.
    ref_mol : rdkit.Chem.Mol, optional
        Reference RDKit molecule for 2D visualization alignment.

    Returns
    -------
    list of int
        MDAnalysis atom indices of the selected anchor atoms.
    """
    
    if out_dir is None:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    import re
    sel_slug = re.sub(r'[^\w]', '_', lig_selection)[:30]
    img_name = f"{out_dir}/pulling_{sel_slug}_{mode}.png"

    if pocket_sel is None:
        pocket_sel = f"protein and around 5 ({lig_selection}) and not name H*"

    ligand_full = u.select_atoms(lig_selection)
    ligand_ha = u.select_atoms(f"({lig_selection}) and not name H*")

    if ligand_full.n_atoms == 0:
        raise ValueError(f"No atoms found for ligand selection '{lig_selection}'.")

    is_multi_residue = ligand_full.n_residues > 1

    # RDKit mol from full ligand (keep Hs so indexing matches MDAnalysis)
    mol = ligand_full.convert_to("RDKIT") if not is_multi_residue else None

    # make sure we are at the last frame
    u.trajectory[-1]
    pocket = u.select_atoms(pocket_sel)
    anchor = []

    # Murcko is only valid for single-residue small molecules
    if mode == "murcko" and is_multi_residue:
        logging.warning(
            f"Murcko scaffold mode is not supported for multi-residue ligand "
            f"('{lig_selection}' spans {ligand_full.n_residues} residues). "
            "Falling back to 'lig_ha'."
        )
        mode = "lig_ha"

    # optional Murcko reduction before anchor selection
    if reduce_before and not is_multi_residue:
        ligand, highlight_rdk_indices, mol = reduce_to_murcko_scaffold(
            u, lig_selection, img_name
        )
        # filter out H for pulling
        ligand = ligand.select_atoms("not name H*")
    else:
        # use heavy atoms for anchor selection
        ligand = ligand_ha
        highlight_rdk_indices = []

    # TODO: implement MM-GBSA residue decomposition as an additional mode.

    if mode == "lig_ha":
        # all heavy atoms of the ligand
        anchor = [a.index for a in ligand_ha.atoms]

    elif mode == "murcko":
        ligand, anchor_indices, mol = reduce_to_murcko_scaffold(
            u, lig_selection, img_name
        )
        # only heavy atoms for anchors
        ligand = ligand.select_atoms("not name H*")
        anchor = [a.index for a in ligand.atoms]

    elif mode == "ca":
        anchor = list(u.select_atoms(f"({lig_selection}) and name CA").indices)

    elif mode == "lig_com":
        com = ligand.center_of_mass()
        dists = np.linalg.norm(ligand.positions - com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    elif mode == "pocket_com":
        pocket_com = pocket.center_of_mass()
        dists = np.linalg.norm(ligand.positions - pocket_com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    elif mode == "contacts":
        contact_counts = _count_pocket_contacts(u, ligand, pocket, frames)
        top_indices = np.argsort(contact_counts)[-n_atoms:]
        anchor = ligand.atoms[top_indices].indices

    elif mode == "inertia":
        coords = ligand.positions - ligand.center_of_mass()
        inertia_tensor = np.dot(coords.T, coords)
        eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
        principal_axis = eigvecs[:, np.argmin(eigvals)]
        projections = np.dot(coords, principal_axis)
        anchor = ligand.atoms[np.argsort(projections)[:n_atoms]].indices

    elif mode == "weighted_com":
        contact_counts = _count_pocket_contacts(u, ligand, pocket, frames)
        top_indices = np.argsort(contact_counts)[-n_atoms:]
        anchor_coords = ligand.positions[top_indices]
        anchor_com = anchor_coords.mean(axis=0)
        dists = np.linalg.norm(ligand.positions - anchor_com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # expand rings in RDKit space if requested (single-residue small molecules only)
    if expand_rings and not is_multi_residue and mol is not None:
        # map MDAnalysis atom index -> RDKit index
        idx_map = {a.index: i for i, a in enumerate(ligand_full.atoms)}
        rdk_anchor_indices = [idx_map[i] for i in anchor if i in idx_map]

        ring_info = mol.GetRingInfo()
        anchor_rings = [
            set(ring) for ring in ring_info.AtomRings()
            if any(i in ring for i in rdk_anchor_indices)
        ]

        expanded_rdk_indices = set()
        for ring in anchor_rings:
            expanded_rdk_indices.update(ring)

        # convert back to MDAnalysis indices, skip hydrogens
        expanded_mda_indices = []
        for ridx in expanded_rdk_indices:
            atom = ligand_full.atoms[ridx]
            if not atom.name.startswith("H"):
                expanded_mda_indices.append(atom.index)

        anchor = sorted(set(anchor).union(expanded_mda_indices))

        if verbose:
            logger.debug(f"[get_ligand_anchor_atoms] Expanded to include rings (heavy only): {expanded_mda_indices}")

    # Draw 2D image (single-residue small molecules only)
    if not is_multi_residue:
        try:
            if ref_mol is not None:
                mol_draw = Chem.RemoveHs(ref_mol)
                Chem.rdDepictor.Compute2DCoords(mol_draw)
                # anchor MDA indices → position among heavy atoms → ref_mol atom index
                ha_indices = list(ligand_ha.atoms.indices)
                highlight_rdk_indices = [ha_indices.index(i) for i in anchor if i in ha_indices]
            else:
                idx_map = {a.index: i for i, a in enumerate(ligand_full.atoms)}
                highlight_rdk_indices = [idx_map[i] for i in anchor if i in idx_map]
                mol_draw = Chem.RemoveHs(mol)
                Chem.rdDepictor.Compute2DCoords(mol_draw)
            img = Chem.Draw.MolToImage(
                mol_draw,
                size=(300, 300),
                highlightAtoms=highlight_rdk_indices,
            )
            img.save(img_name)
        except Exception as e:
            logging.warning(f"Could not generate 2D image with highlights: {e}")
        
    #ligand_ag = u.atoms[anchor]
    
    return anchor


def get_center(positions, atoms, group, weighByMass):
    """Return the center of mass (COM) or center of geometry (COG) for a group of atoms.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n_atoms, 3) with atomic positions in nanometers
        (as returned by ``getPositions(asNumpy=True) / openmmunit.nanometers``).
    atoms : list of openmm.app.topology.Atom
        All atoms in the topology, in index order.
    group : list of int
        Atom indices belonging to the group of interest.
    weighByMass : bool
        If True, compute the mass-weighted center (COM); otherwise compute the
        unweighted geometric center (COG).

    Returns
    -------
    np.ndarray
        Shape (3,) center coordinates in the same units as *positions*.
    """
    group_positions = positions[group]

    if weighByMass:
        masses = np.array([atom.element.mass.value_in_unit(openmmunit.dalton) for atom in (atoms[i] for i in group)])
        if sum(masses) == 0:
            logging.warning("All atoms in the group have zero mass. Using simple mean instead.")
            masses = None
        center = np.average(group_positions, axis=0, weights=masses)
    else:
        center = np.mean(group_positions, axis=0)
    return center
    
def get_COM_dist(simulation,
                 groupA: list[int] = None,
                 groupB: list[int] = None,
                 weighByMass: bool = True
                 ) -> float:
    """Calculate the distance between the centers of mass (or geometry) of two atom groups.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        The running simulation; positions are read from its context.
    groupA : list of int
        Atom indices for the first group.
    groupB : list of int
        Atom indices for the second group.
    weighByMass : bool
        If True, compute center of mass; if False, compute center of geometry.

    Returns
    -------
    float
        Distance in nanometers.

    Note
    ----
    Positions are retrieved via ``getPositions(asNumpy=True) / openmmunit.nanometers``,
    which strips the OpenMM unit wrapper while normalizing to nanometers. The
    returned scalar is therefore dimensionless but numerically in nanometers.
    """
    state = simulation.context.getState(getPositions=True, getVelocities=False)
    positions = state.getPositions(asNumpy=True) / openmmunit.nanometers
    atoms = [atom for atom in simulation.topology.atoms()]

    centerA = get_center(positions, atoms, groupA, weighByMass)
    centerB = get_center(positions, atoms, groupB, weighByMass)
    dist = np.linalg.norm(centerA - centerB)

    return dist


def compute_rmsd(u,
                u_ref,
                alig_select: str = 'backbone',
                groupselections: dict = {},
                aligned_fname: str = None,
                plots_outdir: str = None,
                suffix: str = None
                ) -> pd.DataFrame:
    """Compute RMSD of one or more atom groups relative to a reference structure.

    Parameters
    ----------
    u : mda.Universe
        Mobile trajectory universe.
    u_ref : mda.Universe
        Reference universe (single frame).
    alig_select : str
        MDAnalysis selection string used for alignment (default ``'backbone'``).
    groupselections : dict
        Mapping of label -> MDAnalysis selection string for additional RMSD
        groups to track beyond the alignment selection.
    aligned_fname : str, optional
        If provided, write the aligned trajectory to this path.
    plots_outdir : str, optional
        If provided, save per-group RMSD line plots to this directory.
    suffix : str, optional
        Appended to plot filenames to distinguish runs.

    Returns
    -------
    pd.DataFrame
        Columns: ``frame``, ``time (ps)``, ``RMSD_selected_alignment``, and
        one ``RMSD_<label>`` column per entry in *groupselections*.
    """
    r = RMSD(u,
             u_ref,
             select=alig_select,
             groupselections=list(groupselections.values()),
             ref_frame=0).run()

    rmsd_results = r.results.rmsd  # Do not skip any columns
    columns = ['frame','time (ps)', f'RMSD_selected_alignment'] + [f'RMSD_{group}' for group in groupselections.keys() if group is not None]
    rmsd_df = pd.DataFrame(rmsd_results, columns=columns)

    if aligned_fname is not None:
        # Align the trajectory to the reference and save it
        with mda.Writer(aligned_fname, n_atoms=u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u.atoms)

    if plots_outdir is not None:
        plt.figure(figsize=(10, 5))
        for col in columns[3:]:
            sns.lineplot(x='frame', y=col, data=rmsd_df)
            plt.xlabel('Frame');            plt.ylabel(f'RMSD (Å)')
            plt.title(f'{col} RMSD')
            plt.tight_layout()
            if suffix is not None:
                plt.savefig(f'{plots_outdir}/rmsd_{suffix}_{col}.png')
            else:
                plt.savefig(f'{plots_outdir}/rmsd_{col}.png')
            plt.close()

    return rmsd_df


def write_pocket_pymol(
    u: mda.Universe,
    out_dir: str,
    protein_selection: str = "protein",
    ligand_selection: str = "resname UNK",
    pocket_selection: str = "same residue as protein and (around 4 resname UNK) and (not name H*)",
    show_surface: bool = False,
    ligand_color: str = "yellow",
    pocket_color: str = "orange",
    protein_color: str = "slate",
    com_color: str = "red",
    com_sphere_radius: float = 0.5,
    output_format: str = "pse",
) -> None:
    """Visualize protein, ligand, pocket, and pocket COM in PyMOL.

    Args:
        output_format: ``"pse"`` saves a portable self-contained session;
            ``"pml"`` writes a script + auxiliary PDB files next to it.
    """
    if output_format not in ("pse", "pml"):
        raise ValueError(f"output_format must be 'pse' or 'pml', got '{output_format}'")

    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)

    protein_atoms = u.select_atoms(protein_selection)
    ligand_atoms = u.select_atoms(ligand_selection)
    pocket_atoms = u.select_atoms(pocket_selection)
    pocket_com = pocket_atoms.center_of_mass()
    pocket_resids = sorted(set(pocket_atoms.resids))
    pocket_resi_sel = "+".join(str(r) for r in pocket_resids)


    # ------------------------------------------------------------------ #
    # PSE branch                                                           #
    # ------------------------------------------------------------------ #
    if output_format == "pse":
        try:
            import pymol2
        except ImportError:
            raise ImportError("pymol2 is required. Install open-source PyMOL into your environment.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            protein_pdb = str(tmpdir / "protein.pdb")
            ligand_pdb = str(tmpdir / "ligand.pdb")
            pocket_pdb = str(tmpdir / "pocket.pdb")

            with mda.Writer(protein_pdb, protein_atoms.n_atoms) as w:
                w.write(protein_atoms)
            with mda.Writer(ligand_pdb, ligand_atoms.n_atoms) as w:
                w.write(ligand_atoms)
            with mda.Writer(pocket_pdb, pocket_atoms.n_atoms) as w:
                w.write(pocket_atoms)

            pse_path = str(out_dir / "pocket_view.pse")
            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd
                cmd.bg_color("white")
                cmd.load(protein_pdb, "protein")
                cmd.load(ligand_pdb, "ligand")
                cmd.load(pocket_pdb, "pocket")
                cmd.pseudoatom("pocket_com", pos=list(map(float, pocket_com)))
                cmd.hide("everything")
                cmd.show("cartoon", "protein")
                cmd.show("sticks", "ligand")
                cmd.show("sticks", "pocket")
                cmd.show("spheres", "pocket_com")
                cmd.color(protein_color, "protein")
                cmd.color(pocket_color, f"protein and resi {pocket_resi_sel}")
                cmd.color(ligand_color, "ligand")
                cmd.color(pocket_color, "pocket")
                cmd.color(com_color, "pocket_com")
                cmd.set("sphere_scale", com_sphere_radius, "pocket_com")
                cmd.set("stick_radius", 0.2)
                cmd.set("cartoon_transparency", 0.2)
                if show_surface:
                    cmd.show("surface", "protein")
                    cmd.color("gray70", "protein")
                    cmd.set("transparency", 0.35, "protein")
                cmd.zoom("ligand", 12)
                cmd.save(pse_path)

        return None

    # ------------------------------------------------------------------ #
    # PML branch                                                           #
    # ------------------------------------------------------------------ #
    protein_pdb = str(out_dir / "pocket_protein.pdb")
    ligand_pdb = str(out_dir / "pocket_lig.pdb")
    pocket_pdb = str(out_dir / "pocket_definition.pdb")

    with mda.Writer(protein_pdb, protein_atoms.n_atoms) as w:
        w.write(protein_atoms)
    with mda.Writer(ligand_pdb, ligand_atoms.n_atoms) as w:
        w.write(ligand_atoms)
    with mda.Writer(pocket_pdb, pocket_atoms.n_atoms) as w:
        w.write(pocket_atoms)

    lines = [
        "reinitialize",
        f"load {os.path.basename(protein_pdb)}, protein",
        f"load {os.path.basename(ligand_pdb)}, ligand",
        f"load {os.path.basename(pocket_pdb)}, pocket",
        f"pseudoatom pocket_com, pos=[{pocket_com[0]:.3f},{pocket_com[1]:.3f},{pocket_com[2]:.3f}]",
        "hide everything",
        "show cartoon, protein",
        "show sticks, ligand",
        "show sticks, pocket",
        "show spheres, pocket_com",
        f"color {protein_color}, protein",
        f"color {pocket_color}, protein and resi {pocket_resi_sel}",
        f"color {ligand_color}, ligand",
        f"color {pocket_color}, pocket",
        f"color {com_color}, pocket_com",
        f"set sphere_scale, {com_sphere_radius}, pocket_com",
        "set stick_radius, 0.2",
        "set cartoon_transparency, 0.2",
        "zoom ligand, 12",
        "bg_color white",
    ]
    if show_surface:
        lines.extend([
            "show surface, protein",
            "set surface_color, gray70, protein",
            "set transparency, 0.35, protein",
        ])

    (out_dir / "pocket_view.pml").write_text("\n".join(lines))
    return None


