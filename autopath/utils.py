import os
import shutil
import logging
import tempfile

logger = logging.getLogger("autopath")
import numpy as np
import pandas as pd
from sys import stdout, exit
from glob import glob
from pathlib import Path

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
from MDAnalysis.analysis.rms import RMSD, RMSF
from scipy.spatial.distance import cdist
from MDAnalysis.analysis.distances import distance_array

from pymol import cmd

from scipy.spatial import KDTree

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("fivethirtyeight")

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Scaffolds import MurckoScaffold

from deeptime.clustering import RegularSpace, KMeans

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

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return None

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def align_trajectory_pytraj(
    prmtop_file: str = None,
    traj_file: Union[str, list] = None,
    stride: int = None,
    super_mask: str = "@CA,C,N",
    strip_mask: str = None,  #':HOH,NA,CL,K,POP'
    out_fname: str = None,
) -> None:

    """THis is kinda deprecated since we are now using mdtraj for trajectory processing, but keeping here for now in case we want to add back in some pytraj-specific processing down the line."""
    import pytraj as pt

    ptraj = pt.iterload(traj_file, prmtop_file, stride=stride)
    ptraj = ptraj.autoimage()
    ptraj = ptraj.center()
    ptraj = ptraj.superpose(ref=0, mask=super_mask)

    if strip_mask is not None:
        ptraj = ptraj.strip(strip_mask)
        pt.save(out_fname.replace('.dcd','_dry.prmtop'), ptraj.top, overwrite=True)
    ptraj.save(out_fname)

    return


def wrap_align_save_traj(traj_files, topology, remove_original=True, is_membrane=False):
    import mdtraj as md
    if isinstance(traj_files, str):
        traj_files = [traj_files]
    aligned_paths = []
    for traj_file in traj_files:
        traj = md.load(traj_file, top=topology)
        traj = traj.center_coordinates()
        # image_molecules is prohibitively slow for membrane systems (hundreds of lipids,
        # thousands of atoms); centering is sufficient for sMD/metadynamics analysis.
        if not is_membrane:
            traj = traj.image_molecules(make_whole=True)
        backbone = traj.topology.select("backbone")
        if len(backbone) > 0:
            try:
                traj = traj.superpose(traj[0], atom_indices=backbone)
            except Exception as e:
                logging.warning(f"Superposition failed for {traj_file}: {e}. Proceeding without superposition.")
        out_path = traj_file.replace(".dcd", "_aligned.dcd")
        traj.save(out_path)
        if remove_original:
            os.remove(traj_file)
        aligned_paths.append(out_path)
    return aligned_paths


from autopath.pdb_preprocessor import (
    fetch_smiles,
    get_scrubbed_smile,
    fetch_pdb,
    get_ligand_name,
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

    simulation.saveCheckpoint(f"{out_path}.chk")
    simulation.saveState(f"{out_path}.xml")

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

def save_amber_files(
    topology: app.Topology = None,
    positions: list = None,
    system: System = None,
    out_path: str = None,
) -> None:

    """Saves the OpenMM system and topology to AMBER format files.
    https://parmed.github.io/ParmEd/html/openmm.html
    If system is None, it will not save the data from system but still will save the topology and positions. 
    """
    os.makedirs(out_path, exist_ok=True)

    parmed_structure = parmed.openmm.topsystem.load_topology(
        topology, system, positions
    )

    parmed_structure.save(f"{out_path}/system.prmtop", overwrite=True, format="amber")
    if positions is not None:
        parmed_structure.save(f"{out_path}/system.rst7", overwrite=True, format="rst7")

    return

def select_platform(platform_name: str = None, device_index: str = "0"):

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
    elif platform_name == "CPU":
        platform = Platform.getPlatformByName('CPU')
    elif platform_name == "REFERENCE":
        platform = Platform.getPlatformByName('Reference')
    else:
        raise ValueError(f"Unknown OpenMM platform: {platform_name}")

    return platform


def add_reporters(
    simulation,
    out_dir: str = None,
    suffix: str = None,
    total_steps: int = 250000,
    logperiod: int = 2500,
    verbose: int = 2,
) -> None:
    """Set up the reporters"""

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

def add_barostat(system: System=None, temp: float=300, is_membrane: bool=False) -> System:
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

def get_pocket_atoms_idxs(u:mda.Universe = None,
                     pocket_selection:str = None,
                     ligand_selection:str = None,
                     cutoff: float = 6.0
                     ) -> List[int]:
    """Get the pocket atoms based on a user provided selection 
    or the ligand residue name and some default heuristics."""

    if u is None:
        print("No MDAnalysis Universe provided.")
        exit(1)
        
    u.trajectory[-1]  # set pointer to last frame if its a trajectory

    if pocket_selection is None and ligand_selection is None:
        print("No pocket selection or ligand residue name provided.")
        exit(1)
    # If a custom pocket selection is provided, use it directly
    elif pocket_selection is not None and ligand_selection is None:
        pocket_atoms = u.select_atoms(pocket_selection)
        pocket_atoms_indices = [atom.index for atom in pocket_atoms]
    # If no custom selection, use the ligand residue name to define the pocket
    elif ligand_selection is not None and pocket_selection is None:
        # backbone_names = ["N", "CA", "C", "O"]
        ligand = u.select_atoms(ligand_selection)
        protein_residues = u.select_atoms(f"protein and around {cutoff} group ligand", ligand=ligand).residues
        pocket_atoms_indices = [atom.index for res in protein_residues for atom in res.atoms if atom.name in ['CA']]
    else:
        print("Please provide either a pocket selection or a ligand residue name, not both.")
        exit(1)
        
    if len(pocket_atoms_indices) == 0:
        print(f"No atoms found for the provided pocket selection")
        exit(1)
        
    return pocket_atoms_indices


def _parse_pdb_ss_residues(pdb_path: str) -> set:
    """Return the set of residue numbers declared in HELIX/SHEET PDB records."""
    ss_residues = set()
    with open(pdb_path) as fh:
        for line in fh:
            try:
                if line.startswith("HELIX"):
                    tokens = line.split()
                    start, end = int(tokens[5]), int(tokens[8])
                    ss_residues.update(range(start, end + 1))
                elif line.startswith("SHEET"):
                    tokens = line.split()
                    start, end = int(tokens[6]), int(tokens[9])
                    ss_residues.update(range(start, end + 1))
            except (IndexError, ValueError):
                pass
    return ss_residues


def generate_smd_pocket_selection(
    pdb_path: str,
    sdf_dir: str,
    cutoff: float = 8.0,
    max_pocket_radius: float = 15.0,
    out_dir: str = None,
    com_sphere_radius: float = 1.5,
    verbose: bool = True,
    use_pdb_ss: bool = True,
) -> str:
    """Generate a protein CA selection string for COM-based steered MD pulling.

    Identifies pocket residues as the union of contacts across all docked
    ligands in *sdf_dir*, then restricts them to those within
    *max_pocket_radius* of the ligand ensemble COM to exclude spurious
    contacts from ligand tails or outlier poses. Residues in secondary
    structure elements (helices and sheets) are ranked first to minimise
    pulling artefacts; all selections use CA atoms only.

    When *use_pdb_ss* is True (default), secondary structure is read from the
    HELIX/SHEET records already present in the PDB file. This is preferred over
    DSSP for SMD selection because crystal-structure annotations identify stable
    structural elements regardless of local conformation in any given MD frame.
    Loop residues are additionally restricted to within *cutoff* Å of the
    ligand ensemble COM, so flexible loops that would shift during pulling are
    excluded. Falls back to DSSP if the PDB contains no HELIX/SHEET records.

    Parameters
    ----------
    pdb_path : str
        Path to the receptor PDB file.
    sdf_dir : str
        Directory containing individual ligand SDF files.
    cutoff : float
        Distance cutoff in Angstrom for pocket residue detection.
        Also used as the maximum COM distance for loop residues when
        *use_pdb_ss* is True.
    max_pocket_radius : float
        Maximum distance in Angstrom from the ligand ensemble COM for
        SS residues. Loop residues use *cutoff* instead when *use_pdb_ss*
        is True.
    out_dir : str, optional
        If provided, writes ``ligands_com.pdb``, ``pocket_com.pdb``, and
        ``com_comparison.pml`` for visual QC in PyMOL.
    com_sphere_radius : float
        Sphere scale used for COM pseudoatoms in the PyMOL script.
    verbose : bool
        Log summary statistics via the standard logger.
    use_pdb_ss : bool
        If True, identify secondary structure from HELIX/SHEET records in the
        PDB file and apply a tighter COM-distance filter (*cutoff*) to loop
        residues. Falls back to DSSP when the PDB has no such records.
        If False, use DSSP on the loaded structure for all residues.

    Returns
    -------
    str
        MDAnalysis-compatible selection string, e.g.
        ``'(resid 63 64 65 316 317 318) and name CA'``
    """
    # --- 1. Load protein and assign secondary structure ---
    u = mda.Universe(pdb_path)
    protein_ca = u.select_atoms("protein and name CA")
    ca_residues = protein_ca.residues

    _use_pdb_ss = use_pdb_ss
    if _use_pdb_ss:
        ss_residues = _parse_pdb_ss_residues(pdb_path)
        if ss_residues:
            if verbose:
                n_prot_in_ss = sum(1 for r in ca_residues if r.resnum in ss_residues)
                logging.info(
                    f"PDB HELIX/SHEET records: {n_prot_in_ss} CA residues in SS "
                    f"out of {len(ca_residues)} total"
                )
        else:
            logging.warning(
                "No HELIX/SHEET records found in PDB; falling back to DSSP."
            )
            _use_pdb_ss = False

    if not _use_pdb_ss:
        from MDAnalysis.analysis.dssp import DSSP
        # Exclude terminal cap residues (ACE/NME/NMA): they lack a full N/CA/C/O
        # backbone set and cause DSSP to raise a ValueError on the count check.
        std_prot = u.select_atoms("protein and not resname ACE NME NMA")
        ss_codes = DSSP(std_prot).run().results.dssp[0]
        ss_residues = {
            res.resnum
            for res, code in zip(std_prot.residues, ss_codes)
            if code in ('H', 'E')
        }
        if verbose:
            logging.info(
                f"DSSP: {len(ss_residues)} residues in secondary structure "
                f"(H/E) out of {len(std_prot.residues)} total"
            )

    # --- 2. Load all ligand heavy-atom positions from SDF files ---
    sdf_files = sorted(glob(os.path.join(sdf_dir, "*.sdf")))
    if not sdf_files:
        raise FileNotFoundError(f"No SDF files found in {sdf_dir}")

    all_lig_positions = []
    n_loaded = 0
    for sdf_path in sdf_files:
        for mol in Chem.SDMolSupplier(sdf_path, removeHs=True):
            if mol is None:
                continue
            all_lig_positions.append(mol.GetConformer().GetPositions())
            n_loaded += 1

    if not all_lig_positions:
        raise ValueError("No valid molecules could be loaded from the SDF files.")

    lig_positions = np.vstack(all_lig_positions)   # (N_heavy_atoms, 3)
    # Average of per-ligand COMs so every ligand contributes equally regardless
    # of how many heavy atoms it has (avoids bias from large vs small ligands).
    ligand_com = np.mean([pos.mean(axis=0) for pos in all_lig_positions], axis=0)

    if verbose:
        logging.info(
            f"Loaded {n_loaded} ligands from {len(sdf_files)} SDF files "
            f"({len(lig_positions)} heavy atoms total)"
        )

    # --- 3. Find pocket CA atoms within cutoff of any ligand heavy atom ---
    ca_positions = protein_ca.positions   # (n_CA, 3)
    tree = KDTree(lig_positions)
    hits = tree.query_ball_point(ca_positions, r=cutoff)
    contacted_mask = np.array([len(h) > 0 for h in hits])

    # --- 4. Filter to residues near the ligand COM ---
    # SS residues: within max_pocket_radius.
    # Loop residues (when use_pdb_ss): within the tighter cutoff distance so
    # that flexible loops contacted only by ligand tails are excluded.
    dist_to_com = np.linalg.norm(ca_positions - ligand_com, axis=1)
    if _use_pdb_ss:
        radius_mask = np.array([
            d <= (max_pocket_radius if res.resnum in ss_residues else cutoff)
            for res, d in zip(ca_residues, dist_to_com)
        ])
    else:
        radius_mask = dist_to_com <= max_pocket_radius
    final_mask = contacted_mask & radius_mask

    contacted_resids = [
        res.resnum
        for res, keep in zip(ca_residues, final_mask)
        if keep
    ]
    if not contacted_resids:
        raise ValueError(
            f"No protein CA atoms found within {cutoff} Å of any ligand "
            f"and {max_pocket_radius} Å of the ligand COM. "
            "Try increasing cutoff or max_pocket_radius."
        )

    n_filtered = int(contacted_mask.sum()) - len(contacted_resids)
    if verbose and n_filtered:
        logging.info(
            f"Filtered out {n_filtered} residues beyond "
            f"{max_pocket_radius} Å (SS) / {cutoff} Å (loops) of the ligand COM"
        )

    # --- 5. Sort: secondary structure first, then ascending residue number ---
    contacted_resids.sort(key=lambda r: (r not in ss_residues, r))

    n_ss = sum(1 for r in contacted_resids if r in ss_residues)
    if verbose:
        logging.info(
            f"Pocket residues: {len(contacted_resids)} total, "
            f"{n_ss} in secondary structure"
        )
        logging.info(f"  SS residues  : {[r for r in contacted_resids if r in ss_residues]}")
        logging.info(f"  Loop residues: {[r for r in contacted_resids if r not in ss_residues]}")

    # --- 6. Build MDAnalysis selection string ---
    resid_str = " ".join(str(r) for r in contacted_resids)
    selection = f"(resid {resid_str}) and name CA"

    # --- 6. Write QC output if requested ---
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        # PyMOL selection string for pocket residues (resi uses '+' as separator)
        resi_sel = "+".join(str(r) for r in contacted_resids)

        def _fmt_pos(v: np.ndarray) -> str:
            return f"[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]"

        # Build PML that loads only the receptor and creates selections/
        # pseudoatoms within PyMOL's own coordinate frame — avoids any
        # misalignment that arises from loading separate MDAnalysis-written
        # PDB files (which carry a triclinic CRYST1 that can shift coords).
        # pocket_com is derived from pocket_sel directly so PyMOL computes it.
        pml_lines = [
            "reinitialize",
            f"load {os.path.abspath(pdb_path)}, receptor",
            "hide everything",
            "show cartoon, receptor",
            "color slate, receptor",
            # Pocket CA residues selected directly from the loaded receptor
            f"select pocket_sel, receptor and resi {resi_sel} and name CA",
            "show sticks, pocket_sel",
            "color orange, pocket_sel",
            # Ligand ensemble COM (from Python) and pocket COM derived from selection
            f"pseudoatom ligands_com, pos={_fmt_pos(ligand_com)}",
            "pseudoatom pocket_com, selection=pocket_sel",
            "show spheres, ligands_com",
            "show spheres, pocket_com",
            "color green, ligands_com",
            "color red,   pocket_com",
            f"set sphere_scale, {com_sphere_radius}",
            "set stick_radius, 0.2",
            "set cartoon_transparency, 0.3",
            "zoom ligands_com, 20",
            "bg_color white",
        ]
        Path(os.path.join(out_dir, "com_comparison.pml")).write_text("\n".join(pml_lines))

        if verbose:
            logging.info(f"QC files written to {out_dir}/")
            logging.info(f"  Ligand ensemble COM : {ligand_com.round(3)}")

    return selection


def reduce_to_murcko_scaffold(u, lig_resname: str, img_name: str = None):
    """
    Reduce ligand atoms to their Murcko scaffold representation.

    Returns
    -------
    reduced_ligand : MDAnalysis.AtomGroup
    highlight_rdk_indices : list of int (for RDKit visualization)
    mol : RDKit Mol object (with Hs removed and 2D coords)
    """
    if img_name is None:
        img_name = f"ligand_{lig_resname}_murcko.png"

    ligand_all = u.select_atoms(f"resname {lig_resname}")
    mol = ligand_all.convert_to('RDKIT')
    sel_atoms = mol.GetAtoms()

    try:
        murcko = MurckoScaffold.GetScaffoldForMol(mol)
        murcko_match = mol.GetSubstructMatch(murcko)
        murcko_atom_names = [sel_atoms[i].GetProp('_MDAnalysis_name') for i in murcko_match]
        reduced_ligand = u.select_atoms(f'resname {lig_resname} and name {" ".join(murcko_atom_names)}')

        return reduced_ligand, murcko_match, mol

    except Exception as e:
        logging.warning(f"Could not extract Murcko scaffold: {e}")
        return u.select_atoms(f'resname {lig_resname} and not name H*'), [], mol

def get_ligand_anchor_atoms(
    u,
    lig_resname: str,
    pocket_sel: str = "protein and around 5 resname UNK and not name H*",
    mode: str = "murcko",
    frames: int = 100,
    n_atoms: int = 5,
    reduce_before: bool = False,
    expand_rings: bool = True,
    out_dir: str = None,
    verbose: bool = True,
    ref_mol=None,
):
    """
    Select anchor atoms in the ligand for pulling and optionally visualize them.
    if reduce_before is True, the ligand is first reduced to its Murcko scaffold before
    If mode="murcko", returns Murcko scaffold atoms.
    If expand_rings is True, expands selection to include entire rings containing anchor atoms.
    Only top n_atoms are selected based on the chosen mode.
    Parameters
    ----------

    """
    
    if out_dir is None:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    img_name = f"{out_dir}/pulling_{lig_resname}_{mode}.png"

    ligand_full = u.select_atoms(f"resname {lig_resname}")
    ligand_ha = u.select_atoms(f"resname {lig_resname} and not name H*")

    if ligand_full.n_atoms == 0:
        raise ValueError(f"No atoms found for ligand {lig_resname}.")

    # RDKit mol from full ligand (keep Hs so indexing matches MDAnalysis)
    mol = ligand_full.convert_to("RDKIT")

    # make sure we are at the last frame
    u.trajectory[-1]
    pocket = u.select_atoms(pocket_sel)
    anchor = []

    # optional Murcko reduction before anchor selection
    if reduce_before:
        ligand, highlight_rdk_indices, mol = reduce_to_murcko_scaffold(
            u, lig_resname, img_name
        )
        # filter out H for pulling
        ligand = ligand.select_atoms("not name H*")
    else:
        # use heavy atoms for anchor selection
        ligand = ligand_ha
        highlight_rdk_indices = []

    # This are the difrent modes implemented.
    # TODO implement MMGBSA by residue decomposition 
    # and select top n_atoms from the ligand interacting residues.

    if mode == "lig_ha":
        # all heavy atoms of the ligand
        anchor = [a.index for a in ligand_ha.atoms]

    elif mode == "murcko":
        ligand, anchor_indices, mol = reduce_to_murcko_scaffold(
            u, lig_resname, img_name
        )
        # only heavy atoms for anchors
        ligand = ligand.select_atoms("not name H*")
        anchor = [a.index for a in ligand.atoms]

    elif mode == "lig_com":
        com = ligand.center_of_mass()
        dists = np.linalg.norm(ligand.positions - com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    elif mode == "pocket_com":
        pocket_com = pocket.center_of_mass()
        dists = np.linalg.norm(ligand.positions - pocket_com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    elif mode == "contacts":
        contact_counts = np.zeros(len(ligand))
        for ts in u.trajectory:#[:frames]:
            dmat = distance_array(ligand.positions, pocket.positions)
            contacts = (dmat < 3.5).any(axis=1)
            contact_counts += contacts
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
        contact_counts = np.zeros(len(ligand))
        for ts in u.trajectory[:frames]:
            dmat = distance_array(ligand.positions, pocket.positions)
            contacts = (dmat < 3.5).any(axis=1)
            contact_counts += contacts
        top_indices = np.argsort(contact_counts)[-n_atoms:]
        anchor_coords = ligand.positions[top_indices]
        anchor_com = anchor_coords.mean(axis=0)
        dists = np.linalg.norm(ligand.positions - anchor_com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # expand rings in RDKit space if requested
    if expand_rings:
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
            print(f"[get_ligand_anchor_atoms] Expanded to include rings (heavy only): {expanded_mda_indices}")

    # Draw 2D image
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
    """get indices and names for all non-hydrogen ligand atoms"""

    residues = topology.residues()
    lig_ha_idx = []
    lig_ha_names = []
    for r in residues:
        if r.name == lig_name:
            lig_ha_names = [a.name for a in r.atoms() if not a.name.startswith("H")]
            lig_ha_idx = [a.index for a in r.atoms() if not a.name.startswith("H")]

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

def get_center(positions, atoms, group, weighByMass):
    """Calculate the center of mass (COM) or center of geometry (COG) for a group of atoms in OpenMM."""

    group_positions = positions[group]  # Get positions for the group

    if weighByMass:
        masses = np.array([atom.element.mass.value_in_unit(openmmunit.dalton) for atom in atoms if atom.index in group])
        if sum(masses) == 0:
            logging.warning("All atoms in the group have zero mass. Using simple mean instead.")
            masses = None
        center = np.average(group_positions, axis=0, weights=masses)  # Weighted average for COM
    else:
        center = np.mean(group_positions, axis=0)  # Simple mean for COG
    return center
    
def get_COM_dist(simulation, 
                 groupA:list[int]=None, 
                 groupB:list[int]=None,
                 weighByMass:bool=True
                 ) -> float:
    """Calculate the distance between the centers of mass (COM) or centers of geometry (COG) of two groups of atoms in OpenMM."""
    
    # Get positions
    state = simulation.context.getState(getPositions=True, getVelocities=False)
    positions = state.getPositions(asNumpy=True) / openmmunit.nanometers
    atoms = [atom for atom in simulation.topology.atoms()]

    # Calculate centers for both groups and their distance
    centerA = get_center(positions, atoms, groupA, weighByMass)
    centerB = get_center(positions, atoms, groupB, weighByMass)
    dist = np.linalg.norm(centerA - centerB)

    return dist  # Unitless, but effectively in nanometers because.... openMM


def calculate_com_distance(u, ligand_atoms=None, pocket_atoms=None, weighByMass: bool = True, wrap: bool = True) -> np.ndarray:
    """ Calculate the distance between the center of mass (COM) or center of geometry (COG) between two atom groups in an MDAnalysis Universe."""
    distances = []
    for ts in u.trajectory:
        if weighByMass:
            lig_com = ligand_atoms.center_of_mass(wrap=wrap)
            prot_com = pocket_atoms.center_of_mass(wrap=wrap)
        else:
            lig_com = ligand_atoms.center_of_geometry(wrap=wrap)
            prot_com = pocket_atoms.center_of_geometry(wrap=wrap)

        distances.append(np.linalg.norm(prot_com - lig_com))

    return np.array(distances) # Distance will be in Angstroms because of MDanalysis

def compute_rmsd(u, 
                u_ref,
                alig_select:str='backbone', 
                groupselections:dict={}, 
                aligned_fname:str=None,
                plots_outdir:str=None,
                suffix:str=None
                ) -> pd.DataFrame:
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

def match_cluster_centroids(X:np.ndarray, centroids:np.ndarray, N:int=1):
    """A function to find the N closest points to each centroid in the dataset X.
    Centroids may not be real data points, so we need to find the closest real data points to them.
    """
    kdtree = KDTree(X)
    closest_points = []
    for centroid in centroids:
        _, indices = kdtree.query(centroid, k=N)
        closest_points.append(indices)

    return closest_points


def match_cluster_centroids_unique(X: np.ndarray,
                                   centroids: np.ndarray,
                                   min_frame_separation: int = 10,
                                   ) -> list[int]:
    """Match each centroid to a unique frame index in X.

    This prevents multiple milestones from collapsing onto the same frame and
    optionally enforces a minimum index separation to avoid near-consecutive picks.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_frames, n_features)")
    if centroids.ndim != 2:
        raise ValueError("centroids must be a 2D array of shape (n_centroids, n_features)")

    selected: list[int] = []

    for centroid in centroids:
        distances = np.linalg.norm(X - centroid, axis=1)
        candidate_indices = np.argsort(distances)

        chosen = None
        for idx in candidate_indices:
            idx = int(idx)
            if idx in selected:
                continue
            if min_frame_separation > 0 and any(abs(idx - j) < min_frame_separation for j in selected):
                continue
            chosen = idx
            break

        if chosen is None:
            # Fallback: keep uniqueness if possible, even if separation is violated
            for idx in candidate_indices:
                idx = int(idx)
                if idx not in selected:
                    chosen = idx
                    break

        if chosen is None:
            # Degenerate fallback (e.g., more centers than frames)
            chosen = int(candidate_indices[0])

        selected.append(chosen)

    return selected


def compute_distance_features(u: mda.Universe,
                              ligand_sel: str,
                              pocket_sel: str,
                              stride: int = 1,
                              ) -> np.ndarray:
    """
    Compute flattened pairwise distances between ligand and pocket atoms 
    for each frame in the Universe trajectory.

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe with trajectory loaded.
    ligand_sel : str
        MDAnalysis selection string for ligand atoms.
    pocket_sel : str
        MDAnalysis selection string for pocket atoms.
    stride : int
        Process every `stride`-th frame.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_frames, n_ligand * n_pocket).
        Distances are in nm.
    """
    ligand_atoms = u.select_atoms(ligand_sel)
    pocket_atoms = u.select_atoms(pocket_sel)

    if ligand_atoms.n_atoms == 0 or pocket_atoms.n_atoms == 0:
        raise ValueError(
            f"Empty atom selection: ligand={ligand_atoms.n_atoms}, pocket={pocket_atoms.n_atoms}"
        )

    all_dists = []
    for ts in u.trajectory[::stride]:
        dists = distance_array(ligand_atoms.positions, pocket_atoms.positions)
        all_dists.append(dists.flatten() / 10.0)  # Angstroms -> nm

    return np.array(all_dists)


def extract_milestones(u: mda.Universe,
                       X: np.ndarray,
                       n_milestones: int = 5,
                       min_dist: float = 10.0,
                       out_dir: str = None,
                       prefix: str = "milestone",
                       min_frame_separation: int = 0,
                       ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract milestone frames from trajectory data using RegularSpace clustering
    on a pre-computed feature matrix (e.g., pocket-ligand distances).

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe with trajectories loaded (used for writing PDBs).
    X : np.ndarray
        Feature matrix of shape (n_frames, n_features), typically scaled by caller.
    n_milestones : int
        Maximum number of milestones (max_centers for RegularSpace).
    min_dist : float
        Minimum distance between cluster centers (RegularSpace dmin).
    out_dir : str
        Output directory for milestone PDB files.
    prefix : str
        Prefix for milestone file names.
    min_frame_separation : int
        Optional minimum separation between selected frame indices.

    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each frame.
    sorted_cluster_centers : np.ndarray
        Cluster centers sorted by mean distance (ascending).
    milestone_files : list[str]
        Paths to the written milestone PDB files.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_frames, n_features)")

    # cluster_estimator = RegularSpace(dmin=min_dist, max_centers=n_milestones)
    cluster_estimator = KMeans(n_clusters=n_milestones)
    fitted_model = cluster_estimator.fit(X).fetch_model()

    cluster_centers = fitted_model.cluster_centers
    labels = fitted_model.transform(X)

    # sort by mean distance so milestone 1 = closest to pocket
    mean_dists = cluster_centers.mean(axis=1)
    sorted_indices = np.argsort(mean_dists)
    sorted_cluster_centers = cluster_centers[sorted_indices]

    closest_frames = match_cluster_centroids_unique(
        X,
        sorted_cluster_centers,
        min_frame_separation=min_frame_separation,
    )

    os.makedirs(out_dir, exist_ok=True)
    milestone_files = []
    u.trajectory[0]  # reset
    for i, frame_index in enumerate(closest_frames):
        frame_index = int(frame_index)
        u.trajectory[frame_index]
        fname = os.path.join(out_dir, f"{prefix}_{i+1}_frame_{frame_index}.pdb")
        with mda.Writer(fname, reindex=True) as W:
            W.write(u.atoms)
        milestone_files.append(fname)
        logging.info(f"Wrote milestone {i+1} at frame {frame_index}: {fname}")

    return labels, sorted_cluster_centers, milestone_files


def cluster_sMD_trajectories(u: mda.Universe, 
                             X:np.ndarray, 
                             n_clusters:int = 5, 
                             min_dist:float = 2.0,
                             out_dir:str = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated: use extract_milestones() instead."""
    import warnings
    warnings.warn(
        "cluster_sMD_trajectories is deprecated, use extract_milestones() instead.",
        DeprecationWarning, stacklevel=2,
    )

    # cluster_estimator = KMeans(n_clusters=5)
    cluster_estimator = RegularSpace(dmin=min_dist, max_centers=n_clusters)
    fitted_model = cluster_estimator.fit(X).fetch_model()
    cluster_centers = fitted_model.cluster_centers
    labels = fitted_model.transform(X)

    # sort the array by the second column (COM distance) so milestone 0 is the closest
    sorted_indices = np.argsort(cluster_centers[:, 1])
    sorted_cluster_centers = cluster_centers[sorted_indices]
    closest_frames = match_cluster_centroids(X, sorted_cluster_centers) 

    # Write each representative frame to a PDB
    u.trajectory[0]  # reset
    for i, frame_index in enumerate(closest_frames):
        u.trajectory[frame_index]
        with mda.Writer(os.path.join(f"{out_dir}", 
                                    f"milestone_{i+1}_frame_{frame_index}.pdb",
                                    ), reindex=True) as W:
            W.write(u.atoms)

    return labels, sorted_cluster_centers

def find_closest_points(
    out_dir, x_min, x_max, x_grid_points, x_name, 
    y_min, y_max, y_grid_points, y_name, ref_point, num_neighbors=5
):
    """
    This function finds the closest grid points in the heatmap data to a given reference point.
    
    Parameters:
    - out_dir: Directory containing the heatmap .npy files.
    - x_min, x_max: Min and max values of the x axis in the plot.
    - x_grid_points: Number of grid points along the x axis.
    - x_name: Label name for the x axis.
    - y_min, y_max: Min and max values of the y axis in the plot.
    - y_grid_points: Number of grid points along the y axis.
    - y_name: Label name for the y axis.
    - ref_point: The reference point on the plot (x_ref, y_ref) whose neighbors you want to find.
    - num_neighbors: Number of closest points to retrieve.
    
    Returns:
    A DataFrame of the closest points and their coordinates (x, y) in plot units.
    """
    
    # Extract system name from directory
    sys_name = out_dir.split("/")[0]

    # Generate the x and y axis values (matching the plot)
    x_values = np.linspace(x_min, x_max, x_grid_points)
    y_values = np.linspace(y_min, y_max, y_grid_points)

    # Load the first FE data file (assuming there's one file per walker)
    file_fe = glob(f"{out_dir}/FE_*.npy")[0]
    np_data = np.load(file_fe)
    np_data = np_data * 0.239006  # Convert from KJ to Kcal

    # Reshape np_data into a list of points with (x, y) coordinates
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Reference point provided in plot axis units
    ref_point = np.array([ref_point])  # Ensure it's in the correct shape for cdist

    # Use scipy to calculate the Euclidean distance from each grid point to the reference point
    distances = cdist(grid_points, ref_point, metric='euclidean').ravel()

    # Find the indices of the closest points
    closest_indices = np.argsort(distances)[:num_neighbors]

    # Retrieve the closest points in grid coordinates and their corresponding values in np_data
    closest_points = grid_points[closest_indices]
    closest_values = np_data.ravel()[closest_indices]

    # Prepare a DataFrame with the results
    closest_df = pd.DataFrame({
        'x_value': closest_points[:, 0],
        'y_value': closest_points[:, 1],
        'FE_value': closest_values
    })
    
    return closest_df


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

    # COM pseudoatom written as a one-line PDB regardless of format
    com_pdb_content = (
        "REMARK Pocket center of mass\n"
        f"ATOM      1  COM COM SYS A   1    "
        f"{pocket_com[0]:8.3f}{pocket_com[1]:8.3f}{pocket_com[2]:8.3f}"
        "  1.00  0.00           C\n"
        "END\n"
    )

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
            com_pdb = str(tmpdir / "pocket_com.pdb")

            with mda.Writer(protein_pdb, protein_atoms.n_atoms) as w:
                w.write(protein_atoms)
            with mda.Writer(ligand_pdb, ligand_atoms.n_atoms) as w:
                w.write(ligand_atoms)
            with mda.Writer(pocket_pdb, pocket_atoms.n_atoms) as w:
                w.write(pocket_atoms)
            Path(com_pdb).write_text(com_pdb_content)

            pse_path = str(out_dir / "pocket_view.pse")
            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd
                cmd.bg_color("white")
                cmd.load(protein_pdb, "protein")
                cmd.load(ligand_pdb, "ligand")
                cmd.load(pocket_pdb, "pocket")
                cmd.load(com_pdb, "pocket_com")
                cmd.hide("everything")
                cmd.show("cartoon", "protein")
                cmd.show("sticks", "ligand")
                cmd.show("sticks", "pocket")
                cmd.show("spheres", "pocket_com")
                cmd.color(protein_color, "protein")
                cmd.color(ligand_color, "ligand")
                cmd.color(pocket_color, "pocket")
                cmd.color(com_color, "pocket_com")
                cmd.set("sphere_scale", com_sphere_radius)
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
    com_pdb = str(out_dir / "pocket_com.pdb")

    with mda.Writer(protein_pdb, protein_atoms.n_atoms) as w:
        w.write(protein_atoms)
    with mda.Writer(ligand_pdb, ligand_atoms.n_atoms) as w:
        w.write(ligand_atoms)
    with mda.Writer(pocket_pdb, pocket_atoms.n_atoms) as w:
        w.write(pocket_atoms)
    Path(com_pdb).write_text(com_pdb_content)

    lines = [
        "reinitialize",
        f"load {os.path.basename(protein_pdb)}, protein",
        f"load {os.path.basename(ligand_pdb)}, ligand",
        f"load {os.path.basename(pocket_pdb)}, pocket",
        f"load {os.path.basename(com_pdb)}, pocket_com",
        "hide everything",
        "show cartoon, protein",
        "show sticks, ligand",
        "show sticks, pocket",
        "show spheres, pocket_com",
        f"color {protein_color}, protein",
        f"color {ligand_color}, ligand",
        f"color {pocket_color}, pocket",
        f"color {com_color}, pocket_com",
        f"set sphere_scale, {com_sphere_radius}",
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

    # z_cc in visual coords: where the cone transitions to the cylinder
    z_cc_visual = z_cc - z_offset

    def make_ring(z_level: float, radius: float) -> list:
        rc = center + z_level * axis
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        return [rc + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles]

    # Cone section (0 → z_cc_visual) + three cylinder levels beyond
    z_cone = np.linspace(0.0, max(z_cc_visual, 0.0), n_z)
    z_cyl = [z_cc_visual + extension * t for t in (0.33, 0.67, 1.0)]
    z_all = list(z_cone) + z_cyl

    rings = []
    for z in z_all:
        r = R_cylinder + max(0.0, (z_cc_visual - z)) * np.tan(alpha_rad)
        rings.append(make_ring(z, r))

    # Latitudinal rings
    cgo = [LINEWIDTH, 1.5, COLOR, 0.35, 0.35, 0.35, BEGIN, LINES]
    for pts in rings:
        for i in range(n_pts):
            p1, p2 = pts[i], pts[(i + 1) % n_pts]
            cgo += [VERTEX, float(p1[0]), float(p1[1]), float(p1[2]),
                    VERTEX, float(p2[0]), float(p2[1]), float(p2[2])]
    cgo.append(END)

    # Meridional lines (every n_pts//8 angular index)
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
    ligand_resname: str = "UNK",
    output_format: str = "pse",
) -> str:
    """Create a standalone PyMOL PSE visualising the funnel potential geometry.

    Shows the protein (cartoon), the funnel boundary (CGO wireframe) oriented along the
    PCA-derived unbinding axis, an arrow for the axis, one sphere per milestone at the
    ligand COM, and the sMD COM trajectory as dots coloured blue→red by progress.

    Args:
        funnel_params: Dict returned by
            :func:`~autopath.customForces.generate_funnel_parameters_from_trajectory`.
        reference_pdb: Path to the solvated system PDB (for protein context).
        milestone_files: Ordered list of milestone PDB paths.
        outdir: Directory where ``funnel_visualization.pse`` is written.
        protein_selection: PyMOL/MDAnalysis selection string for the protein.
        ligand_resname: Residue name of the ligand (used to locate atoms in milestone PDBs).
        output_format: Only ``"pse"`` is supported; kept for API consistency.

    Returns:
        Absolute path of the written PSE file.
    """
    try:
        import pymol2
    except ImportError:
        raise ImportError("pymol2 is required for funnel visualization.")

    from pymol.cgo import COLOR, SPHERE, CYLINDER, CONE

    os.makedirs(outdir, exist_ok=True)
    outdir = Path(outdir)

    # --- Unpack funnel parameters ---
    z_cc = funnel_params["z_cc"].value_in_unit(openmmunit.angstrom)
    R_cylinder = funnel_params["R_cylinder"].value_in_unit(openmmunit.angstrom)
    alpha_deg = funnel_params["alpha"].value_in_unit(openmmunit.degrees)
    unbinding_axis = np.array(funnel_params["unbinding_axis"], dtype=float)
    unbinding_axis /= np.linalg.norm(unbinding_axis)
    com_traj = funnel_params["com_trajectory"].astype(float)   # relative Å (guest - host)
    extension = 5.0  # extra cylinder length for context

    # --- Re-anchor to the reference PDB coordinate frame ---
    # com_traj is a relative quantity (guest_com − host_com) computed during
    # generate_funnel_parameters_from_trajectory.  To place it in the reference
    # PDB frame we must add the HOST COM from that PDB — the same atoms used to
    # define the funnel force, stored in funnel_params["host_index"].
    # Using the full-protein COM as the anchor (old behaviour) is WRONG when the
    # host selection is a subset of the protein (e.g. pocket CAs): the two COMs
    # differ by several Å, causing the sMD dots and the funnel cone to appear
    # offset from each other in the visualisation.
    u_ref_anchor = mda.Universe(str(Path(reference_pdb).resolve()))
    prot_anchor = u_ref_anchor.select_atoms(protein_selection)
    prot_com_pdb = prot_anchor.center_of_mass()  # kept for milestone fallback only

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

    # Funnel center: initial ligand COM in the reference frame.
    # com_traj[0] is the bound-state ligand position relative to the host COM,
    # so host_com_ref + com_traj[0] places it correctly in reference-PDB space.
    # This guarantees that the first sMD dot and the funnel tip are co-located.
    funnel_center = host_com_ref + com_traj[0]
    z_offset = float(np.dot(com_traj[0], unbinding_axis))
    z_offset = max(z_offset, 0.0)

    # --- CGO objects ---
    funnel_cgo = _funnel_cgo(
        funnel_center, unbinding_axis, z_cc, R_cylinder, alpha_deg,
        extension=extension, z_offset=z_offset,
    )

    # Unbinding axis arrow: from funnel_center (pocket) outward to cylinder end
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

    import colorsys
    n_ms = len(milestone_files)
    # ms_data: list of (obj_name, lig_atomgroup, positions_in_ref_frame, (r,g,b))
    # Positions are pre-translated so that writing them to a PDB and loading into
    # PyMOL places the ligand correctly in the reference-PDB coordinate frame.
    ms_data = []
    for i, ms_file in enumerate(milestone_files):
        try:
            u_ms = mda.Universe(ms_file)
            lig = u_ms.select_atoms(f"resname {ligand_resname} and not name H*")
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
            # Translate ligand into the reference PDB coordinate frame.
            lig_pos_ref = lig.positions - host_ms.center_of_mass() + host_com_ref
            # Rainbow: blue (bound, i=0) → red (unbound, i=n_ms-1)
            t = i / max(n_ms - 1, 1)
            rgb = colorsys.hsv_to_rgb(2 / 3 * (1.0 - t), 1.0, 1.0)
            ms_data.append((f"milestone_{i}", lig, lig_pos_ref, rgb))
        except Exception as exc:
            logger.warning(f"Could not load milestone {ms_file}: {exc}")
            continue

    # sMD COM trajectory — every 10th frame, blue (start) → red (end).
    # com_traj is relative (guest − host), re-anchored to the host COM in the
    # reference PDB frame so that abs_positions[0] == funnel_center exactly.
    abs_positions = com_traj + host_com_ref
    sampled = abs_positions[::10]
    n_samp = max(1, len(sampled) - 1)
    traj_cgo = []
    for k, pos in enumerate(sampled):
        t = k / n_samp
        traj_cgo += [COLOR, float(t), 0.0, float(1.0 - t),
                     SPHERE, float(pos[0]), float(pos[1]), float(pos[2]), 0.25]

    pse_path = str(outdir / "funnel_visualization.pse")

    # Extract protein atoms to a temp PDB so PyMOL's show/color/set can reference
    # it by the object name "protein" rather than by a selection keyword (which fails
    # in headless pymol2 mode).  Mirrors the pattern in write_pocket_pymol().
    import tempfile as _tempfile
    u_ref = mda.Universe(str(Path(reference_pdb).resolve()))
    prot_atoms = u_ref.select_atoms(protein_selection)

    with _tempfile.TemporaryDirectory() as tmpdir:
        prot_pdb = os.path.join(tmpdir, "protein.pdb")
        with mda.Writer(prot_pdb, prot_atoms.n_atoms) as w:
            w.write(prot_atoms)

        # Write milestone ligand PDBs (translated to reference frame) into tempdir
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

            # Protein — load from temp PDB so object name == "protein"
            c.load(prot_pdb, "protein")
            c.hide("everything", "protein")
            c.show("cartoon", "protein")
            c.color("grey90", "protein")
            c.set("cartoon_transparency", 0.25, "protein")

            # Funnel boundary wireframe
            c.load_cgo(funnel_cgo, "funnel_boundary")

            # Unbinding axis arrow
            c.load_cgo(axis_cgo, "unbinding_axis")

            # Milestone ligands — sticks, rainbow blue (bound) → red (unbound)
            for obj_name, ms_pdb, (r, g, b) in ms_pdb_data:
                c.load(ms_pdb, obj_name)
                c.show("sticks", obj_name)
                c.hide("lines", obj_name)
                color_name = f"ms_col_{obj_name}"
                c.set_color(color_name, [float(r), float(g), float(b)])
                c.color(color_name, obj_name)
                c.set("stick_radius", 0.15, obj_name)

            # sMD COM trajectory dots
            if traj_cgo:
                c.load_cgo(traj_cgo, "smd_com_traj")

            c.zoom("all", 5)
            c.save(pse_path)

    logger.info(f"Funnel visualization saved: {pse_path}")
    return pse_path
