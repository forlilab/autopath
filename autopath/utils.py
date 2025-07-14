import os
import shutil
import logging
import numpy as np
import pandas as pd
from sys import stdout, exit
from glob import glob

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

from scipy.spatial import KDTree

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("fivethirtyeight")

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Scaffolds import MurckoScaffold

from deeptime.clustering import RegularSpace

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
    stride: int = None,
    super_mask: str = "@CA,C,N",
    strip_mask: str = None,  #':HOH,NA,CL,K,POP'
    out_fname: str = None,
) -> None:

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
        logging.info(f"Using {platform_name} platform.")

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
            f"{out_dir}/trajectory_{suffix}.dcd",
            reportInterval=logperiod,
            enforcePeriodicBox=False,  # WARNING this compromises autoimaging afterwards in some cases
        )
    )

    if verbose > 0:

        simulation.reporters.append(
            StateDataReporter(
                f"{out_dir}/statistics_{suffix}.csv",
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

def get_pocket_atoms(u:mda.Universe, 
                     pocket_selection:str, 
                     ligand_selection:str,
                     cutoff: float = 5.0
                     ) -> mda.AtomGroup:
    """Get the pocket atoms based on a user provided selection 
    or the ligand residue name and some default heuristics."""

    u.trajectory[-1]  # set pointer to last frame if its a trajectory

    if pocket_selection is None and ligand_selection is None:
        logging.error("No pocket selection or ligand residue name provided.")
        return []

    # If a custom pocket selection is provided, use it directly
    if pocket_selection is not None:
        pocket_atoms = u.select_atoms(pocket_selection)
        pocket_atoms_indices = [atom.index for atom in pocket_atoms]
    # If no custom selection, use the ligand residue name to define the pocket
    elif ligand_selection is not None:
        backbone_names = ["N", "CA", "C", "O"]
        ligand = u.select_atoms(ligand_selection)
        protein_residues = u.select_atoms(f"protein and around {cutoff} group ligand", ligand=ligand).residues
        pocket_atoms_indices = [atom.index for res in protein_residues 
                                for atom in res.atoms
                                if atom.name in backbone_names]
        
    if len(pocket_atoms_indices) == 0:
        logging.error(f"No atoms found for the provided pocket selection")
        return []
    else:
        # convert to MDAnalysis AtomGroup
        pocket_atoms = u.select_atoms(f"index {' '.join(map(str, pocket_atoms_indices))}")
        return pocket_atoms

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
    mode: str = "com",
    frames: int = 100,
    n_atoms: int = 5,
    reduce_before: bool = True,
    expand_rings: bool = True,
    out_dir: str = None,
    verbose: bool = True,
):
    """
    Select anchor atoms in the ligand for pulling and optionally visualize them.

    If mode="murcko", returns Murcko scaffold atoms.

    Parameters
    ----------
    reduce_before : bool
        If True, reduce ligand to Murcko scaffold before anchor selection.
    """

    img_name = f"{out_dir}/pulling_{lig_resname}_{mode}.png"

    ligand_full = u.select_atoms(f"resname {lig_resname} and not name H*")
    if ligand_full.n_atoms == 0:
        raise ValueError(f"No atoms found for ligand {lig_resname}.")

    # Reduce first if requested
    if reduce_before:
        ligand, highlight_rdk_indices, mol = reduce_to_murcko_scaffold(u, lig_resname, img_name)
    else:
        ligand = ligand_full
        mol = ligand_full.convert_to("RDKIT")
        highlight_rdk_indices = []

    u.trajectory[-1]  # Ensure we are at the last frame
    pocket = u.select_atoms(pocket_sel)
    anchor = []

    if mode == 'ha':
        anchor = ligand_full.indices
        mol = ligand_full.convert_to("RDKIT")

    if mode == "murcko":
        ligand, anchor_indices, mol = reduce_to_murcko_scaffold(u, lig_resname, img_name)
        anchor = [ligand.atoms[i].index for i in range(len(ligand))]

    elif mode == "com":
        com = ligand.center_of_mass()
        dists = np.linalg.norm(ligand.positions - com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    elif mode == "closest":
        pocket_com = pocket.center_of_mass()
        dists = np.linalg.norm(ligand.positions - pocket_com, axis=1)
        anchor = ligand.atoms[np.argsort(dists)[:n_atoms]].indices

    elif mode == "contacts":
        contact_counts = np.zeros(len(ligand))
        for ts in u.trajectory[:frames]:
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
        # anchor = ligand.atoms[np.argsort(projections)[-n_atoms:]].indices

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

    if expand_rings:
        rdk_anchor_indices = []
        idx_map = {a.index: i for i, a in enumerate(ligand_full.atoms)}
        for idx in anchor:
            if idx in idx_map:
                rdk_anchor_indices.append(idx_map[idx])
        ring_info = mol.GetRingInfo()
        anchor_rings = [set(ring) for ring in ring_info.AtomRings() if any(i in ring for i in rdk_anchor_indices)]
        expanded_rdk_indices = set()
        for ring in anchor_rings:
            expanded_rdk_indices.update(ring)
        expanded_mda_indices = [ligand_full.atoms[i].index for i in expanded_rdk_indices]
        anchor = list(set(anchor).union(expanded_mda_indices))
        print(f"[get_ligand_anchor_atoms] Expanded to include rings: {expanded_mda_indices}")

    # Draw 2D image with highlights
    try:
        idx_map = {a.index: i for i, a in enumerate(ligand_full.atoms)}
        highlight_rdk_indices = [idx_map[i] for i in anchor if i in idx_map]
        Chem.rdDepictor.Compute2DCoords(mol)
        mol = Chem.RemoveHs(mol)
        img = Chem.Draw.MolToImage(mol, size=(300, 300), highlightAtoms=highlight_rdk_indices)
        img.save(img_name)
    except Exception as e:
        logging.warning(f"Could not generate 2D image with highlights: {e}")

    if verbose:
        print(f"[get_ligand_anchor_atoms] Anchor atoms selected ({mode}): {anchor}")

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

def compute_rmsd(u, u_ref,
                    alig_select:str='backbone', 
                    groupselections={}, 
                    save_aligned=False,
                    aligned_filename='aligned_trajectory.dcd',
                    do_plot=True,
                    out_dir=None
                    ) -> pd.DataFrame:
    r = RMSD(u, 
             u_ref,
             select=alig_select,
             groupselections=list(groupselections.values()),
             ref_frame=0).run()

    rmsd_results = r.results.rmsd  # Do not skip any columns
    columns = ['frame','time (ps)', f'RMSD_selected_alignment'] + [f'RMSD_{group}' for group in groupselections.keys()]
    rmsd_df = pd.DataFrame(rmsd_results, columns=columns)

    if save_aligned:
        with mda.Writer(aligned_filename, n_atoms=u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u.atoms)

    if do_plot:
        
        plt.figure(figsize=(10, 5))
        for col in columns[3:]:
            sns.lineplot(x='frame', y=col, data=rmsd_df)
            plt.xlabel('Frame');            plt.ylabel(f'RMSD (A)')
            plt.title(f'{col} RMSD')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/{col}.png')
            plt.close()

    return rmsd_df

def plot_atomic_rmsf(u, lig_resname:str='UNK', outname:str='rmsf.png', log_rmsf:bool=False):
    """
    Draws a RMSF (Root Mean Square Fluctuation) plot for a specified ligand and saves it as an image file.
    Parameters:
    -----------
    u : MDAnalysis.Universe
        The MDAnalysis universe object containing the molecular dynamics trajectory and topology.
    lig_resname : str, optional
        The residue name of the ligand to analyze (default is 'UNK').
    outname : str, optional
        The name of the output image file where the RMSF plot will be saved (default is 'rmsf.png').
    log_rmsf : bool, optional
        If True, logs the RMSF values to a CSV file with the same name as the output image (default is False).
    Returns:
    --------
    None
        This function does not return any value. It saves the RMSF plot and optionally logs the RMSF values.
    """

    lig_select = u.select_atoms(f'resname {lig_resname}')
    r = RMSF(atomgroup=lig_select).run()
    probe_mol = lig_select.convert_to('RDKIT')
    probe_mol.Compute2DCoords()
    probe_mol = Chem.RemoveHs(probe_mol)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(probe_mol, r.rmsf, step=0.01, alpha=0.3, contourLines=5) 
    fig.savefig(outname, bbox_inches='tight')
    
    # Optionally, log the RMSF values for further analysis
    if log_rmsf:
        log_fname = os.path.splitext(outname)[0]
        with open(f'{log_fname}.csv', 'w') as f:
            for res_id, rmsf_value in enumerate(r.rmsf):
                f.write(f'{res_id},{rmsf_value:.3f}\n')
    return

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

def cluster_sMD_trajectories(u: mda.Universe, X:np.ndarray, 
                             n_clusters:int, 
                             min_dist:float,
                             out_dir:str
                             ) -> Tuple[np.ndarray, np.ndarray]:

    # cluster_estimator = KMeans(n_clusters=5)
    cluster_estimator = RegularSpace(dmin=min_dist, max_centers=n_clusters)
    fitted_model = cluster_estimator.fit(X).fetch_model()
    cluster_centers = fitted_model.cluster_centers
    labels = fitted_model.transform(X)

    # sort the array by the second column (COM distance) so milestone 0 is the closest
    sorted_indices = np.argsort(cluster_centers[:, 1])
    sorted_cluster_centers = cluster_centers[sorted_indices]
    closest_frames = match_cluster_centroids(X, sorted_cluster_centers) 

    # for i, idx in enumerate(closest_frames):
    #     dist = np.linalg.norm(X[idx] - cluster_centers[i])
    #     logging.info(f"Cluster {i}: Closest frame is {idx} (distance = {dist:.3f})")

    # Write each representative frame to a PDB
    u.trajectory[0]  # reset
    for i, frame_index in enumerate(closest_frames):
        u.trajectory[frame_index]
        with mda.Writer(os.path.join(f"{out_dir}", f"milestone_{i+1}_frame_{frame_index}.pdb"), u.atoms.n_atoms) as W:
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
