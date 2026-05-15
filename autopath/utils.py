import os
import shutil
import logging
import tempfile
import requests
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
from rdkit.Chem import AllChem

from molscrub import Scrub 

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


def fetch_smiles(ligand_name: str) -> str:
    """Gets SMILES string for a given ligand in the CCD

    Args:
        ligand_name (str): name of ligand
    """
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        smiles = data['rcsb_chem_comp_descriptor']['smiles']
        print(f"SMILES for {ligand_name}: {smiles}")
    else:
        print(f"failed to retrieve data for {ligand_name}")
        response.raise_for_status()

    return smiles

def get_scrubbed_smile(ligand_name: str) -> str:
    """Get SMILES string based on tautomer generated by molscrub

    Args:
        ligand_name (str): name of ligand
    """
    smiles = fetch_smiles(ligand_name)
    scrub = Scrub(ph_low=7.4, ph_high=7.4)
    scrubbed_mol = scrub(Chem.MolFromSmiles(smiles))[0] # consider only the first protonation state
    scrubbed_smiles = Chem.MolToSmiles(scrubbed_mol)

    return scrubbed_smiles 
 

def fetch_pdb(pdb_id: str, pdb_chain_ids: str, save_dir: str) -> str:
    """Retreives pdb file for a given pdb id. 
 
    Args:
        pdb_id (str): e.g 1xyz_A or 1xyz - if _X specified in pdb_id, only that chain will be extracted
        pdb_chain_ids (str): e.g. A-B, assumes chains are dash separated
        save_dir (str): e.g ./pdb_structures_folder
    """ 
    os.makedirs(save_dir, exist_ok=True)
    pdb_save_path = f"{save_dir}/{pdb_id}.pdb" 

    cmd.reinitialize()
    if len(pdb_id.split('_')) > 1:
        pdb_id_wo_chain = pdb_id.split('_')[0]
        chain_id = pdb_id.split('_')[1]
        cmd.fetch(pdb_id_wo_chain, async_=0)
        cmd.save(pdb_save_path, f"chain {chain_id} and {pdb_id_wo_chain}")
    else:
        cmd.fetch(pdb_id, async_=0)
        if pdb_chain_ids is not None:
            pdb_chain_ids = pdb_chain_ids.replace('-','+')
            cmd.save(pdb_save_path, f"chain {pdb_chain_ids} and {pdb_id}")
        else:
            cmd.save(pdb_save_path, pdb_id)
    cmd.delete('all')

    if len(pdb_id.split('_')) > 1:
        fetch_pdb_id = pdb_id_wo_chain.lower()
    else:
        fetch_pdb_id = pdb_id.lower()

    fetch_path = f"./{fetch_pdb_id}.cif" 
    if os.path.exists(fetch_path):
        os.remove(fetch_path)    

    return pdb_save_path


def get_ligand_name(pdb_path: str) -> str | None:
    """Returns the name of a ligand in a complex. Raises an error if multiple ligands are present. 
    
    Args:
        pdb_path (str): path to pdb file
    """
    cmd.reinitialize()
    cmd.load(pdb_path, "receptor_w_ligand")
    # cmd.select("org", "(not (polymer or solvent))")
    cmd.select("org", "organic")
    model = cmd.get_model("org")
    ligand_names = list(set(atom.resn for atom in model.atom))
    print(f"the following ligands are present in {pdb_path}:")
    print(ligand_names)

    if len(ligand_names) > 1:
        logging.error(f"Tried to automatically determine name of ligand in complex, but more than 1 ligand is present!")
        exit(1)
    elif len(ligand_names) == 1:
        return ligand_names[0]
    else:
        return None 

def save_receptor_and_ligand_from_pdb(
    pdb_path: str, 
    pdb_id: str, 
    save_dir: str, 
    inorg_cofactor_name: str = None,
    inorg_cofactor_resid: str = None,
    org_colig_name: str = None, 
    org_colig_resid: str = None,
    ignore_colig_simulation: bool = False, 
    ligand_name: str = None, 
    ligand_resid: str = None,
    use_ccd_smiles_for_lig: bool = False,
    use_ccd_smiles_for_colig: bool = False, 
    include_waters: bool = True,
    water_resids: List[str] = None,
    water_chainids: List[str] = None,
)  -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Given a raw pdb file, extracts and saves receptor, co-ligand, and ligand as separate files.  
    
    Args:
        pdb_path (str): path to pdb file 
        pdb_id (str): pdb id 
        save_dir (str): directory to save files
        inorg_cofactor_name (str): name of inorg cofactor
        inorg_cofactor_resid (str): residue number of inorg cofactor - relevant if there are multiple copies
        org_colig_name (str): name of org co-ligand
        org_colig_resid (str): residue number of org co-ligand - relevant if there are multiple copies
        ignore_colig_simulation (bool): treat organic co-ligand as part of receptor as opposed to a separate ligand 
        ligand_name (str): name of ligand to extract
        ligand_resid (str): residue number of ligand to extract - relevant if there are multiple copies 
        include_waters (bool): retain crystallographic waters in receptor 
        water_resids (list): retain crystallographic waters corresponding to these residue ids in receptor 
        water_chainids (list): retain crystallographic waters corresponding to these chain ids in receptor 
    """
    cmd.reinitialize()
    cmd.load(pdb_path, "complex")

    keep_criteria = ["polymer"]
    
    if include_waters:
        if water_resids is None: 
            keep_criteria.append("resn HOH")
        elif water_resids is not None and water_chainids is not None:
            water_resids = "+".join(water_resids)
            water_chainids = "+".join(water_chainids)
            keep_criteria.append(f"resn HOH and resid {water_resids} and chain {water_chainids}")
        elif water_resids is not None:
            water_resids = "+".join(water_resids)
            keep_criteria.append(f"resn HOH and resid {water_resids}")
        elif water_chainids is not None:
            water_chainids = "+".join(water_chainids)
            keep_criteria.append(f"resn HOH and chain {water_chainids}")
        
    if inorg_cofactor_name:
        sel = f"resn {inorg_cofactor_name}"
        if inorg_cofactor_resid:
            if len(inorg_cofactor_resid.split('_')) == 2:
                iocl_resid, iocl_chainid = inorg_cofactor_resid.split('_')
                sel += f" and resid {iocl_resid} and chain {iocl_chainid}"
            else:
                sel += f" and resid {inorg_cofactor_resid}"
        keep_criteria.append(sel)

    if org_colig_name:
        sel = f"resn {org_colig_name}"
        if org_colig_resid:
            if len(org_colig_resid.split('_')) == 2:
                ocl_resid, ocl_chainid = org_colig_resid.split('_')
                sel += f" and resid {ocl_resid} and chain {ocl_chainid}"
            else:
                sel += f" and resid {org_colig_resid}"
        cmd.create("org_colig_obj", sel) 
        org_colig_path = f"{save_dir}/{pdb_id}_org_colig.sdf"
        cmd.save(org_colig_path, "org_colig_obj")
        if use_ccd_smiles_for_colig:
            org_colig_smiles = fetch_smiles(org_colig_name)
        else:
            org_colig_smiles = get_scrubbed_smile(org_colig_name)
        Path(f"{save_dir}/{pdb_id}_org_colig_smiles.txt").write_text(org_colig_smiles)
    else:
        org_colig_path = None
        org_colig_smiles = None

    selection_string = " or ".join(keep_criteria)
    cmd.create("receptor_obj", f"{selection_string}")

    rec_path = f"{save_dir}/{pdb_id}_receptor.pdb" 
    cmd.save(rec_path, "receptor_obj")
    print(f"saved receptor wo/ligand as {rec_path}")

    if ligand_name is None:
        return rec_path, None, None, None, None  
            
    cmd.reinitialize()
    cmd.load(pdb_path) 

    # first try to extract ligand corresponding to a specific conformer (assuming there are multiple conformers) 
    if ligand_resid is None:
        cmd.select("ligand", f"resn {ligand_name} and alt A") 
    else:
        if len(ligand_resid.split('_')) == 2:
            l_resid, l_chainid = ligand_resid.split('_')
            cmd.select("ligand", f"resn {ligand_name} and resid {l_resid} and chain {l_chainid} and alt A") 
        else:
            cmd.select("ligand", f"resn {ligand_name} and resid {ligand_resid} and alt A") 
    objects = cmd.get_object_list("ligand")

    if len(objects) > 0: #multiple conformers exist for ligand 
        print(f"note: multiple conformers exist for ligand {ligand_name}")
        cmd.extract("ligand_obj", "ligand")
        lig_path = f"{save_dir}/{pdb_id}_ligand.sdf" 
        cmd.save(lig_path, "ligand_obj")
        print(f"saved ligand (conformer A) as {lig_path}")
    else:
        if ligand_resid is None:
            cmd.select("ligand", f"resn {ligand_name}") 
        else:
            print(ligand_resid)
            if len(ligand_resid.split('_')) == 2:
                l_resid, l_chainid = ligand_resid.split('_')
                print('here')
                cmd.select("ligand", f"resn {ligand_name} and resid {l_resid} and chain {l_chainid}") 
            else:
                cmd.select("ligand", f"resn {ligand_name} and resid {ligand_resid}") 
        objects = cmd.get_object_list("ligand")
        if len(objects) > 0:
            cmd.extract("ligand_obj", "ligand")
            lig_path = f"{save_dir}/{pdb_id}_ligand.sdf" 
            cmd.save(lig_path, "ligand_obj")
            print(f"saved ligand as {lig_path}")
        else:
            logging.error(f"No ligand with name {ligand_name}")
            exit(1)

    if use_ccd_smiles_for_lig:
        lig_smiles = fetch_smiles(ligand_name)
    else:
        lig_smiles = get_scrubbed_smile(ligand_name)
    Path(f"{save_dir}/{pdb_id}_ligand_smiles.txt").write_text(lig_smiles)

    return rec_path, lig_path, lig_smiles, org_colig_path, org_colig_smiles 


def save_receptor_and_ligand_from_openmm(
    pdb_path: str, 
    save_dir: str, 
    inorg_cofactor_name: str = None,
    org_colig_name: str = None,
    remove_H_colig: bool = True, 
    ligand_name: str = None,
    water_resids: List[str] = None,
    water_chainids: List[str] = None,
    output_fname_rec: str = None,
    output_fname_lig: str = None,
)  -> None:
    """Given a OpenMM generated pdb file, extracts and saves receptor and ligand as separate files.  
    
    Args:
        pdb_path (str): path to pdb file 
        save_dir (str): directory to save files
        inorg_cofactor_name (str): name of inorganic cofactor
        org_colig_name (str): name of organic co-ligand
        remove_H_colig (bool): remove hydrogens from co-ligand. set to True by default to facilitate Meeko's automated parameterization of unknown residues. 
        ligand_name (str): name of ligand to extract
        water_resids (list): retain waters corresponding to these residue ids in receptor 
        water_chainids (list): retain crystallographic waters corresponding to these chain ids in receptor 
        output_fname_rec (str): output filename for receptor (assumes extension is present)
        output_fname_lig (str): output filename for ligand (assumes extension is present)
    """
    fname = os.path.splitext(os.path.basename(pdb_path))[0]
    if output_fname_rec is None:
        rec_path = f"{save_dir}/{fname}_receptor.pdb" 
    if output_fname_lig is None:
        lig_path = f"{save_dir}/{fname}_ligand.pdb" 
    if output_fname_rec is not None:
        rec_path = f"{save_dir}/{output_fname_rec}"
    if output_fname_lig is not None:
        lig_path = f"{save_dir}/{output_fname_lig}"

    cmd.reinitialize()
    cmd.load(pdb_path, "complex")

    keep_criteria = ["polymer"]

    if water_resids is not None and water_chainids is not None:
        water_resids = "+".join(water_resids)
        water_chainids = "+".join(water_chainids)
        keep_criteria.append(f"resn HOH and resid {water_resids} and chain {water_chainids}")
    elif water_resids is not None:
        water_resids = "+".join(water_resids)
        keep_criteria.append(f"resn HOH and resid {water_resids}")
    elif water_chainids is not None:
        water_chainids = "+".join(water_chainids)
        keep_criteria.append(f"resn HOH and chain {water_chainids}")
    
    if inorg_cofactor_name:
        sel = f"resn {inorg_cofactor_name}"
        keep_criteria.append(sel)

    if org_colig_name:
        if remove_H_colig:
            sel = f"resn {org_colig_name} and not elem H"
        else:
            sel = f"resn {org_colig_name}"
        keep_criteria.append(sel)

    selection_string = " or ".join(keep_criteria)
    cmd.create("receptor_obj", f"{selection_string}")
    
    cmd.save(rec_path, "receptor_obj")
    print(f"saved receptor wo/ligand as {rec_path}")

    if ligand_name:
        cmd.create("ligand_obj",  f"resn {ligand_name}") 
        cmd.save(lig_path, "ligand_obj")
        print(f"saved ligand as {lig_path}")

    return 


def _create_cap_universe(n_atoms, name, resname, positions, resids, segid):
    u_new = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_atoms,
        atom_resindex=np.arange(n_atoms),
        residue_segindex=np.arange(n_atoms),
        n_segments=n_atoms,
        trajectory=True,
    )
    u_new.add_TopologyAttr('name', name)
    u_new.add_TopologyAttr('resid', resids)
    u_new.add_TopologyAttr('resname', resname)
    u_new.atoms.positions = positions
    u_new.add_TopologyAttr('segid', n_atoms * [segid])
    u_new.add_TopologyAttr('chainID', n_atoms * [segid])
    return u_new


def _get_nme_pos(end_residue):
    if "OXT" in end_residue.names:
        index = np.where(end_residue.names == "OXT")[0][0]
        N_position = end_residue.positions[index]
        index_c = np.where(end_residue.names == "C")[0][0]
        carbon_position = end_residue.positions[index_c]
        vector = N_position - carbon_position
        vector /= np.sqrt(sum(vector**2))
        C_position = N_position + vector * 1.36
    else:
        index_o = np.where(end_residue.names == "O")[0][0]
        index_ca = np.where(end_residue.names == "CA")[0][0]
        mid_point = (end_residue.positions[index_o] + end_residue.positions[index_ca]) / 2
        index_c = np.where(end_residue.names == "C")[0][0]
        vector = end_residue.positions[index_c] - mid_point
        vector /= np.sqrt(sum(vector**2))
        N_position = end_residue.positions[index_c] + 1.36 * vector
        C_position = N_position + 1.36 * vector
    return N_position, C_position


def _get_ace_pos(end_residue):
    index_ca = np.where(end_residue.names == "CA")[0][0]
    index_n = np.where(end_residue.names == "N")[0][0]
    vector = end_residue.positions[index_n] - end_residue.positions[index_ca]
    vector /= np.sqrt(sum(vector**2))
    C1_position = end_residue.positions[index_n] + 1.36 * vector

    xa, ya, za = end_residue.positions[index_ca]
    xg, yg, zg = C1_position

    orientation = np.array([2 * np.random.rand() - 1, 2 * np.random.rand() - 1, 2 * np.random.rand() - 1])
    nx, ny, nz = orientation / np.sqrt(sum(orientation**2))

    x1 = xg - (xa - xg) / 2 + np.sqrt(3) * (ny * (za - zg) - nz * (ya - yg)) / 2
    y1 = yg - (ya - yg) / 2 + np.sqrt(3) * (nz * (xa - xg) - nx * (za - zg)) / 2
    z1 = zg - (za - zg) / 2 + np.sqrt(3) * (nx * (ya - yg) - ny * (xa - xg)) / 2

    x2 = xg - (xa - xg) / 2 - np.sqrt(3) * (ny * (za - zg) - nz * (ya - yg)) / 2
    y2 = yg - (ya - yg) / 2 - np.sqrt(3) * (nz * (xa - xg) - nx * (za - zg)) / 2
    z2 = zg - (za - zg) / 2 - np.sqrt(3) * (nx * (ya - yg) - ny * (xa - xg)) / 2

    C2_position = np.array([x1, y1, z1])
    O_position = np.array([x2, y2, z2])

    vector = C2_position - C1_position
    vector /= np.sqrt(sum(vector**2))
    C2_position = C1_position + 1.36 * vector

    vector = O_position - C1_position
    vector /= np.sqrt(sum(vector**2))
    O_position = C1_position + 1.36 * vector

    return C1_position, C2_position, O_position


def _apply_caps(pdb_in: str, pdb_out: str) -> None:
    """Add ACE (N-terminus) and NME (C-terminus) capping groups to all protein chains.

    Expects a hydrogen-free input PDB. Non-protein atoms (waters, ions) are
    preserved unchanged. Writes the capped structure to pdb_out.
    Implementation from https://github.com/ibrahim-mohd/Add-NME-ACE-residues-to-protein-terminal-residues
    
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(pdb_in)

    # Work on protein only for capping; preserve non-protein atoms separately
    protein_sel = u.select_atoms("protein")
    non_protein_sel = u.select_atoms("not protein")

    if len(protein_sel) == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u.atoms.write(pdb_out)
        return

    protein_u = mda.Merge(protein_sel)

    res_start = 0
    segment_universes = []

    for seg in protein_u.segments:
        segid = seg.segid

        # ACE at N-terminus
        first_res = seg.residues[0].atoms
        ace_positions = _get_ace_pos(first_res)
        ace_names = ["C", "CH3", "O"]
        resid = seg.residues[0].resid
        ace_universe = _create_cap_universe(
            n_atoms=len(ace_positions),
            name=ace_names,
            resname=len(ace_names) * ["ACE"],
            positions=ace_positions,
            resids=resid * np.ones(len(ace_names)),
            segid=segid,
        )

        # NME at C-terminus
        last_res = seg.residues[-1].atoms
        nme_positions = _get_nme_pos(last_res)
        nme_names = ["N", "C"]
        resid = seg.residues[-1].resid + 2
        nme_universe = _create_cap_universe(
            n_atoms=len(nme_names),
            name=nme_names,
            resname=len(nme_names) * ["NME"],
            positions=nme_positions,
            resids=resid * np.ones(len(nme_names)),
            segid=segid,
        )

        # Remove OXT if present before merging
        if "OXT" in last_res.names:
            oxt_index = last_res.select_atoms("name OXT")[0].index
            Chain = seg.atoms.select_atoms(f"not index {oxt_index}")
        else:
            Chain = seg.atoms

        u_all = mda.Merge(ace_universe.atoms, Chain, nme_universe.atoms)

        resids_ace = [res_start + 1] * 3
        resids_pro = np.arange(resids_ace[0] + 1, Chain.residues.n_residues + resids_ace[0] + 1)
        resids_nme = [resids_pro[-1] + 1, resids_pro[-1] + 1]
        u_all.atoms.residues.resids = np.concatenate([resids_ace, resids_pro, resids_nme])

        res_start = u_all.atoms.residues.resids[-1]
        segment_universes.append(u_all)

    parts = [seg.atoms for seg in segment_universes]
    if len(non_protein_sel) > 0:
        parts.append(non_protein_sel)
    all_uni = mda.Merge(*parts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_uni.atoms.write(pdb_out)


def fix_pdb(
    pdbfile: str,
    replace_nonstandard_residues: bool = True,
    keep_heterogens: bool = False,
    ignore_terminal_missing_residues: bool = False,
    pH: float = 7.4,
    discard_input_hydrogens: bool = False,
    cap_termini: bool = False,
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
        discard_input_hydrogens (bool): removes all hydrogens from input structure (then readd with PDBFixer)
        cap_termini (bool): if True, add ACE and NME neutral terminal capping groups to each chain before adding hydrogens.
    """

    fixer = PDBFixer(str(pdbfile))

    if discard_input_hydrogens:
        modeller = Modeller(fixer.topology, fixer.positions)
        h_atoms = [a for a in modeller.topology.atoms() if a.element.symbol == 'H']
        modeller.delete(h_atoms)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

    fixer.findMissingResidues()

    if ignore_terminal_missing_residues:
        # if missing terminal residues shall be ignored, remove them from the dictionary
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

    if replace_nonstandard_residues:
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=True)

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if cap_termini:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_in, \
             tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_out:
            tmp_in_path = tmp_in.name
            tmp_out_path = tmp_out.name
        app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp_in_path, keepIds=True)
        _apply_caps(tmp_in_path, tmp_out_path)
        fixer = PDBFixer(tmp_out_path)
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)

    fixer.addMissingHydrogens(pH)

    return fixer


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
    if platform_name == None or platform_name == "FASTEST":
        platform_name = get_fastest_platform().getName()

    try:
        platform = Platform.getPlatformByName(platform_name)
        logging.info(f"Using {platform_name} platform.")

        if platform_name in ["OPENCL"]:
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
        ss_codes = DSSP(u).run().results.dssp[0]  # shape (n_residues,)
        ss_residues = {
            res.resnum
            for res, code in zip(ca_residues, ss_codes)
            if code in ('H', 'E')
        }
        if verbose:
            logging.info(
                f"DSSP: {len(ss_residues)} residues in secondary structure "
                f"(H/E) out of {len(ca_residues)} total"
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
    ligand_com = lig_positions.mean(axis=0)

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

def assign_bondOrders(mol: Chem.Mol=None, template_smiles: str=None):
    """Assign bond orders from a template molecule to a target molecule."""

    try:
        template_mol = Chem.MolFromSmiles(template_smiles)
    except:
        logging.error(f"Could not generate template molecule from {template_smiles}")
        return mol
    
    # Assign bond orders from the template to the target molecule
    new_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, mol)
    new_mol = Chem.AddHs(new_mol, addCoords=True)
    
    if new_mol is None:
        logging.error(f"Could not assign bond orders from template {template_smiles}")
        return mol
            
    return new_mol


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
