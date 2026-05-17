import os
import logging
import tempfile
import warnings
import requests
from sys import exit
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import MDAnalysis as mda
import openmm.app as app
from openmm.app import Modeller
from pdbfixer.pdbfixer import PDBFixer

from pymol import cmd

from rdkit import Chem
from rdkit.Chem import AllChem

from molscrub import Scrub


class PDBPreprocessor:
    """Prepare a PDB for MD simulation: fix missing atoms/residues, add hydrogens,
    and optionally cap chain termini with ACE/NME."""

    def __init__(self, pdbfile: str):
        self.fixer = PDBFixer(str(pdbfile))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fix(
        self,
        replace_nonstandard_residues: bool = True,
        keep_heterogens: bool = False,
        ignore_terminal_missing_residues: bool = False,
        pH: float = 7.4,
        discard_input_hydrogens: bool = False,
        cap_termini: bool = False,
    ) -> PDBFixer:
        """Fix common PDB problems and optionally cap chain termini.

        Args:
            replace_nonstandard_residues: Replace non-standard residues with
                their standard equivalents.
            keep_heterogens: If False, remove all heterogens except water.
            ignore_terminal_missing_residues: If True, missing residues at the
                very beginning or end of a chain are not built.
            pH: pH used to determine protonation states when adding hydrogens.
            discard_input_hydrogens: Remove all hydrogens from the input before
                re-adding them at the requested pH.
            cap_termini: Add ACE (N-terminus) and NME (C-terminus) capping
                groups to each chain. Chains that already have ACE or NME are
                not double-capped.

        Returns:
            The PDBFixer object with the fixed (and optionally capped and
            protonated) structure.
        """
        if discard_input_hydrogens:
            modeller = Modeller(self.fixer.topology, self.fixer.positions)
            h_atoms = [a for a in modeller.topology.atoms() if a.element.symbol == 'H']
            modeller.delete(h_atoms)
            self.fixer.topology = modeller.topology
            self.fixer.positions = modeller.positions

        self.fixer.findMissingResidues()

        if ignore_terminal_missing_residues:
            chains = list(self.fixer.topology.chains())
            for key in list(self.fixer.missingResidues.keys()):
                chain = chains[key[0]]
                if key[1] == 0 or key[1] == len(list(chain.residues())):
                    del self.fixer.missingResidues[key]

        if replace_nonstandard_residues:
            self.fixer.findNonstandardResidues()
            self.fixer.replaceNonstandardResidues()

        if not keep_heterogens:
            self.fixer.removeHeterogens(keepWater=True)

        self.fixer.findMissingAtoms()
        self.fixer.addMissingAtoms()

        if cap_termini:
            # Strip only terminal-specific H atoms (N-terminal NH3+ and HXT)
            # so that user-defined protonation states on other residues are kept.
            modeller_tmp = Modeller(self.fixer.topology, self.fixer.positions)
            terminal_h = []
            for chain in modeller_tmp.topology.chains():
                residues = list(chain.residues())
                if not residues:
                    continue
                for atom in residues[0].atoms():
                    if atom.element.symbol == 'H' and atom.name in (
                        'H1', 'H2', 'H3', 'HN1', 'HN2', 'HN3'
                    ):
                        terminal_h.append(atom)
                for atom in residues[-1].atoms():
                    if atom.element.symbol == 'H' and atom.name == 'HXT':
                        terminal_h.append(atom)
            if terminal_h:
                modeller_tmp.delete(terminal_h)

            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_in, \
                 tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_out:
                tmp_in_path = tmp_in.name
                tmp_out_path = tmp_out.name
            app.PDBFile.writeFile(
                modeller_tmp.topology, modeller_tmp.positions, tmp_in_path, keepIds=True
            )
            self._apply_caps(tmp_in_path, tmp_out_path)
            self.fixer = PDBFixer(tmp_out_path)
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

        self.fixer.addMissingHydrogens(pH)
        return self.fixer

    # ------------------------------------------------------------------
    # Cap placement helpers (static — no instance state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ace_pos(end_residue):
        """Return (C, CH3, O) positions for an ACE cap at the N-terminus.

        Geometry is derived deterministically from the CA–N backbone axis
        and the CA–N–C1 plane, avoiding random perpendicular vectors.
        """
        ca_pos = end_residue.positions[np.where(end_residue.names == "CA")[0][0]]
        n_pos  = end_residue.positions[np.where(end_residue.names == "N" )[0][0]]

        v_can = (n_pos - ca_pos) / np.linalg.norm(n_pos - ca_pos)
        C1_pos = n_pos + 1.36 * v_can  # ACE carbonyl C

        # Deterministic in-plane perpendicular via CA–N–C1 normal
        v_c1n  = (n_pos  - C1_pos) / np.linalg.norm(n_pos  - C1_pos)
        v_c1ca = (ca_pos - C1_pos) / np.linalg.norm(ca_pos - C1_pos)
        n_plane = np.cross(v_c1n, v_c1ca)
        if np.linalg.norm(n_plane) < 1e-6:  # degenerate: fall back to any perpendicular
            ref = np.array([1., 0., 0.]) if abs(v_c1n[0]) < 0.9 else np.array([0., 1., 0.])
            n_plane = np.cross(v_c1n, ref)
        n_plane /= np.linalg.norm(n_plane)
        v_perp = np.cross(n_plane, v_c1n)
        v_perp /= np.linalg.norm(v_perp)

        # sp2: O and CH3 at ±120° from C1–N axis; correct bond lengths
        O_pos   = C1_pos + 1.23 * (-0.5 * v_c1n + 0.866 * v_perp)  # C=O  1.23 Å
        CH3_pos = C1_pos + 1.52 * (-0.5 * v_c1n - 0.866 * v_perp)  # C–CH3 1.52 Å

        return C1_pos, CH3_pos, O_pos  # order matches ace_names = ["C", "CH3", "O"]

    @staticmethod
    def _get_nme_pos(end_residue):
        """Return (N, C) positions for an NME cap at the C-terminus.

        When OXT is present it is used as a guide (original behaviour).
        When OXT is absent, sp2 geometry at the carbonyl C is used instead
        of the less accurate midpoint(O, CA) fallback.
        """
        if "OXT" in end_residue.names:
            oxt_pos = end_residue.positions[np.where(end_residue.names == "OXT")[0][0]]
            c_pos   = end_residue.positions[np.where(end_residue.names == "C"  )[0][0]]
            vector  = (oxt_pos - c_pos) / np.linalg.norm(oxt_pos - c_pos)
            N_position = oxt_pos
            C_position = N_position + vector * 1.47  # NME CH3 along C–N axis
        else:
            # sp2 geometry: NME N is placed symmetrically to the carbonyl O
            # relative to the terminal C (mirrors OXT placement logic).
            c_pos  = end_residue.positions[np.where(end_residue.names == "C" )[0][0]]
            o_pos  = end_residue.positions[np.where(end_residue.names == "O" )[0][0]]
            ca_pos = end_residue.positions[np.where(end_residue.names == "CA")[0][0]]
            v_co  = (o_pos  - c_pos) / np.linalg.norm(o_pos  - c_pos)
            v_cca = (ca_pos - c_pos) / np.linalg.norm(ca_pos - c_pos)
            n_dir = -(v_co + v_cca)
            n_dir /= np.linalg.norm(n_dir)
            N_position = c_pos + 1.36 * n_dir
            C_position = N_position + 1.47 * n_dir   # NME CH3 along C–N axis
        return N_position, C_position

    @staticmethod
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

    # ------------------------------------------------------------------
    # Cap application
    # ------------------------------------------------------------------

    def _apply_caps(self, pdb_in: str, pdb_out: str) -> None:
        """Add ACE (N-terminus) and NME (C-terminus) capping groups.

        Skips termini that already carry ACE or NME/NMA — no double-capping.
        Non-protein atoms (waters, ions) are preserved unchanged.
        Expects terminal residues to be hydrogen-free (caller's responsibility).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(pdb_in)

        protein_sel     = u.select_atoms("protein")
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
            segid    = seg.segid
            skip_ace = seg.residues[0].resname  in ("ACE",)
            skip_nme = seg.residues[-1].resname in ("NME", "NMA")

            if skip_ace and skip_nme:
                segment_universes.append(mda.Merge(seg.atoms))
                res_start = seg.residues.resids[-1]
                continue

            last_res = seg.residues[-1].atoms

            # Remove OXT before merging (NME takes its place)
            if not skip_nme and "OXT" in last_res.names:
                oxt_index = last_res.select_atoms("name OXT")[0].index
                Chain = seg.atoms.select_atoms(f"not index {oxt_index}")
            else:
                Chain = seg.atoms

            parts = []

            if not skip_ace:
                first_res    = seg.residues[0].atoms
                ace_positions = self._get_ace_pos(first_res)
                ace_names    = ["C", "CH3", "O"]
                ace_universe = self._create_cap_universe(
                    n_atoms=len(ace_positions),
                    name=ace_names,
                    resname=len(ace_names) * ["ACE"],
                    positions=ace_positions,
                    resids=seg.residues[0].resid * np.ones(len(ace_names)),
                    segid=segid,
                )
                parts.append(ace_universe.atoms)

            parts.append(Chain)

            if not skip_nme:
                nme_positions = self._get_nme_pos(last_res)
                nme_names    = ["N", "C"]
                nme_universe = self._create_cap_universe(
                    n_atoms=len(nme_names),
                    name=nme_names,
                    resname=len(nme_names) * ["NME"],
                    positions=nme_positions,
                    resids=(seg.residues[-1].resid + 2) * np.ones(len(nme_names)),
                    segid=segid,
                )
                parts.append(nme_universe.atoms)

            u_all = mda.Merge(*parts)

            # Assign sequential residue IDs
            new_resids = []
            if not skip_ace:
                new_resids.extend([res_start + 1] * 3)
                chain_start = res_start + 2
            else:
                chain_start = res_start + 1
            chain_resids = list(np.arange(chain_start, Chain.residues.n_residues + chain_start))
            new_resids.extend(chain_resids)
            if not skip_nme:
                nme_id = chain_resids[-1] + 1
                new_resids.extend([nme_id, nme_id])

            u_all.atoms.residues.resids = np.array(new_resids)
            res_start = u_all.atoms.residues.resids[-1]
            segment_universes.append(u_all)

        parts_final = [seg.atoms for seg in segment_universes]
        if len(non_protein_sel) > 0:
            parts_final.append(non_protein_sel)
        all_uni = mda.Merge(*parts_final)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_uni.atoms.write(pdb_out)


# ---------------------------------------------------------------------------
# Standalone PDB / structure utilities
# ---------------------------------------------------------------------------

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
