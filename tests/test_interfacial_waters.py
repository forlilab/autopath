"""find_interfacial_waters must survive a site with no long-lived water."""
import os
import numpy as np
import pytest
import MDAnalysis as mda

from autopath.ap_PLIP import ProteinLigandAnalyzer


def _two_frame_universe(tmp_path, bridge=False):
    """Protein + ligand + two waters. With bridge=False no water sits between them."""
    top = mda.Universe.empty(8, n_residues=4, atom_resindex=[0, 0, 1, 1, 2, 2, 3, 3],
                             residue_segindex=[0, 0, 0, 0], trajectory=True)
    top.add_TopologyAttr('name', ['N', 'CA', 'N', 'CA', 'O', 'H1', 'O', 'H1'])
    top.add_TopologyAttr('resname', ['ALA', 'ALA', 'HOH', 'HOH'])
    top.add_TopologyAttr('resid', [1, 2, 3, 4])
    top.add_TopologyAttr('type', ['N', 'C', 'N', 'C', 'O', 'H', 'O', 'H'])
    # receptor at x=0, ligand at x=6; waters either between them or far away
    wx = 3.0 if bridge else 40.0
    pos = np.array([[0., 0, 0], [1.5, 0, 0],          # receptor resid 1
                    [6., 0, 0], [7.5, 0, 0],          # ligand   resid 2
                    [wx, 0, 0], [wx + 0.95, 0, 0],    # water    resid 3
                    [wx, 5, 0], [wx + 0.95, 5, 0]])   # water    resid 4
    top.atoms.positions = pos
    top.dimensions = [80., 80., 80., 90., 90., 90.]
    return top


def test_no_persistent_water_still_writes_a_pdb(tmp_path):
    """An interface with no long-lived water is a normal result. The empty resid
    list used to make the selection string unparseable and raise SelectionError."""
    u = _two_frame_universe(tmp_path, bridge=False)
    out = str(tmp_path / "nowater.pdb")
    res = ProteinLigandAnalyzer.find_interfacial_waters(
        u, ligand_sel="resid 2", protein_sel="resid 1", cutoff=3.5, pdb_fname=out)
    assert res == {}, "no water bridges this interface"
    assert os.path.exists(out), "a site with no persistent water must still write"
    v = mda.Universe(out)
    assert v.select_atoms("resname HOH").n_residues == 0
    assert v.atoms.n_atoms == 4            # receptor + ligand only


def test_a_bridging_water_is_found_and_written(tmp_path):
    """Guard must not swallow the normal case."""
    u = _two_frame_universe(tmp_path, bridge=True)
    out = str(tmp_path / "water.pdb")
    res = ProteinLigandAnalyzer.find_interfacial_waters(
        u, ligand_sel="resid 2", protein_sel="resid 1", cutoff=3.5, pdb_fname=out)
    assert 3 in res and res[3] == pytest.approx(1.0)
    v = mda.Universe(out)
    assert v.select_atoms("resname HOH").n_residues == 1
