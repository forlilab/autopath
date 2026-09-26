"""Regression tests for SteeredMD._get_pocket_atoms.

The pocket subset feeds numpy fancy-indexing (positions_nm[subset]) in
_compute_nc/_compute_mindist, so it must be integer atom indices, not
openmm Atom objects.
"""
import numpy as np
import pytest
from types import SimpleNamespace

import openmm.unit as openmmunit

from autopath.pulling.steered_md import SteeredMD


class _Element:
    def __init__(self, symbol):
        self.symbol = symbol


class _Residue:
    def __init__(self, name, idx):
        self.name = name
        self.index = idx


class _Atom:
    def __init__(self, index, name, symbol, residue):
        self.index = index
        self.name = name
        self.element = _Element(symbol)
        self.residue = residue


class _Topology:
    def __init__(self, atoms):
        self._atoms = atoms

    def atoms(self):
        return iter(self._atoms)


def _make_system():
    """Two protein residues (CA + CB each), a ligand atom, a water, a Ca2+ ion.

    Residue 0's CA sits 0.2 nm from the ligand (inside the 0.6 nm cutoff);
    residue 1's CA sits 5 nm away (outside).
    """
    r0, r1 = _Residue("ALA", 0), _Residue("LEU", 1)
    rlig, rwat, rion = _Residue("UNK", 2), _Residue("HOH", 3), _Residue("CA", 4)
    atoms = [
        _Atom(0, "CA", "C", r0),
        _Atom(1, "CB", "C", r0),
        _Atom(2, "CA", "C", r1),
        _Atom(3, "CB", "C", r1),
        _Atom(4, "C1", "C", rlig),      # ligand (groupA)
        _Atom(5, "O", "O", rwat),
        _Atom(6, "CA", "Ca", rion),     # calcium ion: named CA but not a carbon
    ]
    pos = np.array([
        [0.2, 0.0, 0.0],   # near CA
        [0.3, 0.0, 0.0],
        [5.0, 0.0, 0.0],   # far CA
        [5.1, 0.0, 0.0],
        [0.0, 0.0, 0.0],   # ligand
        [1.0, 1.0, 1.0],
        [0.25, 0.0, 0.0],  # ion, close to ligand but must be excluded
    ])
    state = SimpleNamespace(
        getPositions=lambda asNumpy=False: pos * openmmunit.nanometers)
    sim = SimpleNamespace(context=SimpleNamespace(
        getState=lambda **kw: state))
    smd = SimpleNamespace(topology=_Topology(atoms), groupA_atoms=[4])
    return smd, sim, pos


def test_get_pocket_atoms_returns_integer_indices():
    smd, sim, pos = _make_system()
    subset, residues = SteeredMD._get_pocket_atoms(smd, sim, cutoff=0.6)

    subset = np.asarray(subset)
    assert np.issubdtype(subset.dtype, np.integer), (
        f"pocket subset must be integer indices, got dtype {subset.dtype}")
    # only residue 0's CA is within 0.6 nm; the Ca2+ ion must not be picked up
    assert subset.tolist() == [0]
    # and the subset must be usable as a numpy index (what _compute_nc does)
    np.testing.assert_allclose(pos[subset], pos[[0]])
    assert [r.name for r in residues] == ["ALA"]


def _make_protein_protein_system():
    """A two-residue pulled peptide (groupA = its own CAs), plus a near and a
    far receptor residue. The pulled CAs sit right next to each other, so
    before the fix every pulled atom is trivially "within cutoff of itself".
    """
    rA0, rA1 = _Residue("GLY", 0), _Residue("GLY", 1)
    rNear, rFar = _Residue("ALA", 2), _Residue("LEU", 3)
    atoms = [
        _Atom(0, "CA", "C", rA0),   # pulled group (groupA)
        _Atom(1, "CA", "C", rA1),   # pulled group (groupA)
        _Atom(2, "CA", "C", rNear),  # receptor, within cutoff
        _Atom(3, "CA", "C", rFar),   # receptor, outside cutoff
    ]
    pos = np.array([
        [0.00, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [5.00, 0.0, 0.0],
    ])
    state = SimpleNamespace(
        getPositions=lambda asNumpy=False: pos * openmmunit.nanometers)
    sim = SimpleNamespace(context=SimpleNamespace(
        getState=lambda **kw: state))
    smd = SimpleNamespace(topology=_Topology(atoms), groupA_atoms=[0, 1])
    return smd, sim, pos


def test_get_pocket_atoms_excludes_self_for_protein_protein_pull():
    smd, sim, pos = _make_protein_protein_system()
    subset, residues = SteeredMD._get_pocket_atoms(smd, sim, cutoff=0.6)

    subset = np.asarray(subset)
    # the pulled group's own CAs must never appear in its own pocket
    assert set(subset.tolist()).isdisjoint(smd.groupA_atoms)
    assert subset.tolist() == [2]
    min_dist = np.min(np.linalg.norm(
        pos[smd.groupA_atoms][:, np.newaxis, :] - pos[subset][np.newaxis, :, :], axis=-1))
    assert min_dist > 0.0
    assert [r.name for r in residues] == ["ALA"]


def test_get_pocket_atoms_unaffected_for_small_molecule_pull():
    # small-molecule case: groupA shares no residue with any protein_CA atom,
    # so exclusion must be a no-op and the result byte-identical to before.
    smd, sim, pos = _make_system()
    subset, residues = SteeredMD._get_pocket_atoms(smd, sim, cutoff=0.6)
    assert subset.tolist() == [0]
    assert [r.name for r in residues] == ["ALA"]
