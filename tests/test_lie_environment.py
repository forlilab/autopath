"""LIE environment selection: 0-based Atom.resid vs 1-based Amber masks.

pytraj exposes ``Atom.resid`` 0-based, but an Amber ``:N`` mask is 1-based. Pasting the
raw attribute into a mask shifts the whole environment by one residue. Measured on a
6E22 benzene box, that moved the LIE total from -9.78 to -7.58 kcal/mol (29%), so this
is a wrong-answer bug rather than a cosmetic one.

The helper is exercised against a stub topology so the test needs no prmtop.
"""

import pytest

from autopath.ap_PLIP import environment_resids


class _Atom:
    def __init__(self, resid):
        self.resid = resid          # 0-based, as pytraj reports it


class _StubTopology:
    """Minimal duck-type of pytraj's Topology: select() -> indices, [] -> atoms."""

    def __init__(self, atom_resids, selected):
        self._atoms = [_Atom(r) for r in atom_resids]
        self._selected = selected
        self.last_mask = None

    def select(self, mask):
        self.last_mask = mask
        return list(self._selected)

    def __getitem__(self, i):
        return self._atoms[i]


def test_resids_are_returned_one_based():
    # atoms 0..3 belong to 0-based residues 182,183,183,201
    top = _StubTopology([182, 183, 183, 201], selected=[0, 1, 2, 3])
    assert environment_resids(top) == [183, 184, 202]


def test_duplicates_collapse_and_output_is_sorted():
    top = _StubTopology([5, 5, 2, 9, 2], selected=[0, 1, 2, 3, 4])
    assert environment_resids(top) == [3, 6, 10]


def test_empty_selection_gives_an_empty_environment():
    top = _StubTopology([1, 2, 3], selected=[])
    assert environment_resids(top) == []


def test_mask_includes_cutoff_ligand_and_exclusion():
    top = _StubTopology([0], selected=[0])
    environment_resids(top, ligand_amber_selection=":810",
                       exclude_amber_selection=":NA,CL", cutoff=6.0)
    mask = top.last_mask
    assert "(:810<:6.0)" in mask
    assert "!(:NA,CL)" in mask
    assert "!(:810)" in mask


def test_water_is_not_excluded_by_default():
    """LIE scores the ligand against its whole surroundings; solvent belongs in it."""
    top = _StubTopology([0], selected=[0])
    environment_resids(top)
    for water in ("WAT", "HOH", "SOL"):
        assert water not in top.last_mask


def test_no_exclusion_mask_is_allowed():
    top = _StubTopology([0], selected=[0])
    environment_resids(top, exclude_amber_selection=None)
    assert "!()" not in top.last_mask
