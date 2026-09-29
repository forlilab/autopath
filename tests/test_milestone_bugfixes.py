"""Regression tests for the milestone bugs found in the final branch review."""
import os
import numpy as np
import pytest
import MDAnalysis as mda

from autopath.utils import _reimaging_ligand
from autopath.clustering import order_milestones_bound_to_unbound


def _lig_universe():
    """One protein CA plus a 2-atom ligand residue (heavy atom + its hydrogen)."""
    u = mda.Universe.empty(3, n_residues=2, atom_resindex=[0, 1, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "C1", "H1"])
    u.add_TopologyAttr("resname", ["ALA", "LIG"])
    u.add_TopologyAttr("resid", [1, 2])
    u.dimensions = [40, 40, 40, 90, 90, 90]
    # ligand sits a box length away from the protein, so re-imaging must fire
    u.atoms.positions = np.array([[0., 0., 0.], [34., 0., 0.], [35., 0., 0.]])
    return u


def test_reimaging_moves_ligand_hydrogens_with_their_heavy_atoms():
    """A heavy-atom selection must not strand the ligand's hydrogens.

    Callers pass "(ligand) and not name H*". Translating only the selected atoms
    left the H a box length behind, giving a 41 A C-H bond in the structure that
    then gets serialized and minimised.
    """
    u = _lig_universe()
    assert _reimaging_ligand(u, "resname LIG and not name H*") is True

    c1, h1 = u.atoms[1].position, u.atoms[2].position
    assert np.linalg.norm(h1 - c1) == pytest.approx(1.0, abs=1e-4)
    # and it really did move, i.e. the test is not passing because nothing happened
    assert c1[0] == pytest.approx(-6.0, abs=1e-4)
    assert h1[0] == pytest.approx(-5.0, abs=1e-4)


def test_reimaging_leaves_protein_untouched():
    u = _lig_universe()
    _reimaging_ligand(u, "resname LIG and not name H*")
    np.testing.assert_allclose(u.atoms[0].position, [0., 0., 0.], atol=1e-4)


def test_reimaging_noop_when_ligand_already_at_minimum_image():
    u = _lig_universe()
    u.atoms.positions = np.array([[0., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
    assert _reimaging_ligand(u, "resname LIG and not name H*") is False
    np.testing.assert_allclose(u.atoms[1].position, [3., 0., 0.], atol=1e-4)


# --- milestone direction anchoring -----------------------------------------

def _write_state_pair(tmp_path, seps):
    """Write one .pdb per requested pocket-ligand separation (in nm).

    Atom 0 is the pocket, atom 1 the ligand; PDB coordinates are Angstrom.
    """
    paths = []
    for i, sep in enumerate(seps):
        p = tmp_path / f"milestone_{i + 1}_frame_{i}.pdb"
        with open(p, "w") as f:
            f.write(f"ATOM      1  C1  POC A   1    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
            f.write(f"ATOM      2  C2  LIG A   2    {sep * 10:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
            f.write("END\n")
        paths.append(str(p))
    return paths


def test_bound_to_unbound_order_is_left_alone(tmp_path):
    paths = _write_state_pair(tmp_path, [0.4, 0.9, 1.5])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == paths


def test_unbound_to_bound_order_is_reversed(tmp_path):
    """PC1's sign is arbitrary, so the centroid order is as likely to be reversed."""
    paths = _write_state_pair(tmp_path, [1.5, 0.9, 0.4])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == list(reversed(paths))


def test_single_milestone_is_returned_unchanged(tmp_path):
    paths = _write_state_pair(tmp_path, [0.7])
    assert order_milestones_bound_to_unbound(paths, [0], [1]) == paths


# --- sigma='auto' ----------------------------------------------------------

def _auto_sigma(com_dists):
    """The sigma='auto' arithmetic from pathCV_cv, against which we assert."""
    steps = np.diff(np.asarray(com_dists, dtype=float))
    gaps = np.abs(steps)
    usable = gaps[gaps > 1e-6]
    if usable.size == 0:
        raise ValueError("no usable spacing")
    return 0.5 * float(usable.min())


def test_auto_sigma_is_positive_for_a_non_monotone_milestone_list():
    """A backwards step must not produce a negative sigma.

    cvpack accepts a negative sigma without complaint, so this failed silently.
    """
    sigma = _auto_sigma([0.40, 1.20, 0.50, 0.90, 1.10])
    assert sigma > 0
    assert sigma == pytest.approx(0.5 * 0.20, abs=1e-9)


def test_auto_sigma_matches_half_the_minimum_gap_when_monotone():
    assert _auto_sigma([0.4, 0.9, 1.5]) == pytest.approx(0.25, abs=1e-9)


def test_auto_sigma_rejects_all_identical_milestones():
    with pytest.raises(ValueError):
        _auto_sigma([0.5, 0.5, 0.5])


def test_pathcv_source_guards_sigma_and_warns_on_non_monotone():
    """Pin the guard in the real code, since the arithmetic above is a copy."""
    import inspect
    from autopath.metadynamics import CV
    src = inspect.getsource(CV.pathCV_cv)
    assert "np.abs(steps)" in src
    assert "cannot auto-tune sigma" in src
    assert "not monotonically increasing" in src


# --- no loadState anywhere -------------------------------------------------

def test_no_module_calls_load_state():
    """loadState restores global parameters by name, so it silently keeps stale
    values on a system whose forces were stripped and re-added."""
    import autopath
    root = os.path.dirname(os.path.abspath(autopath.__file__))
    offenders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path) as f:
                for lineno, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    if ".loadState(" in code:
                        offenders.append(f"{os.path.relpath(path, root)}:{lineno}")
    assert offenders == [], f"loadState is forbidden; found at {offenders}"


# --- medoid_info.json lookup ------------------------------------------------

def test_core_looks_for_medoid_info_under_path_analysis():
    """PathModel writes medoid_info.json into path_analysis/.

    Reading it from the analysis dir instead silently fell back to using every
    sMD trajectory rather than the medoids, with only a warning.
    """
    import inspect
    from autopath import autopath_core
    from autopath.pulling import PathModel as pm

    writer = inspect.getsource(pm)
    assert '"path_analysis", "medoid_info.json"' in writer

    reader = inspect.getsource(autopath_core)
    assert 'path_analysis/medoid_info.json' in reader
