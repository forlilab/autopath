from itertools import product

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.lib.mdamath import triclinic_vectors

from autopath.utils import min_image_distance


def _universe(positions, chains, resnames, box):
    n = len(positions)
    u = mda.Universe.empty(n, n_residues=n, atom_resindex=np.arange(n), trajectory=True)
    u.add_TopologyAttr("resnames", resnames)
    u.add_TopologyAttr("chainIDs", chains)
    u.add_TopologyAttr("names", ["X"] * n)
    u.atoms.positions = np.asarray(positions, dtype=float)
    u.dimensions = box
    return u


def test_self_image_distance_cubic():
    # one chain, atoms 2 A apart in a 5 A box: nearest own image is 3 A away
    u = _universe([[0.5, 0, 0], [2.5, 0, 0]], ["A", "A"], ["PRO", "PRO"], [5, 5, 5, 90, 90, 90])
    assert min_image_distance(u, "resname PRO") == pytest.approx([3.0])


def test_ligand_wrapped_across_boundary_is_coimaged():
    # ligand bound 1.5 A from the protein but wrapped to the far side of a 10 A box
    u = _universe([[1, 5, 5], [9.5, 5, 5]], ["A", "B"], ["PRO", "LIG"], [10, 10, 10, 90, 90, 90])
    assert min_image_distance(u, "resname LIG", "resname PRO") == pytest.approx([8.5])


def test_matches_brute_force_triclinic():
    rng = np.random.default_rng(0)
    box = np.array([30, 30, 30, 60, 60, 90], dtype=float)
    prot = rng.normal(10, 3, (40, 3))
    lig = prot[:5].mean(axis=0) + rng.normal(0, 1, (5, 3))
    u = _universe(np.vstack([prot, lig]), ["A"] * 40 + ["B"] * 5,
                  ["PRO"] * 40 + ["LIG"] * 5, box)
    vecs = triclinic_vectors(box)
    shifts = [np.array(n) @ vecs for n in product(range(-2, 3), repeat=3) if any(n)]
    ref = min(np.linalg.norm(lig[:, None] - (prot + t)[None], axis=-1).min() for t in shifts)
    assert min_image_distance(u, "resname LIG", "resname PRO") == pytest.approx([ref])


def test_aligned_trajectory_rejected(tmp_path):
    u = _universe([[0.5, 0, 0], [2.5, 0, 0]], ["A", "A"], ["PRO", "PRO"], [5, 5, 5, 90, 90, 90])
    pdb, dcd = tmp_path / "s.pdb", tmp_path / "x_aligned.dcd"
    u.atoms.write(pdb)
    with mda.Writer(str(dcd), u.atoms.n_atoms) as w:
        w.write(u.atoms)
    with pytest.raises(ValueError, match="superposed"):
        min_image_distance(mda.Universe(str(pdb), str(dcd)), "all")


def test_periodic_image_report_flags_frames_inside_cutoff(caplog):
    import logging
    from openmm import NonbondedForce, System, unit
    from autopath.utils import periodic_image_report

    system = System()
    nb = NonbondedForce()
    nb.setCutoffDistance(0.4 * unit.nanometer)
    system.addForce(nb)
    u = _universe([[0.5, 0, 0], [2.5, 0, 0]], ["A", "A"], ["ALA", "ALA"], [5, 5, 5, 90, 90, 90])
    with caplog.at_level(logging.WARNING, logger="autopath"):
        df = periodic_image_report(u, system)
    assert df.attrs["cutoff_A"] == pytest.approx(4.0)
    assert df["protein_self"].tolist() == pytest.approx([3.0])
    assert "inside the cutoff" in caplog.text


def test_partial_bonds_do_not_split_a_chain():
    # a single CONECT-style bond must not turn the other atoms into separately imaged molecules
    u = _universe([[0.5, 0, 0], [2.0, 0, 0], [5.5, 0, 0]], ["A"] * 3, ["PRO"] * 3, [8, 8, 8, 90, 90, 90])
    u.add_TopologyAttr("bonds", [(0, 1)])
    # whole chain spans 0.5..5.5: own image 5.5 -> 8.5 is 3.0 A (3.25 if atom 2 were re-imaged alone)
    assert min_image_distance(u, "resname PRO") == pytest.approx([3.0])
