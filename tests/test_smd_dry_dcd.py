"""Dry sMD DCDs (protein+ligand prefix) and the topology lookup used by the analysis."""
import MDAnalysis as mda
import numpy as np
import openmm.unit as u
import pytest
from openmm.app import Element, PDBFile, Topology

from autopath.pulling.SMDData import DRY_TOP_NAME, smd_topology
from autopath.pulling.steered_md import SteeredMD

BOX = np.diag([3.0, 3.0, 3.0])


def _topology(order=("ALA", "UNK", "HOH", "HOH", "NA")):
    top = Topology()
    chain = top.addChain()
    for name in order:
        res = top.addResidue(name, chain)
        if name == "HOH":
            atoms = [("O", "O"), ("H1", "H"), ("H2", "H")]
        elif name == "NA":
            atoms = [("NA", "Na")]
        else:
            atoms = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
        for an, el in atoms:
            top.addAtom(an, Element.getBySymbol(el), res)
    top.setPeriodicBoxVectors(BOX * u.nanometers)
    return top


def _smd(top, out_dir):
    smd = SteeredMD.__new__(SteeredMD)
    smd.topology, smd.out_dir, smd.dcd_dry, smd._dry_top = top, str(out_dir), True, None
    return smd


def _pos(n, seed=0):
    return np.random.default_rng(seed).uniform(0.2, 2.8, (n, 3)) * u.nanometers


def test_dry_subset_is_solute_prefix(tmp_path):
    top = _topology()
    idx, dry = _smd(top, tmp_path)._dry_subset()
    assert idx == list(range(8))
    assert [r.name for r in dry.residues()] == ["ALA", "UNK"]


def test_dry_subset_rejects_solvent_inside_solute(tmp_path):
    with pytest.raises(ValueError, match="before all solvent"):
        _smd(_topology(("ALA", "HOH", "UNK", "NA")), tmp_path)._dry_subset()


def test_dry_pdb_and_topology_lookup(tmp_path):
    top = _topology()
    n = top.getNumAtoms()
    pos = _pos(n)
    full_pdb = tmp_path / "eq.pdb"
    with open(full_pdb, "w") as f:
        PDBFile.writeFile(top, pos, f)

    trajdir = tmp_path / "trajectories"
    trajdir.mkdir()
    smd = _smd(top, trajdir)
    smd._save_dry_pdb(pos, BOX * u.nanometers, str(trajdir / DRY_TOP_NAME))
    dry_u = mda.Universe(str(trajdir / DRY_TOP_NAME))
    full_u = mda.Universe(str(full_pdb))
    assert dry_u.atoms.n_atoms == 8
    # same atoms at the same indices, so index and name selections agree
    np.testing.assert_allclose(dry_u.atoms.positions, full_u.atoms.positions[:8], atol=1e-3)
    assert dry_u.select_atoms("index 4:7").resnames.tolist() == ["UNK"] * 4

    dry_dcd = trajdir / "sMD_replica-1_v0.01_forward.dcd"
    with mda.Writer(str(dry_dcd), n_atoms=8) as w:
        for _ in range(3):
            w.write(dry_u.atoms)
    full_dcd = trajdir / "full.dcd"
    with mda.Writer(str(full_dcd), n_atoms=n) as w:
        w.write(full_u.atoms)

    assert smd_topology(str(full_pdb), str(full_dcd)) == str(full_pdb)
    top_for_dry = smd_topology(str(full_pdb), str(dry_dcd))
    assert top_for_dry.endswith(DRY_TOP_NAME)
    assert mda.Universe(top_for_dry, str(dry_dcd)).trajectory.n_frames == 3

    (trajdir / DRY_TOP_NAME).unlink()
    with pytest.raises(ValueError, match="no dry_top.pdb"):
        smd_topology(str(full_pdb), str(dry_dcd))
