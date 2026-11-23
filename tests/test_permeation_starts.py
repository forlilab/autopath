import pytest

from autopath.pulling.permeation_starts import check_same_topology, end_to_end_atoms


def test_end_to_end_atoms_are_the_heavy_chain_ends():
    from rdkit import Chem
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))     # heavy atoms 0..6; ends are C0 and O6
    assert set(end_to_end_atoms(mol)) == {0, 6}


def _pdb(path, names):
    lines = [f"HETATM{i + 1:5d} {n:<4s} UNK A   1       0.000   0.000 {i:7.3f}  1.00  0.00           {n[0]}"
             for i, n in enumerate(names)]
    path.write_text("\n".join(lines + ["END"]) + "\n")
    return str(path)


def test_check_same_topology_accepts_matches_and_names_the_first_mismatch(tmp_path):
    a = _pdb(tmp_path / "a.pdb", ["C1", "O1"])
    check_same_topology([a, _pdb(tmp_path / "b.pdb", ["C1", "O1"])])
    with pytest.raises(ValueError, match="atom 1"):
        check_same_topology([a, _pdb(tmp_path / "c.pdb", ["C1", "N1"])])
    with pytest.raises(ValueError, match="has 3 atoms"):
        check_same_topology([a, _pdb(tmp_path / "d.pdb", ["C1", "O1", "C2"])])
