import os

import MDAnalysis as mda
import pytest

from autopath import SystemPreparation

LIG = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "3ptb.sdf")


def test_requires_protein_or_ligands(tmp_path):
    with pytest.raises(ValueError, match="protein PDB, ligands, or both"):
        SystemPreparation(out_dir=str(tmp_path)).run(protein=None, ligands=None)


def test_ligand_only_membrane_not_supported(tmp_path):
    with pytest.raises(ValueError, match="not supported with a membrane"):
        SystemPreparation(is_membrane=True, out_dir=str(tmp_path)).run(protein=None, ligands=LIG)


@pytest.mark.slow
def test_ligand_only_box(tmp_path):
    system, topology = SystemPreparation(padding=1.0, out_dir=str(tmp_path)).run(
        protein=None, ligands=[("LIG", LIG, None, False)])
    u = mda.Universe(str(tmp_path / "system.pdb"))
    assert set(u.residues.resnames) - {"HOH", "NA", "CL"} == {"LIG"}
    assert len(u.select_atoms("protein")) == 0
    assert system.getNumParticles() == u.atoms.n_atoms
    for f in ("system.xml", "system.prmtop"):
        assert (tmp_path / f).is_file()
