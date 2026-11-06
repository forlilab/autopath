"""NC autostop, pocket and geom features use the whole ligand; the anchor only sets the pulled COM."""
import numpy as np

from autopath.pulling.steered_md import SteeredMD


def _smd(groupA, ligand_atoms):
    smd = SteeredMD.__new__(SteeredMD)
    smd.groupA_atoms = groupA
    smd.ligand_atoms = ligand_atoms
    smd.subset_protein_CA = np.array([0])
    return smd


def test_feature_atoms_fall_back_to_anchor():
    assert list(_smd([3, 4], None)._feature_atoms()) == [3, 4]
    assert list(_smd([3, 4], [1, 2, 3, 4])._feature_atoms()) == [1, 2, 3, 4]


def test_nc_counts_non_anchor_contacts():
    pos = np.array([[0.0, 0, 0],      # pocket CA
                    [0.3, 0, 0],      # ligand atom off the anchor, still in contact
                    [3.0, 0, 0],      # anchor atom, far away
                    [3.1, 0, 0]])
    anchor_only = _smd([2, 3], None)._compute_nc(pos)
    whole = _smd([2, 3], [1, 2, 3])._compute_nc(pos)
    assert anchor_only < 1e-3 < 0.5 < whole
