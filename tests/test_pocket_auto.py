"""Auto pocket = protein CA atoms within the cutoff of ligand heavy atoms."""
import numpy as np
import MDAnalysis as mda

from autopath.utils import get_pocket_atoms_idxs


def test_auto_pocket_uses_ca_distance_to_ligand_heavy_atoms():
    # residues: ALA1 (CA 6 A), ALA2 (CA 11 A, side chain 3 A), ALA3 (CA 9 A from heavy, 7 A from H), ligand UNK
    names = ["CA", "CA", "CB", "CA", "C1", "H1"]
    resindex = [0, 1, 1, 2, 3, 3]
    u = mda.Universe.empty(6, n_residues=4, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", ["ALA", "ALA", "ALA", "UNK"])
    u.add_TopologyAttr("resid", [1, 2, 3, 4])
    u.atoms.positions = np.array([[6.0, 0, 0], [11.0, 0, 0], [3.0, 0, 0],
                                  [0, 9.0, 0], [0, 0, 0], [0, 2.0, 0]])
    u.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
    assert get_pocket_atoms_idxs(u, ligand_selection="resname UNK", cutoff=8.0) == [0]
