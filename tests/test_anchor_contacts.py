import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from autopath.utils import get_ligand_anchor_atoms

BOX = 30.0
N_FRAMES = 10
MODES = ["contacts", "weighted_com"]


def _universe(a_pos, b_pos):
    # atoms: 0 = ALA CA (P1), 1 = ALA CA (P2), 2 = LIG C1 (A), 3 = LIG C2 (B)
    u = mda.Universe.empty(4, n_residues=4, atom_resindex=[0, 1, 2, 3],
                           trajectory=True)
    u.add_TopologyAttr("name", ["CA", "CA", "C1", "C2"])
    u.add_TopologyAttr("resname", ["ALA", "ALA", "LIG", "LIG"])
    u.add_TopologyAttr("resid", [1, 2, 100, 101])
    u.add_TopologyAttr("masses", [12.0] * 4)

    coords = np.zeros((N_FRAMES, 4, 3), dtype=np.float32)
    for f in range(N_FRAMES):
        coords[f, 0] = (0.5, 15, 15)
        coords[f, 1] = (15, 5, 15)
        coords[f, 2] = a_pos(f)
        coords[f, 3] = b_pos(f)
    u.load_new(coords, format=MemoryReader,
               dimensions=np.array([BOX, BOX, BOX, 90, 90, 90]))
    return u


def _anchor(u, mode, tmp_path):
    return list(get_ligand_anchor_atoms(u, "resname LIG", mode=mode,
                                        n_atoms=1, out_dir=str(tmp_path)))


@pytest.mark.parametrize("mode", MODES)
def test_uses_minimum_image(mode, tmp_path):
    # A contacts P1 only across the periodic boundary; B never within 3.5 Å
    u = _universe(lambda f: (29.5, 15, 15), lambda f: (15, 9.5, 15))
    assert _anchor(u, mode, tmp_path) == [2]


@pytest.mark.parametrize("mode", MODES)
def test_uses_second_half(mode, tmp_path):
    # A contacts in frames 0-6 (7 total, 2 in second half);
    # B contacts in frames 7-9 (3 total, all in second half)
    u = _universe(lambda f: (3.5, 15, 15) if f <= 6 else (5.0, 15, 15),
                  lambda f: (15, 8.0, 15) if f >= 7 else (15, 9.5, 15))
    assert _anchor(u, mode, tmp_path) == [3]
