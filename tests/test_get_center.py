"""get_center pairs masses with positions in group order, sorted or not."""
import numpy as np
from openmm.app import Element, Topology

from autopath.utils import get_center


def test_unsorted_group_mass_weighted_center():
    top = Topology()
    chain = top.addChain()
    res = top.addResidue("X", chain)
    for el in ("H", "C", "O", "N"):
        top.addAtom(el, Element.getBySymbol(el), res)
    atoms = list(top.atoms())
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    group = [2, 0, 1]
    m = np.array([atoms[i].element.mass._value for i in group])
    expected = np.average(pos[group], axis=0, weights=m)
    np.testing.assert_allclose(get_center(pos, atoms, group, True), expected)
