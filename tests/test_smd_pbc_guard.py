"""In-loop PBC guard: the half-box limit follows the current COM-COM direction."""
import numpy as np
import pytest

from autopath.pulling.steered_md import _com_reduction_margin

_BOX = np.diag([8.0, 8.0, 6.0])


def test_limit_follows_current_direction():
    w = np.array([1.0])
    d, lim_x = _com_reduction_margin(np.array([[0, 0, 0], [2.0, 0, 0]]), _BOX, [1], w, [0], w)
    assert d == pytest.approx(2.0) and lim_x == pytest.approx(4.0)
    d, lim_z = _com_reduction_margin(np.array([[0, 0, 0], [0, 0, 2.0]]), _BOX, [1], w, [0], w)
    assert d == pytest.approx(2.0) and lim_z == pytest.approx(3.0)


def test_mass_weighted_com():
    pos = np.array([[0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    d, _ = _com_reduction_margin(pos, _BOX, [1, 2], np.array([0.5, 0.5]), [0], np.array([1.0]))
    assert d == pytest.approx(2.0)
