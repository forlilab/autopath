# tests/test_learned_features.py
import numpy as np
import pytest

from autopath.metadynamics.learned.features import (Feature, evaluate, features_from_dict,
                                                     features_to_dict, CONTACT_CUTOFF_FACTOR,
                                                     CONTACT_SWITCH_FACTOR)


def _frame(*xyz):
    return np.array([xyz], dtype=float)          # (1, n_atoms, 3)


def test_pair_contact_value():
    X = _frame([0, 0, 0], [0.35, 0, 0], [0, 0.7, 0])
    f = Feature("p", "pair", "F1", pairs=[(0, 1), (0, 2)], r0=0.35)
    assert evaluate([f], X)[0, 0] == pytest.approx(0.5 + 1 / (1 + 2 ** 6))


def test_contacts_switch_matches_openmm_polynomial():
    r0 = 0.3
    rs, rc = CONTACT_SWITCH_FACTOR * r0, CONTACT_CUTOFF_FACTOR * r0
    r = 0.5 * (rs + rc)
    X = _frame([0, 0, 0], [r, 0, 0], [0, 2 * rc, 0])
    f = Feature("c", "contacts", "F2", group_a=[0], group_b=[1, 2], r0=r0)
    x = 0.5
    expected = (1 / (1 + (r / r0) ** 6)) * (1 - 10 * x ** 3 + 15 * x ** 4 - 6 * x ** 5)
    assert evaluate([f], X)[0, 0] == pytest.approx(expected)


def test_centroid_distance_unweighted():
    X = _frame([0, 0, 0], [0.2, 0, 0], [1.1, 0, 0])
    f = Feature("d", "centroid_distance", "F3", groups=[[0, 1], [2]])
    assert evaluate([f], X)[0, 0] == pytest.approx(1.0)


def test_centroid_count():
    X = _frame([0, 0, 0], [0.5, 0, 0], [0, 1.0, 0])
    f = Feature("w", "centroid_count", "F5", groups=[[0]], group_b=[1, 2], r0=0.5)
    assert evaluate([f], X)[0, 0] == pytest.approx(0.5 + 1 / (1 + 2 ** 8))


def test_cos_perpendicular_and_parallel():
    X = _frame([0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [2, 0, 0])
    perp = Feature("o", "cos", "F8", groups=[[0], [1], [2], [3]])
    par = Feature("o2", "cos", "F8", groups=[[0], [1], [4], [5]])
    v = evaluate([perp, par], X)[0]
    assert v[0] == pytest.approx(0.0, abs=1e-12) and v[1] == pytest.approx(1.0)


def test_round_trip_through_npz(tmp_path):
    fs = [Feature("p", "pair", "F1", pairs=[(0, 1)], r0=0.35),
          Feature("o", "cos", "F8", groups=[[0], [1], [2], [3]])]
    np.savez(tmp_path / "f.npz", **features_to_dict(fs))
    back = features_from_dict(dict(np.load(tmp_path / "f.npz", allow_pickle=False)))
    assert back == fs


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        evaluate([Feature("x", "torsion", "F9")], _frame([0, 0, 0]))
