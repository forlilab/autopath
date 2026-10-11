import numpy as np
import pytest

from autopath.metadynamics.learned.features import (CN_DMAX, CN_R0, Feature, cn_switch, evaluate,
                                                    features_from_dict, features_to_dict)


def _ref(r, r0=CN_R0, d=CN_DMAX, n=6, m=10):
    f = lambda x: (1 - x ** n) / (1 - x ** m)
    c = f(d / r0)
    out = (f(r / r0) - c) / (1 - c)
    return np.where(r < d, out, 0.0)


def _ref_c(d=CN_DMAX, r0=CN_R0):
    x = d / r0
    return (1 - x ** 6) / (1 - x ** 10)


@pytest.mark.parametrize("r", [0.1, 0.3, 0.34, 0.36, 0.5, 0.9, 0.99])
def test_cn_switch_matches_the_shifted_rational(r):
    assert cn_switch(np.array([r]))[0] == pytest.approx(float(_ref(np.array([r]))[0]), rel=1e-12)


def test_cn_switch_is_finite_and_continuous_at_r0():
    r = CN_R0 * np.array([1 - 2e-3, 1 - 1e-9, 1.0, 1 + 1e-9, 1 + 2e-3])
    v = cn_switch(r)
    assert np.all(np.isfinite(v))
    lim = (6 / 10 - _ref_c()) / (1 - _ref_c())
    assert v[2] == pytest.approx(lim, rel=1e-6)
    assert abs(v[1] - v[3]) < 1e-6


def test_cn_switch_is_zero_at_and_beyond_dmax():
    assert np.all(cn_switch(np.array([CN_DMAX, CN_DMAX + 0.01, 3.0])) == 0.0)


def test_cn_feature_sums_terms_with_scales_and_minimum_image():
    box = np.diag([3.0, 3.0, 3.0])
    X = np.zeros((1, 4, 3))
    X[0, 1] = [0.4, 0, 0]
    X[0, 2] = [2.8, 0, 0]          # 0.2 nm from atom 0 through the boundary
    X[0, 3] = [0, 0.5, 0]
    f = Feature("w", "cn", "W3", terms=[[[0], [1, 2], 1.0], [[0], [3], 0.5]])
    v = evaluate([f], X, box=box[None])[0, 0]
    exp = _ref(np.array([0.4]))[0] + _ref(np.array([0.2]))[0] + 0.5 * _ref(np.array([0.5]))[0]
    assert v == pytest.approx(exp, rel=1e-12)


def test_cn_triclinic_minimum_image():
    box = np.array([[3.0, 0, 0], [1.0, 3.0, 0], [0.5, 0.5, 3.0]])
    X = np.zeros((1, 2, 3))
    X[0, 1] = box[1] + [0.2, 0.1, 0.0]     # one b-vector away plus a small offset
    f = Feature("w", "cn", "W3", terms=[[[0], [1], 1.0]])
    v = evaluate([f], X, box=box[None])[0, 0]
    assert v == pytest.approx(_ref(np.array([np.hypot(0.2, 0.1)]))[0], rel=1e-9)


def test_cn_skips_self_pairs_and_excluded_pairs():
    box = np.diag([3.0, 3.0, 3.0])[None]
    X = np.zeros((1, 3, 3))
    X[0, 1] = [0.4, 0, 0]
    X[0, 2] = [0, 0.5, 0]
    f = Feature("w", "cn", "W3", terms=[[[0, 1], [0, 1, 2], 1.0]])
    v = evaluate([f], X, box=box)[0, 0]
    exp = 2 * _ref(np.array([0.4]))[0] + _ref(np.array([0.5]))[0] + _ref(np.array([np.hypot(0.4, 0.5)]))[0]
    assert v == pytest.approx(exp, rel=1e-12)


def test_cn_exclude_drops_the_listed_pair_in_either_order():
    box = np.diag([3.0, 3.0, 3.0])[None]
    X = np.zeros((1, 2, 3))
    X[0, 1] = [0.4, 0, 0]
    for ex in ([[0, 1]], [[1, 0]]):
        f = Feature("w", "cn", "W3", terms=[[[0], [1], 1.0]], exclude=ex)
        assert evaluate([f], X, box=box)[0, 0] == 0.0
    f = Feature("w", "cn", "W3", terms=[[[0], [1], 1.0]], exclude=[[0, 2]])
    assert evaluate([f], X, box=box)[0, 0] == pytest.approx(_ref(np.array([0.4]))[0], rel=1e-12)


def test_cn_needs_a_box():
    f = Feature("w", "cn", "W1", terms=[[[0], [1], 1.0]])
    with pytest.raises(ValueError, match="box"):
        evaluate([f], np.zeros((1, 2, 3)))


def test_cn_terms_and_exclude_survive_the_json_round_trip():
    f = Feature("w", "cn", "W3", terms=[[[0, 1], [2], 0.63]], exclude=[[3, 1], [0, 2]])
    g = features_from_dict(features_to_dict([f]))[0]
    assert g == f
    assert g.exclude == [[0, 2], [1, 3]] or g.exclude == [[1, 3], [0, 2]]
    assert all(a < b for a, b in g.exclude)


def _brute(X, box, a, b, scale=1.0):
    rng = range(-2, 3)
    sh = np.array([[i, j, k] for i in rng for j in rng for k in rng], float) @ box
    tot = 0.0
    for i in a:
        for j in b:
            if i == j:
                continue
            d = np.linalg.norm((X[j] - X[i])[None] + sh, axis=1).min()
            tot += float(cn_switch(np.array([d]))[0])
    return scale * tot


def _octa(d=7.0):
    return np.array([[d, 0, 0], [d / 3, 2 * np.sqrt(2) * d / 3, 0],
                     [-d / 3, np.sqrt(2) * d / 3, np.sqrt(6) * d / 3]])


@pytest.mark.parametrize("box", [_octa(), np.array([[6, 0, 0], [3, 5.6, 0], [-3, 2.8, 5.2]], float)])
def test_cn_triclinic_many_atoms_matches_image_search(box):
    rng = np.random.default_rng(3)
    n = 400
    X = (rng.random((n, 3)) * 1.3 - 0.15) @ box
    a, b = list(range(0, 30)), list(range(15, 400))
    f = Feature("w", "cn", "W3", terms=[[a, b, 0.7]])
    v = evaluate([f], X[None], box=box[None])[0, 0]
    assert v == pytest.approx(_brute(X, box, a, b, 0.7), rel=1e-10)


def test_cn_rejects_a_box_too_small_for_dmax():
    f = Feature("w", "cn", "W1", terms=[[[0], [1], 1.0]])
    with pytest.raises(ValueError, match="not supported"):
        evaluate([f], np.zeros((1, 2, 3)), box=np.diag([2.0] * 3)[None])


def test_cn_box_frame_count_must_match():
    f = Feature("w", "cn", "W1", terms=[[[0], [1], 1.0]])
    with pytest.raises(ValueError, match="frames"):
        evaluate([f], np.zeros((2, 2, 3)), box=np.tile(np.diag([3.0] * 3), (3, 1, 1)))
