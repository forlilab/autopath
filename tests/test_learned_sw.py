# tests/test_learned_sw.py
"""s_w: TS-component choice, sign, z-correlation check and z-residualised refit."""
import numpy as np
import pytest

from autopath.metadynamics.learned.reduce import HLDA
from autopath.metadynamics.learned.sw import (SW_CORR_MAX, choose_component, fit_sw, fit_z_splines,
                                              z_spline_values)

ORDER = ["B", "TS", "U"]


def _three_class(n=400, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.array(ORDER, dtype=object), n)
    x1 = np.select([lab == "B", lab == "U"], [-4.0, 4.0], 0.0) + rng.normal(0, 1, 3 * n)     # B vs U
    x2 = np.where(lab == "TS", 2.0, 0.0) + rng.normal(0, 1, 3 * n)                          # TS vs rest
    x3 = rng.normal(0, 1, 3 * n)
    return np.c_[x1, x2, x3], lab


class _Eye:
    """Reducer stub: component k = feature k."""
    def fit(self, X, labels, order):
        return np.eye(X.shape[1])[:, :2]


def _z_data(n=3000, bump=0.1, z_coupled=True, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, 1.0, n)
    lab = np.full(n, "", dtype=object)
    lab[z < 0.3] = "B"
    lab[z > 0.7] = "U"
    mid = np.flatnonzero((z >= 0.3) & (z <= 0.7))
    lab[mid[::2]] = "TS"
    ts = (lab == "TS").astype(float)
    base = z if z_coupled else rng.normal(0.0, 0.3, n)
    f0 = base + bump * ts + rng.normal(0.0, 0.01, n)
    return np.c_[f0, rng.normal(0.0, 1.0, n)], z, lab


def test_choose_component_picks_the_ts_separating_axis():
    X, lab = _three_class()
    W = HLDA(n_components=2).fit(X, lab, ORDER)
    assert abs(W[0, 0]) > 0.9                              # HLDA's first axis is the B/U one
    k, ratios = choose_component(X @ W, lab, ORDER)
    assert k == 1 and ratios[1] > 5 * ratios[0]
    assert abs(W[1, k]) > 0.9


def test_fit_sw_orients_ts_above_b():
    X, lab = _three_class()
    z = np.random.default_rng(1).uniform(0, 1, len(lab))   # unrelated to s_w
    out = fit_sw(X, z, lab, ORDER)
    s = out["X"] @ out["w"]
    assert out["component"] == 1 and not out["residualised"]
    assert s[lab == "B"].mean() < s[lab == "TS"].mean()
    assert out["weights"].shape == (3, 2) and np.isnan(out["corr_resid"]) and out["tables"].shape == (0, 0)


def test_fit_sw_ignores_unlabelled_frames():
    X, lab = _three_class()
    z = np.random.default_rng(1).uniform(0, 1, len(lab))
    ref = fit_sw(X, z, lab, ORDER)
    X2 = np.r_[X, np.full((50, 3), 100.0)]
    lab2 = np.r_[lab, np.full(50, "", dtype=object)]
    z2 = np.r_[z, np.full(50, 0.5)]
    out = fit_sw(X2, z2, lab2, ORDER)
    assert out["component"] == ref["component"]
    assert np.allclose(out["w"], ref["w"]) and out["corr_plain"] == pytest.approx(ref["corr_plain"])


def test_strong_z_coupling_triggers_residualisation():
    X, z, lab = _z_data()
    out = fit_sw(X, z, lab, ORDER, reducer=_Eye())
    assert abs(out["corr_plain"]) > SW_CORR_MAX and out["residualised"]
    assert out["plain_component"] == 0 and out["component"] == 0
    assert abs(out["corr_resid"]) < 0.1
    m = lab != ""
    s = out["X"] @ out["w"]
    assert np.corrcoef(s[m], z[m])[0, 1] == pytest.approx(out["corr_resid"])
    assert out["tables"].shape == (2, 200) and out["z_min"] == z.min() and out["z_max"] == z.max()
    assert s[lab == "B"].mean() < s[lab == "TS"].mean()


def test_weak_z_coupling_keeps_the_plain_fit():
    X, z, lab = _z_data(bump=1.0, z_coupled=False)
    out = fit_sw(X, z, lab, ORDER, reducer=_Eye())
    assert abs(out["corr_plain"]) <= SW_CORR_MAX and not out["residualised"]
    assert np.isnan(out["corr_resid"]) and out["tables"].shape == (0, 0)
    assert np.array_equal(out["X"], X)


def test_spline_residuals_are_uncorrelated_with_z():
    rng = np.random.default_rng(2)
    z = rng.uniform(-0.5, 2.0, 5000)
    Y = np.c_[z, np.sin(3 * z), z ** 3] + rng.normal(0, 0.05, (5000, 3))
    lo, hi, tables = fit_z_splines(Y, z)
    R = Y - z_spline_values(tables, lo, hi, z)
    assert tables.shape == (3, 200) and (lo, hi) == (z.min(), z.max())
    for k in range(3):
        assert abs(np.corrcoef(R[:, k], z)[0, 1]) < 0.01
        assert R[:, k].std() < 0.06                        # z-dependence removed, noise remains


def test_spline_values_clamp_outside_the_table_range():
    tables = np.vstack([np.linspace(0.0, 1.0, 50), np.linspace(2.0, -2.0, 50)])
    v = z_spline_values(tables, 0.0, 1.0, np.array([-3.0, 0.0, 1.0, 5.0]))
    assert v.shape == (4, 2)
    assert v[0] == pytest.approx(v[1]) and v[3] == pytest.approx(v[2])
    assert v[2] == pytest.approx([1.0, -2.0])


def test_too_few_distinct_z_values_is_an_error():
    z = np.repeat([0.0, 1.0], 100)
    with pytest.raises(ValueError, match="distinct"):
        fit_z_splines(np.c_[z], z)


def test_fit_sw_keeps_the_plain_fit_after_a_refit():
    X, z, lab = _z_data()
    out = fit_sw(X, z, lab, ORDER, reducer=_Eye())
    assert out["residualised"]
    m = lab != ""
    k = out["plain_component"]
    s = X @ out["plain_w"]
    assert s[lab == "B"].mean() < s[lab == "TS"].mean()
    assert out["plain_w"].shape == (2,) and out["plain_fisher_ts"].shape == (2,)
    assert out["corr_plain"] == pytest.approx(np.corrcoef(s[m], z[m])[0, 1])
    assert out["plain_fisher_ts"][k] == pytest.approx(choose_component(X @ np.eye(2), lab, ORDER)[1][k])
    assert not np.allclose(out["fisher_ts"], out["plain_fisher_ts"])


def test_constant_sw_over_labelled_frames_is_an_error():
    X, z, lab = _z_data()
    X = X.copy()
    X[:] = 1.0
    with pytest.raises(ValueError, match="constant"):
        fit_sw(X, z, lab, ORDER, reducer=_Eye())


def test_non_finite_z_is_an_error():
    X, z, lab = _z_data()
    z = z.copy()
    z[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_z_splines(X, z)
    with pytest.raises(ValueError, match="non-finite"):
        fit_sw(X, z, lab, ORDER, reducer=_Eye())


def test_unlabelled_frames_enter_the_pooled_z_spline():
    X, z, lab = _z_data()
    ref = fit_sw(X, z, lab, ORDER, reducer=_Eye())
    X2 = X.copy()
    un = lab == ""
    assert un.any() and ref["residualised"]
    X2[un, 1] += 5.0 * np.sin(6 * z[un])                   # z-dependence only in unlabelled frames
    out = fit_sw(X2, z, lab, ORDER, reducer=_Eye())
    assert out["residualised"]
    assert not np.allclose(out["tables"][1], ref["tables"][1], atol=0.1)
