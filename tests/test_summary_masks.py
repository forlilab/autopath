import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pytest

from autopath.metadynamics.summary.analysis import bands, fes_reweighted, hill_mask, fes_marginal
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.plots import fig_bias, fig_fes2d
from autopath.metadynamics.summary.report import report
from ._summary_fixtures import make_run

OFF = dict(min_count=0, min_bin_ess=0, min_density=0, bins=60, hill_mask_sigmas=np.inf, smooth_sigma=0)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in ("files", "run_dir", "params", "fes_masked", "fes_kept", "fmax_kcal", "bias_unsampled", "fes_bins",
                                                                  "binding_bias", "dg_bias_series", "estimator_gap",
                                                                  "bias", "estimator_spread", "estimator_gaps")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def test_report_numbers_unchanged_by_masks(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), cross=True)
    a = report(rd, out_dir=str(tmp_path / "a"))
    b = report(rd, out_dir=str(tmp_path / "b"), **OFF)
    assert json.dumps(_strip(a), sort_keys=True, default=str) == json.dumps(_strip(b), sort_keys=True, default=str)
    assert a["params"]["smooth_sigma"] == 2.0 and a["fmax_kcal"] is not None and "pooled" in a["fes_kept"]["map"]
    assert a["params"]["min_count"] == 0 and a["params"]["min_density"] == 0.05 and a["params"]["bins_per_hill"] == 4
    assert a["fes_bins"]["map"]["funnel_z"]["rule"] == "hill"
    assert a["params"]["hill_mask_sigmas"] == 2.0 and "pooled" in a["fes_masked"]["map"]


def test_sparse_bin_masked_and_colour_max_excludes_it(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), n_walkers=1))
    w = run.walkers[0]
    w.bias = w.ct.copy()
    w.cvs["funnel_z"] = np.concatenate([np.zeros(len(w.t) - 2), [2.3, 2.3]])
    w.cvs["learned_sw"] = np.concatenate([np.zeros(len(w.t) - 2), [3.9, 3.9]])
    off = fes_reweighted(run, "funnel_z", "learned_sw", burn=0.0, min_count=0, min_bin_ess=0, smooth_sigma=0)
    on = fes_reweighted(run, "funnel_z", "learned_sw", burn=0.0, smooth_sigma=0, min_count=5, min_bin_ess=3.0)
    assert np.isfinite(off.pooled[2]).sum() == 2 and np.isfinite(on.pooled[2]).sum() == 1
    assert on.n_masked == 1 and np.nanmin(on.pooled[2]) == 0.0
    _, axes, _, nums = fig_fes2d(run, bands(run), pair=("funnel_z", "learned_sw"), burn=0.0,
                                 smooth_sigma=0, min_count=5, min_bin_ess=3.0, out_path=str(tmp_path / "f.png"))
    assert nums["fes_masked"]["map"] == {"pooled": 1}
    prim = on.pooled[2][np.isfinite(on.pooled[2])] / 4.184               # cap from the c(t) map only
    assert nums["fmax"] == pytest.approx(np.percentile(prim, 99))
    assert np.nanmax(off.pooled[2]) / 4.184 > 1.5 and np.nanmax(on.pooled[2]) == 0.0     # masked bin leaves the map


def test_bin_ess_mask():
    from autopath.metadynamics.summary.analysis import _masked_hist
    x = np.array([[0.1]] * 10 + [[0.9]] * 10)
    w = np.array([1.0] * 10 + [100.0] + [1e-6] * 9)
    _, F, nm, kept = _masked_hist(x, w / w.sum(), [2], 2.5, 5, 3.0, 0.0)
    assert np.isnan(F[1]) and np.isfinite(F[0]) and nm == 1


def _half_grid_raw(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), n_walkers=1)
    run = load_run(rd)
    h = run.walkers[0].hills_raw[0]
    n = len(h["height_kj"])
    cv = np.zeros((n, 2))
    cv[:, 0] = np.linspace(-0.3, 0.8, n)          # left half of the funnel_z grid (-0.4..2.4)
    cv[:, 1] = np.linspace(-1.9, 3.9, n)
    h["cv"] = cv
    return run, [h]


def test_hill_mask_half_grid_and_marginal(tmp_path):
    run, raw = _half_grid_raw(tmp_path)
    m = hill_mask(raw, 2.0)
    xs = np.linspace(-0.4, 2.4, 101)
    assert m.shape == (101, 101) and not m[:, xs > 1.5].any() and m[:, xs < 0.5].any()
    F = np.where(m, 1.0, np.nan)
    marg = fes_marginal(F, 0, 2.5)
    assert np.isnan(marg[xs > 1.5]).all() and np.isfinite(marg[m.any(axis=0)]).all()
    assert np.isnan(fes_marginal(np.full((3, 3), np.nan), 0, 2.5)).all()


def test_hill_mask_until_and_inf(tmp_path):
    run, raw = _half_grid_raw(tmp_path)
    h = raw[0]
    early = {h["run_id"]: float(h["time_ns"][len(h["time_ns"]) // 4])}
    assert hill_mask(raw, 2.0, early).sum() < hill_mask(raw, 2.0).sum()
    assert hill_mask(raw, np.inf).all()


def test_hill_mask_periodic_wraps(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=("torsion",), monitors=(), n_walkers=1)
    h = load_run(rd).walkers[0].hills_raw[0]
    h["cv"] = np.full((len(h["height_kj"]), 1), np.pi - 0.01)
    m = hill_mask([h], 2.0)
    assert m[0] and m[-1] and not m[50]


def test_fig_bias_blanks_unsampled(tmp_path):
    run, raw = _half_grid_raw(tmp_path)
    run.walkers[0].hills_raw[0] = raw[0]
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"), n_snap=2)
    y = axes[0, 2].lines[-1].get_ydata()
    xs = np.linspace(-0.4, 2.4, 101)
    assert np.isnan(y[xs > 1.5]).all() and np.isfinite(y[xs < 0.5]).all()


def test_bias_map_unmasked_with_outline(tmp_path):
    run, raw = _half_grid_raw(tmp_path)
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"), n_snap=2)
    a = axes[0, 1]
    assert not any(np.isnan(np.ma.getdata(c.get_paths()[0].vertices)).any() for c in a.collections if c.get_paths())
    dashed = [c for c in a.collections if len(c.get_linestyle()) and c.get_linestyle()[0][1] is not None
              and c.get_linestyle()[0][1] and c.get_linewidths()[0] == pytest.approx(0.8)]
    assert len(dashed) == 1
    cb = [x for x in axes[0, 1].figure.axes if x.get_label() == "<colorbar>"][0]
    assert cb.get_ylim()[0] == pytest.approx(0.0, abs=1e-6)


def _spike(smooth, **kw):
    from autopath.metadynamics.summary.analysis import _masked_hist
    pts = np.full((1, 2), 0.5)
    pts = np.vstack([pts, [[0.0, 0.0], [1.0, 1.0]]])
    return _masked_hist(pts, np.ones(3) / 3, [11, 11], 2.5, 0, 0, smooth, **kw)


def test_smoothing_spreads_single_spike():
    _, F0, _, _ = _spike(0.0)
    _, F2, _, _ = _spike(1.0)
    assert np.isfinite(F0).sum() == 3 and np.isfinite(F2).sum() > 3 * 5
    assert np.isfinite(F2[5, 4]) and np.isnan(F0[5, 4])


def test_smooth_zero_reproduces_unsmoothed(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), cross=True))
    a = fes_reweighted(run, "funnel_z", "learned_sw", min_count=0, min_bin_ess=0, smooth_sigma=0).pooled[2]
    b = fes_reweighted(run, "funnel_z", "learned_sw", min_count=0, min_bin_ess=0, smooth_sigma=1e-9).pooled[2]
    np.testing.assert_allclose(a, b, atol=1e-6)
    s = fes_reweighted(run, "funnel_z", "learned_sw", min_count=0, min_bin_ess=0, smooth_sigma=2).pooled[2]
    assert np.isfinite(s).sum() > np.isfinite(a).sum()


def test_neighbourhood_mask_and_kept_fraction():
    from autopath.metadynamics.summary.analysis import _masked_hist
    x = np.array([[0.1]] * 30 + [[0.9]] * 2)
    w = np.ones(32) / 32
    _, F, nm, kept = _masked_hist(x, w, [10], 2.5, 5, 3.0, 0.0)
    assert np.isnan(F[9]) and kept == pytest.approx(0.5) and nm == 1


def test_fmax_is_99th_percentile_and_override(tmp_path):
    from autopath.metadynamics.summary.plots import _cap
    F = np.arange(101, dtype=float).reshape(1, -1)
    assert _cap([F]) == pytest.approx(np.percentile(F, 99))
    assert _cap([F], 7.5) == 7.5
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), cross=True))
    pair = ("funnel_z", "learned_sw")
    _, _, _, n1 = fig_fes2d(run, bands(run), pair=pair, out_path=str(tmp_path / "a.png"))
    F2 = fes_reweighted(run, *pair).pooled[2] / 4.184
    assert n1["fmax"] == pytest.approx(np.percentile(F2[np.isfinite(F2)], 99))
    f, axes, _, n2 = fig_fes2d(run, bands(run), pair=pair, fmax=4.0, out_path=str(tmp_path / "b.png"))
    assert n2["fmax"] == 4.0
    cb = [a for a in f.axes if a.get_label() == "<colorbar>"][0]
    assert cb.get_ylim()[1] == pytest.approx(4.0)


def test_neighbourhood_ess_equals_count_for_equal_weights_in_one_bin():
    from autopath.metadynamics.summary.analysis import _neighbourhood
    for shape in ((21,), (21, 21)):
        n = np.zeros(shape); c = (10,) * len(shape)
        n[c] = 12.0
        for sig in (0.0, 2.0):
            nb, ess = _neighbourhood(0.3 * n, n, 0.09 * n, sig)
            assert nb[c] == pytest.approx(12.0, rel=1e-3) and ess[c] == pytest.approx(12.0, rel=1e-3)


def test_neighbourhood_dominant_frame_ess_one():
    from autopath.metadynamics.summary.analysis import _neighbourhood
    n = np.zeros((21, 21)); n[10, 10] = 30
    p = np.full_like(n, 1e-8); p[10, 10] = 1.0
    n_nb, ess = _neighbourhood(p, n + 1, p ** 2, 2.0)
    assert ess[10, 10] == pytest.approx(1.0, abs=1e-3)


def test_neighbourhood_sigma_zero_is_raw_rule():
    from autopath.metadynamics.summary.analysis import _neighbourhood
    rng = np.random.default_rng(1)
    n = rng.integers(0, 9, (8, 8)).astype(float); w = rng.random((8, 8)) * n
    s2 = w ** 2 / np.maximum(n, 1) + 0.1 * n
    nb, ess = _neighbourhood(w, n, s2, 0.0)
    np.testing.assert_array_equal(nb, n)
    np.testing.assert_allclose(ess[s2 > 0], (w ** 2 / np.where(s2 > 0, s2, 1))[s2 > 0])


def _binned(tmp_path, **kw):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=("r_xy",), n_walkers=1))
    return run, fes_reweighted(run, "funnel_z", "r_xy", burn=0.0, **kw)


def test_bin_width_is_hill_over_bins_per_hill(tmp_path):
    run, c = _binned(tmp_path)
    hill = {s.name: s.sigma for s in run.cvs}["funnel_z"]
    xc, _, _ = c.pooled
    assert np.diff(xc) == pytest.approx(hill / 4)
    assert c.bin_info["funnel_z"]["rule"] == "hill"
    _, c8 = _binned(tmp_path / "b", bins_per_hill=8)
    assert np.diff(c8.pooled[0]) == pytest.approx(hill / 8)


def test_monitored_cv_uses_fallback_bins(tmp_path):
    _, c = _binned(tmp_path, bins_fallback=17)
    assert len(c.pooled[1]) == 17 and c.bin_info["r_xy"] == dict(width=pytest.approx(np.diff(c.pooled[1])[0]),
                                                                 n_bins=17, rule="fallback")


def test_explicit_bins_override(tmp_path):
    _, c = _binned(tmp_path, bins=9)
    assert F_shape(c) == (9, 9) and c.bin_info["funnel_z"]["rule"] == "explicit"
    _, c = _binned(tmp_path / "d", bins={"funnel_z": 7, "r_xy": 5})
    assert F_shape(c) == (7, 5)


def F_shape(c):
    return c.pooled[2].shape


def test_min_density_masks_empty_keeps_sparse():
    from autopath.metadynamics.summary.analysis import _masked_hist
    x = np.array([[0.0]] * 40 + [[1.0]] * 40 + [[0.5]])
    w = np.ones(81) / 81
    _, F, _, _ = _masked_hist(x, w, [21], 2.5, 0, 0, 1.0, 0.05)
    assert np.isfinite(F[0]) and np.isfinite(F[10]) and np.isnan(F[7])
    _, F0, _, _ = _masked_hist(x, w, [21], 2.5, 0, 0, 1.0, 0.0)
    assert np.isfinite(F0[7])
