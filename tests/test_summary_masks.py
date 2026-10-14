import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pytest

from autopath.metadynamics.summary.analysis import bands, fes_reweighted, hill_mask, fes_marginal
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.plots import fig_bias, fig_fe
from autopath.metadynamics.summary.report import report
from ._summary_fixtures import make_run

OFF = dict(min_count=0, min_bin_ess=0, hill_mask_sigmas=np.inf)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in ("files", "run_dir", "params", "fes_masked", "bias_unsampled")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def test_report_numbers_unchanged_by_masks(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), cross=True)
    a = report(rd, out_dir=str(tmp_path / "a"))
    b = report(rd, out_dir=str(tmp_path / "b"), **OFF)
    assert json.dumps(_strip(a), sort_keys=True, default=str) == json.dumps(_strip(b), sort_keys=True, default=str)
    assert a["params"]["min_count"] == 5 and a["params"]["min_bin_ess"] == 3.0
    assert a["params"]["hill_mask_sigmas"] == 2.0 and "pooled" in a["fes_masked"]["map"]


def test_sparse_bin_masked_and_colour_max_excludes_it(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), n_walkers=1))
    w = run.walkers[0]
    w.bias = w.ct.copy()
    w.cvs["funnel_z"] = np.concatenate([np.zeros(len(w.t) - 2), [2.3, 2.3]])
    w.cvs["learned_sw"] = np.concatenate([np.zeros(len(w.t) - 2), [3.9, 3.9]])
    off = fes_reweighted(run, "funnel_z", "learned_sw", burn=0.0, **dict(min_count=0, min_bin_ess=0))
    on = fes_reweighted(run, "funnel_z", "learned_sw", burn=0.0)
    assert np.isfinite(off.pooled[2]).sum() == 2 and np.isfinite(on.pooled[2]).sum() == 1
    assert on.n_masked == 1 and np.nanmin(on.pooled[2]) == 0.0
    _, axes, _, nums = fig_fe(run, bands(run), pair=("funnel_z", "learned_sw"), burn=0.0,
                              out_path=str(tmp_path / "f.png"))
    assert nums["fes_masked"]["map"] == {"pooled": 1}
    cb = [a for a in axes["map"][0].figure.axes if a.get_label() == "<colorbar>"][0]
    assert np.nanmax(off.pooled[2]) / 4.184 > 1.5 and cb.get_ylim()[1] <= 1.0
    assert np.nanmax(on.pooled[2]) == 0.0


def test_bin_ess_mask():
    from autopath.metadynamics.summary.analysis import _masked_hist
    x = np.array([[0.1]] * 10 + [[0.9]] * 10)
    w = np.array([1.0] * 10 + [100.0] + [1e-6] * 9)
    _, F, nm = _masked_hist(x, w / w.sum(), [2], 2.5, 5, 3.0)
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
