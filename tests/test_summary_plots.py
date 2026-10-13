import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pytest

from autopath.metadynamics.Hills import bias_from_hills
from autopath.metadynamics.summary.analysis import bands
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.plots import fig_cvs, fig_bias, _until
from ._summary_fixtures import make_run

@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


MODES = [dict(), dict(layout="isolated", n_walkers=3), dict(cvs=("funnel_z", "learned_sw"), monitors=()),
         dict(cvs=("funnel_z", "learned_sw"), monitors=(), layout="isolated", n_walkers=3),
         dict(tempering="wt+tt"), dict(tempering="tt"), dict(n_walkers=8), dict(extend=True),
         dict(n_walkers=1), dict(cvs=("dist",), monitors=("funnel_z",))]


@pytest.mark.parametrize("kw", MODES)
def test_fig_cvs_and_bias_write(tmp_path, kw):
    run = load_run(make_run(str(tmp_path / "r"), **kw))
    for fn, args in ((fig_cvs, (run, bands(run))), (fig_bias, (run,))):
        p = str(tmp_path / f"{fn.__name__}.png")
        fig, axes, skipped = fn(*args, out_path=p)
        assert os.path.getsize(p) > 1000 and isinstance(skipped, list)


def test_tt_panel_skipped_without_vstar(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), tempering="wt"))
    _, _, skipped = fig_bias(run, out_path=str(tmp_path / "b.png"))
    assert any("V*" in s for s in skipped)


def test_tt_panel_drawn(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), tempering="wt+tt"))
    _, axes, skipped = fig_bias(run, out_path=str(tmp_path / "b.png"))
    assert not any("V*" in s for s in skipped) and len(axes[0, 3].lines) == len(run.walkers)


def test_overlaid_when_many_walkers(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=8))
    _, axes, _ = fig_cvs(run, bands(run), out_path=str(tmp_path / "c.png"))
    assert axes.shape[0] == 1


def test_one_row_per_walker(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=3))
    _, axes, _ = fig_cvs(run, bands(run), out_path=str(tmp_path / "c.png"))
    assert axes.shape[0] == 3


def test_isolated_bias_drawn_per_walker(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"), n_snap=4)
    assert len(axes[0, 2].lines) == 3 * 4


def test_isolated_2d_one_map_per_walker(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3,
                            cvs=("funnel_z", "learned_sw"), monitors=()))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"))
    assert all(len(axes[g, 2].images) == 1 for g in range(3))


def test_bias_growth_with_extension(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), extend=True, n_walkers=1))
    w = run.walkers[0]
    assert len(w.segments) == 1
    until = _until(run.walkers, float(w.hill_t[-1]))
    assert set(until) == {"w0", "w0_x2"}
    np.testing.assert_allclose(bias_from_hills(w.hills_raw, until=until), bias_from_hills(w.hills_raw))
    # a time inside the first segment keeps no extension hills
    half = _until(run.walkers, w.segments[0] / 2)
    assert float(half["w0_x2"]) < 0
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"), n_snap=3)
    last = axes[0, 2].lines[-1].get_ydata()
    np.testing.assert_allclose(last, np.asarray(bias_from_hills(w.hills_raw)).ravel())


def _set_hill_cv(rd, rid, value):
    f = os.path.join(rd, f"HILLS_{rid}.npz")
    with np.load(f) as d:
        rec = {k: d[k] for k in d.files}
    rec["cv"] = np.full_like(rec["cv"], value)
    np.savez(f, **rec)


def test_isolated_clamp_per_walker(tmp_path):
    from autopath.metadynamics.summary.analysis import clamp_fractions
    rd = make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2)
    _set_hill_cv(rd, "w0", 2.4)
    _set_hill_cv(rd, "w1", 1.0)
    run = load_run(rd)
    assert clamp_fractions(run) == {"funnel_z": {"w0": 1.0, "w1": 0.0}}
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"))
    assert len(axes[0, 1].patches) >= 2 and "w0 1.000" in axes[0, 1].texts[0].get_text()


def test_shared_clamp_pooled(tmp_path):
    from autopath.metadynamics.summary.analysis import clamp_fractions
    rd = make_run(str(tmp_path / "r"), n_walkers=2)
    _set_hill_cv(rd, "w0", 2.4)
    _set_hill_cv(rd, "w1", 1.0)
    assert clamp_fractions(load_run(rd)) == {"funnel_z": {"pooled": 0.5}}


def test_vstar_length_mismatch_skipped(tmp_path):
    rd = make_run(str(tmp_path / "r"), tempering="wt+tt", n_walkers=1)
    f = os.path.join(rd, "HILLS_w0.npz")
    with np.load(f) as d:
        rec = {k: d[k] for k in d.files}
    rec["vstar_kj"] = rec["vstar_kj"][:-3]
    np.savez(f, **rec)
    _, axes, skipped = fig_bias(load_run(rd), out_path=str(tmp_path / "b.png"))
    assert "V* length != hill count: not supported" in skipped and not axes[0, 3].lines


def test_2d_marginals_per_walker(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2,
                            cvs=("funnel_z", "learned_sw"), monitors=()))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"), n_snap=4)
    assert all(len(axes[g, k].lines) == 4 for g in range(2) for k in (3, 4))
