import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pytest

from autopath.metadynamics.summary.analysis import bands, fes_bias
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.plots import MAX_ROWS, _bias_map, fig_fes2d
from autopath.metadynamics.summary.report import report
from ._summary_fixtures import make_run

CVS = ("funnel_z", "learned_sw")
PAIR = CVS


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _run(tmp_path, **kw):
    return load_run(make_run(str(tmp_path / "r"), cvs=CVS, monitors=kw.pop("monitors", ()), **kw))


def _levels(ax):
    return np.asarray(ax.collections[0].levels)


def _cbars(fig):
    return [a for a in fig.axes if a.get_label() == "<colorbar>"]


def test_shared_two_panels_share_limits_levels_colourbar(tmp_path):
    run = _run(tmp_path)
    fig, axes, skipped, nums = fig_fes2d(run, bands(run), out_path=str(tmp_path / "m.png"))
    assert axes.shape == (1, 2) and os.path.getsize(str(tmp_path / "m.png")) > 1000
    l, r = axes[0]
    assert l.get_xlim() == r.get_xlim() and l.get_ylim() == r.get_ylim()
    assert np.array_equal(_levels(l), _levels(r)) and _levels(l)[-1] == pytest.approx(nums["fmax"], abs=1.0)
    assert len(_cbars(fig)) == 1
    assert l.get_title("left").startswith("all: reweighted F (ESS ") and "smoothed σ=2 bins)" in l.get_title("left")
    assert r.get_title("left").startswith("all: bias-derived F = −") and r.get_title("left").endswith("·V (final bias)")
    assert not any("bias-derived" in s for s in skipped) and "funnel outline: no funnel" in skipped


def test_isolated_two_panels_per_walker(tmp_path):
    run = _run(tmp_path, layout="isolated", n_walkers=3)
    fig, axes, _, nums = fig_fes2d(run, bands(run))
    assert axes.shape == (3, 2)
    lv = [_levels(a) for a in axes.flat]
    assert all(np.array_equal(lv[0], x) for x in lv[1:])
    assert len({a.get_xlim() for a in axes.flat}) == 1 and len({a.get_ylim() for a in axes.flat}) == 1
    for i, rid in enumerate(("w0", "w1", "w2")):
        assert axes[i, 0].get_title("left").startswith(f"{rid}: reweighted F")
        assert axes[i, 1].get_title("left").startswith(f"{rid}: bias-derived F")
    assert set(nums["fes_ess"]["map"]) == {"w0", "w1", "w2"} and len(_cbars(fig)) == 1


def test_isolated_rows_capped(tmp_path):
    run = _run(tmp_path, layout="isolated", n_walkers=MAX_ROWS + 1)
    _, axes, skipped, _ = fig_fes2d(run, bands(run))
    assert axes.shape == (MAX_ROWS, 2) and any("1 of 7 isolated walkers not drawn" in s for s in skipped)


def test_non_biased_pair_skips_bias_panel(tmp_path):
    run = _run(tmp_path, monitors=("r_xy",))
    _, axes, skipped, nums = fig_fes2d(run, bands(run), pair=("funnel_z", "r_xy"))
    assert "bias-derived map: pair is not the biased CVs" in skipped
    assert not axes[0, 1].axison and not axes[0, 1].collections and axes[0, 0].collections
    assert nums["fmax"] is not None


def test_one_d_bias_skips_figure(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")))
    fig, axes, skipped, nums = fig_fes2d(run, bands(run))
    assert fig is None and axes is None and any("2D maps" in s and "needs 2" in s for s in skipped)
    assert nums["fmax"] is None and nums["fes_ess"] is None


@pytest.mark.parametrize("pair", [PAIR, PAIR[::-1]])
def test_bias_map_orientation_off_centre_hill(tmp_path, pair):
    run = _run(tmp_path, n_walkers=1)
    h = run.walkers[0].hills_raw[0]
    x0, y0 = 1.7, 3.0                                        # off-centre in both CV grids
    h["cv"] = np.tile([x0, y0], (len(h["height_kj"]), 1))
    bm, why = _bias_map(run, pair, 2.0)
    assert why is None
    gx, gy, F = bm["pooled"]
    iy, ix = np.unravel_index(np.nanargmin(F), F.shape)
    want = (x0, y0) if pair == PAIR else (y0, x0)
    assert gx[ix] == pytest.approx(want[0], abs=0.1) and gy[iy] == pytest.approx(want[1], abs=0.1)
    assert np.nanmin(F) == pytest.approx(0.0)
    # drawn contours: lowest band is centred on the hill
    _, axes, _, _ = fig_fes2d(run, bands(run), pair=pair)
    v = axes[0, 1].collections[0].get_paths()[0].vertices
    assert v[:, 0].mean() == pytest.approx(want[0], abs=0.1) and v[:, 1].mean() == pytest.approx(want[1], abs=0.15)


def test_bias_map_is_scaled_final_bias_in_kcal(tmp_path):
    run = _run(tmp_path)
    bm, _ = _bias_map(run, PAIR, 2.0)
    grids, F = fes_bias(run).pooled
    np.testing.assert_allclose(bm["pooled"][2], F / 4.184)
    assert bm["pooled"][2].shape == (len(grids[1]), len(grids[0]))


def test_numbers_keys(tmp_path):
    run = _run(tmp_path)
    _, _, _, nums = fig_fes2d(run, bands(run), fmax=3.0)
    assert nums["fmax"] == 3.0 and set(nums["fes_bins"]["map"]) == set(CVS)
    assert "pooled" in nums["fes_masked"]["map"] and "pooled" in nums["fes_kept"]["map"]
    assert nums["fes_ess"]["map"] > 0


def test_report_writes_fes2d(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=CVS, monitors=(), cross=True)
    s = report(rd, out_dir=str(tmp_path / "o"), step_ns=0.5)
    for k in ("cvs", "bias", "fe", "fes2d"):
        assert os.path.getsize(s["files"][k]) > 1000
    assert s["files"]["fes2d"].endswith("metad_fes-2d.png")
    d = json.load(open(s["files"]["summary"]))
    assert d["fmax_kcal"] is not None and "pooled" in d["fes_kept"]["map"] and "w" in d["fes_kept"]
    assert set(d["fes_ess"]) == {"map", "w"} and set(d["fes_bins"]) == {"map", "w"}


def test_report_one_d_bias_has_no_fes2d(tmp_path):
    s = report(make_run(str(tmp_path / "r")), out_dir=str(tmp_path / "o"), step_ns=0.5)
    assert s["files"]["fes2d"] is None and any("2D maps" in x for x in s["skipped_panels"])
    assert not os.path.exists(str(tmp_path / "o" / "metad_fes-2d.png"))
