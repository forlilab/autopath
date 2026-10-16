import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pytest
from openmm import unit

from autopath.metadynamics.Hills import bias_from_hills
from autopath.metadynamics.summary.analysis import bands
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.plots import fig_cvs, fig_bias, fig_fe, _until
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
    assert not any("V*" in s for s in skipped) and len(axes[0, 2].lines) == len(run.walkers)


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
    assert len(axes[0, 1].lines) == 3 * 4


def test_isolated_2d_one_map_per_walker(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3,
                            cvs=("funnel_z", "learned_sw"), monitors=()))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"))
    assert all(len(axes[g, 1].collections) >= 2 for g in range(3))


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
    last = axes[0, 1].lines[-1].get_ydata()
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
    assert axes.shape[1] == 2                                   # no deposition column


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
    assert "V* length != hill count: not supported" in skipped and not axes[0, 2].lines


def test_2d_marginals_per_walker(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2,
                            cvs=("funnel_z", "learned_sw"), monitors=()))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"), n_snap=4)
    assert all(len(axes[g, k].lines) == 4 for g in range(2) for k in (2, 3))


FP = {"z_cc": 12.0 * unit.angstrom, "z_max": 24.0 * unit.angstrom, "z_low": -1.5 * unit.angstrom,
      "R_cylinder": 1.0 * unit.angstrom, "alpha": 31.5 * unit.degrees}


@pytest.mark.parametrize("kw", MODES)
def test_fig_fe_writes(tmp_path, kw):
    run = load_run(make_run(str(tmp_path / "r"), **kw), funnel=FP)
    fig, axes, skipped, nums = fig_fe(run, bands(run), out_path=str(tmp_path / "fe.png"), step_ns=0.5)
    assert os.path.getsize(str(tmp_path / "fe.png")) > 1000 and "binding" in nums


def test_fig_fe_isolated_never_pools(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3), funnel=FP)
    _, _, _, nums = fig_fe(run, bands(run), out_path=str(tmp_path / "fe.png"), step_ns=0.5)
    assert nums["binding"]["pooled"] is None and len(nums["binding"]["per_walker"]) == 3
    assert set(nums["blocks"]) == {"w0", "w1", "w2"}


def test_fig_fe_without_reweight_skips(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), reweight=False), funnel=FP)
    _, _, skipped, nums = fig_fe(run, bands(run), out_path=str(tmp_path / "fe.png"))
    assert any("no REWEIGHT data" in s for s in skipped) and nums["binding"] is None


def test_fig_fe_short_series_recorded(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    _, _, skipped, nums = fig_fe(run, bands(run), step_ns=1.0)
    assert any("< 12 points" in s for s in skipped) and nums["blocks"] is None


def test_fig_fe_blocks_when_long_enough(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    _, _, skipped, nums = fig_fe(run, bands(run), step_ns=0.1)
    assert not any("< 12" in s for s in skipped)
    assert nums["blocks"]["unit"] == "kJ/mol" and "n_valid_blocks" in nums["blocks"]


def _legend_texts(ax):
    return [x.get_text() for x in ax.get_legend().get_texts()]


def test_fig_fe_no_funnel_says_band_dF(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")))
    _, axes, skipped, nums = fig_fe(run, bands(run), step_ns=0.1)
    assert nums["binding"]["pooled"]["kind"] == "band_dF"
    txt = " ".join(_legend_texts(axes["w"]) + [axes["dg"].get_ylabel(), axes["w"].get_title()])
    assert "band ΔF" in txt and "ΔG°" not in txt
    assert "funnel outline: no funnel" in skipped


def test_fig_fe_shown_value_equals_binding(tmp_path):
    from autopath.metadynamics.summary.analysis import binding
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3), funnel=FP)
    _, axes, _, nums = fig_fe(run, bands(run), step_ns=0.1)
    bd = binding(run, bands(run))
    for rid, v in bd.per_walker.items():
        np.testing.assert_equal(nums["binding"]["per_walker"][rid]["value"], v["value"])
        want = f"{v['value'] / 4.184:.2f}" if np.isfinite(v["value"]) else "n/a"
        assert any(s.startswith(f"{rid}: ΔG° {want}") for s in _legend_texts(axes["w"]))
    np.testing.assert_equal(nums["binding"]["mean"], bd.mean) and axes["bars"] is not None


def test_fig_fe_2d_without_rxy_records_outline_skip(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()), funnel=FP)
    _, _, skipped, _ = fig_fe(run, bands(run), step_ns=0.5)
    assert "funnel outline: needs the (funnel_z, r_xy) pair" in skipped
    assert "default pair: r_xy not logged" in skipped


def test_fig_fe_rxy_monitor_default_pair_draws_outline(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), monitors=("funnel_z", "r_xy")), funnel=FP)
    _, axes, skipped, _ = fig_fe(run, bands(run), step_ns=0.5)
    assert not any("funnel outline" in s or "default pair" in s for s in skipped)
    assert "funnel_z" in axes["map"][0].get_xlabel() and "r_xy" in axes["map"][0].get_ylabel()
    assert any(l.get_linestyle() == "--" for l in axes["map"][0].get_lines())


def test_fig_fe_wt_tt_numbers(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), tempering="wt+tt"), funnel=FP)
    _, _, _, nums = fig_fe(run, bands(run), step_ns=0.1)
    assert nums["binding"]["pooled"]["kind"] == "dG0" and nums["blocks"]["n_valid_blocks"] >= 1
    assert len(nums["blocks"]["block_dg"]) == 10


def test_fig_fe_uses_given_numbers(tmp_path):
    from autopath.metadynamics.summary.analysis import fe_numbers
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    b = bands(run)
    nums = fe_numbers(run, b, step_ns=0.1, n_boot=20)
    nums["binding"].pooled["value"] = -41.84                       # marker: -10.00 kcal/mol
    _, axes, skipped, out = fig_fe(run, b, numbers=nums)
    assert any("ΔG° -10.00" in t for t in _legend_texts(axes["w"]))
    assert out["blocks"] is nums["blocks"] and out["series"]["pooled"]["t"][-1] == nums["series"].pooled["t"][-1]


def test_fig_fe_text_box_shows_ess(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2), funnel=FP)
    _, axes, _, nums = fig_fe(run, bands(run), step_ns=0.1, show_numbers=True)
    txt = " ".join(t.get_text() for t in axes["text"].texts)
    for rid, v in nums["binding"]["per_walker"].items():
        assert f"{rid}:" in txt and "final ΔG°" in txt and f"ESS {v['ess']:.0f}" in txt
    assert "min block ESS" in txt and set(nums["fes_ess"]["map"]) == {"w0", "w1"}


def test_fig_fe_accepts_axes(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    fig, ax = plt.subplots(1, 4)
    given = dict(ct=ax[0], map=ax[1], w=ax[2], dg=ax[3])
    f, axes, _, _ = fig_fe(run, bands(run), step_ns=0.5, axes=given)
    assert f is fig and axes["w"] is ax[2] and ax[2].lines and ax[3].lines
    with pytest.raises(ValueError, match="not supported"):
        fig_fe(run, bands(run), step_ns=0.5, axes=dict(ct=ax[0], map=ax[1], w=ax[2]))


def test_fig_fe_isolated_axes_without_bars(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2), funnel=FP)
    fig, ax = plt.subplots(2, 4)
    _, _, skipped, _ = fig_fe(run, bands(run), step_ns=0.5,
                              axes=dict(ct=list(ax[:, 0]), map=list(ax[:, 1]), w=ax[0, 2], dg=ax[0, 3]))
    assert "per-walker bars: no axes given" in skipped


def test_fig_fe_all_nan_no_warnings(tmp_path):
    import warnings
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()), funnel=FP)
    for w in run.walkers:
        w.cvs["funnel_z"][:] = 0.1                                 # never reaches bulk: every ΔG° NaN
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        _, _, _, nums = fig_fe(run, bands(run), step_ns=0.1, out_path=str(tmp_path / "fe.png"))
    assert np.isnan(nums["binding"]["pooled"]["value"]) and nums["blocks"]["n_valid_blocks"] == 0


def test_overlay_shows_recrossings(tmp_path):
    from autopath.metadynamics.summary.analysis import recrossings
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=8))
    b = bands(run)
    _, axes, _ = fig_cvs(run, b, out_path=str(tmp_path / "c.png"))
    txt = " ".join(t.get_text() for t in axes[0, 0].texts)
    for w in run.walkers:
        assert f"{w.rid}: {recrossings(w, 'funnel_z', b['funnel_z'])['n_cycles']} cyc" in txt


def test_outline_starts_at_shared_z_low(tmp_path):
    from autopath.metadynamics.summary.analysis import funnel_z_low
    from autopath.metadynamics.summary.plots import _funnel_outline
    fp = {k: v for k, v in FP.items() if k != "z_low"}
    run = load_run(make_run(str(tmp_path / "r")), funnel=fp)
    z, _ = _funnel_outline(run, 10.0)
    assert z[0] == pytest.approx(run.cvs[0].grid_min) == pytest.approx(funnel_z_low(run, 10.0) / 10.0)
    assert bands(run)["funnel_z"].bound[0] == pytest.approx(z[0])


def _stars_on(ax):
    return [l for l in ax.get_lines() if l.get_marker() == "*"]


def _drop_basins(rd):
    for f in os.listdir(rd):
        if f.startswith("HILLS_") and f.endswith(".npz"):
            with np.load(os.path.join(rd, f)) as d:
                rec = {k: d[k] for k in d.files if k != "tt_basins"}
            np.savez(os.path.join(rd, f), **rec)


def _tt_basins(rd, pts):
    for f in os.listdir(rd):
        if f.startswith("HILLS_") and f.endswith(".npz"):
            with np.load(os.path.join(rd, f)) as d:
                rec = {k: d[k] for k in d.files}
            rec["tt_basins"] = np.asarray(pts, float)
            np.savez(os.path.join(rd, f), **rec)


def test_bias_figure_has_no_deposition_column(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"))
    assert axes.shape == (1, 4)
    assert not any("deposition on" in a.get_title("left") for a in axes.flat)
    assert "F marginal on funnel_z (from bias)" in axes[0, 2].get_title("left")
    assert axes[0, 2].get_ylabel() == "F (kcal/mol)" and axes[0, 1].xaxis.get_gridlines()[0].get_visible()


def test_bias_marginal_min_zero_and_matches_helper(tmp_path):
    from autopath.metadynamics.summary.analysis import fes_marginal, bias_fes_scale, KT_PER_K
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"), n_snap=3, hill_mask_sigmas=np.inf)
    y = axes[0, 2].lines[-1].get_ydata()
    raw = [h for w in run.walkers for h in w.hills_raw]
    F = -bias_fes_scale(run) * np.asarray(bias_from_hills(raw))
    np.testing.assert_allclose(y, fes_marginal(F, 0, KT_PER_K * 300.0) / 4.184)
    assert y.min() == pytest.approx(0.0)


def test_fes_marginal_separable():
    from autopath.metadynamics.summary.analysis import fes_marginal
    x, y = np.linspace(0, 1, 40), np.linspace(-1, 2, 60)
    f, g = 3.0 * np.sin(4 * x) + 5.0, (y - 0.5) ** 2 * 2.0
    F = f[None, :] + g[:, None]                     # F[iy, ix]
    m = fes_marginal(F, 0, 2.5)
    np.testing.assert_allclose(m, f - f.min(), atol=1e-9)
    m1 = fes_marginal(F, 1, 2.5)
    np.testing.assert_allclose(m1, g - g.min(), atol=1e-9)


def test_marginal_integrates_boltzmann():
    from autopath.metadynamics.summary.analysis import fes_marginal
    F = np.array([[0.0, 2.0], [1.0, 3.0]])
    kt = 1.7
    want = -kt * np.log(np.exp(-F / kt).sum(axis=0))
    np.testing.assert_allclose(fes_marginal(F, 0, kt), want - want.min())


def test_hills_record_basins_loaded(tmp_path):
    from autopath.metadynamics.summary.analysis import tt_basins
    rd = make_run(str(tmp_path / "r"), tempering="tt")
    _tt_basins(rd, [[0.3], [1.5]])
    np.testing.assert_allclose(tt_basins(load_run(rd)), [[0.3], [1.5]])
    assert tt_basins(load_run(make_run(str(tmp_path / "w"), tempering="wt"))) is None


def test_fig_bias_stars_2d(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), tempering="wt+tt")
    _tt_basins(rd, [[0.2, 0.0], [1.5, 3.0]])
    _, axes, skipped = fig_bias(load_run(rd), out_path=str(tmp_path / "b.png"))
    for k in (1, 2, 3):
        st = _stars_on(axes[0, k])
        assert len(st) == 1 and len(st[0].get_xdata()) == 2 and st[0].get_markersize() == 14
    assert not any("TT basins" in s for s in skipped)


def test_fig_bias_basins_not_recorded(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), tempering="wt+tt")
    _drop_basins(rd)
    _, axes, skipped = fig_bias(load_run(rd), out_path=str(tmp_path / "b.png"))
    assert "TT basins: not recorded" in skipped and not _stars_on(axes[0, 1])


def test_non_tt_has_no_stars_or_skip(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()), funnel=FP)
    _, axes, skipped = fig_bias(run, out_path=str(tmp_path / "b.png"))
    assert not _stars_on(axes[0, 1]) and not any("TT basins" in s for s in skipped)
    _, axes, skipped, _ = fig_fe(run, bands(run), step_ns=0.5)
    assert not _stars_on(axes["map"][0]) and not _stars_on(axes["w"]) and not any("TT basins" in s for s in skipped)


def test_fig_fe_stars_and_no_band_spans(tmp_path):
    rd = make_run(str(tmp_path / "r"), tempering="wt+tt", monitors=("funnel_z", "r_xy"))
    _tt_basins(rd, [[0.2], [1.5]])
    run = load_run(rd, funnel=FP)
    _, axes, skipped, _ = fig_fe(run, bands(run), step_ns=0.5)
    m = axes["map"][0]
    assert not any(isinstance(p, plt.Polygon) for p in m.patches)    # axvspan/axhspan make Polygons
    assert not m.patches
    assert len(_stars_on(m)) == 0 and any("TT basins" in s for s in skipped)   # r_xy is not a biased CV
    assert len(_stars_on(axes["w"])) == 1 and len(_stars_on(axes["w"])[0].get_xdata()) == 2


def test_fig_fe_map_stars_biased_pair(tmp_path):
    rd = make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=(), tempering="wt+tt")
    _tt_basins(rd, [[0.2, 0.0], [1.5, 3.0]])
    run = load_run(rd, funnel=FP)
    _, axes, skipped, _ = fig_fe(run, bands(run), pair=("funnel_z", "learned_sw"), step_ns=0.5)
    st = _stars_on(axes["map"][0])
    assert len(st) == 1 and list(st[0].get_ydata()) == [0.0, 3.0]
    assert not axes["map"][0].patches


def test_fig_fe_basins_not_recorded(tmp_path):
    rd = make_run(str(tmp_path / "r"), tempering="tt")
    _drop_basins(rd)
    run = load_run(rd, funnel=FP)
    _, axes, skipped, _ = fig_fe(run, bands(run), step_ns=0.5)
    assert "TT basins: not recorded" in skipped and not _stars_on(axes["w"])


def test_fig_fe_ct_panel(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=2, extend=True), funnel=FP)
    _, axes, _, _ = fig_fe(run, bands(run), step_ns=0.5)
    a = axes["ct"][0]
    assert len(a.lines) == 2 * 2 + 2 * len(run.walkers[0].segments)    # c(t), V per walker + boundaries
    np.testing.assert_allclose([l for l in a.lines if l.get_label() == f"{run.walkers[0].rid} c(t)"][0].get_ydata(), run.walkers[0].ct)
    assert "c(t) bold" in a.get_title("left")
    iso = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=2), funnel=FP)
    _, axes, _, _ = fig_fe(iso, bands(iso), step_ns=0.5)
    assert len(axes["ct"]) == 2 and all(len(a.lines) >= 2 for a in axes["ct"])


def test_fig_fe_maps_share_limits_and_levels(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3,
                            cvs=("funnel_z", "learned_sw"), monitors=()), funnel=FP)
    _, axes, _, _ = fig_fe(run, bands(run), pair=("funnel_z", "learned_sw"), step_ns=0.5, min_count=0, min_bin_ess=0)
    ms = axes["map"]
    assert len({m.get_xlim() for m in ms}) == 1 and len({m.get_ylim() for m in ms}) == 1
    lv = [np.asarray(m.collections[0].levels) for m in ms]
    assert all(np.array_equal(lv[0], l) for l in lv[1:])


def test_fig_bias_maps_share_levels(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3,
                            cvs=("funnel_z", "learned_sw"), monitors=()))
    _, axes, _ = fig_bias(run, out_path=str(tmp_path / "b.png"))
    lv = [np.asarray(axes[g, 1].collections[0].levels) for g in range(3)]
    assert all(np.array_equal(lv[0], l) for l in lv[1:])
    assert len({axes[g, 1].get_xlim() for g in range(3)}) == 1


def test_fig_fe_ct_linestyles_distinct_and_on_top(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=3), funnel=FP)
    _, axes, _, _ = fig_fe(run, bands(run), step_ns=0.5)
    ct = [l for l in axes["ct"][0].get_lines() if l.get_label().endswith("c(t)")]
    assert len({l.get_linestyle() for l in ct}) == 3
    faint = [l for l in axes["ct"][0].get_lines() if l.get_alpha() == 0.35]
    assert min(l.get_zorder() for l in ct) > max(l.get_zorder() for l in faint)


def test_fig_fe_dg_panel_plain_labels(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    _, axes, _, _ = fig_fe(run, bands(run), step_ns=0.1, show_numbers=True)
    d = axes["dg"]
    leg = _legend_texts(d)
    assert "reweighted ΔG°(t) from all frames up to t" in leg and "start of statistics (t_rej)" in leg
    assert "block ΔG° (colour = min(site, bulk) ESS)" in leg and "block undefined" in leg
    assert d.get_title("left") == "ΔG° vs simulation time"
    assert all(t.get_text().isdigit() for t in d.texts)      # only round-trip counts; numbers live outside
    txt = axes["text"].texts[0].get_text().split("\n")
    for k in ("final ΔG°", "mean ± SD after t_rej", "block bootstrap SD", "last two blocks differ by",
              "valid blocks", "ESS"):
        assert any(l.startswith(k) for l in txt), k


def test_fig_fe_numbers_outside_data_axes(tmp_path):
    for kw in (dict(), dict(layout="isolated", n_walkers=2)):
        run = load_run(make_run(str(tmp_path / f"r{len(kw)}"), **kw), funnel=FP)
        _, axes, _, _ = fig_fe(run, bands(run), step_ns=0.1, show_numbers=True)
        assert all(t.get_text().isdigit() for t in axes["dg"].texts) and not axes["w"].texts and axes["text"].texts


def test_fig_fe_given_axes_without_text_skips(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    fig, ax = plt.subplots(1, 4)
    _, _, skipped, _ = fig_fe(run, bands(run), step_ns=0.5, axes=dict(ct=ax[0], map=ax[1], w=ax[2], dg=ax[3]), show_numbers=True)
    assert "ΔG° numbers: no text axes given" in skipped and not ax[3].texts


def test_w_ylim_ignores_values_outside_bands():
    from autopath.metadynamics.summary.plots import _w_ylim
    from autopath.metadynamics.summary.analysis import Band
    fig, ax = plt.subplots()
    x = np.linspace(-1.0, 3.0, 81)
    y = np.where(x < -0.5, 60.0, np.sin(x) + 2.0)        # wall far outside the bands
    ax.plot(x, y)
    _w_ylim(ax, Band(bound=(-0.2, 0.5), unbound=(1.2, 2.3), bound_end="low", source="test"))
    lo, hi = ax.get_ylim()
    inside = y[(x >= -0.2) & (x <= 2.3)].max()
    assert hi == pytest.approx(1.15 * inside + 0.2) and hi < 10 and lo < 0


def test_fig_fe_no_numbers_by_default(tmp_path):
    for kw in (dict(), dict(layout="isolated", n_walkers=2)):
        run = load_run(make_run(str(tmp_path / f"n{len(kw)}"), **kw), funnel=FP)
        fig, axes, skipped, nums = fig_fe(run, bands(run), step_ns=0.1)
        assert axes["text"] is None and not any(not t.get_text().isdigit() for a in fig.axes if a.get_visible() for t in a.texts)
        assert nums["blocks"] is not None and not any("numbers" in s for s in skipped)
        if kw:                                            # bars take the whole bottom row
            pb, pw = axes["bars"].get_position(), axes["w"].get_position()
            assert pb.x0 == pytest.approx(pw.x0) and pb.x1 == pytest.approx(pw.x1)


@pytest.mark.parametrize("layout", ["shared", "isolated"])
def test_fig_fe_dg_panel_has_colorbar_and_ess_axis(tmp_path, layout):
    run = load_run(make_run(str(tmp_path / "r"), layout=layout, n_walkers=2), funnel=FP)
    fig, axes, _, nums = fig_fe(run, bands(run), step_ns=0.1)
    assert axes["cbar"].get_ylabel() == "block min(site, bulk) ESS"
    assert axes["ess"].get_ylabel() == "ESS/N" and axes["ess"].get_yscale() == "log"
    assert axes["ess"] in fig.axes and axes["ess"].get_lines()
    d, e = axes["dg"], axes["ess"]
    assert d.get_shared_x_axes().joined(d, e)
    assert e.get_position().y1 <= d.get_position().y0 + 1e-6
    assert e.get_position().height < d.get_position().height
    assert len(fig.axes) == len(set(fig.axes)) and not any(a is not d and a.bbox.bounds == d.bbox.bounds for a in fig.axes)
    blk = nums["blocks"] if layout == "shared" else nums["blocks"]["w0"]
    assert len(blk["block_recross"]) == 10
