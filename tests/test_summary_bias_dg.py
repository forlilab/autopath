import json
import math

import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics.Hills import bias_from_hills
from autopath.metadynamics.summary import check_convergence, estimators_agree, report
from autopath.metadynamics.summary.analysis import (C0_PER_A3, KT_PER_K, _bias_curve, _bias_value, bands,
                                                    bias_fes_scale, binding, binding_bias, dg_bias_series, dg_series,
                                                    fes_bias, fe_numbers)
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.plots import fig_fe
from ._summary_fixtures import make_run

A = unit.angstrom
FP = {"z_cc": 12.0 * A, "z_max": 24.0 * A, "z_low": -1.5 * A, "R_cylinder": 1.0 * A, "alpha": 31.5 * unit.degrees}
ALL = float("inf")                                   # mask off: the whole grid counts as sampled


@pytest.mark.parametrize("temp,scale", [("tt", 1.0), ("wt", 20.0 / 19.0), ("wt+tt", 20.0 / 19.0)])
def test_fes_scale_per_tempering(tmp_path, temp, scale):
    run = load_run(make_run(str(tmp_path / "r"), tempering=temp), funnel=FP)
    assert bias_fes_scale(run) == pytest.approx(scale)
    grids, F = fes_bias(run, ALL).pooled
    V = np.asarray(bias_from_hills([h for w in run.walkers for h in w.hills_raw]))
    np.testing.assert_allclose(F, -scale * V - (-scale * V).min())


def test_1d_bias_has_no_marginal(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    grids, F = fes_bias(run, ALL).pooled
    z, W = _bias_curve(run, "funnel_z", grids, F)
    np.testing.assert_array_equal(W, F)
    np.testing.assert_array_equal(z, grids[0])


def test_2d_marginal_over_axis0_matches_separable_case(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "r_xy"), monitors=()), funnel=FP)
    kt = KT_PER_K * run.temperature
    zg, rg = np.linspace(-0.4, 2.4, 101), np.linspace(0.0, 1.5, 101)
    a = 3.0 * np.sin(rg * 4.0)                       # axis 0 (second CV)
    b = 5.0 * (zg - 1.0) ** 2                        # axis 1 (first CV)
    F = a[:, None] + b[None, :]
    z, W = _bias_curve(run, "funnel_z", [zg, rg], F)
    np.testing.assert_allclose(W, b - b.min(), atol=1e-9)
    # the other CV integrates out too
    z2, W2 = _bias_curve(run, "r_xy", [zg, rg], F)
    np.testing.assert_allclose(W2, a - a.min(), atol=1e-9)
    # NaN entries leave the integral
    Fn = F.copy(); Fn[:50, 10] = np.nan
    _, Wn = _bias_curve(run, "funnel_z", [zg, rg], Fn)
    want = -kt * np.log(np.exp(-Fn[50:, 10] / kt).sum()) + 0.0
    ref = -kt * np.log(np.exp(-F[:, 40] / kt).sum())
    assert Wn[10] - Wn[40] == pytest.approx(want - ref)


def test_bias_grid_shape_asserted(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "r_xy"), monitors=()), funnel=FP)
    with pytest.raises(ValueError, match="not supported"):
        _bias_curve(run, "funnel_z", [np.zeros(101), np.zeros(50)], np.zeros((101, 101)))
    with pytest.raises(ValueError, match="not supported"):
        _bias_curve(run, "dist", [np.zeros(101), np.zeros(101)], np.zeros((101, 101)))


def test_known_bias_gives_known_dG0(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    b = bands(run)["funnel_z"]
    kt = KT_PER_K * run.temperature
    z = np.linspace(-0.4, 2.4, 2801)                  # nm
    # flat everywhere: ΔG° = -kT ln(C0 π R² L)
    v = _bias_value(run, bands(run), "funnel_z", z, np.zeros_like(z))
    site = (z >= b.bound[0]) & (z <= b.bound[1])
    dz = (z[1] - z[0]) * 10
    assert v["kind"] == "dG0" and v["unit"] == "kJ/mol"
    assert v["value"] == pytest.approx(-kt * np.log(C0_PER_A3 * np.pi * site.sum() * dz), abs=1e-9)
    # well of depth D in the site, flat bulk: ΔG° shifts by -D at fixed site width
    D = 8.0
    W = np.where(site, -D, 0.0)
    w = _bias_value(run, bands(run), "funnel_z", z, W)
    assert w["value"] - v["value"] == pytest.approx(-D, abs=1e-9)
    # a constant offset of W does not matter
    assert _bias_value(run, bands(run), "funnel_z", z, W + 7.0)["value"] == pytest.approx(w["value"], abs=1e-9)


def test_band_dF_without_funnel(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=()))
    c = binding_bias(run, bands(run), cv="dist", hill_mask_sigmas=ALL)
    assert c.pooled["kind"] == "band_dF"
    grids, F = fes_bias(run, ALL).pooled
    b = bands(run)["dist"]
    kt = KT_PER_K * run.temperature
    g = grids[0]
    sb = np.exp(-F[(g >= b.bound[0]) & (g <= b.bound[1])] / kt).sum()
    su = np.exp(-F[(g >= b.unbound[0]) & (g <= b.unbound[1])] / kt).sum()
    assert c.pooled["value"] == pytest.approx(-kt * np.log(sb / su))


def test_shared_pools_and_isolated_per_walker(tmp_path):
    sh = load_run(make_run(str(tmp_path / "s"), layout="shared", n_walkers=3), funnel=FP)
    c = binding_bias(sh, bands(sh), hill_mask_sigmas=ALL)
    assert c.per_walker == {} and np.isfinite(c.pooled["value"])
    iso = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=3), funnel=FP)
    ci = binding_bias(iso, bands(iso), hill_mask_sigmas=ALL)
    assert ci.pooled is None and set(ci.per_walker) == {"w0", "w1", "w2"}
    vals = [v["value"] for v in ci.per_walker.values()]
    assert ci.mean == pytest.approx(np.mean(vals)) and ci.n_defined == 3
    one = load_run(make_run(str(tmp_path / "o"), layout="isolated", n_walkers=3), funnel=FP)
    one.walkers, one.layout = one.walkers[:1], "single"
    assert binding_bias(one, bands(one), hill_mask_sigmas=ALL).pooled["value"] == \
        pytest.approx(ci.per_walker["w0"]["value"])


@pytest.mark.parametrize("layout", ["shared", "isolated"])
def test_series_ends_at_binding_bias(tmp_path, layout):
    run = load_run(make_run(str(tmp_path / "r"), layout=layout, n_walkers=2), funnel=FP)
    b = bands(run)
    s = dg_bias_series(run, b, n=8, hill_mask_sigmas=ALL)
    bb = binding_bias(run, b, hill_mask_sigmas=ALL)
    pairs = [(s.pooled, bb.pooled)] if layout == "shared" else [(s.per_walker[k], bb.per_walker[k]) for k in s.per_walker]
    for ser, fin in pairs:
        assert len(ser["t"]) == 8 and np.all(np.diff(ser["t"]) >= 0)
        assert ser["dg"][-1] == pytest.approx(fin["value"])
        assert ser["unit"] == "kJ/mol"
        ends = [w.hill_t.max() for w in run.walkers]
        assert ser["t"][-1] == pytest.approx(sum(ends) if layout == "shared" else ends[0])


def _sequential(tmp_path):
    path = make_run(str(tmp_path / "r"), layout="shared", n_walkers=2)
    f = np.load(f"{path}/HILLS_w1.npz")
    d = dict(f)
    d["wall_s"] = d["wall_s"] + 1e5                  # walker 1 deposits after walker 0 has finished
    np.savez(f"{path}/HILLS_w1.npz", **d)
    return load_run(path, funnel=FP)


def test_shared_bias_series_x_is_aggregate_time(tmp_path):
    run = _sequential(tmp_path)
    b = bands(run)
    s = dg_bias_series(run, b, n=8, hill_mask_sigmas=ALL).pooled
    assert np.all(np.diff(s["t"]) > 0) and len(set(s["t"])) == 8
    np.testing.assert_array_equal(s["t"], s["t_aggregate"])
    assert s["t"][-1] == pytest.approx(sum(w.hill_t.max() for w in run.walkers))
    assert s["dg"][-1] == pytest.approx(binding_bias(run, b, hill_mask_sigmas=ALL).pooled["value"])


def test_reweighted_t_aggregate_monotonic(tmp_path):
    run = _sequential(tmp_path)
    for w in run.walkers[1:]:
        w.t = w.t[: len(w.t) // 2]; w.bias = w.bias[: len(w.t)]; w.ct = w.ct[: len(w.t)]
        w.cvs = {k: v[: len(w.t)] for k, v in w.cvs.items()}
    s = dg_series(run, bands(run), step_ns=0.25).pooled
    assert np.all(np.diff(s["t_aggregate"]) > 0)
    assert s["t_aggregate"][-1] == pytest.approx(sum(w.t[-1] for w in run.walkers))
    assert s["t_aggregate"][0] == pytest.approx(2 * 0.25)
    iso = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=2), funnel=FP)
    assert "t_aggregate" not in dg_series(iso, bands(iso)).per_walker["w0"]


def test_summary_json_has_t_aggregate_shared(tmp_path):
    s = report(make_run(str(tmp_path / "r"), layout="shared", n_walkers=2), funnel=FP,
               out_dir=str(tmp_path / "o"), step_ns=0.1, hill_mask_sigmas=ALL)
    d = json.load(open(s["files"]["summary"]))
    assert len(d["dg_bias_series"]["pooled"]["t_aggregate"]) == len(d["dg_bias_series"]["pooled"]["t"])
    ta = d["series"]["pooled"]["t_aggregate"]
    assert len(ta) == len(d["series"]["pooled"]["t"]) and np.all(np.diff(ta) > 0)


@pytest.mark.parametrize("n", [2, 3])
def test_fig_fe_isolated_layout(tmp_path, n):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=n), funnel=FP)
    fig, axes, _, _ = fig_fe(run, bands(run), step_ns=0.1, show_numbers=True)
    fig.canvas.draw()
    p = {k: axes[k].get_position() for k in ("w", "dg", "bars", "text")}
    assert p["w"].x0 == pytest.approx(p["dg"].x0) and p["w"].x1 == pytest.approx(p["dg"].x1)
    assert p["w"].y0 > p["dg"].y1 - 1e-9 and p["dg"].y0 > p["bars"].y1 - 1e-9
    assert p["bars"].y0 == pytest.approx(p["text"].y0) and p["bars"].y1 == pytest.approx(p["text"].y1)
    assert p["text"].x0 >= p["bars"].x1 - 1e-9
    assert p["bars"].x0 == pytest.approx(p["w"].x0) and p["text"].x1 == pytest.approx(p["w"].x1)
    assert axes["ct"][0].get_position().x1 <= p["w"].x0 + 1e-9
    assert p["w"].height == pytest.approx(p["dg"].height, rel=0.05)


def test_hill_mask_applies_to_series(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    b = bands(run)
    s = dg_bias_series(run, b, n=5, hill_mask_sigmas=0.5).pooled
    assert len(s["dg"]) == 5       # early cuts may be undefined; must not raise


def test_reweighted_numbers_unchanged(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    b = bands(run)
    n = fe_numbers(run, b, step_ns=0.1, n_boot=20)
    assert n["binding"].pooled["value"] == binding(run, b).pooled["value"]
    g = n["estimator_gap"]
    assert g["unit"] == "kJ/mol" and g["pooled"] == pytest.approx(
        n["binding"].pooled["value"] - n["binding_bias"].pooled["value"], nan_ok=True)


def test_summary_json_records_estimators(tmp_path):
    s = report(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2, tempering="wt+tt"), funnel=FP,
               out_dir=str(tmp_path / "o"), step_ns=0.1, hill_mask_sigmas=ALL)
    d = json.load(open(s["files"]["summary"]))
    assert d["tempering"] == "wt+tt" and d["fes_scale"] == pytest.approx(20.0 / 19.0)
    assert set(d["binding_bias"]["per_walker"]) == {"w0", "w1"} and d["binding_bias"]["pooled"] is None
    assert d["binding_bias"]["per_walker"]["w0"]["kind"] == "dG0"
    assert set(d["dg_bias_series"]["per_walker"]) == {"w0", "w1"}
    assert d["estimator_gap"]["unit"] == "kJ/mol" and d["estimator_gap"]["pooled"] is None
    for k, v in d["estimator_gap"]["per_walker"].items():
        assert v == pytest.approx(d["binding"]["per_walker"][k]["value"] - d["binding_bias"]["per_walker"][k]["value"])


def test_fig_fe_overlays_bias_curves(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2), funnel=FP)
    _, axes, _, nums = fig_fe(run, bands(run), step_ns=0.1, show_numbers=True)
    for key in ("w", "dg"):
        dashed = [ln for ln in axes[key].get_lines() if ln.get_linestyle() == "--"]
        assert len(dashed) == 2
    leg = [t.get_text() for t in axes["w"].get_legend().get_texts()]
    assert all(" rw / " in t and " bias (kcal/mol)" in t for t in leg)
    txt = axes["text"].texts[0].get_text()
    assert txt.count("bias-derived ΔG°") == 2 and txt.count("estimator gap") == 2
    assert nums["estimator_gap"]["pooled"] is None


def _gap_summary(layout, gap, rids=("w0", "w1")):
    return dict(layout=layout, walkers={r: {} for r in rids}, estimator_gap=gap)


def test_estimators_agree_unit():
    c = estimators_agree(1.0)
    assert c.__name__ == "estimators_agree(1.0)"
    ok, _ = c(_gap_summary("shared", dict(pooled=4.0, per_walker={}, unit="kJ/mol")))        # 0.96 kcal
    assert ok
    ok, why = c(_gap_summary("shared", dict(pooled=-5.0, per_walker={}, unit="kJ/mol")))
    assert not ok and "all: gap -1.20" in why
    ok, why = c(_gap_summary("shared", dict(pooled=None, per_walker={}, unit="kJ/mol")))
    assert not ok and "undefined" in why
    ok, why = c(_gap_summary("isolated", dict(pooled=None, per_walker=dict(w0=1.0, w1=-9.0), unit="kJ/mol")))
    assert not ok and "w1" in why and "w0" not in why
    ok, why = c(_gap_summary("isolated", dict(pooled=None, per_walker=dict(w0=1.0), unit="kJ/mol")))
    assert not ok and "w1: undefined" in why
    ok, _ = c(_gap_summary("shared", None))
    assert not ok


@pytest.mark.parametrize("layout", ["shared", "isolated"])
def test_estimators_agree_end_to_end(tmp_path, layout):
    s = report(make_run(str(tmp_path / "r"), layout=layout, n_walkers=2), funnel=FP, out_dir=str(tmp_path / "o"),
               step_ns=0.1, hill_mask_sigmas=ALL)
    g = s["estimator_gap"]
    gaps = [g["pooled"]] if layout == "shared" else list(g["per_walker"].values())
    big = max(abs(x) for x in gaps) / 4.184
    assert not check_convergence(s, [estimators_agree(big + 0.5)]).needs_more_sampling
    v = check_convergence(s, [estimators_agree(big / 2)])
    assert v.needs_more_sampling and not v.passed[f"estimators_agree({big / 2})"]
    s["estimator_gap"] = None
    v = check_convergence(s, [estimators_agree(100.0)])
    assert v.needs_more_sampling and "undefined" in v.reasons[0]


def test_no_reweight_leaves_gap_undefined(tmp_path):
    s = report(make_run(str(tmp_path / "r"), reweight=False), funnel=FP, out_dir=str(tmp_path / "o"),
               hill_mask_sigmas=ALL)
    assert s["estimator_gap"] is None and s["binding_bias"] is not None
    v = check_convergence(s, [estimators_agree(1.0)])
    assert v.needs_more_sampling and "undefined" in v.reasons[0]
