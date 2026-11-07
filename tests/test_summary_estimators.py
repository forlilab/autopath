import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics.Hills import bias_from_hills, fes_scale
from autopath.metadynamics.summary import check_convergence, estimators_agree, report
from autopath.metadynamics.summary.analysis import (ESTIMATORS, KT_PER_K, LABELS, RW_METHODS, _final_at_frames,
                                                    balanced_offsets, bands, binding, binding_bias, fe_numbers,
                                                    fes_reweighted, hill_volumes, log_weights)
from autopath.metadynamics.summary.data import load_run
from matplotlib.colors import to_hex

from autopath.metadynamics.summary.plots import ESTIMATOR_COLORS, fig_estimators, fig_fes2d
from ._summary_fixtures import TOY_BANDS, TOY_GRID, make_run, make_wt_toy, toy_potential, write_walker

A = unit.angstrom
FP = {"z_cc": 12.0 * A, "z_max": 24.0 * A, "z_low": -1.5 * A, "R_cylinder": 1.0 * A, "alpha": 31.5 * unit.degrees}
ALL = float("inf")
KT = KT_PER_K * 300.0


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _toy_grid():
    lo, hi, n, _ = TOY_GRID["x"]
    return np.linspace(lo, hi, n)


def _static_run(path, tempering):
    """Hills all at t=0; frames on grid points with counts ∝ exp(-β(F + V)), F = -fes_scale·V; c(t) const."""
    g = _toy_grid()
    hc, hh = np.array([0.6, 0.65, 0.7, 1.2]), np.array([3.0, 3.0, 2.0, 1.0])
    V = (hh[:, None] * np.exp(-(g[None, :] - hc[:, None]) ** 2 / (2 * TOY_GRID["x"][3] ** 2))).sum(0)
    sc = fes_scale(dict(tempering=tempering, bias_factor=20.0))
    p = np.exp(V * (sc - 1.0) / KT)
    counts = np.round(400 * p / p.min()).astype(int)
    X = np.repeat(g, counts)
    order = np.random.default_rng(0).permutation(len(X))
    X = np.r_[X[order], X[order][:1]]                        # the loader drops the last row
    bias = np.interp(X, g, V)
    write_walker(path, "w0", ("x",), X[:, None], TOY_GRID, hc, hh, np.zeros(4), bias=bias,
                 ct=np.full(len(X), 3.0), tempering=tempering)
    return load_run(path)


@pytest.mark.parametrize("tempering", ["wt", "tt", "wt+tt"])
def test_static_bias_all_estimators_agree(tmp_path, tempering):
    run = _static_run(str(tmp_path / "r"), tempering)
    b = bands(run, TOY_BANDS)
    v = {m: binding(run, b, cv="x", burn=0.0, method=m).pooled["value"] for m in RW_METHODS}
    v["bias"] = binding_bias(run, b, cv="x", hill_mask_sigmas=ALL).pooled["value"]
    assert np.isfinite(v["rw_ct"]) and abs(v["rw_ct"]) > 1.0
    assert v["rw_final"] == pytest.approx(v["rw_ct"], abs=1e-9)          # c(t) const: rw_final == rw_ct
    assert v["rw_balanced"] == pytest.approx(v["rw_ct"], abs=1e-9)       # offset const after the last hill
    assert v["bias"] == pytest.approx(v["rw_ct"], abs=0.05)              # rounding of the frame counts


def test_rw_final_equals_rw_ct_log_weights_up_to_constant(tmp_path):
    run = _static_run(str(tmp_path / "r"), "wt")
    w = run.walkers[0]
    d = log_weights(run, w, "rw_final") - log_weights(run, w, "rw_ct")
    np.testing.assert_allclose(d, d[0], atol=1e-9)


def _hills_run(path, cvs, grid, hc, hh, ht, bounds=None, n_frames=50):
    X = np.tile([np.mean(grid[c][:2]) for c in cvs], (n_frames + 1, 1))
    write_walker(path, "w0", cvs, X, grid, hc, hh, ht, bias=np.zeros(n_frames + 1), ct=np.zeros(n_frames + 1),
                 bounds=bounds, dt=0.01)
    return load_run(path)


@pytest.mark.parametrize("bounds,L", [(None, 3.0), ({"x": (0.5, 2.5)}, 2.0)])
def test_balanced_offset_uniform_hills(tmp_path, bounds, L):
    rng = np.random.default_rng(1)
    n = 200
    hc, hh = rng.uniform(1.0, 2.0, n), rng.uniform(0.5, 1.5, n)
    ht = np.linspace(0.0, 0.4, n)
    run = _hills_run(str(tmp_path / "r"), ("x",), TOY_GRID, hc, hh, ht, bounds)
    sig = TOY_GRID["x"][3]
    off = balanced_offsets(run, run.walkers)["w0"]
    want = hh.sum() * np.sqrt(2 * np.pi) * sig / L
    assert off[-1] == pytest.approx(want, rel=1e-3)
    t = run.walkers[0].t
    k = np.searchsorted(ht, t, side="right")
    np.testing.assert_allclose(off, np.r_[0.0, np.cumsum(hh)][k] * np.sqrt(2 * np.pi) * sig / L, rtol=1e-3)


def test_balanced_offset_uniform_hills_2d(tmp_path):
    grid = {"x": (0.0, 3.0, 151, 0.06), "y": (-1.0, 1.0, 81, 0.05)}
    rng = np.random.default_rng(2)
    n = 300
    hc = np.c_[rng.uniform(1.0, 2.0, n), rng.uniform(-0.5, 0.5, n)]
    hh = rng.uniform(0.5, 1.5, n)
    run = _hills_run(str(tmp_path / "r"), ("x", "y"), grid, hc, hh, np.zeros(n), bounds={"y": (-0.8, 0.8)})
    vol = hill_volumes(run, run.walkers[0].hills_raw)
    np.testing.assert_allclose(vol, 2 * np.pi * 0.06 * 0.05 / (3.0 * 1.6), rtol=1e-3)


def _two_walkers(path, wall1, n_frames=40, extend=False):
    """Shared bias: w0 then w1; ten hills each at walker times 0.05, 0.1, ..."""
    for k, wall in enumerate((np.arange(10.0), wall1)):
        ht = 0.05 * np.arange(1, 11)
        write_walker(path, f"w{k}", ("x",), np.full((n_frames + 1, 1), 1.5), TOY_GRID,
                     np.full(10, 1.0 + 0.5 * k), np.full(10, 1.0 + k), ht, bias=np.zeros(n_frames + 1),
                     ct=np.zeros(n_frames + 1), wall=wall, bias_dir=path, dt=0.0123)
    return load_run(path)


def test_sequential_shared_walkers_wall_clock_order(tmp_path):
    run = _two_walkers(str(tmp_path / "r"), 1000.0 + np.arange(10.0))
    assert run.layout == "shared"
    vol = hill_volumes(run, [h for w in run.walkers for h in w.hills_raw])
    c0, c1 = 1.0 * vol[:10], 2.0 * vol[10:]
    off = balanced_offsets(run, run.walkers)
    w0, w1 = run.walkers
    k0 = np.searchsorted(w0.hill_t, w0.t, side="right")
    np.testing.assert_allclose(off["w0"], np.r_[0.0, np.cumsum(c0)][k0])          # w1 has not deposited yet
    k1 = np.searchsorted(w1.hill_t, w1.t, side="right")
    np.testing.assert_allclose(off["w1"], c0.sum() + np.r_[0.0, np.cumsum(c1)][k1])  # all of w0 already in
    assert off["w1"][0] == pytest.approx(c0.sum())


def test_non_monotonic_wall_not_supported(tmp_path):
    run = _two_walkers(str(tmp_path / "r"), 1000.0 - np.arange(10.0))
    with pytest.raises(ValueError, match="non-monotonic hill wall times: not supported"):
        balanced_offsets(run, run.walkers)
    b = bands(run, TOY_BANDS)
    n = fe_numbers(run, b, cv="x", step_ns=0.1, n_boot=5, min_points=2)
    assert "rw_balanced" not in n["estimators"] and {"rw_ct", "rw_final", "bias"} <= set(n["estimators"])
    assert any("non-monotonic hill wall times" in s for s in n["skipped"])


def test_extended_segments_keep_their_own_wall_times(tmp_path):
    path = str(tmp_path / "r")
    _two_walkers(path, 100.0 + np.arange(10.0))
    ht = 0.05 * np.arange(1, 11)
    write_walker(path, "w0_x2", ("x",), np.full((41, 1), 1.5), TOY_GRID, np.full(10, 1.0), np.full(10, 1.0), ht,
                 bias=np.zeros(41), ct=np.zeros(41), wall=200.0 + np.arange(10.0), bias_dir=path, dt=0.0123)
    run = load_run(path, funnel=None)
    w0 = next(w for w in run.walkers if w.rid == "w0")
    off = balanced_offsets(run, run.walkers)["w0"]
    vol = hill_volumes(run, [h for w in run.walkers for h in w.hills_raw])[0]
    seg2 = w0.segment_of_record == 1
    # segment 2 starts after all of w0's first segment and all of w1
    assert off[seg2][0] == pytest.approx(vol * (10 * 1.0 + 10 * 2.0), rel=1e-9)
    assert off[seg2][-1] == pytest.approx(vol * (19 * 1.0 + 10 * 2.0), rel=1e-9)        # 9 of 10 own hills


def test_rw_final_2d_orientation_and_grid_nodes(tmp_path):
    grid = {"x": (0.0, 3.0, 151, 0.06), "y": (-1.0, 1.0, 81, 0.05)}
    gx, gy = np.linspace(0, 3, 151), np.linspace(-1, 1, 81)
    X = np.c_[gx[[10, 40, 70, 100, 100]], gy[[5, 20, 40, 60, 60]]]
    write_walker(str(tmp_path / "r"), "w0", ("x", "y"), X, grid, np.array([[1.2, 0.2]]), np.array([5.0]),
                 np.zeros(1), bias=np.zeros(5), ct=np.zeros(5))
    run = load_run(str(tmp_path / "r"))
    V = np.asarray(bias_from_hills(run.walkers[0].hills_raw))
    assert V.shape == (81, 151)
    got = _final_at_frames(run, run.walkers[0])
    np.testing.assert_allclose(got, V[[5, 20, 40, 60], [10, 40, 70, 100]], atol=1e-12)


def test_rw_final_outside_grid(tmp_path):
    grid = {"x": TOY_GRID["x"]}
    X = np.array([[1.0], [3.2], [1.0]])
    write_walker(str(tmp_path / "a"), "w0", ("x",), X, grid, np.array([1.0]), np.array([1.0]), np.zeros(1),
                 bias=np.zeros(3), ct=np.zeros(3))
    run = load_run(str(tmp_path / "a"))
    with pytest.raises(ValueError, match="frame CV x outside the bias grid: not supported"):
        _final_at_frames(run, run.walkers[0])
    write_walker(str(tmp_path / "b"), "w0", ("x",), X, grid, np.array([1.0]), np.array([1.0]), np.zeros(1),
                 bias=np.zeros(3), ct=np.zeros(3), bounds={"x": (0.5, 2.5)})
    run = load_run(str(tmp_path / "b"))
    V = np.asarray(bias_from_hills(run.walkers[0].hills_raw))
    assert _final_at_frames(run, run.walkers[0])[1] == pytest.approx(V[-1])      # clamped CV: clipped


def test_rw_final_uses_the_bias_of_the_layout(tmp_path):
    sh = load_run(make_run(str(tmp_path / "s"), n_walkers=2), funnel=FP)
    iso = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=2), funnel=FP)
    for run, group in ((sh, lambda w: sh.walkers), (iso, lambda w: [w])):
        for w in run.walkers:
            V = np.asarray(bias_from_hills([h for x in group(w) for h in x.hills_raw]))
            want = np.interp(w.cvs["funnel_z"], np.linspace(-0.4, 2.4, 101), V) / KT
            np.testing.assert_allclose(log_weights(run, w, "rw_final"), want, rtol=1e-10)


@pytest.mark.parametrize("method", list(RW_METHODS))
def test_isolated_never_pooled(tmp_path, method):
    iso = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=3), funnel=FP)
    b = bands(iso)
    c = binding(iso, b, method=method)
    assert c.pooled is None and set(c.per_walker) == {"w0", "w1", "w2"}
    one = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=3), funnel=FP)
    one.walkers, one.layout = [w for w in one.walkers if w.rid == "w1"], "single"
    assert binding(one, bands(one), method=method).pooled["value"] == pytest.approx(c.per_walker["w1"]["value"])
    f = fes_reweighted(iso, "funnel_z", method=method)
    assert f.pooled is None and set(f.per_walker) == {"w0", "w1", "w2"}


def test_isolated_fe_numbers_per_walker_for_every_estimator(tmp_path):
    run = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=2), funnel=FP)
    n = fe_numbers(run, bands(run), step_ns=0.1, n_boot=10)
    assert set(n["estimators"]) == set(ESTIMATORS)
    for m, e in n["estimators"].items():
        assert e["binding"].pooled is None and set(e["binding"].per_walker) == {"w0", "w1"}
        assert e["series"].pooled is None and set(e["series"].per_walker) == {"w0", "w1"}
        assert (e["blocks"] is None) == (m == "bias")
    sp = n["estimator_spread"]
    assert sp["pooled"] is None
    for rid in ("w0", "w1"):
        v = [n["estimators"][m]["binding"].per_walker[rid]["value"] for m in ESTIMATORS]
        assert sp["per_walker"][rid] == pytest.approx(max(v) - min(v))
    g = n["estimator_gaps"]["rw_ct-bias"]["per_walker"]
    assert g == pytest.approx(n["estimator_gap"]["per_walker"])


def test_fe_numbers_keeps_old_keys(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    b = bands(run)
    n = fe_numbers(run, b, step_ns=0.1, n_boot=10)
    assert n["primary"] == "rw_ct"
    assert n["binding"] is n["estimators"]["rw_ct"]["binding"] and n["series"] is n["estimators"]["rw_ct"]["series"]
    assert n["blocks"] is n["estimators"]["rw_ct"]["blocks"]
    assert n["binding_bias"] is n["estimators"]["bias"]["binding"]
    assert n["binding"].pooled["value"] == binding(run, b).pooled["value"]
    v = [n["estimators"][m]["binding"].pooled["value"] for m in ESTIMATORS]
    assert n["estimator_spread"]["pooled"] == pytest.approx(max(v) - min(v))
    with pytest.raises(ValueError, match="not supported"):
        fe_numbers(run, b, primary="bias")


def test_wt_toy_all_estimators_converge(tmp_path):
    run = load_run(make_wt_toy(str(tmp_path / "t"), n_steps=600_000, h0=0.3, stride=100, step=0.03, seed=0))
    g = np.linspace(0, 3, 30001)
    p = np.exp(-toy_potential(g) / KT)
    bd, ub = TOY_BANDS["x"]["bound"], TOY_BANDS["x"]["unbound"]
    exact = -KT * np.log(p[(g >= bd[0]) & (g <= bd[1])].sum() / p[(g >= ub[0]) & (g <= ub[1])].sum())
    n = fe_numbers(run, bands(run, TOY_BANDS), cv="x", step_ns=0.3, n_boot=10)
    v = {m: e["binding"].pooled["value"] for m, e in n["estimators"].items()}
    assert set(v) == set(ESTIMATORS)
    for m, x in v.items():
        assert x == pytest.approx(exact, abs=1.0), m                    # kJ/mol
    assert n["estimator_spread"]["pooled"] < 0.8


def _est_summary(layout, vals, rids=("w0", "w1")):
    """Summary with binding values (kJ/mol) per estimator: ``vals[m]`` is a float (pooled) or {rid: float}."""
    est = {}
    for m, v in vals.items():
        if isinstance(v, dict):
            est[m] = dict(binding=dict(pooled=None, per_walker={k: dict(value=x) for k, x in v.items()}))
        else:
            est[m] = dict(binding=dict(pooled=dict(value=v), per_walker={}))
    return dict(layout=layout, walkers={r: {} for r in rids}, estimators=est)


def test_estimators_agree_pass_fail_undefined():
    c = estimators_agree(1.0)
    assert c.__name__ == "estimators_agree(1.0)"
    ok, why = c(_est_summary("shared", dict(rw_ct=0.0, rw_final=2.0, rw_balanced=-1.0, bias=3.0)))   # 0.96 kcal
    assert ok, why
    ok, why = c(_est_summary("shared", dict(rw_ct=0.0, rw_final=2.0, rw_balanced=-1.0, bias=4.5)))
    assert not ok and "all: spread 1.31 kcal/mol (bias - rw_balanced)" in why
    ok, why = c(_est_summary("shared", dict(rw_ct=0.0, rw_final=None, rw_balanced=-1.0, bias=1.0)))
    assert not ok and "rw_final undefined" in why
    ok, why = c(_est_summary("shared", dict(rw_ct=0.0, bias=1.0)))
    assert not ok and "rw_final, rw_balanced undefined" in why
    c2 = estimators_agree(1.0, methods=("rw_ct", "bias"))
    assert c2.__name__ == "estimators_agree(1.0, rw_ct,bias)"
    ok, _ = c2(_est_summary("shared", dict(rw_ct=0.0, bias=1.0)))
    assert ok
    iso = _est_summary("isolated", dict(rw_ct=dict(w0=0.0, w1=0.0), rw_final=dict(w0=0.5, w1=9.0),
                                        rw_balanced=dict(w0=0.1, w1=0.0), bias=dict(w0=0.2)))
    ok, why = c(iso)
    assert not ok and "w1: bias undefined" in why and "w0" not in why
    ok, why = estimators_agree(1.0, methods=("rw_ct",))(iso)
    assert not ok and "fewer than 2" in why
    ok, why = c(dict(layout="shared", walkers={"w0": {}}))
    assert not ok and "undefined" in why


def test_convergence_follows_primary():
    from autopath.metadynamics.summary.convergence import min_ess
    s = dict(layout="shared", walkers={"w0": {}}, primary="rw_final", binding=dict(pooled=dict(ess=1000.0)),
             estimators=dict(rw_ct=dict(binding=dict(pooled=dict(ess=1000.0))),
                             rw_final=dict(binding=dict(pooled=dict(ess=5.0)))))
    ok, why = min_ess(10)(s)
    assert not ok and "ESS 5" in why
    s["primary"] = "rw_ct"
    assert min_ess(10)(s)[0]


@pytest.mark.parametrize("layout,n", [("shared", 2), ("isolated", 3)])
def test_fig_estimators_axes(tmp_path, layout, n):
    run = load_run(make_run(str(tmp_path / "r"), layout=layout, n_walkers=n), funnel=FP)
    nums = fe_numbers(run, bands(run), step_ns=0.1, n_boot=5)
    fig, axes, skipped = fig_estimators(run, nums, out_path=str(tmp_path / "e.png"))
    rows = 1 if layout == "shared" else n
    assert axes.shape == (rows, 2) and os.path.getsize(str(tmp_path / "e.png")) > 1000
    for r in range(rows):
        left, right = axes[r]
        assert [t.get_text() for t in left.get_legend().get_texts()] == [LABELS[m] for m in ESTIMATORS]
        est_lines = [ln for ln in left.get_lines() if ln.get_linewidth() > 2.0]
        assert [ln.get_linestyle() for ln in est_lines] == ["-"] * 4
        assert [to_hex(ln.get_color()).upper() for ln in est_lines] == [ESTIMATOR_COLORS[m].upper() for m in ESTIMATORS]
        assert [t.get_text() for t in right.get_yticklabels()] == [LABELS[m] for m in ESTIMATORS]
        stems = [c_.get_segments()[0] for c_ in right.collections]     # lollipop stems from 0 to each value
        heads = [ln for ln in right.get_lines() if ln.get_marker() == "o"]
        assert len(stems) == len(heads) >= 1 and all(float(sg[0][0]) == 0.0 for sg in stems)
        assert {to_hex(ln.get_color()).upper() for ln in heads} <= {c_.upper() for c_ in ESTIMATOR_COLORS.values()}
        assert [float(sg[1][0]) for sg in stems] == pytest.approx([float(h.get_xdata()[0]) for h in heads])
        assert len(right.texts) == 4                          # values or "undefined"
        assert "spread" in right.get_title("left")
    assert len({tuple(axes[r, 1].get_xlim()) for r in range(rows)}) == 1       # one x-scale across rows
    want = "aggregate walker time" if layout == "shared" else "walker time (ns)"
    assert want in axes[0, 0].get_xlabel()
    if layout == "isolated":
        assert [axes[i, 0].get_title("left").split(":")[0] for i in range(n)] == ["w0", "w1", "w2"]


def test_fes2d_four_maps_and_primary_cap(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()))
    b = bands(run)
    fig, axes, skipped, nums = fig_fes2d(run, b)
    assert axes.shape == (1, 4)
    titles = [a.get_title("left") for a in axes[0]]
    assert titles[0].startswith("all: reweight c(t) F (ESS ")
    assert titles[1].startswith("all: reweight final bias F (ESS ")
    assert titles[2].startswith("all: reweight balanced F (ESS ")
    assert titles[3].startswith("all: bias-derived F")
    assert all(a.collections for a in axes[0])
    lv = [np.asarray(a.collections[0].levels) for a in axes[0]]
    assert all(np.array_equal(lv[0], x) for x in lv[1:])
    F = fes_reweighted(run, "funnel_z", "learned_sw").pooled[2] / 4.184
    assert nums["fmax"] == pytest.approx(np.percentile(F[np.isfinite(F)], 99))
    assert len([a for a in fig.axes if a.get_label() == "<colorbar>"]) == 1


def test_fes2d_isolated_rows(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2, cvs=("funnel_z", "learned_sw"),
                            monitors=()))
    _, axes, _, _ = fig_fes2d(run, bands(run), fmax=4.0)
    assert axes.shape == (2, 4)
    for i, rid in enumerate(("w0", "w1")):
        assert all(a.get_title("left").startswith(f"{rid}: ") for a in axes[i])
        assert np.asarray(axes[i, 1].collections[0].levels)[-1] == pytest.approx(4.0)


def test_report_writes_estimators(tmp_path):
    s = report(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2), funnel=FP, out_dir=str(tmp_path / "o"),
               step_ns=0.1, hill_mask_sigmas=ALL)
    assert os.path.getsize(s["files"]["estimators"]) > 1000
    assert s["files"]["estimators"].endswith("metad_estimators.png")
    d = json.load(open(s["files"]["summary"]))
    assert d["primary"] == "rw_ct" and d["params"]["primary"] == "rw_ct"
    assert set(d["estimators"]) == set(ESTIMATORS)
    assert d["estimators"]["rw_ct"]["binding"] == d["binding"]
    assert d["estimators"]["bias"]["binding"] == d["binding_bias"]
    assert d["estimators"]["rw_final"]["label"] == "reweight final bias"
    assert set(d["estimator_spread"]["per_walker"]) == {"w0", "w1"} and d["estimator_spread"]["pooled"] is None
    assert set(d["estimator_gaps"]) == {"rw_ct-rw_final", "rw_ct-rw_balanced", "rw_ct-bias", "rw_final-rw_balanced",
                                        "rw_final-bias", "rw_balanced-bias"}
    big = max(d["estimator_spread"]["per_walker"].values()) / 4.184
    assert not check_convergence(s, [estimators_agree(big + 0.01)]).needs_more_sampling
    assert check_convergence(s, [estimators_agree(big - 0.01)]).needs_more_sampling


def test_report_primary_rw_balanced(tmp_path):
    s = report(make_run(str(tmp_path / "r")), funnel=FP, out_dir=str(tmp_path / "o"), step_ns=0.1,
               hill_mask_sigmas=ALL, primary="rw_balanced")
    assert s["primary"] == "rw_balanced"
    assert s["estimators"]["rw_balanced"]["blocks"] is not None
    assert s["binding"] == s["estimators"]["rw_ct"]["binding"]


def test_shared_extension_skips_balanced_with_reason(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=2, extend=True), funnel=FP)
    n = fe_numbers(run, bands(run), step_ns=0.5, n_boot=5)
    assert "rw_balanced" not in n["estimators"]
    assert any("reweight balanced ΔG°: non-monotonic hill wall times: not supported" in s for s in n["skipped"])
    assert np.isnan(n["estimator_spread"]["pooled"])
