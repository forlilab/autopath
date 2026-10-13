import os
import warnings

import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics.summary.analysis import (KT_PER_K, _frames, _frames_in_window, _frames_in_windows, band_dF, bands, binding,
                                                    block_analysis, dg_from_frames, dg_series, _value)
from autopath.metadynamics.summary.data import load_run
from ._summary_fixtures import make_run

A = unit.angstrom
FP = {"z_cc": 12.0 * A, "z_max": 24.0 * A, "z_low": -1.5 * A, "R_cylinder": 1.0 * A}


def test_dg_matches_formula_on_analytic_profile():
    kt = 0.0083144626 * 300.0
    z = np.linspace(-2, 24, 26001)
    W = np.minimum(0.0, -20.0 + 4.0 * (z - 2) ** 2)
    p = np.exp(-W / kt); w = p / p.sum()
    bound, bulk = (-2.0, 5.0), (12.0, 22.5)
    m = (z >= -2) & (z <= 5)
    integ = np.trapz(np.exp(-W[m] / kt), z[m]) if hasattr(np, "trapz") else np.trapezoid(np.exp(-W[m] / kt), z[m])
    ref = -kt * np.log(1 / 1661.0 * np.pi * 1.0 ** 2 * integ)
    assert dg_from_frames(z, w, bound, bulk, 1.0, 300.0) == pytest.approx(ref, abs=0.05)


def test_dg_nan_without_bulk():
    z = np.zeros(100); w = np.full(100, 0.01)
    assert np.isnan(dg_from_frames(z, w, (-1, 1), (12, 22), 1.0, 300.0))


def test_isolated_walker_without_bulk(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3), funnel=FP)
    run.walkers[1].cvs["funnel_z"][:] = 0.1
    c = binding(run, bands(run))
    assert c.pooled is None and np.isnan(c.per_walker["w1"]["value"]) and c.per_walker["w1"]["reason"]
    finite = [v["value"] for v in c.per_walker.values() if np.isfinite(v["value"])]
    assert c.mean == pytest.approx(np.mean(finite))


def test_no_funnel_reports_band_difference(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=()))
    c = binding(run, bands(run), cv="dist")
    assert c.pooled["kind"] == "band_dF"


def test_band_dF_sign():
    from autopath.metadynamics.summary.analysis import Band
    b = Band((0, 1), (2, 3), "low", "t")
    x = np.array([0.5, 2.5]); w = np.array([0.8, 0.2])
    assert band_dF(x, w, b, 300.0) == pytest.approx(-KT_PER_K * 300.0 * np.log(4.0))


def _bv(t, dg):
    def f(lo, hi):
        m = (t >= lo) & (t < hi)
        return float(np.mean(dg[m])) if m.any() else float("nan")
    return f


def _wv(t, dg):
    def f(wins):
        m = np.concatenate([dg[(t >= lo) & (t < hi)] for lo, hi in wins])
        return float(m.mean()) if m.size else float("nan")
    return f


def _synthetic(n=101, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 100, n)
    return t, -30 + rng.normal(0, 1, n)


def test_block_analysis_values():
    t, dg = _synthetic()
    r = block_analysis(t, dg, _bv(t, dg), t_rej=0.2, n_blocks=10, n_boot=500, seed=0,
                       windows_value=_wv(t, dg))
    edges = np.linspace(20, 100, 11)
    exp = [dg[(t >= lo) & (t < hi)].mean() for lo, hi in zip(edges[:-2], edges[1:-1])]
    exp.append(dg[(t >= edges[-2]) & (t <= 100)].mean())
    assert r["block_dg"] == pytest.approx(exp)
    rng = np.random.default_rng(0)
    wins = [(lo, hi if k < 9 else np.nextafter(hi, np.inf)) for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    boots = []
    for _ in range(500):
        idx = rng.integers(0, 10, 10)
        boots.append(np.concatenate([dg[(t >= wins[i][0]) & (t < wins[i][1])] for i in idx]).mean())
    assert r["boot_sd"] == pytest.approx(np.std(boots))
    assert r["last_blocks_diff"] == pytest.approx(abs(exp[-1] - exp[-2]))
    assert r["time_mean"] == pytest.approx(dg[t >= 20].mean()) and r["n_finite"] == int((t >= 20).sum())
    assert r["n_valid_blocks"] == 10 and r["nan_blocks"] == []
    assert r == block_analysis(t, dg, _bv(t, dg), t_rej=0.2, n_blocks=10, n_boot=500, seed=0,
                       windows_value=_wv(t, dg))


def test_block_analysis_one_nan_block():
    t, dg = _synthetic()
    bv = _bv(t, dg)
    r = block_analysis(t, dg, lambda lo, hi: float("nan") if lo == pytest.approx(44.0) else bv(lo, hi),
                       t_rej=0.2, n_blocks=10, n_boot=200, windows_value=_wv(t, dg))
    assert r["nan_blocks"] == [3] and r["n_valid_blocks"] == 9 and np.isfinite(r["boot_sd"])


def test_block_analysis_all_nan_no_warnings():
    t, dg = _synthetic()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = block_analysis(t, np.full_like(dg, np.nan), lambda lo, hi: float("nan"), n_boot=50,
                           windows_value=lambda wins: float("nan"))
    assert np.isnan(r["boot_sd"]) and r["n_valid_blocks"] == 0 and np.isnan(r["time_mean"]) and r["n_finite"] == 0


def test_blocks_partition_frames():
    t = np.linspace(0, 10, 101)
    seen = []
    block_analysis(t, np.zeros(101), lambda lo, hi: seen.append(((t >= lo) & (t < hi)).astype(int)) or 0.0,
                   t_rej=0.2, n_blocks=7, n_boot=2, windows_value=lambda wins: 0.0)
    assert (np.sum(seen, 0) == (t >= 2.0).astype(int)).all()


def _hand_dg(run, t_max=None, burn=0.2):
    kt = KT_PER_K * run.temperature
    zs, lws = [], []
    for w in run.walkers:
        keep = np.ones(len(w.t), bool) if t_max is None else w.t <= t_max
        z, b, c = w.cvs["funnel_z"][keep], w.bias[keep], w.ct[keep]
        n0 = int(burn * len(z))
        zs.append(z[n0:] * 10.0); lws.append((b[n0:] - c[n0:]) / kt)
    z, lw = np.concatenate(zs), np.concatenate(lws)
    w = np.exp(lw - lw.max()); w /= w.sum()
    site = (z >= -1.5) & (z <= 4.8); blk = (z >= 12.0) & (z <= 22.5)
    return -kt * np.log(np.pi * 1.0 ** 2 / 1661.0 * w[site].sum() / (w[blk].sum() / 10.5))


def test_binding_value_by_hand(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    v = binding(run, bands(run)).pooled
    assert v["kind"] == "dG0" and v["unit"] == "kJ/mol"
    assert v["value"] == pytest.approx(_hand_dg(run), abs=1e-9)


def test_dg_series_value_by_hand(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    for w in run.walkers:
        w.bias = np.zeros_like(w.bias); w.ct = np.zeros_like(w.ct)
    s = dg_series(run, bands(run), step_ns=0.5).pooled
    k = int(np.argmin(abs(s["t"] - 2.0)))
    assert s["dg"][k] == pytest.approx(_hand_dg(run, t_max=s["t"][k]), abs=1e-9)


def test_funnel_without_r_cylinder_not_supported(tmp_path):
    fp = {k: v for k, v in FP.items() if k != "R_cylinder"}
    run = load_run(make_run(str(tmp_path / "r")), funnel=fp)
    with pytest.raises(ValueError, match="not supported"):
        binding(run, bands(run))


def test_block_value_window(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    z, w = _frames_in_window(run.walkers, run, "funnel_z", 1.0, 2.0)
    assert len(z) == sum(int(((x.t >= 1) & (x.t < 2)).sum()) for x in run.walkers)
    assert w.sum() == pytest.approx(1.0)


def test_pooled_weights_keep_relative_scale(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    kt = KT_PER_K * run.temperature
    a, b = run.walkers
    a.bias = np.zeros_like(a.bias); a.ct = np.zeros_like(a.ct)
    b.bias = np.full_like(b.bias, kt * np.log(4.0)); b.ct = np.zeros_like(b.ct)
    for fn in (lambda: _frames(run.walkers, run, "funnel_z", 0.0),
               lambda: _frames_in_window(run.walkers, run, "funnel_z", 0.0, 1e9)):
        _, w = fn()
        n = len(a.t)
        assert w[n:].sum() / w[:n].sum() == pytest.approx(4.0)


def test_dg_series_ends_at_T_and_blocks_cover_frames(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    T = max(w.t[-1] for w in run.walkers)
    assert abs(T / 0.5 - round(T / 0.5)) > 1e-6
    s = dg_series(run, bands(run), step_ns=0.5).pooled
    assert s["t"][-1] == pytest.approx(T)
    counts = []

    def bv(lo, hi):
        counts.append(len(_frames_in_window(run.walkers, run, "funnel_z", lo, hi)[0]))
        return 0.0
    block_analysis(s["t"], s["dg"], bv, t_rej=0.2, n_blocks=10, n_boot=2, windows_value=lambda wins: 0.0)
    assert sum(counts) == sum(int((w.t >= 0.2 * T).sum()) for w in run.walkers)


def test_block_analysis_needs_windows_value():
    t, dg = _synthetic()
    with pytest.raises(ValueError, match="not supported"):
        block_analysis(t, dg, _bv(t, dg))


def test_frame_bootstrap_on_run_matches_hand_loop(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    b = bands(run)
    s = dg_series(run, b, step_ns=0.5).pooled
    bvf = lambda lo, hi: _value(run, b, "funnel_z", *_frames_in_window(run.walkers, run, "funnel_z", lo, hi))["value"]
    wvf = lambda wins: _value(run, b, "funnel_z", *_frames_in_windows(run.walkers, run, "funnel_z", wins, {}))["value"]
    r = block_analysis(s["t"], s["dg"], bvf, n_boot=40, seed=3, windows_value=wvf)
    T = s["t"][-1]; edges = np.linspace(0.2 * T, T, 11)
    wins = [(lo, hi if k < 9 else np.nextafter(hi, np.inf)) for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(40):
        idx = rng.integers(0, 10, 10)
        z = np.concatenate([w.cvs["funnel_z"][(w.t >= wins[i][0]) & (w.t < wins[i][1])] for i in idx for w in run.walkers])
        lw = np.concatenate([((w.bias - w.ct) / (KT_PER_K * run.temperature))[(w.t >= wins[i][0]) & (w.t < wins[i][1])]
                             for i in idx for w in run.walkers])
        p = np.exp(lw - lw.max()); p /= p.sum()
        vals.append(_value(run, b, "funnel_z", z, p)["value"])
    vals = np.array(vals); fin = np.isfinite(vals)
    assert r["boot_sd"] == pytest.approx(np.std(vals[fin])) and r["n_boot_nan"] == int((~fin).sum())


def test_repeated_window_doubles_weight(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    kt = KT_PER_K * run.temperature
    for w in run.walkers:
        w.bias = np.zeros_like(w.bias); w.ct = np.zeros_like(w.ct)
    z1, w1 = _frames_in_windows(run.walkers, run, "funnel_z", [(1.0, 2.0), (2.0, 3.0)])
    z2, w2 = _frames_in_windows(run.walkers, run, "funnel_z", [(1.0, 2.0), (1.0, 2.0), (2.0, 3.0)])
    n1 = sum(int(((x.t >= 1) & (x.t < 2)).sum()) for x in run.walkers)
    n2 = sum(int(((x.t >= 2) & (x.t < 3)).sum()) for x in run.walkers)
    assert len(z2) == len(z1) + n1 and len(z1) == n1 + n2
    # uniform raw weights: the doubled window carries twice the weight mass
    assert w2[:n1].sum() / w2[-n2:].sum() == pytest.approx(n1 / n2)
    assert w2[:2 * n1].sum() / w2[-n2:].sum() == pytest.approx(2 * n1 / n2)


def test_contacts_band_dF_sign(tmp_path):
    from autopath.metadynamics.summary.analysis import Band
    run = load_run(make_run(str(tmp_path / "r"), cvs=("contacts",), monitors=(), bounds={"contacts": (0.0, 6.0)}))
    for w in run.walkers:
        w.bias = np.zeros_like(w.bias); w.ct = np.zeros_like(w.ct)
        n = len(w.t)
        x = np.full(n, 3.0)
        x[: int(0.4 * n)] = 5.8                                  # bound (high contacts): 4x the unbound time
        x[int(0.4 * n): int(0.5 * n)] = 0.2
        w.cvs["contacts"] = x
    b = bands(run)
    assert b["contacts"].bound_end == "high"
    v = binding(run, b, cv="contacts", burn=0.0).pooled
    pb = sum(int((w.cvs["contacts"] == 5.8).sum()) for w in run.walkers)
    pu = sum(int((w.cvs["contacts"] == 0.2).sum()) for w in run.walkers)
    assert v["kind"] == "band_dF" and v["value"] < 0
    assert v["value"] == pytest.approx(-KT_PER_K * 300.0 * np.log(pb / pu))


def test_isolated_aggregates_count_undefined_walkers():
    import types
    from autopath.metadynamics.summary.analysis import apply
    W = [types.SimpleNamespace(rid=r) for r in ("a", "b", "c")]
    run = types.SimpleNamespace(layout="isolated", walkers=W)
    vals = {"a": -20.0, "b": float("nan"), "c": float("nan")}
    c = apply(run, lambda ws: {"value": vals[ws[0].rid]}, scalar=lambda r: r["value"])
    assert c.n_walkers == 3 and c.n_defined == 1 and c.mean == -20.0
    assert np.isnan(c.sd) and np.isnan(c.spread)


def test_series_ends_at_longest_walker_and_equals_binding(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    f = os.path.join(d, "COLVAR_w1.npy")
    np.save(f, np.load(f)[:120])                               # late-joining walker
    run = load_run(d, funnel=FP)
    b = bands(run)
    s = dg_series(run, b, step_ns=0.1).pooled
    assert s["t"][-1] == pytest.approx(max(w.t[-1] for w in run.walkers))
    assert s["dg"][-1] == binding(run, b).pooled["value"]
    assert s["ess"][-1] == binding(run, b).pooled["ess"] and len(s["ess"]) == len(s["t"])


def test_series_shorter_walker_contributes_all_frames(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    f = os.path.join(d, "COLVAR_w1.npy")
    np.save(f, np.load(f)[:120])
    run = load_run(d, funnel=FP)
    for w in run.walkers:
        w.bias = np.zeros_like(w.bias); w.ct = np.zeros_like(w.ct)
    s = dg_series(run, bands(run), step_ns=0.5).pooled
    k = int(np.argmin(abs(s["t"] - 3.0)))
    assert s["t"][k] > run.walkers[1].t[-1]
    assert s["dg"][k] == pytest.approx(_hand_dg(run, t_max=s["t"][k]), abs=1e-9)
    n_used = sum(int((w.t <= s["t"][k]).sum()) - int(0.2 * (w.t <= s["t"][k]).sum()) for w in run.walkers)
    assert s["ess"][k] == pytest.approx(n_used)                 # uniform weights


def test_late_walker_keeps_blocks(tmp_path):
    from autopath.metadynamics.summary.analysis import fe_numbers
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    from ._summary_fixtures import add_fresh
    add_fresh(d, "w2", n_hills=1, n_rows=4)
    run = load_run(d, funnel=FP)
    assert len(run.walkers) == 3 and len(run.walkers[2].t) == 3
    out = fe_numbers(run, bands(run), step_ns=0.1)
    assert out["blocks"] is not None and len(out["series"].pooled["t"]) > 12


def test_kish_ess():
    from autopath.metadynamics.summary.analysis import kish_ess
    assert kish_ess(np.full(250, 0.3)) == pytest.approx(250)
    w = np.full(1000, 1e-9); w[0] = 1.0
    assert kish_ess(w) == pytest.approx(1.0, abs=1e-3)
    assert kish_ess(np.zeros(0)) == 0.0


def test_binding_carries_ess(tmp_path):
    from autopath.metadynamics.summary.analysis import kish_ess
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    v = binding(run, bands(run)).pooled
    _, w = _frames(run.walkers, run, "funnel_z", 0.2)
    assert v["ess"] == pytest.approx(kish_ess(w)) and v["n_frames"] == len(w)
    assert 1.0 <= v["ess"] <= len(w)


def test_fe_numbers_blocks_have_ess_and_min_points(tmp_path):
    from autopath.metadynamics.summary.analysis import fe_numbers, kish_ess
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2), funnel=FP)
    b = bands(run)
    out = fe_numbers(run, b, step_ns=0.1, n_boot=20)
    assert out["series"].pooled is None and set(out["blocks"]) == {"w0", "w1"}
    blk = out["blocks"]["w0"]
    assert len(blk["block_ess"]) == 10
    T = out["series"].per_walker["w0"]["t"][-1]
    lo, hi = 0.2 * T, 0.2 * T + 0.08 * T
    w0 = run.walkers[0]
    assert blk["block_ess"][0] == pytest.approx(kish_ess(_frames_in_window([w0], run, "funnel_z", lo, hi)[1]))
    out = fe_numbers(run, b, step_ns=0.1, min_points=10 ** 6)
    assert out["blocks"] == {"w0": None, "w1": None}
    assert any(f"< {10 ** 6} points" in s for s in out["skipped"])


def test_isolated_different_walkers_not_pooled(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=2), funnel=FP)
    for w, frac in zip(run.walkers, (0.5, 0.2)):
        w.bias = np.zeros_like(w.bias); w.ct = np.zeros_like(w.ct)
        n = len(w.t)
        z = np.full(n, 1.8)                                        # bulk
        z[: int(frac * n)] = 0.1                                   # bound
        w.cvs["funnel_z"] = z
    b = bands(run)
    c = binding(run, b, burn=0.0)
    v0, v1 = c.per_walker["w0"]["value"], c.per_walker["w1"]["value"]
    assert c.pooled is None and np.isfinite(v0) and np.isfinite(v1) and abs(v0 - v1) > 1.0
    pooled = _value(run, b, "funnel_z", *_frames(run.walkers, run, "funnel_z", 0.0))["value"]
    assert min(abs(pooled - v0), abs(pooled - v1)) > 0.1
    assert c.mean == pytest.approx((v0 + v1) / 2) and c.spread == pytest.approx(abs(v0 - v1))
