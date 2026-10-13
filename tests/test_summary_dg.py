import warnings

import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics.summary.analysis import (KT_PER_K, _frames, _frames_in_window, band_dF, bands, binding,
                                                    block_analysis, dg_from_frames, dg_series)
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


def _synthetic(n=101, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 100, n)
    return t, -30 + rng.normal(0, 1, n)


def test_block_analysis_values():
    t, dg = _synthetic()
    r = block_analysis(t, dg, _bv(t, dg), t_rej=0.2, n_blocks=10, n_boot=500, seed=0)
    edges = np.linspace(20, 100, 11)
    exp = [dg[(t >= lo) & (t < hi)].mean() for lo, hi in zip(edges[:-2], edges[1:-1])]
    exp.append(dg[(t >= edges[-2]) & (t <= 100)].mean())
    assert r["block_dg"] == pytest.approx(exp)
    rng = np.random.default_rng(0); eb = np.array(exp)
    boots = [eb[rng.integers(0, 10, 10)].mean() for _ in range(500)]
    assert r["boot_sd"] == pytest.approx(np.std(boots))
    assert r["last_blocks_diff"] == pytest.approx(abs(exp[-1] - exp[-2]))
    assert r["time_mean"] == pytest.approx(dg[t >= 20].mean()) and r["n_finite"] == int((t >= 20).sum())
    assert r["n_valid_blocks"] == 10 and r["nan_blocks"] == []
    assert r == block_analysis(t, dg, _bv(t, dg), t_rej=0.2, n_blocks=10, n_boot=500, seed=0)


def test_block_analysis_one_nan_block():
    t, dg = _synthetic()
    bv = _bv(t, dg)
    r = block_analysis(t, dg, lambda lo, hi: float("nan") if lo == pytest.approx(44.0) else bv(lo, hi),
                       t_rej=0.2, n_blocks=10, n_boot=200)
    assert r["nan_blocks"] == [3] and r["n_valid_blocks"] == 9 and np.isfinite(r["boot_sd"])


def test_block_analysis_all_nan_no_warnings():
    t, dg = _synthetic()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = block_analysis(t, np.full_like(dg, np.nan), lambda lo, hi: float("nan"), n_boot=50)
    assert np.isnan(r["boot_sd"]) and r["n_valid_blocks"] == 0 and np.isnan(r["time_mean"]) and r["n_finite"] == 0


def test_blocks_partition_frames():
    t = np.linspace(0, 10, 101)
    seen = []
    block_analysis(t, np.zeros(101), lambda lo, hi: seen.append(((t >= lo) & (t < hi)).astype(int)) or 0.0,
                   t_rej=0.2, n_blocks=7, n_boot=2)
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
    T = min(w.t[-1] for w in run.walkers)
    assert abs(T / 0.5 - round(T / 0.5)) > 1e-6
    s = dg_series(run, bands(run), step_ns=0.5).pooled
    assert s["t"][-1] == pytest.approx(T)
    counts = []

    def bv(lo, hi):
        counts.append(len(_frames_in_window(run.walkers, run, "funnel_z", lo, hi)[0]))
        return 0.0
    block_analysis(s["t"], s["dg"], bv, t_rej=0.2, n_blocks=10, n_boot=2)
    assert sum(counts) == sum(int((w.t >= 0.2 * T).sum()) for w in run.walkers)
