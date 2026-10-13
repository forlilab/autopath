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


def test_block_analysis_reproducible():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 100, 101); dg = -30 + rng.normal(0, 1, 101)

    def block_value(mask_t):
        return float(dg[mask_t].mean())
    a = block_analysis(t, dg, block_value, t_rej=0.2, n_blocks=10, n_boot=500, seed=0)
    b = block_analysis(t, dg, block_value, t_rej=0.2, n_blocks=10, n_boot=500, seed=0)
    assert a == b and len(a["block_dg"]) == 10 and a["boot_sd"] > 0
    assert a["time_mean"] == pytest.approx(dg[t >= 20].mean())


def test_dg_series_shape(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    s = dg_series(run, bands(run), step_ns=0.5).pooled
    assert len(s["t"]) == len(s["dg"]) and s["t"][0] > 0


def test_block_value_window(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")), funnel=FP)
    z, w = _frames_in_window(run.walkers, run, "funnel_z", 1.0, 2.0)
    assert len(z) == sum(int(((x.t >= 1) & (x.t <= 2)).sum()) for x in run.walkers)
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
