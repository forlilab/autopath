import numpy as np
import pytest
from autopath.metadynamics.summary.data import load_run
from autopath.metadynamics.summary.analysis import weights, apply, fes_bias, fes_reweighted
from ._summary_fixtures import make_run


def test_weights_are_ct_reweighting(tmp_path):
    run = load_run(make_run(str(tmp_path / "r")))
    w0 = run.walkers[0]
    m, w = weights(w0, 300.0, burn=0.2)
    lw = (w0.bias[m] - w0.ct[m]) / (0.0083144626 * 300.0)
    ref = np.exp(lw - lw.max()); ref /= ref.sum()
    assert np.allclose(w, ref) and m.sum() == len(w0.t) - int(0.2 * len(w0.t))


def test_apply_pools_shared_and_never_isolated(tmp_path):
    shared = load_run(make_run(str(tmp_path / "s"), layout="shared", n_walkers=3))
    iso = load_run(make_run(str(tmp_path / "i"), layout="isolated", n_walkers=3))
    calls = []
    f = lambda ws: calls.append(len(ws)) or len(ws)
    c = apply(shared, f)
    assert c.pooled == 3 and c.per_walker == {} and calls == [3]
    calls.clear()
    c = apply(iso, f, scalar=float)
    assert c.pooled is None and calls == [1, 1, 1] and set(c.per_walker) == {"w0", "w1", "w2"}
    assert c.mean == 1.0 and c.sd == 0.0 and c.spread == 0.0


def test_fes_bias_prefactor(tmp_path):
    run_wt = load_run(make_run(str(tmp_path / "a"), tempering="wt"))
    run_tt = load_run(make_run(str(tmp_path / "b"), tempering="tt"))
    (_, fwt), (_, ftt) = fes_bias(run_wt).pooled, fes_bias(run_tt).pooled
    assert np.nanmax(fwt) == pytest.approx(np.nanmax(ftt) * 20.0 / 19.0, rel=1e-6)


def test_fes_reweighted_1d_and_2d(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()))
    xc, F = fes_reweighted(run, "funnel_z").pooled
    assert F.shape == xc.shape and np.nanmin(F) == 0.0
    xc, yc, F2 = fes_reweighted(run, "funnel_z", "learned_sw").pooled
    assert F2.shape == (len(xc), len(yc))


def test_fes_reweighted_needs_reweight(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), reweight=False))
    with pytest.raises(ValueError, match="no REWEIGHT data"):
        fes_reweighted(run, "funnel_z")
