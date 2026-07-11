import numpy as np
import pandas as pd
import pytest
from autopath.pulling.Estimators import ForceEstimator, ESTIMATOR_REGISTRY, FrictionEstimator
from autopath.pulling.SMDData import SMDData


def test_estimate_dG_is_raw_mean_work():
    W = np.array([10.0, 12.0, 8.0, 14.0])
    out = ForceEstimator.estimate_dG(W, beta=0.4)
    assert out['Wmean'] == pytest.approx(W.mean())
    assert out['dG'] == pytest.approx(W.mean())          # raw work, no correction
    assert np.isnan(out['Wdiss'])                         # friction is force-based, not Wdiss
    assert ForceEstimator.estimate_dG(np.array([]), beta=0.4) is None


def test_force_in_registry():
    assert ESTIMATOR_REGISTRY['force'] is ForceEstimator
    assert ForceEstimator().name == 'force'


class _StubSMD:
    """Minimal SMDData stand-in for exercising ForceEstimator.fit_transform."""
    def __init__(self, raw_data, protocol_grids, beta=0.4):
        self.raw_data = raw_data
        self.protocol_grids = protocol_grids
        self.beta = beta
        self.captured = None
        self.captured_name = None

    def add_estimator_results(self, name, df):
        # mirror the real SMDData: tag rows with the estimator name
        df = df.copy()
        df['estimator'] = name
        self.captured_name = name
        self.captured = df


def test_fit_transform_records_fmean_and_dG_equals_wmean():
    raw = pd.DataFrame({
        'step':  [0, 0, 1, 1],
        'speed': [0.01, 0.01, 0.01, 0.01],
        'path':  ['p0', 'p0', 'p0', 'p0'],
        'work':  [10.0, 20.0, 30.0, 50.0],
        'force': [100.0, 200.0, 300.0, 500.0],
    })
    grids = {0.01: pd.DataFrame({'step': [0, 1],
                                 'r_target_protocol': [1.0, 1.1]})}
    stub = _StubSMD(raw, grids)
    ForceEstimator().fit_transform(stub)
    res = stub.captured.sort_values('step').reset_index(drop=True)
    assert stub.captured_name == 'force'
    assert 'Fmean' in res.columns
    # step 0: force mean = 150, work mean = 15 ; step 1: force mean = 400, work mean = 40
    np.testing.assert_allclose(res['Fmean'].to_numpy(), [150.0, 400.0])
    np.testing.assert_allclose(res['dG'].to_numpy(), [15.0, 40.0])   # dG == Wmean
    assert res['Wdiss'].isna().all()
    assert (res['estimator'] == 'force').all()


def _results(**dg_by_est):
    frames = []
    for est, dg in dg_by_est.items():
        frames.append(pd.DataFrame({'estimator': est, 'dG': dg,
                                     'step': range(len(dg))}))
    return pd.concat(frames, ignore_index=True)


def test_reference_prefers_fewer_negative_bins():
    # cumulant has negatives, jarzynski none -> jarzynski (the betasigma~10 regime)
    r = _results(cumulant=[-1.0, 5.0, 10.0], jarzynski=[1.0, 6.0, 11.0])
    assert SMDData.choose_reference_estimator(r) == 'jarzynski'


def test_reference_ties_go_to_cumulant():
    r = _results(cumulant=[1.0, 5.0, 10.0], jarzynski=[1.0, 6.0, 11.0])
    assert SMDData.choose_reference_estimator(r) == 'cumulant'


def test_reference_never_selects_force():
    # force dG is all non-negative -> 0 negatives, but must be excluded
    r = _results(cumulant=[-1.0, 5.0], jarzynski=[-1.0, 6.0], force=[9.0, 10.0])
    assert SMDData.choose_reference_estimator(r) in ('cumulant', 'jarzynski')


def _force_results_linear(feq_of_step, gamma_of_step, speeds, paths=('p0',)):
    """Build force results where Fmean(step,speed) = Feq(step) + Gamma(step)*speed."""
    rows = []
    for step, (feq, gam) in enumerate(zip(feq_of_step, gamma_of_step)):
        for sp in speeds:
            for p in paths:
                rows.append(dict(estimator='force', step=step, path=p,
                                 speed=sp, r_coord=float(step) * 0.1,
                                 Fmean=feq + gam * sp))
    return pd.DataFrame(rows)


def test_force_regression_recovers_feq_and_gamma():
    speeds = [0.001, 0.005, 0.01]
    feq = [0.0, 100.0, 250.0, 300.0, 260.0]
    gam = [500.0, 800.0, 1200.0, 900.0, 400.0]
    fr = _force_results_linear(feq, gam, speeds)
    weights = {sp: {'p0': 1.0} for sp in speeds}
    out = FrictionEstimator().gamma_from_force_regression(fr, weights)
    out = out.sort_values('step')
    np.testing.assert_allclose(out['Feq'].to_numpy(), feq, atol=1e-6)
    np.testing.assert_allclose(out['Gamma'].to_numpy(), gam, atol=1e-6)
    assert (out['method'] == 'regression').all()
    assert (out['estimator'] == 'force').all()


def test_force_derivative_matches_slope():
    speeds = [0.001, 0.005, 0.01]
    feq = [0.0, 100.0, 250.0]
    gam = [500.0, 800.0, 1200.0]
    fr = _force_results_linear(feq, gam, speeds)
    weights = {sp: {'p0': 1.0} for sp in speeds}
    out = FrictionEstimator().gamma_from_force_derivative(fr, weights)
    # For exactly-linear data, per-speed (Fbar-Feq)/v == gamma at every speed.
    for sp in speeds:
        g = out[out['speed'] == sp].sort_values('step')
        np.testing.assert_allclose(g['Gamma'].to_numpy(), gam, atol=1e-6)
    assert (out['method'] == 'derivative').all()
