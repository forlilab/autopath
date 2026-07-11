import numpy as np
import pandas as pd
import pytest
from autopath.pulling.Estimators import ForceEstimator, ESTIMATOR_REGISTRY


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
