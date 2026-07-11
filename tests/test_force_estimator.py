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
