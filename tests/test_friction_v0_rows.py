import numpy as np
import pandas as pd
import pandas.testing as pdt

from autopath.pulling.Estimators import FrictionEstimator

R = np.linspace(1.0, 2.0, 21)


def _pmfs(with_v0):
    rows = []
    for v in (0.005, 0.01, 0.02):
        rows.append(pd.DataFrame({"step": np.arange(R.size), "r_coord": R, "speed": v,
                                  "estimator": "jarzynski", "Wdiss": 1e3 * v * (R - 1.0)}))
    if with_v0:  # off-line v->0 rows, as folded into mixture_pmfs
        rows.append(pd.DataFrame({"step": np.arange(R.size), "r_coord": R, "speed": 0.0,
                                  "estimator": "jarzynski", "Wdiss": 50.0}))
    return pd.concat(rows, ignore_index=True)


def test_v0_rows_do_not_enter_friction():
    fe = FrictionEstimator(use_spline=False)
    deriv = fe.gamma_from_wdiss_derivative(_pmfs(True), estimator="jarzynski")
    assert (deriv["speed"] > 0).all()
    assert np.isfinite(deriv["Gamma"]).all()
    pdt.assert_frame_equal(fe.gamma_from_wdiss_regression(_pmfs(True), estimator="jarzynski"),
                           fe.gamma_from_wdiss_regression(_pmfs(False), estimator="jarzynski"))
