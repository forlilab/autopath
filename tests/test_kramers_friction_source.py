import numpy as np
import pandas as pd
import pytest

from autopath.pulling.Estimators import KramersEstimator

R = np.round(np.arange(1.0, 2.0 + 1e-9, 0.01), 6)
STEP = np.arange(R.size)
G_DERIV = {0.01: 3e3, 0.02: 5e3}
G_REG = 9e3


G_FORCE = 7e3


def _inputs(speeds, force=True):
    dG = 30.0 * np.exp(-((R - 1.5) / 0.1) ** 2)
    pmf = pd.concat([pd.DataFrame({"step": STEP, "r_coord": R, "dG": dG, "speed": v,
                                   "estimator": "cumulant"}) for v in speeds])
    fric = [pd.DataFrame({"step": STEP, "r_coord": R, "Gamma": G_DERIV[v], "speed": v,
                          "method": "derivative", "estimator": "cumulant"}) for v in speeds]
    if len(speeds) > 1:
        fric.append(pd.DataFrame({"step": STEP, "r_coord": R, "Gamma": G_REG, "speed": np.nan,
                                  "method": "regression", "estimator": "cumulant"}))
        if force:
            fric.append(pd.DataFrame({"step": STEP, "r_coord": R, "Gamma": G_FORCE,
                                      "speed": np.nan, "method": "regression",
                                      "estimator": "force"}))
    return pmf, pd.concat(fric)


def _gammas(speeds, source, force=True):
    pmf, fric = _inputs(speeds, force)
    prof = KramersEstimator._iter_profiles(pmf, fric, None, "cumulant", False, source)
    return {s: set(p["Gamma"]) for p, _, s in prof}


def test_auto_two_speeds_uses_force_friction():
    assert _gammas([0.01, 0.02], "auto") == {0.01: {G_FORCE}, 0.02: {G_FORCE}}


def test_auto_single_speed_uses_derivative():
    assert _gammas([0.01], "auto") == {0.01: {G_DERIV[0.01]}}


def test_explicit_derivative_two_speeds():
    assert _gammas([0.01, 0.02], "derivative") == {v: {g} for v, g in G_DERIV.items()}


def test_explicit_regression_uses_wdiss_regression():
    assert _gammas([0.01, 0.02], "regression") == {0.01: {G_REG}, 0.02: {G_REG}}


def test_auto_without_force_raises():
    pmf, fric = _inputs([0.01, 0.02], force=False)
    with pytest.raises(ValueError, match="force estimator"):
        KramersEstimator().compute(pmf, fric)


def test_compute_runs_and_rejects_bad_source():
    pmf, fric = _inputs([0.01, 0.02])
    out = KramersEstimator().compute(pmf, fric, friction_source_work="regression")
    assert len(out) == 2
    with pytest.raises(ValueError):
        KramersEstimator().compute(pmf, fric, friction_source_work="bogus")
