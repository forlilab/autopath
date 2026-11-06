import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

import autopath.pulling.Estimators as E
from autopath.pulling.Estimators import FrictionEstimator, extrapolate_to_v0

SPEEDS = (0.005, 0.01, 0.02)
N_STEP = 40


def _table(n_rep=4, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    true = 1000 + 400 * np.sin(np.linspace(0, 3, N_STEP))
    rows = []
    for s in range(N_STEP):
        for v in SPEEDS:
            for _ in range(n_rep):
                rows.append((s, 1.0 + 0.01 * s, v, "jarzynski",
                             5 * np.cos(s / 7) + true[s] * v + rng.normal(0, noise)))
    return pd.DataFrame(rows, columns=["step", "r_coord", "speed", "estimator", "Wdiss"]), true


def test_mixed_reads_random_slopes():
    df, true = _table()
    mixed = extrapolate_to_v0(df, param="Wdiss", model="mixed")
    assert (mixed["model"] == "mixedlm").all()
    s_mix = mixed.sort_values("step")["Wdiss_slope"].to_numpy()
    # a fixed-slope-only reading (the random slopes dropped) is a constant
    assert np.std(s_mix) > 0.25 * np.std(true)
    assert np.sqrt(np.mean((s_mix - true) ** 2)) < 0.9 * np.sqrt(np.mean((true.mean() - true) ** 2))
    # random intercepts are read too: intercept profile is not flat
    assert np.std(mixed.sort_values("step")["Wdiss"]) > 1.0


def test_ols_default_unchanged():
    df, _ = _table(n_rep=1)
    out = extrapolate_to_v0(df, param="Wdiss").sort_values("step")
    for s, g in df.groupby("step"):
        lr = linregress(g["speed"], g["Wdiss"])
        row = out[out["step"] == s].iloc[0]
        assert row["Wdiss_slope"] == pytest.approx(lr.slope)
        assert row["Wdiss"] == pytest.approx(lr.intercept)
        assert row["model"] == "linear"
    fe = FrictionEstimator(use_spline=False)
    assert fe.friction_model == "ols"
    assert (fe.gamma_from_wdiss_regression(df, estimator="jarzynski")["model"] == "linear").all()


def test_mixed_failure_falls_back_to_ols(monkeypatch, caplog):
    df, _ = _table(n_rep=1)

    def boom(*a, **k):
        raise RuntimeError("no fit")

    monkeypatch.setattr(E, "_mixed_step_rows", boom)
    with caplog.at_level("WARNING"):
        out = extrapolate_to_v0(df, param="Wdiss", model="mixed")
    assert (out["model"] == "ols_fallback").all()
    assert "mixed-effects fit failed" in caplog.text
    ols = extrapolate_to_v0(df, param="Wdiss")
    np.testing.assert_allclose(out["Wdiss_slope"], ols["Wdiss_slope"])


def test_friction_estimator_mixed_and_bad_value():
    df, _ = _table()
    out = FrictionEstimator(use_spline=False, friction_model="mixed") \
        .gamma_from_wdiss_regression(df, estimator="jarzynski")
    assert (out["model"] == "mixedlm").all() and np.isfinite(out["Gamma"]).all()
    with pytest.raises(ValueError):
        FrictionEstimator(friction_model="bogus")
