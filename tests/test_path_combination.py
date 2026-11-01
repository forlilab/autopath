"""Per-path profile recombination: Eq. 21 (exponential, p_neq) vs linear (p_eq)."""
import numpy as np
import pandas as pd
import pytest

from autopath.pulling.AnalysisSMD import SMDAnalysis
from autopath.pulling.Estimators import (
    JarzynskiEstimator,
    calculate_weighted_pmf,
)
from autopath.pulling.SMDData import SMDData

SPEED = 0.005
BETA = 1.0 / (0.008314462618 * 300.0)


class _Stub(SMDData):
    """SMDData built from an in-memory work table (no log files)."""

    def __init__(self, raw_data: pd.DataFrame, steps):
        self.beta = BETA
        self.raw_data = raw_data
        self.results = None
        r = np.linspace(1.0, 2.0, len(steps))
        self.protocol_grids = {
            SPEED: pd.DataFrame({"step": list(steps), "r_target_protocol": r})
        }


def _work_table(n_by_path, mean_by_path, sd_by_path, steps, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for path, n in n_by_path.items():
        for i in range(n):
            for s in steps:
                w = (s + 1) * mean_by_path[path] + rng.normal(0.0, sd_by_path[path])
                rows.append({"trajname": f"{path}_t{i}", "speed": SPEED,
                             "step": s, "path": path, "work": w})
    return pd.DataFrame(rows)


def _results_stub(per_path, n_by_path, estimator="jarzynski"):
    """Stub with given per-path results ``{path: (dG, Wmean)}`` arrays."""
    steps = list(range(len(next(iter(per_path.values()))[0])))
    raw = pd.DataFrame([
        {"trajname": f"{p}_t{i}", "speed": SPEED, "step": 0, "path": p, "work": 0.0}
        for p, n in n_by_path.items() for i in range(n)
    ])
    smd = _Stub(raw, steps)
    r = smd.protocol_grids[SPEED]["r_target_protocol"].to_numpy()
    rows = []
    for p, (dG, Wm) in per_path.items():
        for s in steps:
            wdiss = np.nan if estimator == "force" else Wm[s] - dG[s]
            rows.append({"step": s, "speed": SPEED, "path": p, "r_coord": r[s],
                         "n_samples": n_by_path[p], "dG": dG[s], "Wmean": Wm[s],
                         "Wdiss": wdiss, "estimator": estimator})
    smd.results = pd.DataFrame(rows)
    return smd, steps


def _weights(p_eq, estimator="jarzynski"):
    return {estimator: {SPEED: dict(p_eq)}}


PER_PATH = {
    "pA": (np.array([0.0, 5.0, 12.0, 20.0]), np.array([1.0, 9.0, 20.0, 35.0])),
    "pB": (np.array([0.0, 8.0, 10.0, 30.0]), np.array([2.0, 15.0, 30.0, 60.0])),
}
N_BY_PATH = {"pA": 3, "pB": 7}
P_EQ = {"pA": 0.8, "pB": 0.2}


def test_eq21_matches_closed_form_with_unequal_paths():
    smd, _ = _results_stub(PER_PATH, N_BY_PATH)
    out = calculate_weighted_pmf(smd, weights_by_estimator=_weights(P_EQ),
                                 path_combination="eq21").sort_values("step")
    pA, pB = 0.3, 0.7
    (dA, wA), (dB, wB) = PER_PATH["pA"], PER_PATH["pB"]
    dG_exp = -np.log(pA * np.exp(-BETA * dA) + pB * np.exp(-BETA * dB)) / BETA
    wdiss_exp = (pA * wA + pB * wB) - dG_exp
    np.testing.assert_allclose(out["dG"].to_numpy(), dG_exp, rtol=0, atol=1e-10)
    np.testing.assert_allclose(out["Wdiss"].to_numpy(), wdiss_exp, rtol=0, atol=1e-10)


def test_eq21_renormalises_over_paths_present():
    dB = PER_PATH["pB"][0].copy()
    dB[2] = np.nan
    smd, _ = _results_stub({"pA": PER_PATH["pA"], "pB": (dB, PER_PATH["pB"][1])},
                           N_BY_PATH)
    out = calculate_weighted_pmf(smd, weights_by_estimator=_weights(P_EQ),
                                 path_combination="eq21").set_index("step")
    assert out.loc[2, "dG"] == pytest.approx(PER_PATH["pA"][0][2], abs=1e-10)
    assert out.loc[2, "Wdiss"] == pytest.approx(
        PER_PATH["pA"][1][2] - PER_PATH["pA"][0][2], abs=1e-10)


def test_linear_is_peq_weighted_mean():
    smd, _ = _results_stub(PER_PATH, N_BY_PATH)
    out = calculate_weighted_pmf(smd, weights_by_estimator=_weights(P_EQ),
                                 path_combination="linear").sort_values("step")
    (dA, wA), (dB, wB) = PER_PATH["pA"], PER_PATH["pB"]
    dG_exp = (0.8 * dA + 0.2 * dB) / 1.0
    wdiss_exp = 0.8 * (wA - dA) + 0.2 * (wB - dB)
    np.testing.assert_allclose(out["dG"].to_numpy(), dG_exp, atol=1e-12)
    np.testing.assert_allclose(out["Wdiss"].to_numpy(), wdiss_exp, atol=1e-12)
    assert list(out.columns) == ["step", "speed", "estimator", "r_coord", "dG", "Wdiss"]


def test_force_estimator_mixed_linearly_in_eq21_mode():
    smd, _ = _results_stub(PER_PATH, N_BY_PATH, estimator="force")
    w = _weights(P_EQ, estimator="force")
    eq21 = calculate_weighted_pmf(smd, weights_by_estimator=w, path_combination="eq21")
    lin = calculate_weighted_pmf(smd, weights_by_estimator=w, path_combination="linear")
    pd.testing.assert_frame_equal(eq21, lin)
    assert eq21["Wdiss"].isna().all()


def test_single_path_eq21_equals_linear():
    per = {"pA": PER_PATH["pA"]}
    smd, _ = _results_stub(per, {"pA": 4})
    w = _weights({"pA": 1.0})
    eq21 = calculate_weighted_pmf(smd, weights_by_estimator=w, path_combination="eq21")
    lin = calculate_weighted_pmf(smd, weights_by_estimator=w, path_combination="linear")
    pd.testing.assert_frame_equal(eq21, lin, check_exact=False, atol=1e-10)
    full = calculate_weighted_pmf(smd, weights_by_estimator=w, path_combination="eq21",
                                  weight_cols=["dG", "Wdiss", "Wmean"])
    np.testing.assert_allclose(full["Wdiss"], full["Wmean"] - full["dG"], atol=1e-12)


def _two_path_jarzynski(steps):
    n_by_path = {"pA": 4, "pB": 9}
    raw = _work_table(n_by_path, {"pA": 10.0, "pB": 14.0}, {"pA": 1.0, "pB": 3.0}, steps)
    smd = _Stub(raw, steps)
    JarzynskiEstimator().fit_transform(smd)
    return smd, raw, n_by_path


def test_eq21_reproduces_pooled_jarzynski():
    steps = list(range(6))
    smd, raw, _ = _two_path_jarzynski(steps)
    out = calculate_weighted_pmf(smd, weights_by_estimator=_weights({"pA": 0.9, "pB": 0.1}),
                                 path_combination="eq21").set_index("step")
    for s in steps:
        pooled = JarzynskiEstimator.estimate_dG(raw.loc[raw["step"] == s, "work"].to_numpy(), BETA)
        assert out.loc[s, "dG"] == pytest.approx(pooled["dG"], abs=1e-10)
        assert out.loc[s, "Wdiss"] == pytest.approx(pooled["Wdiss"], abs=1e-10)


def test_convergence_series_reproduces_pooled_jarzynski():
    steps = list(range(6))
    smd, raw, n_by_path = _two_path_jarzynski(steps)
    for col in ("dG", "Wdiss"):
        series = SMDAnalysis._weighted_series_from_results(
            smd.results, dict(n_by_path), value_col=col, beta=BETA,
            path_combination="eq21",
        )
        for s in steps:
            pooled = JarzynskiEstimator.estimate_dG(
                raw.loc[raw["step"] == s, "work"].to_numpy(), BETA)
            assert series.loc[s] == pytest.approx(pooled[col], abs=1e-10)


def test_convergence_series_linear_is_peq_weighted_mean():
    steps = list(range(6))
    smd, _, n_by_path = _two_path_jarzynski(steps)
    p_neq = SMDData._compute_p_neq(n_by_path)
    p_eq = SMDAnalysis._compute_p_eq(smd.results, p_neq, BETA)
    series = SMDAnalysis._weighted_series_from_results(
        smd.results, dict(n_by_path), value_col="dG", beta=BETA,
        path_combination="linear",
    )
    res = smd.results.set_index(["step", "path"])["dG"]
    for s in steps:
        exp = sum(p_eq[p] * res.loc[(s, p)] for p in p_eq) / sum(p_eq.values())
        assert series.loc[s] == pytest.approx(exp, abs=1e-12)


def test_invalid_path_combination_raises(tmp_path):
    smd, _ = _results_stub(PER_PATH, N_BY_PATH)
    with pytest.raises(ValueError):
        calculate_weighted_pmf(smd, weights_by_estimator=_weights(P_EQ),
                               path_combination="mean")
    with pytest.raises(ValueError):
        SMDAnalysis._weighted_series_from_results(
            smd.results, dict(N_BY_PATH), value_col="dG", beta=BETA,
            path_combination="mean")
    with pytest.raises(ValueError):
        SMDAnalysis(path_model="null", outdir=str(tmp_path / "a"),
                    path_combination="mean")
    assert SMDAnalysis(path_model="null", outdir=str(tmp_path / "b"),
                       path_combination="linear").path_combination == "linear"
