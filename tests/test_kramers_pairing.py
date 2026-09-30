import numpy as np
import pandas as pd
import pytest

from autopath.pulling.Estimators import KramersEstimator

KBT = 8.314e-3 * 300.0
R = np.round(np.arange(1.0, 2.0 + 1e-9, 0.002), 6)
STEP = np.arange(R.size)


def _profile(dG, gamma):
    return pd.DataFrame({"step": STEP, "r_coord": R, "dG": dG, "Gamma": gamma})


def _barrier(r0=1.5, h=30.0, w=0.1):
    return h * np.exp(-((R - r0) / w) ** 2)


# --- kramers_mfpt -----------------------------------------------------------

def test_barrier_is_measured_from_start_not_from_the_minimum():
    well = -10.0 * np.exp(-((R - 1.15) / 0.05) ** 2)
    res = KramersEstimator.kramers_mfpt(_profile(_barrier() + well, np.full(R.size, 1e4)),
                                        1.0, 1.9, None, KBT, 1.0)
    assert res["barrier_kjmol"] == pytest.approx(30.0, abs=0.1)
    assert res["min_dF_kjmol"] == pytest.approx(-10.0, abs=0.1)


def test_tau_is_invariant_to_a_constant_offset():
    G = np.full(R.size, 1e4)
    a = KramersEstimator.kramers_mfpt(_profile(_barrier(), G), 1.0, 1.9, None, KBT, 1.0)
    b = KramersEstimator.kramers_mfpt(_profile(_barrier() + 55.0, G), 1.0, 1.9, None, KBT, 1.0)
    assert a["tau_ps"] == pytest.approx(b["tau_ps"], rel=1e-12)
    assert a["barrier_kjmol"] == pytest.approx(b["barrier_kjmol"])


def test_profile_beyond_absorbing_boundary_is_ignored():
    # A huge rise past abs_r must neither trip the overflow guard nor the flag
    G = np.full(R.size, 1e4)
    tail = np.where(R > 1.9, 5000.0 * (R - 1.9), 0.0)
    a = KramersEstimator.kramers_mfpt(_profile(_barrier(), G), 1.0, 1.8, None, KBT, 1.0)
    b = KramersEstimator.kramers_mfpt(_profile(_barrier() + tail, G), 1.0, 1.8, None, KBT, 1.0)
    assert np.isfinite(b["tau_ps"])
    assert b["tau_ps"] == pytest.approx(a["tau_ps"], rel=1e-12)
    assert b["max_beta_dF"] == pytest.approx(a["max_beta_dF"])
    assert b["max_beta_dF"] < 100


def test_max_beta_dF_is_the_span_inside_the_window():
    well = -10.0 * np.exp(-((R - 1.15) / 0.05) ** 2)
    res = KramersEstimator.kramers_mfpt(_profile(_barrier() + well, np.full(R.size, 1e4)),
                                        1.0, 1.9, None, KBT, 1.0)
    assert res["max_beta_dF"] == pytest.approx(40.0 / KBT, rel=1e-3)


def test_gamma_clip_count_covers_only_the_integration_window():
    G = np.full(R.size, 1e4)
    G[R > 1.95] = -5.0                       # outside [start, abs]: irrelevant
    res = KramersEstimator.kramers_mfpt(_profile(_barrier(), G), 1.0, 1.9, None, KBT, 1.0)
    assert res["n_gamma_clipped"] == 0
    G[(R > 1.3) & (R < 1.32)] = -5.0         # inside
    res = KramersEstimator.kramers_mfpt(_profile(_barrier(), G), 1.0, 1.9, None, KBT, 1.0)
    assert res["n_gamma_clipped"] == int(((R > 1.3) & (R < 1.32)).sum())
    assert res["n_gamma_points"] == int(((R >= 1.0) & (R <= 1.9)).sum())


# --- compute(): pairing -----------------------------------------------------

def _tables(estimators=("cumulant", "force"), with_v0=True):
    pmf, fric = [], []
    for est in estimators:
        for v in ([0.0] if with_v0 else []) + [0.001, 0.005]:
            pmf.append(pd.DataFrame({"estimator": est, "speed": v, "step": STEP,
                                     "r_coord": R, "dG": _barrier() + 100 * v}))
            # AutoPath writes all-NaN derivative rows at speed 0
            fric.append(pd.DataFrame({"estimator": est, "speed": v, "step": STEP, "r_coord": R,
                                      "Gamma": np.nan if v == 0 else 1e4 * (1 + v),
                                      "method": "derivative"}))
        if with_v0:
            fric.append(pd.DataFrame({"estimator": est, "speed": 0.0, "step": STEP, "r_coord": R,
                                      "Gamma": 5e3, "method": "regression"}))
    pmf = pd.concat(pmf, ignore_index=True)
    return pmf, pd.concat(fric, ignore_index=True), (pmf[pmf.speed == 0.0] if with_v0 else None)


def test_no_per_speed_row_at_v0_and_force_only_at_v0():
    pmf, fric, v0 = _tables()
    out = KramersEstimator(temperature=300).compute(pmf, fric, dG_extrapolated=v0, abs_r=1.9)
    got = set(zip(out.estimator, out.source))
    assert ("cumulant", "per_speed(v=0.0)") not in got
    assert {("cumulant", "v0_extrapolation"), ("cumulant", "per_speed(v=0.001)"),
            ("cumulant", "per_speed(v=0.005)")} <= got
    assert {s for e, s in got if e == "force"} == {"v0_extrapolation"}
    assert out["tau_mfpt_ps"].notna().all()
    assert {"n_gamma_clipped", "n_gamma_points", "min_dF_kjmol"} <= set(out.columns)


def test_force_without_v0_is_skipped_not_run_per_speed():
    pmf, fric, _ = _tables(with_v0=False)
    k = KramersEstimator(temperature=300)
    out = k.compute(pmf, fric, dG_extrapolated=None, abs_r=1.9)
    assert "force" not in set(out.estimator)
    out1 = k.compute(pmf, fric, dG_extrapolated=None, abs_r=1.9, all_speeds=False)
    assert "force" not in set(out1.estimator)
    assert set(out1.source) == {"per_speed(v=0.001)"}


# --- compute_per_path(): each path with its own friction ---------------------

def _per_path_results(v=0.01):
    rows = []
    for path, gam in (("path-0", 1e4), ("path-1", 4e4)):
        wd = gam * v * (R - R[0])            # Wdiss = Γ v Δr  →  Γ = dWdiss/dr / v
        for est in ("cumulant", "force"):
            rows.append(pd.DataFrame({"estimator": est, "speed": v, "path": path, "step": STEP,
                                      "r_coord": R, "dG": _barrier(),
                                      "Wdiss": wd if est != "force" else np.nan}))
    return pd.concat(rows, ignore_index=True)


def test_per_path_uses_each_paths_own_friction_and_skips_force():
    res = _per_path_results()
    w = {"cumulant": {0.01: {"path-0": 0.5, "path-1": 0.5}}, "force": {0.01: {"path-0": 0.5, "path-1": 0.5}}}
    out = KramersEstimator(temperature=300).compute_per_path(res, w)
    per = out[out.source == "per_path"].set_index("path")
    assert "force" not in set(out.estimator)
    # constant Γ: τ ∝ Γ, so the 4× friction path is 4× slower
    assert per.loc["path-1", "tau_mfpt_ps"] / per.loc["path-0", "tau_mfpt_ps"] == pytest.approx(4.0, rel=0.02)
