"""Kramers MFPT: end-of-range absorbing boundary, mirror extension, plateau diagnostic."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid

from autopath.config import Config
from autopath.pulling.Estimators import KramersEstimator

KBT = 8.314e-3 * 300.0
GAMMA = 1.0e3
X0, XB, GB = 1.0, 1.3, 25.0               # well minimum, cusp barrier position / height
KAPPA = 2.0 * GB / (XB - X0) ** 2         # exactly quadratic well up to the barrier
P, W, DIP, XD, WD = 5.0, 0.1, 4.0, 1.6, 0.05


def G(x):
    """Quadratic well (symmetric about X0), cusp barrier, shallow post-barrier dip, plateau."""
    x = np.asarray(x, dtype=float)
    out = P + (GB - P) * np.exp(-((x - XB) ** 2) / (2 * W ** 2)) \
        - DIP * np.exp(-((x - XD) ** 2) / (2 * WD ** 2))
    return np.where(x <= XB, 0.5 * KAPPA * (x - X0) ** 2, out)


def mfpt_reference(b, lo=X0 - 0.6, n=400_001):
    """MFPT from X0 to b with the whole well (x < X0 included), by fine-grid quadrature."""
    x = np.linspace(lo, b, n)
    g = G(x) / KBT
    inner = cumulative_trapezoid(np.exp(-g), x, initial=0.0)
    outer = np.exp(g) * GAMMA / KBT * inner
    m = x >= X0
    return float(np.trapz(outer[m], x[m]))


def pull_profile(hi=2.0, dr=0.001):
    """What a forward pull from the well minimum sees: r >= X0 only."""
    r = X0 + dr * np.arange(int(round((hi - X0) / dr)) + 1)
    return pd.DataFrame({"step": np.arange(r.size), "r_coord": r, "dG": G(r),
                         "Gamma": np.full(r.size, GAMMA)})


def tau(prof, abs_r, ext):
    return KramersEstimator.kramers_mfpt(prof, X0, abs_r, None, KBT, 1.0,
                                         bound_extension=ext)["tau_ps"]


# --- plateau diagnostic ----------------------------------------------------------

NOISE = np.random.default_rng(7).normal(0.0, 0.3, 1001)
R1 = X0 + 0.001 * np.arange(1001)


def G_mono(x):
    """Symmetric quadratic well joined to a monotone tanh rise onto a plateau."""
    x = np.asarray(x, dtype=float)
    up = 15.0 * np.tanh((x - X0) ** 2 / 0.08)
    return np.where(x >= X0, up, 15.0 * (x - X0) ** 2 / 0.08)


def test_plateau_reached_on_a_plateaued_profile():
    assert KramersEstimator.plateau_reached(R1, G_mono(R1) + NOISE)


def test_plateau_not_reached_on_a_rising_profile():
    assert not KramersEstimator.plateau_reached(R1, 10.0 * (R1 - X0) + NOISE)


# --- mirror extension, against the full-well reference -------------------------

def test_mirror_recovers_the_exact_mfpt():
    prof = pull_profile()
    assert tau(prof, 2.0, "mirror") == pytest.approx(mfpt_reference(2.0), rel=0.01)


def test_reflecting_at_the_start_halves_a_symmetric_well():
    prof = pull_profile()
    assert tau(prof, 2.0, "none") / mfpt_reference(2.0) == pytest.approx(0.5, abs=0.03)


def test_extension_fraction_is_reported():
    prof = pull_profile()
    res = KramersEstimator.kramers_mfpt(prof, X0, 1.6, None, KBT, 1.0, bound_extension="mirror")
    assert res["bound_ext_frac"] == pytest.approx(0.5)
    res = KramersEstimator.kramers_mfpt(prof, X0, 1.6, None, KBT, 1.0)
    assert res["bound_ext_frac"] == 0.0


def test_extension_rejects_a_reflecting_wall_away_from_start():
    prof = pull_profile()
    with pytest.raises(ValueError, match="reflect_r"):
        KramersEstimator.kramers_mfpt(prof, 1.1, 1.6, 1.0, KBT, 1.0, bound_extension="mirror")
    with pytest.raises(ValueError, match="bound_extension"):
        KramersEstimator.kramers_mfpt(prof, X0, 1.6, None, KBT, 1.0, bound_extension="harmonic")


# --- compute() / compute_per_path() wiring --------------------------------------

def _tables(dG, gamma=GAMMA):
    prof = pull_profile()
    pmf = pd.DataFrame({"estimator": "jarzynski", "speed": 0.0, "step": prof.step,
                        "r_coord": prof.r_coord, "dG": dG})
    fric = pd.DataFrame({"estimator": "jarzynski", "speed": 0.0, "step": prof.step,
                         "r_coord": prof.r_coord, "Gamma": gamma, "method": "regression"})
    return pmf, fric


@pytest.mark.parametrize("dG", [pull_profile().dG.values,                    # barrier
                                10.0 * (pull_profile().r_coord.values - X0)])  # still rising
def test_end_boundary_is_r_max_for_any_profile(dG):
    pmf, fric = _tables(dG)
    row = KramersEstimator(temperature=300).compute(
        pmf, fric, dG_extrapolated=pmf, boundary_method="end", bound_extension="mirror").iloc[0]
    assert row.abs_method == "end" and row.bound_extension == "mirror"
    assert row.abs_r_nm == pytest.approx(pmf.r_coord.max())
    prof = pmf.assign(Gamma=GAMMA)
    assert row.tau_mfpt_ps == pytest.approx(
        KramersEstimator.kramers_mfpt(prof, X0, 2.0, None, KBT, 1.0,
                                      bound_extension="mirror")["tau_ps"], rel=1e-9)


def test_end_boundary_ignores_trailing_nan_friction():
    gamma = np.full(len(pull_profile()), GAMMA)
    gamma[-50:] = np.nan
    pmf, fric = _tables(pull_profile().dG.values, gamma)
    row = KramersEstimator(temperature=300).compute(
        pmf, fric, dG_extrapolated=pmf, boundary_method="end").iloc[0]
    assert row.abs_r_nm == pytest.approx(pmf.r_coord.iloc[-51])


def test_plateau_reached_is_reported_but_does_not_move_the_boundary():
    k = KramersEstimator(temperature=300)
    rising, fric = _tables(10.0 * (pull_profile().r_coord.values - X0) + NOISE)
    flat, _ = _tables(G_mono(pull_profile().r_coord.values) + NOISE)
    a = k.compute(rising, fric, dG_extrapolated=rising, boundary_method="end").iloc[0]
    b = k.compute(flat, fric, dG_extrapolated=flat, boundary_method="end").iloc[0]
    assert (bool(a.plateau_reached), bool(b.plateau_reached)) == (False, True)
    assert a.abs_r_nm == b.abs_r_nm == pytest.approx(2.0)


def test_uncomputable_profile_is_skipped():
    pmf, fric = _tables(np.full(len(pull_profile()), np.nan))
    k = KramersEstimator(temperature=300)
    assert k.compute(pmf, fric, dG_extrapolated=pmf, boundary_method="end").empty
    with pytest.raises(ValueError, match="boundary_method"):
        k.compute(pmf, fric, dG_extrapolated=pmf, boundary_method="bulk")


def test_compute_legacy_rule_is_unchanged():
    pmf, fric = _tables(pull_profile().dG.values)
    row = KramersEstimator(temperature=300).compute(pmf, fric, dG_extrapolated=pmf).iloc[0]
    ts = KramersEstimator.detect_ts(pmf.r_coord.values, pmf.dG.values, KBT)
    assert row.abs_method == "pmf_peak" and row.bound_extension == "none"
    assert row.abs_r_nm == pytest.approx(ts) and row.reflect_r_nm == pytest.approx(X0)
    assert row.tau_mfpt_ps == pytest.approx(tau(pull_profile(), ts, "none"), rel=1e-9)


def test_per_path_end_boundary():
    prof = pull_profile()
    v = 0.01
    res = pd.DataFrame({"estimator": "cumulant", "speed": v, "path": "path-0", "step": prof.step,
                        "r_coord": prof.r_coord, "dG": prof.dG,
                        "Wdiss": GAMMA * v * (prof.r_coord - X0)})
    w = {"cumulant": {v: {"path-0": 1.0}}}
    out = KramersEstimator(temperature=300).compute_per_path(
        res, w, boundary_method="end", bound_extension="mirror")
    row = out[out.source == "per_path"].iloc[0]
    assert row.abs_r_nm == pytest.approx(prof.r_coord.max(), abs=0.002)
    assert "plateau_reached" in out.columns


# --- configuration --------------------------------------------------------------

def test_config_defaults_and_cli_forwarding(tmp_path, monkeypatch):
    from autopath.cli import run_AutoPath

    c = Config()
    assert (c.sMD_kramers_abs, c.sMD_kramers_bound) == ("end", "mirror")
    assert not hasattr(c, "sMD_kramers_force_fraction")
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self, lig):
            pass

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"sMD": {"sMD_kramers_abs": "force_plateau",
                                       "sMD_kramers_bound": "none"}}))
    lig = tmp_path / "lig.sdf"
    lig.write_text("")
    monkeypatch.setattr(run_AutoPath, "AutoPath", Recorder)
    monkeypatch.setattr(sys, "argv", ["run_AutoPath", "-c", str(cfg), "-l", str(lig)])
    run_AutoPath.main()
    assert (seen["sMD_kramers_abs"], seen["sMD_kramers_bound"]) == ("force_plateau", "none")
