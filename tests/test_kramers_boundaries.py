"""Kramers MFPT boundaries: bulk absorbing boundary and bound-state extension."""
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


# --- absorbing boundary -------------------------------------------------------

def test_bulk_boundary_is_the_post_barrier_minimum():
    prof = pull_profile()
    abs_r, ts_r, at_end = KramersEstimator.bulk_boundary(prof.r_coord.values, prof.dG.values, KBT)
    x = np.linspace(XB, 2.0, 100_001)
    assert ts_r == pytest.approx(XB, abs=0.003)
    assert abs_r == pytest.approx(x[np.argmin(G(x))], abs=0.003)
    assert not at_end


def test_bulk_boundary_flags_an_unresolved_descent():
    prof = pull_profile(hi=1.45)           # ends while still descending from the barrier
    abs_r, _, at_end = KramersEstimator.bulk_boundary(prof.r_coord.values, prof.dG.values, KBT)
    assert at_end and abs_r == pytest.approx(prof.r_coord.iloc[-1])


NOISE = np.random.default_rng(7).normal(0.0, 0.3, 1001)
R1 = X0 + 0.001 * np.arange(1001)


def test_constant_slope_rise_to_the_end_is_not_supported():
    with pytest.raises(ValueError, match="still rising"):
        KramersEstimator.bulk_boundary(R1, 10.0 * (R1 - X0) + NOISE, KBT)


def test_flat_profile_is_not_supported():
    with pytest.raises(ValueError, match="flat from the start"):
        KramersEstimator.bulk_boundary(R1, NOISE, KBT)


def G_mono(x):
    """Symmetric quadratic well joined to a monotone tanh rise onto a plateau."""
    x = np.asarray(x, dtype=float)
    up = 15.0 * np.tanh((x - X0) ** 2 / 0.08)
    return np.where(x >= X0, up, 15.0 * (x - X0) ** 2 / 0.08)


def test_monotone_profile_absorbs_at_the_plateau_onset():
    abs_r, ts_r, at_end = KramersEstimator.bulk_boundary(R1, G_mono(R1) + NOISE, KBT)
    assert np.isnan(ts_r) and not at_end
    # reached the plateau within the noise, and not deep inside it
    assert 15.0 - G_mono(abs_r) < 1.0
    assert abs_r < 1.75


def test_monotone_mfpt_with_mirror_matches_full_well_reference():
    abs_r, _, _ = KramersEstimator.bulk_boundary(R1, G_mono(R1) + NOISE, KBT)
    prof = pd.DataFrame({"r_coord": R1, "dG": G_mono(R1), "Gamma": GAMMA})
    x = np.linspace(X0 - 0.6, abs_r, 400_001)
    g = G_mono(x) / KBT
    inner = cumulative_trapezoid(np.exp(-g), x, initial=0.0)
    m = x >= X0
    ref = float(np.trapz((np.exp(g) * GAMMA / KBT * inner)[m], x[m]))
    assert tau(prof, abs_r, "mirror") == pytest.approx(ref, rel=0.02)


def test_bump_below_a_higher_plateau_uses_the_plateau():
    dG = G_mono(R1) + NOISE + 3.0 * np.exp(-((R1 - 1.15) / 0.02) ** 2) \
        - 3.0 * np.exp(-((R1 - 1.2) / 0.02) ** 2)
    abs_r, ts_r, _ = KramersEstimator.bulk_boundary(R1, dG, KBT)
    assert np.isnan(ts_r) and abs_r > 1.2


# --- bound-state extension, against the full-well reference --------------------

@pytest.mark.parametrize("ext", ["mirror", "harmonic"])
def test_extension_recovers_the_exact_mfpt(ext):
    prof = pull_profile()
    abs_r, _, _ = KramersEstimator.bulk_boundary(prof.r_coord.values, prof.dG.values, KBT)
    assert tau(prof, abs_r, ext) == pytest.approx(mfpt_reference(abs_r), rel=0.01)


def test_reflecting_at_the_start_halves_a_symmetric_well():
    prof = pull_profile()
    abs_r, _, _ = KramersEstimator.bulk_boundary(prof.r_coord.values, prof.dG.values, KBT)
    assert tau(prof, abs_r, "none") / mfpt_reference(abs_r) == pytest.approx(0.5, abs=0.03)


def test_old_boundaries_underestimate_and_new_ones_do_not():
    prof = pull_profile()
    ts = KramersEstimator.detect_ts(prof.r_coord.values, prof.dG.values, KBT)
    abs_new, _, _ = KramersEstimator.bulk_boundary(prof.r_coord.values, prof.dG.values, KBT)
    ref = mfpt_reference(abs_new)
    assert tau(prof, ts, "none") / ref < 0.4      # barrier top + reflect at start
    assert tau(prof, abs_new, "mirror") / ref == pytest.approx(1.0, rel=0.01)


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
        KramersEstimator.kramers_mfpt(prof, X0, 1.6, None, KBT, 1.0, bound_extension="flat")


def test_harmonic_rejects_a_non_convex_well():
    r = X0 + 0.001 * np.arange(501)
    dG = 20.0 * np.sin(np.pi * (r - X0) / 0.5)    # concave from the start
    prof = pd.DataFrame({"r_coord": r, "dG": dG, "Gamma": GAMMA})
    with pytest.raises(ValueError, match="curvature"):
        KramersEstimator.kramers_mfpt(prof, X0, 1.4, None, KBT, 1.0, bound_extension="harmonic")


# --- compute() / compute_per_path() wiring --------------------------------------

def _tables(dG):
    prof = pull_profile()
    pmf = pd.DataFrame({"estimator": "jarzynski", "speed": 0.0, "step": prof.step,
                        "r_coord": prof.r_coord, "dG": dG})
    fric = pd.DataFrame({"estimator": "jarzynski", "speed": 0.0, "step": prof.step,
                         "r_coord": prof.r_coord, "Gamma": GAMMA, "method": "regression"})
    return pmf, fric


def test_compute_bulk_mirror_matches_direct_call():
    pmf, fric = _tables(pull_profile().dG.values)
    out = KramersEstimator(temperature=300).compute(
        pmf, fric, dG_extrapolated=pmf, boundary_method="bulk", bound_extension="mirror")
    row = out.iloc[0]
    abs_r, _, _ = KramersEstimator.bulk_boundary(pmf.r_coord.values, pmf.dG.values, KBT)
    assert row.abs_method == "bulk" and row.bound_extension == "mirror"
    assert row.abs_kind == "post_barrier_min"
    assert row.abs_r_nm == pytest.approx(abs_r)
    assert not row.abs_at_profile_end
    assert row.tau_mfpt_ps == pytest.approx(tau(pull_profile(), abs_r, "mirror"), rel=1e-9)


def test_compute_legacy_rule_is_unchanged():
    pmf, fric = _tables(pull_profile().dG.values)
    row = KramersEstimator(temperature=300).compute(pmf, fric, dG_extrapolated=pmf).iloc[0]
    ts = KramersEstimator.detect_ts(pmf.r_coord.values, pmf.dG.values, KBT)
    assert row.abs_method == "pmf_peak" and row.bound_extension == "none"
    assert row.abs_r_nm == pytest.approx(ts) and row.reflect_r_nm == pytest.approx(X0)
    assert row.tau_mfpt_ps == pytest.approx(tau(pull_profile(), ts, "none"), rel=1e-9)


def test_compute_skips_barrier_free_profiles_under_bulk():
    pmf, fric = _tables(10.0 * (pull_profile().r_coord.values - X0))
    k = KramersEstimator(temperature=300)
    assert k.compute(pmf, fric, dG_extrapolated=pmf, boundary_method="bulk").empty
    with pytest.raises(ValueError, match="boundary_method"):
        k.compute(pmf, fric, dG_extrapolated=pmf, boundary_method="plateau")


def test_per_path_bulk_and_extension():
    prof = pull_profile()
    v = 0.01
    res = pd.DataFrame({"estimator": "cumulant", "speed": v, "path": "path-0", "step": prof.step,
                        "r_coord": prof.r_coord, "dG": prof.dG,
                        "Wdiss": GAMMA * v * (prof.r_coord - X0)})
    w = {"cumulant": {v: {"path-0": 1.0}}}
    k = KramersEstimator(temperature=300)
    new = k.compute_per_path(res, w, boundary_method="bulk", bound_extension="mirror")
    old = k.compute_per_path(res, w)
    new_row = new[new.source == "per_path"].iloc[0]
    old_row = old[old.source == "per_path"].iloc[0]
    abs_r, _, _ = KramersEstimator.bulk_boundary(prof.r_coord.values, prof.dG.values, KBT)
    assert new_row.abs_r_nm == pytest.approx(abs_r)
    assert new_row.tau_mfpt_ps == pytest.approx(mfpt_reference(abs_r), rel=0.02)
    assert old_row.tau_mfpt_ps < 0.4 * new_row.tau_mfpt_ps


# --- configuration --------------------------------------------------------------

def test_config_defaults_and_cli_forwarding(tmp_path, monkeypatch):
    from autopath.cli import run_AutoPath

    c = Config()
    assert (c.sMD_kramers_abs, c.sMD_kramers_bound) == ("bulk", "mirror")
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
