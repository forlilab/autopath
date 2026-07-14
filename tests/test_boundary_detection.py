import numpy as np
import pandas as pd
import pytest
from autopath.pulling.Estimators import KramersEstimator


def _force_df(force_of_r, n=3000, r0=0.5, r1=2.5, speed=0.01):
    r = np.linspace(r0, r1, n)
    return pd.DataFrame({"r_coord": r, "force": force_of_r(r), "speed": speed})


def test_force_plateau_finds_decay_region_not_end():
    """Force peaks then decays -> boundary lands in the decay region, off the end."""
    # peak at r=1.0, linear decay to ~0 by r=2.0, flat small tail after.
    def f(r):
        pk = 500.0
        out = np.where(r <= 1.0, pk * (r - 0.5) / 0.5,
                       np.clip(pk * (2.0 - r) / 1.0, 20.0, None))
        return out
    df = _force_df(f)
    r_ts = KramersEstimator.force_plateau_boundary(df, 0.01, 0.5, 2.5, frac=0.3)
    assert r_ts is not None
    assert 1.0 < r_ts < 2.4          # past the peak, before the coordinate end


def test_force_plateau_higher_frac_is_nearer_peak():
    def f(r):
        pk = 500.0
        return np.where(r <= 1.0, pk * (r - 0.5) / 0.5,
                        np.clip(pk * (2.0 - r) / 1.0, 20.0, None))
    df = _force_df(f)
    r_low = KramersEstimator.force_plateau_boundary(df, 0.01, 0.5, 2.5, frac=0.2)
    r_high = KramersEstimator.force_plateau_boundary(df, 0.01, 0.5, 2.5, frac=0.5)
    assert r_high is not None and r_low is not None
    assert r_high < r_low            # decaying to 50% happens earlier than to 20%


def test_boundary_buffer_extends_past_rupture_but_not_past_end():
    def f(r):
        pk = 500.0
        return np.where(r <= 1.0, pk * (r - 0.5) / 0.5,
                        np.clip(pk * (2.0 - r) / 1.0, 20.0, None))
    df = _force_df(f)
    r_ts = KramersEstimator.force_plateau_boundary(df, 0.01, 0.5, 2.5, frac=0.4)
    r_hi = 2.5
    for buf in (0.0, 0.1, 0.25):
        cap = min(r_ts * (1.0 + buf), r_hi)
        assert cap >= r_ts          # buffer only extends the window
        assert cap <= r_hi          # never past the coordinate end


def test_force_plateau_monotonic_returns_none():
    """A force that never decays below frac*peak -> None (caller falls back)."""
    df = _force_df(lambda r: 100.0 * r)   # monotonically increasing, peak at end
    assert KramersEstimator.force_plateau_boundary(df, 0.01, 0.5, 2.5, frac=0.3) is None


def test_force_peak_is_before_absorbing_boundary():
    """TS (force peak) sits before the Kramers absorbing boundary (decay point)."""
    def f(r):
        pk = 500.0
        return np.where(r <= 1.0, pk * (r - 0.5) / 0.5,
                        np.clip(pk * (2.0 - r) / 1.0, 20.0, None))
    df = _force_df(f)
    peak = KramersEstimator.force_peak_r(df, 0.01, 0.5, 2.5)
    absb = KramersEstimator.force_plateau_boundary(df, 0.01, 0.5, 2.5, frac=0.4)
    assert peak is not None and absb is not None
    assert peak == pytest.approx(1.0, abs=0.1)   # rupture at the constructed peak
    assert peak < absb                            # TS precedes the absorbing boundary


def test_force_peak_none_when_unusable():
    df = _force_df(lambda r: 100.0 * r, n=50)     # <100 frames -> unusable
    assert KramersEstimator.force_peak_r(df, 0.01, 0.5, 2.5) is None
