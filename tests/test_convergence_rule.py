import numpy as np
import pandas as pd
import pytest

from autopath.pulling.Convergence import (
    first_streak,
    tail_converged,
    window_mean_pmf,
    window_reference_scalar,
)


def test_first_streak_needs_consecutive_passes():
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6, 7],
                       "converged": [False, True, False, True, True]})
    assert first_streak(df, k_consec=2) == 7.0


def test_first_streak_k3_rejects_a_flickering_series():
    """The failure mode this whole change exists to fix."""
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6, 7, 8],
                       "converged": [True, True, False, True, True, False]})
    assert first_streak(df, k_consec=2) == 4.0
    assert np.isnan(first_streak(df, k_consec=3))


def test_first_streak_fires_on_a_genuine_plateau():
    df = pd.DataFrame({"n_replicas": [3, 4, 5, 6, 7],
                       "converged": [False, False, True, True, True]})
    assert first_streak(df, k_consec=3) == 7.0


def test_first_streak_is_nan_when_shorter_than_the_streak():
    df = pd.DataFrame({"n_replicas": [3, 4], "converged": [True, True]})
    assert np.isnan(first_streak(df, k_consec=3))


def test_first_streak_sorts_by_replica_count():
    df = pd.DataFrame({"n_replicas": [7, 3, 4], "converged": [True, True, True]})
    assert first_streak(df, k_consec=3) == 7.0


def test_window_mean_pmf_of_one_series_is_that_series():
    s = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    pd.testing.assert_series_equal(window_mean_pmf([s]), s)


def test_window_mean_pmf_averages_on_the_shared_index():
    a = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    b = pd.Series([3.0, 4.0], index=[11, 12])
    out = window_mean_pmf([a, b])
    assert list(out.index) == [11, 12]
    assert out.tolist() == [2.5, 3.5]


def test_window_reference_scalar_single_value_is_itself():
    assert window_reference_scalar([2.5]) == 2.5


def test_window_reference_scalar_single_nan_is_nan():
    assert np.isnan(window_reference_scalar([float("nan")]))


def test_window_reference_scalar_ignores_nans_among_finites():
    assert window_reference_scalar([1.0, float("nan"), 3.0]) == 2.0


def test_window_reference_scalar_all_nan_is_nan():
    assert np.isnan(window_reference_scalar([float("nan"), float("nan")]))


# --- tail_converged: the DEPLOYMENT predicate -------------------------------
# first_streak answers the offline-calibration question ("at which rung would
# we first have stopped?"); tail_converged answers deployment's ("are we
# converged right now?"). They must not be confused.


def test_tail_converged_restart_scenario_contrasts_with_first_streak():
    """The exact failure first_streak caused in the deployment loop.

    A campaign restarted with 48 replicas on disk: rungs 46 and 47 passed but
    the latest rung, 48, failed. first_streak still finds the old 46-47 pair
    and would stop the run; the tail-anchored predicate correctly refuses.
    """
    df = pd.DataFrame({"n_replicas": [44, 45, 46, 47, 48],
                       "converged": [False, False, True, True, False]})
    assert first_streak(df, k_consec=2) == 47.0      # a streak exists...
    assert tail_converged(df, k_consec=2) is False   # ...but not at the tail


def test_tail_converged_fires_when_the_latest_rungs_pass():
    df = pd.DataFrame({"n_replicas": [44, 45, 46, 47, 48],
                       "converged": [False, True, False, True, True]})
    assert tail_converged(df, k_consec=2) is True
    assert tail_converged(df, k_consec=3) is False


def test_tail_converged_is_false_when_shorter_than_the_streak():
    df = pd.DataFrame({"n_replicas": [3, 4], "converged": [True, True]})
    assert tail_converged(df, k_consec=3) is False
    assert tail_converged(pd.DataFrame({"n_replicas": [], "converged": []}),
                          k_consec=2) is False


def test_tail_converged_sorts_by_replica_count():
    """Row order in conv_df is not guaranteed; replica order decides."""
    df = pd.DataFrame({"n_replicas": [7, 5, 6], "converged": [False, True, True]})
    assert tail_converged(df, k_consec=2) is False   # rung 7 (the tail) failed
    df2 = pd.DataFrame({"n_replicas": [7, 5, 6], "converged": [True, False, True]})
    assert tail_converged(df2, k_consec=2) is True   # rungs 6,7 pass


def test_tail_converged_window_of_one_is_just_the_last_rung():
    df = pd.DataFrame({"n_replicas": [5, 6], "converged": [False, True]})
    assert tail_converged(df, k_consec=1) is True
    df2 = pd.DataFrame({"n_replicas": [5, 6], "converged": [True, False]})
    assert tail_converged(df2, k_consec=1) is False


def test_tail_converged_never_passes_on_a_degenerate_streak():
    """Defensive: k_consec<1 must not make the predicate vacuously true."""
    df = pd.DataFrame({"n_replicas": [5, 6], "converged": [False, False]})
    assert tail_converged(df, k_consec=0) is False
