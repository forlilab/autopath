import numpy as np
import pandas as pd
import pytest

from autopath.pulling.Convergence import (
    first_streak,
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
    a = pd.Series([1.0, 3.0, 3.0], index=[10, 11, 12])
    b = pd.Series([3.0, 4.0], index=[11, 12])
    out = window_mean_pmf([a, b])
    assert list(out.index) == [11, 12]
    assert out.tolist() == [3.0, 3.5]


def test_window_reference_scalar_single_value_is_itself():
    assert window_reference_scalar([2.5]) == 2.5


def test_window_reference_scalar_single_nan_is_nan():
    assert np.isnan(window_reference_scalar([float("nan")]))


def test_window_reference_scalar_ignores_nans_among_finites():
    assert window_reference_scalar([1.0, float("nan"), 3.0]) == 2.0


def test_window_reference_scalar_all_nan_is_nan():
    assert np.isnan(window_reference_scalar([float("nan"), float("nan")]))
