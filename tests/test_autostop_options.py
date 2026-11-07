import numpy as np
import pandas as pd
import pytest

from autopath.pulling.Convergence import first_streak, validate_autostop_options


def test_deployment_decision_is_a_streak_not_a_pair():
    """The exact series that stopped 6dy7/contacts at v=0.01 must not stop at k=3."""
    df = pd.DataFrame({"n_replicas": [12, 13, 14, 15, 16],
                       "converged": [True, True, False, False, True]})
    assert first_streak(df, k_consec=2) == 13.0
    assert np.isnan(first_streak(df, k_consec=3))


def test_per_speed_estimators_are_legal_either_way():
    for est in ("cumulant", "jarzynski"):
        for alt in (True, False):
            validate_autostop_options(est, alt, [0.005, 0.01, 0.015])


def test_force_requires_alternating_speeds():
    with pytest.raises(ValueError, match="alternate_speeds"):
        validate_autostop_options("force", False, [0.005, 0.01, 0.015])


def test_force_requires_at_least_two_speeds():
    with pytest.raises(ValueError, match="two speeds"):
        validate_autostop_options("force", True, [0.005])


def test_force_is_legal_with_alternation_and_two_speeds():
    validate_autostop_options("force", True, [0.005, 0.01])


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="Unknown autostop estimator"):
        validate_autostop_options("mbar", False, [0.005, 0.01])


def test_round_robin_visits_every_live_speed_before_repeating():
    """Ordering contract for alternate_speeds=True, tested on the pure helper."""
    from autopath.pulling.Convergence import round_robin_order

    live = {0.005: True, 0.01: True, 0.015: True}
    assert round_robin_order(live) == [0.005, 0.01, 0.015]

    live[0.01] = False          # retired
    assert round_robin_order(live) == [0.005, 0.015]

    assert round_robin_order({0.005: False}) == []


# --- numeric knobs ----------------------------------------------------------


def test_conv_streak_zero_is_rejected():
    """A zero-length tail is trivially all-True: the run would stop at rung 1."""
    with pytest.raises(ValueError, match="sMD_conv_streak must be >= 1"):
        validate_autostop_options("cumulant", False, [0.005, 0.01], conv_streak=0)


def test_conv_window_zero_is_rejected():
    with pytest.raises(ValueError, match="sMD_conv_window must be >= 1"):
        validate_autostop_options("cumulant", False, [0.005, 0.01], conv_window=0)


def test_negative_numeric_knobs_are_rejected():
    with pytest.raises(ValueError, match="sMD_conv_window must be >= 1"):
        validate_autostop_options("cumulant", False, [0.005, 0.01], conv_window=-3)
    with pytest.raises(ValueError, match="sMD_conv_streak must be >= 1"):
        validate_autostop_options("cumulant", False, [0.005, 0.01], conv_streak=-1)


def test_minimal_legal_numeric_knobs_are_accepted():
    """conv_window=1 (pairwise reference) + streak of 1 is legal, if permissive."""
    validate_autostop_options("cumulant", False, [0.005, 0.01],
                              conv_window=1, conv_streak=1)
