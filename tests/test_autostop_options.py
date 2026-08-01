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
