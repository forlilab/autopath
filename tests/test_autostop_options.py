import numpy as np
import pandas as pd

from autopath.pulling.Convergence import first_streak


def test_deployment_decision_is_a_streak_not_a_pair():
    """The exact series that stopped 6dy7/contacts at v=0.01 must not stop at k=3."""
    df = pd.DataFrame({"n_replicas": [12, 13, 14, 15, 16],
                       "converged": [True, True, False, False, True]})
    assert first_streak(df, k_consec=2) == 13.0
    assert np.isnan(first_streak(df, k_consec=3))
