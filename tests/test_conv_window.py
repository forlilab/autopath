"""The window reference, and the fixture proving conv_window=1 is a no-op."""
import numpy as np
import pytest

from autopath.pulling.Convergence import window_mean_pmf, window_reference_scalar


def test_window_lags_a_drifting_series_more_than_a_pairwise_reference():
    """Why the window fixes the bug: on a steady drift the pairwise step looks
    small while the distance to the window average is w/2 times larger."""
    import pandas as pd
    drift = 1.0
    rungs = [pd.Series([k * drift, k * drift], index=[10, 11]) for k in range(6)]
    pairwise_gap = abs(rungs[5].iloc[0] - rungs[4].iloc[0])
    window_gap = abs(rungs[5].iloc[0] - window_mean_pmf(rungs[0:5]).iloc[0])
    assert pairwise_gap == pytest.approx(1.0)
    assert window_gap == pytest.approx(3.0)
    assert window_gap > pairwise_gap


def test_window_of_one_reproduces_the_pairwise_reference():
    import pandas as pd
    prev = pd.Series([5.0, 6.0], index=[10, 11])
    assert window_mean_pmf([prev]).equals(prev)
    assert window_reference_scalar([7.0]) == 7.0


@pytest.mark.slow
def test_conv_window_1_reproduces_the_deployment_fixture(tmp_path):
    """Backward-compatibility gate. Real data; ~60 s."""
    import sys
    sys.path.insert(0, "scratch/paper_figures")
    import wdr5_conv_lib as lib

    df = lib.conv_api("6dy7_A", "murcko", "cumulant",
                      outdir=str(tmp_path), speeds=[0.015], conv_window=1)
    assert len(df) == 47
    r46 = df[df["n_replicas"] == 46].iloc[0]
    r47 = df[df["n_replicas"] == 47].iloc[0]
    r48 = df[df["n_replicas"] == 48].iloc[0]
    assert abs(r46["dG_weighted-rmsd"] - 2.077676) < 1e-4
    assert abs(r46["barrier_delta"] - 2.677670) < 1e-4
    assert abs(r46["r_ts_delta"] - 0.000619) < 1e-4
    assert bool(r46["converged"]) is True
    assert abs(r47["dG_weighted-rmsd"] - 1.880280) < 1e-4
    assert bool(r47["converged"]) is True
    assert bool(r48["converged"]) is False
