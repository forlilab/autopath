"""Regression tests for build_protocol_grids' r0 determinism.

`.iloc[0]` used to pick whichever replica's row happened to sort first in
the concatenated raw_data DataFrame, which depends on the order the log
files were passed in (glob() order is filesystem-dependent; a caller using
sorted(glob()) gets a different row than autopath_core's bare glob()). That
made r_target_protocol - and hence r_coord read off it - shift by up to the
replica-to-replica scatter in r_target at step0.

The fix uses the median of r_target across replicas at step0, matching the
consensus build_analysis_coord already uses for r_coord.
"""
import numpy as np
import pandas as pd
import pytest

from autopath.pulling.SMDData import SMDData


def _bare_smd():
    """An SMDData instance with __init__ skipped (build_protocol_grids uses
    no other instance state), so it can be called directly on synthetic
    raw_data without touching disk."""
    return SMDData.__new__(SMDData)


def _make_raw_data(r0_by_traj, speed=0.01, steps=(0, 1, 2, 3)):
    """Build a synthetic raw_data DataFrame: several trajnames at one speed,
    each with a fixed dx=0.1 nm/step but a different r0 at step0."""
    dx = 0.1
    rows = []
    for trajname, r0 in r0_by_traj.items():
        for step in steps:
            rows.append({
                "trajname": trajname,
                "speed": speed,
                "step": step,
                "r_target": r0 + dx * (step - steps[0]),
            })
    return pd.DataFrame(rows)


def test_protocol_grid_is_deterministic_under_log_ordering():
    """The real regression: row order must not change the reconstructed grid."""
    r0_by_traj = {
        "traj0": 1.000,
        "traj1": 1.001,
        "traj2": 1.002,
        "traj3": 1.003,
    }
    raw_data = _make_raw_data(r0_by_traj)
    permuted = raw_data.sample(frac=1.0, random_state=0).reset_index(drop=True)

    smd = _bare_smd()
    grids_a = smd.build_protocol_grids(raw_data)
    grids_b = smd.build_protocol_grids(permuted)

    pd.testing.assert_frame_equal(grids_a[0.01], grids_b[0.01])


def test_protocol_grid_r0_uses_median_not_first_row_or_mean():
    """Pick r0 values where median != mean so the test discriminates."""
    # sorted: 1.000, 1.001, 1.002, 1.010 -> median 1.0015, mean 1.00325
    r0_by_traj = {
        "trajC": 1.002,
        "trajA": 1.000,
        "trajD": 1.010,
        "trajB": 1.001,
    }
    raw_data = _make_raw_data(r0_by_traj)
    expected_median = np.median(list(r0_by_traj.values()))
    expected_mean = np.mean(list(r0_by_traj.values()))
    assert expected_median != pytest.approx(expected_mean)

    smd = _bare_smd()
    grid = smd.build_protocol_grids(raw_data)[0.01]

    r0 = grid.loc[grid["step"] == 0, "r_target_protocol"].iloc[0]
    # not the first-inserted row's value (trajC's r0)
    assert r0 != pytest.approx(r0_by_traj["trajC"])
    # not the mean either
    assert r0 != pytest.approx(expected_mean)
    assert r0 == pytest.approx(expected_median)
