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
    """Backward-compatibility gate. Real data; ~60 s.

    Provenance of the hardcoded numbers
    -----------------------------------
    Re-pinned 2026-08-03, after two reproducibility fixes:

    1. ``build_protocol_grids`` now takes r0 as the *median* r_target across
       replicas at step0 instead of ``.iloc[0]``, so the grid no longer
       depends on the order log files were concatenated in.
    2. Replicas are now ordered chronologically (``_replica_start_datetime``)
       instead of by the bare HHMMSS in the filename, so rung k really is
       "the first k replicas run".

    Fix 1 shifts ``protocol_grid`` against a ``boundary_cap`` derived
    independently from ``r_coord``, which flips steps in and out of the fit
    set. That is a deliberate, accepted behaviour change, not a regression.

    The superseded pre-fix values, recorded for provenance only:
        k=46: rmsd 2.077676, barrier_delta 2.677670, r_ts_delta 0.000619
        k=47: rmsd 1.880280, converged True

    The numbers are pinned to an explicit 50-log subset (the original
    2026-07-18 batch) so they do not drift as replicas are added.
    """
    import sys
    sys.path.insert(0, "scratch/paper_figures")
    import wdr5_conv_lib as lib
    from autopath.pulling.SMDData import SMDData

    # Pin to a fixed 50-replica subset instead of "whatever is on disk":
    # 6dy7_A/murcko at v=0.015 has since grown from 50 to 100 replicas as
    # more campaigns landed, and check_convergence walks rungs k=4..N, so
    # letting N float would change both the row count AND (because the DTW
    # path model is refit on whatever replicas it is given) the clustering
    # -- and therefore the PMF -- at every rung. This is a fixed-data
    # invariant: it must reproduce the numbers below regardless of how much
    # new data is added later. check_convergence itself filters `logs` by
    # speed and sorts by _replica_start_datetime before building rung k from
    # the first k replicas, so replicating that ordering here selects exactly
    # the replica set the API would have used when there were 50 -- namely
    # the original 2026-07-18 batch.
    all_logs = lib.logs_for("6dy7_A", "murcko")
    v015_logs = sorted(
        (f for f in all_logs if SMDData._speed_from_log(f) == 0.015),
        key=SMDData._replica_start_datetime,
    )
    logs = v015_logs[:50]

    df = lib.conv_api("6dy7_A", "murcko", "cumulant",
                      outdir=str(tmp_path), speeds=[0.015], logs=logs,
                      conv_window=1)
    assert len(df) == 47
    r46 = df[df["n_replicas"] == 46].iloc[0]
    r47 = df[df["n_replicas"] == 47].iloc[0]
    r48 = df[df["n_replicas"] == 48].iloc[0]
    assert abs(r46["dG_weighted-rmsd"] - 1.471416) < 1e-4
    assert abs(r46["barrier_delta"] - 1.795663) < 1e-4
    assert abs(r46["r_ts_delta"] - 0.000619) < 1e-4
    assert bool(r46["converged"]) is True
    assert abs(r47["dG_weighted-rmsd"] - 18.065981) < 1e-4
    assert bool(r47["converged"]) is False
    assert bool(r48["converged"]) is False


# --- end-to-end window plumbing (synthetic, no simulation data) -------------
# The deque/history wiring inside check_convergence is otherwise only covered
# by the `slow` real-data fixture above. These stubs drive the same code path
# with a hand-built, monotonically drifting PMF series.

import os
from unittest import mock

import numpy as np

import autopath.pulling.AnalysisSMD as A

_SPEED = 0.01
_N_LOGS = 8
_STEPS = list(range(20))
_R = np.linspace(1.0, 2.5, len(_STEPS))
_PMF_STEPS = _STEPS[2:15]          # 13 steps > min_common_points (5)
_DRIFT = 2.5                       # kJ/mol added to every step, per rung


class _StubSMD:
    """SMDData stand-in: _N_LOGS replicas on one path, flat protocol grid.

    Carries no 'force'/'r_coord' columns, so check_convergence finds no
    force-plateau boundary and leaves the RMSD window uncapped.
    """

    def __init__(self, logs, **kw):
        import pandas as pd
        self.beta = 1.0 / (0.0083145 * 300.0)
        self.raw_data = pd.DataFrame([
            {"trajname": f"traj{i}", "speed": _SPEED, "step": s,
             "work": 5.0 + 0.5 * s + 0.1 * i, "path": "p0"}
            for i in range(_N_LOGS) for s in _STEPS
        ])
        self.protocol_grids = {
            _SPEED: pd.DataFrame({"step": _STEPS, "r_target_protocol": _R})}

    @staticmethod
    def _speed_from_log(fn):
        return _SPEED

    @staticmethod
    def _replica_idx_from_log(fn):
        return int("".join(c for c in os.path.basename(fn) if c.isdigit()) or 0)

    @staticmethod
    def _replica_start_datetime(fn):
        # check_convergence sorts replicas chronologically; these synthetic
        # logs have no real mtime, so reuse the filename index as the key.
        return _StubSMD._replica_idx_from_log(fn)


class _StubDTW:
    def __init__(self, *a, **k):
        pass

    def fit_transform(self, feat_df, **kw):
        return {f"traj{i}": "p0" for i in range(_N_LOGS)}


def _drifting_conv_df(conv_window, outdir):
    """Run check_convergence where rung k's PMF is rung (k-1)'s + _DRIFT.

    The PMF *shape* is identical at every rung (a uniform offset), so the
    barrier and TS position never move and the RMSD alone decides.
    """
    import pandas as pd

    logs = [f"traj{i}.log" for i in range(_N_LOGS)]
    ana = A.SMDAnalysis(sysname="t", temperature=300.0, outdir=str(outdir),
                        do_plots=False)
    shape = pd.Series({s: 0.3 * s for s in _PMF_STEPS})
    calls = {"n": 0}

    def _pmf(*a, **k):
        calls["n"] += 1
        series = shape + _DRIFT * (calls["n"] - 1)
        if k.get("return_paths"):
            per_path = {int(s): {"p0": v} for s, v in series.items()}
            return series, per_path, 1
        return series

    with mock.patch.object(A, "SMDData", _StubSMD), \
         mock.patch.object(A, "DTWPathModel", _StubDTW), \
         mock.patch.object(A.SMDAnalysis, "_build_cluster_feature_df",
                           return_value=(pd.DataFrame(), None)), \
         mock.patch.object(A.SMDAnalysis, "_results_from_running_stats",
                           return_value=pd.DataFrame({"step": _STEPS,
                                                      "path": "p0",
                                                      "n_samples": _N_LOGS})), \
         mock.patch.object(A.SMDAnalysis, "_weighted_series_from_results",
                           autospec=True, side_effect=_pmf):
        conv_df, _ = ana.check_convergence(
            logs=logs, speeds=[_SPEED], estimator_name="cumulant",
            min_replicas=5, trace_min_replicas=3, trim_fraction=0.0,
            boundary_method="pmf_peak", conv_window=conv_window,
        )
    return conv_df


def test_wider_conv_window_is_stricter_on_a_drifting_pmf(tmp_path):
    """The plumbing test: a steady drift passes pairwise, fails windowed.

    Each rung moves the PMF by 2.5 kJ/mol, under the 4.0 kJ/mol RMSD
    tolerance, so conv_window=1 declares convergence on every rung. Against a
    5-rung mean the distance grows to ~w/2 * drift and the later rungs fail.
    """
    from autopath.pulling.Convergence import tail_converged

    narrow = _drifting_conv_df(1, tmp_path / "w1")
    wide = _drifting_conv_df(5, tmp_path / "w5")

    # rungs 3..8 with rung 3 seeding the history -> 5 comparison rows each
    assert len(narrow) == 5
    assert len(wide) == 5
    assert list(narrow["n_replicas"]) == list(wide["n_replicas"]) == [4, 5, 6, 7, 8]

    # pairwise: every rung looks converged
    assert bool(narrow["converged"].all()) is True
    assert narrow["dG_weighted-rmsd"].round(6).eq(_DRIFT).all()

    # windowed: the reference lags, so the same series stops passing
    assert bool(wide["converged"].all()) is False
    assert list(wide["converged"]) == [True, True, False, False, False]

    # the window reference is strictly further away from rung 3 onwards
    late = wide["n_replicas"] >= 5
    assert (wide.loc[late, "dG_weighted-rmsd"].to_numpy()
            > narrow.loc[late.to_numpy(), "dG_weighted-rmsd"].to_numpy()).all()

    # and the deployment predicate inherits the stricter verdict
    assert tail_converged(narrow, k_consec=3) is True
    assert tail_converged(wide, k_consec=3) is False


def test_conv_window_1_matches_the_pairwise_reference_end_to_end(tmp_path):
    """conv_window=1 must be a no-op: the audit hinge of the whole change.

    With a uniform per-rung offset the pairwise RMSD is exactly the drift,
    and barrier/TS deltas stay at zero because the PMF shape never changes.
    """
    narrow = _drifting_conv_df(1, tmp_path / "w1")
    assert narrow["dG_weighted-rmsd"].round(6).eq(_DRIFT).all()
    assert narrow["barrier_delta"].abs().max() < 1e-9
    assert narrow["r_ts_delta"].abs().max() < 1e-9
