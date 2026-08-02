"""Two-path synthetic fixture for the n_paths / per-path trace diagnostics.

Exercises check_convergence end-to-end (real _results_from_running_stats and
_weighted_series_from_results — nothing mocked past SMDData/DTWPathModel)
with two genuinely distinct paths so that, once both clear the support-policy
floor, the mixture PMF is built from >1 contributing path. This is the case
Change 1/2 in AnalysisSMD.py are meant to make diagnosable: without it, a
stalled RMSD looks identical whether one path is under-sampled or the DTW
path count itself just changed.
"""
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import autopath.pulling.AnalysisSMD as A
from autopath.pulling.SMDData import SMDData as _RealSMDData

SPEED = 0.01
STEPS = list(range(20))
R = np.linspace(1.0, 2.5, len(STEPS))
NLOGS = 10  # traj0-4 -> p0, traj5-9 -> p1


class _StubSMDTwoPaths(_RealSMDData):
    """Two-path stand-in with low-variance, monotonically increasing work.

    Subclasses the real SMDData (rather than replacing it outright) so that
    static helpers used by the real, un-mocked _weighted_series_from_results
    (``_compute_p_neq``) stay available under the patched name.

    Small per-replica jitter keeps cumulant dG >= 0 at every step for both
    paths, so both carry non-zero equilibrium weight once estimable.
    """

    def __init__(self, logs, **kw):
        self.beta = 1.0 / (0.0083145 * 300.0)
        rows = []
        for i in range(NLOGS):
            base = 5.0 if i < 5 else 9.0
            jitter = 0.05 * (i % 5)
            for s in STEPS:
                rows.append({"trajname": f"traj{i}", "speed": SPEED, "step": s,
                             "work": base + 0.5 * s + jitter})
        self.raw_data = pd.DataFrame(rows)
        self.protocol_grids = {SPEED: pd.DataFrame({"step": STEPS, "r_target_protocol": R})}

    @staticmethod
    def _speed_from_log(fn):
        return SPEED

    @staticmethod
    def _replica_idx_from_log(fn):
        return int("".join(c for c in os.path.basename(fn) if c.isdigit()) or 0)


class _StubDTWTwoPaths:
    def __init__(self, *a, **k):
        pass

    def fit_transform(self, feat_df, **kw):
        return {f"traj{i}": ("p0" if i < 5 else "p1") for i in range(NLOGS)}


class TestNPathsDiagnostics(unittest.TestCase):
    def _run(self):
        logs = [f"traj{i}.log" for i in range(NLOGS)]
        ana = A.SMDAnalysis(sysname="t", temperature=300.0,
                            outdir="/tmp/npaths_diag_test", do_plots=False)
        with mock.patch.object(A, "SMDData", _StubSMDTwoPaths), \
             mock.patch.object(A, "DTWPathModel", _StubDTWTwoPaths), \
             mock.patch.object(A.SMDAnalysis, "_build_cluster_feature_df",
                               return_value=(pd.DataFrame(), None)):
            return ana.check_convergence(
                logs=logs, speeds=[SPEED], estimator_name="cumulant",
                min_replicas=5, trace_min_replicas=3, trim_fraction=0.0,
                boundary_method="pmf_peak",
            )

    def test_n_paths_present_and_positive_on_metric_rows(self):
        conv_df, _ = self._run()
        self.assertIn("n_paths", conv_df.columns)
        self.assertGreater(len(conv_df), 0)
        # Every emitted metric row must carry a positive path count.
        self.assertTrue((conv_df["n_paths"] > 0).all())
        # p1 needs 3 replicas to clear min_samples_per_step_conv (default 3);
        # that lands at n_replicas=8 (traj5,6,7). From then on both paths
        # are estimable and non-negative, so both should contribute.
        late = conv_df[conv_df["n_replicas"] >= 8]
        self.assertGreater(len(late), 0)
        self.assertTrue((late["n_paths"] == 2).all())
        # Before p1 clears the floor, only p0 contributes.
        early = conv_df[conv_df["n_replicas"] == 4]
        if len(early) > 0:
            self.assertTrue((early["n_paths"] == 1).all())

    def test_traces_have_mixture_and_per_path_rows_for_same_step(self):
        _, traces_df = self._run()
        self.assertFalse(traces_df.empty)
        expected_cols = {"step", "r_coord", "speed", "path", "n_replicas",
                         "quantity", "value"}
        self.assertTrue(expected_cols.issubset(traces_df.columns))

        mixture = traces_df[traces_df["path"] == "mixture"]
        self.assertGreater(len(mixture), 0)

        # At a late rung (both paths contributing), every mixture row's
        # (step, n_replicas) must have at least one companion per-path row.
        late_mix = mixture[mixture["n_replicas"] >= 8]
        self.assertGreater(len(late_mix), 0)
        found_both = False
        for _, r in late_mix.iterrows():
            per_path = traces_df[
                (traces_df["step"] == r["step"]) &
                (traces_df["n_replicas"] == r["n_replicas"]) &
                (traces_df["path"] != "mixture")
            ]
            self.assertGreaterEqual(len(per_path), 1)
            if set(per_path["path"]) == {"p0", "p1"}:
                found_both = True
        self.assertTrue(
            found_both,
            "expected at least one (step, n_replicas) with both p0 and p1 "
            "per-path trace rows alongside the mixture row",
        )


if __name__ == "__main__":
    unittest.main()
