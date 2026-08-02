import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import autopath.pulling.AnalysisSMD as A
from autopath.pulling.support import SupportPolicy

SPEED = 0.01
STEPS = list(range(0, 20))
R = np.linspace(1.0, 2.5, len(STEPS))
NLOGS = 6


class StubSMD:
    def __init__(self, logs, **kw):
        self.beta = 1.0 / (0.0083145 * 300.0)
        rows = []
        for i in range(NLOGS):
            for s in STEPS:
                rows.append({"trajname": f"traj{i}", "speed": SPEED, "step": s,
                             "work": float(5.0 + 0.5 * s + 0.1 * i), "path": "p0"})
        self.raw_data = pd.DataFrame(rows)
        self.protocol_grids = {SPEED: pd.DataFrame({"step": STEPS, "r_target_protocol": R})}

    @staticmethod
    def _speed_from_log(fn):
        return SPEED

    @staticmethod
    def _replica_idx_from_log(fn):
        return int("".join(c for c in os.path.basename(fn) if c.isdigit()) or 0)


class StubDTW:
    def __init__(self, *a, **k):
        pass

    def fit_transform(self, feat_df, **kw):
        return {f"traj{i}": "p0" for i in range(NLOGS)}


class TestConvergenceSupportGate(unittest.TestCase):
    def _run(self):
        logs = [f"traj{i}.log" for i in range(NLOGS)]
        ana = A.SMDAnalysis(sysname="t", temperature=300.0,
                            outdir="/tmp/conv_support_test", do_plots=False)
        good = pd.Series({s: float(2.0 + 0.3 * s) for s in STEPS[2:15]})
        call = {"n": 0}

        def weighted(*a, **k):
            call["n"] += 1
            series = pd.Series(dtype=float) if call["n"] == 1 else good.copy()
            if k.get("return_paths"):
                per_path = {int(s): {"p0": v} for s, v in series.items()}
                return series, per_path, (0 if series.empty else 1)
            return series

        with mock.patch.object(A, "SMDData", StubSMD), \
             mock.patch.object(A, "DTWPathModel", StubDTW), \
             mock.patch.object(A.SMDAnalysis, "_build_cluster_feature_df",
                               return_value=(pd.DataFrame(), None)), \
             mock.patch.object(A.SMDAnalysis, "_results_from_running_stats",
                               return_value=pd.DataFrame({"step": STEPS, "path": "p0",
                                                          "n_samples": 6})), \
             mock.patch.object(A.SMDAnalysis, "_weighted_series_from_results",
                               autospec=True, side_effect=weighted):
            return ana.check_convergence(
                logs=logs, speeds=[SPEED], estimator_name="cumulant",
                min_replicas=5, trace_min_replicas=3, trim_fraction=0.0,
                boundary_method="pmf_peak",
            )

    def test_no_crash_and_finite_barrier(self):
        conv_df, _ = self._run()
        self.assertGreater(len(conv_df), 0)
        self.assertEqual(str(conv_df["barrier_height"].dtype), "float64")
        self.assertFalse(conv_df["barrier_height"].map(lambda x: x is None).any())

    def test_results_from_running_stats_skips_under_floor_cells(self):
        # Real _results_from_running_stats (not mocked) must drop n < policy floor.
        pol = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)
        running_stats = {(0, "p0"): {"n": 1}, (1, "p0"): {"n": 4}}
        running_samples = {(0, "p0"): [5.0], (1, "p0"): [5.0, 6.0, 7.0, 8.0]}
        grid = pd.Series({0: 1.0, 1: 1.1})
        df = A.SMDAnalysis._results_from_running_stats(
            running_stats, running_samples, SPEED, grid, "cumulant",
            1.0 / (0.0083145 * 300.0), policy=pol,
        )
        # step 0 (n=1) dropped; step 1 (n=4) kept
        self.assertEqual(set(df["step"]), {1})

    def test_weighted_series_renormalizes_over_survivors(self):
        pol = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)
        # p1 is a singleton path → must be excluded; p_neq renormalized over p0 only.
        results = pd.DataFrame([
            {"step": 0, "path": "p0", "n_samples": 5, "dG": 1.0, "r_coord": 1.0},
            {"step": 1, "path": "p0", "n_samples": 5, "dG": 2.0, "r_coord": 1.1},
            {"step": 0, "path": "p1", "n_samples": 1, "dG": 9.0, "r_coord": 1.0},
        ])
        counts = {"p0": 5, "p1": 1}
        series = A.SMDAnalysis._weighted_series_from_results(
            results, counts, value_col="dG", beta=1.0 / (0.0083145 * 300.0),
            trim_fraction=0.0, policy=pol,
        )
        # Only p0 contributes → weighted value equals p0's dG at each step.
        self.assertAlmostEqual(series.loc[0], 1.0, places=6)
        self.assertAlmostEqual(series.loc[1], 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
