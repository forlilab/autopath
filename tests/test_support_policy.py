import unittest
import pandas as pd
from autopath.pulling.support import SupportPolicy


class TestSupportPolicy(unittest.TestCase):
    def setUp(self):
        # 3 paths: p0 has 5 trajs, p1 has 2, p2 has 1 (singleton).
        self.counts = {"p0": 5, "p1": 2, "p2": 1}
        # per (step, path) n_samples rows (one estimator's results table)
        self.df = pd.DataFrame([
            {"step": 0, "path": "p0", "n_samples": 5, "dG": 1.0},
            {"step": 1, "path": "p0", "n_samples": 4, "dG": 2.0},
            {"step": 0, "path": "p1", "n_samples": 2, "dG": 1.5},
            {"step": 1, "path": "p1", "n_samples": 1, "dG": 9.0},  # under floor
            {"step": 0, "path": "p2", "n_samples": 1, "dG": 3.0},  # singleton path
        ])

    def test_predicates(self):
        pol = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)
        self.assertTrue(pol.usable_path(2))
        self.assertFalse(pol.usable_path(1))
        self.assertTrue(pol.estimable_step(3))
        self.assertFalse(pol.estimable_step(2))

    def test_apply_drops_singleton_path_and_low_sample_rows(self):
        pol = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)
        gdf, gcounts = pol.apply(self.df, self.counts)
        # p2 (1 traj) dropped entirely from counts and rows
        self.assertNotIn("p2", gcounts)
        self.assertEqual(set(gcounts), {"p0", "p1"})
        self.assertNotIn("p2", set(gdf["path"]))
        # p1 step-1 row (n_samples=1 < 3) dropped; p1 step-0 (n=2 < 3) also dropped
        p1_rows = gdf[gdf["path"] == "p1"]
        self.assertEqual(len(p1_rows), 0)
        # p0 step-0 (n=5) kept; p0 step-1 (n=4) kept
        self.assertEqual(set(gdf[gdf["path"] == "p0"]["step"]), {0, 1})

    def test_apply_is_pure_no_dg_dependency(self):
        # Flipping dG values must not change which rows/paths survive.
        pol = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)
        gdf_a, ca = pol.apply(self.df, self.counts)
        df2 = self.df.copy()
        df2["dG"] = -df2["dG"] * 100.0
        gdf_b, cb = pol.apply(df2, self.counts)
        self.assertEqual(ca, cb)
        self.assertEqual(
            set(map(tuple, gdf_a[["step", "path"]].values.tolist())),
            set(map(tuple, gdf_b[["step", "path"]].values.tolist())),
        )

    def test_gated_counts_renormalize_to_one(self):
        # Survivor counts feed p_neq = count / total → sums to 1.
        pol = SupportPolicy(min_samples_per_step=3, min_trajs_per_path=2)
        _, gcounts = pol.apply(self.df, self.counts)
        total = sum(gcounts.values())
        p_neq = {p: c / total for p, c in gcounts.items()}
        self.assertAlmostEqual(sum(p_neq.values()), 1.0, places=12)

    def test_noop_on_fully_supported_data(self):
        # Regression guard for run(): data already at/above 5/5 is unchanged.
        pol = SupportPolicy(min_samples_per_step=5, min_trajs_per_path=5)
        full = pd.DataFrame([
            {"step": 0, "path": "p0", "n_samples": 6, "dG": 1.0},
            {"step": 1, "path": "p0", "n_samples": 6, "dG": 2.0},
        ])
        counts = {"p0": 6}
        gdf, gcounts = pol.apply(full, counts)
        self.assertEqual(gcounts, counts)
        pd.testing.assert_frame_equal(
            gdf.reset_index(drop=True), full.reset_index(drop=True)
        )


if __name__ == "__main__":
    unittest.main()
