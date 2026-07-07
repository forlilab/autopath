"""Centralized, count-based data-support policy for SMD PMF estimation.

The support gate answers a pure trajectory-count question — "does this step
have enough samples, does this path have enough trajectories?" — so the
surviving (step, path) set is identical for every estimator. Equilibrium
weights (p_eq) remain estimator-specific: they are computed *over* the gated
survivors but still use each estimator's own dG profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SupportPolicy:
    """Count-based estimability gate shared by the run() and convergence pipelines.

    Parameters
    ----------
    min_samples_per_step : int
        Minimum work-value count in a (step, path) cell for its dG to be
        estimable (variance needs >= this many samples).
    min_trajs_per_path : int
        Minimum number of trajectories for a path to enter the mixture.

    Notes
    -----
    The production ``run()`` pipeline sources THRESHOLDS from this policy
    (via ``min_samples_per_step`` / ``min_trajs_per_path`` / ``usable_path``),
    but NOT the ``apply()`` gate itself — ``run()`` keeps its own
    ``trim_results_by_n_samples_support`` + ``_path_filtering`` to preserve
    the verified no-op behavior. Routing ``run()`` through ``apply()`` would
    risk changing production output.
    """

    min_samples_per_step: int
    min_trajs_per_path: int

    def usable_path(self, n_trajs: int) -> bool:
        return n_trajs >= self.min_trajs_per_path

    def estimable_step(self, n_samples: int) -> bool:
        return n_samples >= self.min_samples_per_step

    def apply(
        self,
        results_df: pd.DataFrame,
        path_traj_counts: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """Gate a results table and its path→traj-count map.

        Returns ``(gated_results_df, gated_path_traj_counts)``:
        - paths with ``count < min_trajs_per_path`` are removed from both,
        - ``(step, path)`` rows with ``n_samples < min_samples_per_step`` are
          dropped from the results table.

        Pure and count-based — never inspects dG/energy columns.
        """
        gated_counts = {
            path: count
            for path, count in path_traj_counts.items()
            if self.usable_path(count)
        }
        if results_df is None or results_df.empty:
            return (
                results_df.copy() if results_df is not None else pd.DataFrame(),
                gated_counts,
            )
        mask = (
            results_df["path"].isin(gated_counts.keys())
            & (results_df["n_samples"] >= self.min_samples_per_step)
        )
        return results_df[mask].copy(), gated_counts
