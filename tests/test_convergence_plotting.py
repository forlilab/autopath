"""Regression tests for the vALL convergence plotting crash fixes.

These cover the shape mismatch between the per-speed estimators and the
`force` convergence ladder: an empty traces frame/file, and a metrics
frame carrying an extra string `estimator` column.
"""
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from autopath.pulling.Diagnostics import (
    plot_convergence_traces,
    plot_convergence_metrics,
)


def test_plot_convergence_traces_handles_empty_only_file(tmp_path):
    """The force ladder writes an empty traces CSV; the plotter must not
    crash with 'No objects to concatenate' when that is the only file."""
    empty_traces = tmp_path / "sMD_conv_vALL_traces.csv"
    empty_traces.write_text("")

    plot_convergence_traces([str(empty_traces)], outdir=str(tmp_path))


def test_plot_convergence_metrics_ignores_string_estimator_column(tmp_path):
    """The force ladder's metrics frame carries a string 'estimator' column
    (value 'force'). It must be excluded from the metrics to plot, not
    averaged with .mean() (which would raise on an object dtype column)."""
    metrics_df = pd.DataFrame({
        "speed": [0.01, 0.01, 0.02, 0.02],
        "path": ["p0", "p0", "p0", "p0"],
        "n_replicas": [1, 2, 1, 2],
        "converged": [False, True, False, True],
        "decision_quantity": ["dG", "dG", "dG", "dG"],
        "n_common_points": [10, 10, 10, 10],
        "reason": ["", "", "", ""],
        "estimator": ["force", "force", "force", "force"],
        "rmsd": [1.5, 0.4, 1.2, 0.3],
    })
    metrics_csv = tmp_path / "sMD_conv_vALL_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    plot_convergence_metrics([str(metrics_csv)], outdir=str(tmp_path))

    out_svg = tmp_path / "sMD_convergence_metrics.svg"
    assert out_svg.exists()
