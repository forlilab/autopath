"""Per-path large-barrier warning is emitted once per (estimator, speed)."""
import logging

import numpy as np
import pandas as pd

from autopath.pulling.Estimators import KramersEstimator


def _results(paths, speed, height):
    r = np.linspace(0.0, 2.0, 41)
    dG = height * np.sin(np.pi * r / 2.0)
    rows = []
    for p in paths:
        rows.append(pd.DataFrame({
            "estimator": "cumulant", "speed": speed, "path": p,
            "step": np.arange(r.size), "r_coord": r, "dG": dG,
            "Wdiss": 5.0 * r * speed * 1e3,
        }))
    return pd.concat(rows, ignore_index=True)


def test_large_barrier_aggregated(caplog):
    df = _results(["path-0", "path-1", "path-2"], 0.01, 200.0)
    weights = {"cumulant": {0.01: {p: 1 / 3 for p in ("path-0", "path-1", "path-2")}}}
    with caplog.at_level(logging.WARNING, logger="autopath"):
        KramersEstimator(barrier_warn_kjmol=80.0).compute_per_path(df, weights)
    msgs = [r.getMessage() for r in caplog.records if "Large barrier" in r.getMessage()]
    assert len(msgs) == 1
    assert "3 path(s)" in msgs[0] and "estimator=cumulant" in msgs[0] and "speed=0.01" in msgs[0]


def test_clipped_gamma_aggregated(caplog):
    paths = ("path-0", "path-1", "path-2")
    df = _results(list(paths), 0.01, 20.0)
    df["Wdiss"] = -5.0 * df["r_coord"] * 0.01 * 1e3  # negative friction, floored
    weights = {"cumulant": {0.01: {p: 1 / 3 for p in paths}}}
    with caplog.at_level(logging.WARNING, logger="autopath"):
        out = KramersEstimator(barrier_warn_kjmol=1e9).compute_per_path(df, weights)
    msgs = [r.getMessage() for r in caplog.records if "clipped" in r.getMessage()]
    assert len(msgs) == 1
    assert "3 path(s)" in msgs[0] and "estimator=cumulant" in msgs[0] and "speed=0.01" in msgs[0]
    assert (out[out["source"] == "per_path"]["n_gamma_clipped"] > 0).all()


def test_warn_false_silences_per_path(caplog):
    paths = ("path-0", "path-1")
    df = _results(list(paths), 0.01, 200.0)
    weights = {"cumulant": {0.01: {p: 0.5 for p in paths}}}
    with caplog.at_level(logging.WARNING, logger="autopath"):
        out = KramersEstimator(barrier_warn_kjmol=80.0).compute_per_path(df, weights, warn=False)
    assert not caplog.records
    assert (out["barrier_kjmol"] > 80.0).any()


def test_kramers_warning_helper():
    from autopath.pulling.AnalysisSMD import kramers_warning
    mix = pd.DataFrame({
        "estimator": ["a", "a", "b"], "barrier_kjmol": [100.0, 50.0, 10.0],
        "n_gamma_clipped": [5, 0, 0], "n_gamma_points": [50, 50, 50],
    })
    paths = pd.DataFrame({
        "estimator": ["a", "a", "a", "b"], "source": ["per_path"] * 3 + ["per_path"],
        "barrier_kjmol": [90.0, 120.0, 20.0, 10.0],
        "n_gamma_clipped": [10, 0, 0, 0], "n_gamma_points": [50, 50, 50, 50],
    })
    msg = kramers_warning("a", mix, paths, 80.0, 1.0)
    assert msg == ("[Kramers] a: barrier 20.0-120.0 kJ/mol, 1/2 rows and 2/3 paths above 80 kJ/mol "
                   "(likely sampling-limited); Γ clipped to floor=1.0 at 10%-20% of points")
    assert kramers_warning("b", mix, paths, 80.0, 1.0) is None
    assert "paths" not in kramers_warning("a", mix, None, 80.0, 1.0)
