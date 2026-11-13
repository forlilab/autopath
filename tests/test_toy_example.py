"""End-to-end run of examples/08_Toy_validation (slow)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_example_end_to_end(tmp_path):
    """Small run of the example: pulls at two speeds, two escapes, analysis, validate table."""
    script = os.path.join(ROOT, "examples", "08_Toy_validation", "run_toy.py")
    env = {**os.environ, "PYTHONPATH": ROOT}
    subprocess.run([sys.executable, script, "--scenario", "Open-HW", "--speeds", "5", "10",
                    "--n-pulls", "12", "--n-escapes", "2", "--out", str(tmp_path)],
                   check=True, env=env, cwd=tmp_path)
    v = pd.read_csv(tmp_path / "Open-HW" / "validate.csv")
    assert set(v.estimator) == {"cumulant", "jarzynski", "force"}
    assert np.isfinite(v.barrier_err).all() and np.isfinite(v.tau_exact_ns).all()
    assert v.n_pulls.max() == 24
