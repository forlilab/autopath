import numpy as np
import pandas as pd
import pytest

from autopath.pulling.hummer_szabo import (
    f_jarzynski, g0_profile, hs_log_weights, slice_table, slice_weights,
)

from .hs_toys import harmonic, protocol_grids, simulate_pulls

K = 3765.6
BETA = 1.0 / (0.008314462618 * 300.0)


def _g0(rows, F, c):
    slices = slice_table(rows, protocol_grids(rows), BETA)
    cv = slice_weights(slices, c)
    if F == "jarzynski":
        Fv = f_jarzynski(slices, BETA)
    else:
        raise AssertionError(F)
    logw = hs_log_weights(rows, slices, Fv, cv, K, BETA)
    return g0_profile(rows.r_after.to_numpy(), logw, rows.trajname.to_numpy(), BETA)


def _max_dev(out, G0):
    m = (out.ess_pulls >= 30).to_numpy()
    assert m.sum() > 3
    d = out.G0.to_numpy()[m] - G0(out.r_bin.to_numpy()[m])
    return np.max(np.abs(d - d.mean()))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unequal_pull_counts_across_speeds(seed):
    # Unequal coverage: a slice weight that scales with n_pulls biases where the fast windows end.
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = pd.concat([
        simulate_pulls(G0, dG0, n_pulls=30, speed=0.0025, r_start=0.9, length_nm=0.2, seed=seed),
        simulate_pulls(G0, dG0, n_pulls=90, speed=0.005, r_start=0.9, length_nm=0.1, seed=seed + 10),
    ], ignore_index=True)
    assert _max_dev(_g0(rows, F="jarzynski", c="ess"), G0) < 1.0
