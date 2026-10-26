import numpy as np
import pytest

from autopath.pulling.hummer_szabo import (
    f_jarzynski, g0_profile, hs_log_weights, slice_table, slice_weights,
)

from .hs_toys import flat, harmonic, protocol_grids, simulate_pulls

K = 3765.6
BETA = 1.0 / (0.008314462618 * 300.0)


def _chain(rows):
    slices = slice_table(rows, protocol_grids(rows), BETA)
    F = f_jarzynski(slices, BETA)
    c = slice_weights(slices, "uniform")
    return slices, F, c


def _logw(rows):
    slices, F, c = _chain(rows)
    return hs_log_weights(rows, slices, F, c, K, BETA)


def _g0(rows, F):
    assert F == "jarzynski"
    return g0_profile(rows.r_after.to_numpy(), _logw(rows), rows.trajname.to_numpy(), BETA)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flat_landscape_is_recovered(seed):
    rows = simulate_pulls(*flat(), n_pulls=40, speed=0.01, length_nm=0.4, seed=seed)
    out = _g0(rows, F="jarzynski")
    m = out.ess_pulls >= 30
    assert m.sum() > 5
    assert np.ptp(out.G0[m]) < 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harmonic_well_is_recovered(seed):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=60, speed=0.005, r_start=0.9, length_nm=0.2, seed=seed)
    out = _g0(rows, F="jarzynski")
    m = out.ess_pulls >= 30
    assert m.sum() > 3
    ref = G0(out.r_bin[m].to_numpy())
    d = out.G0[m].to_numpy() - ref
    assert np.max(np.abs(d - d.mean())) < 1.0


def test_incomplete_support_is_not_supported():
    rows = simulate_pulls(*flat(), n_pulls=4, speed=0.01, length_nm=0.1, stop_steps=[50, 99, 99, 99])
    with pytest.raises(ValueError, match="not supported"):
        slice_table(rows, protocol_grids(rows), BETA)


def test_constant_work_offset_leaves_weights_unchanged():
    rows = simulate_pulls(*flat(), n_pulls=8, speed=0.01, length_nm=0.1, seed=3)
    a = _logw(rows)
    b = _logw(rows.assign(work=rows.work + 1e4))
    np.testing.assert_allclose(a - a.max(), b - b.max(), atol=1e-9)


def test_row_far_from_every_lambda_gets_finite_weight():
    rows = simulate_pulls(*flat(), n_pulls=4, speed=0.01, length_nm=0.1, seed=4)
    rows.loc[rows.index[0], "r_after"] = rows.r_target.max() + 1.0
    assert np.isfinite(_logw(rows)).all()


def _slice_table_reference(rows, beta):
    # Straightforward per-group computation, the definition slice_table must reproduce.
    from scipy.special import logsumexp
    out = []
    for (speed, step), w in rows.groupby(["speed", "step"], sort=True)["work"]:
        a = -beta * w.to_numpy()
        n = len(a)
        out.append((speed, step, n, logsumexp(a) - np.log(n),
                    np.exp(2 * logsumexp(a) - logsumexp(2 * a)),
                    w.mean(), w.var(ddof=1) if n > 1 else np.nan))
    import pandas as pd
    return pd.DataFrame(out, columns=["speed", "step", "n_pulls", "lnZ", "ess", "Wmean", "Wvar"])


def test_slice_table_matches_per_group_reference():
    import pandas as pd
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = pd.concat([
        simulate_pulls(G0, dG0, n_pulls=7, speed=0.005, r_start=0.9, length_nm=0.05, seed=21),
        simulate_pulls(G0, dG0, n_pulls=1, speed=0.01, r_start=0.9, length_nm=0.05, seed=22),
        simulate_pulls(G0, dG0, n_pulls=4, speed=0.02, r_start=0.9, length_nm=0.03, seed=23),
    ], ignore_index=True)
    rows["work"] *= 40.0      # spread -beta W over many kT so the max shift matters
    rows = rows.sample(frac=1.0, random_state=0)
    got = slice_table(rows, protocol_grids(rows), BETA)
    ref = _slice_table_reference(rows, BETA)
    pd.testing.assert_frame_equal(got[["speed", "step", "n_pulls"]], ref[["speed", "step", "n_pulls"]],
                                  check_dtype=False)
    for col in ("lnZ", "ess", "Wmean", "Wvar"):
        np.testing.assert_allclose(got[col].to_numpy(), ref[col].to_numpy(), rtol=1e-12, atol=0,
                                   equal_nan=True, err_msg=col)
    assert got.loc[got.speed == 0.01, "Wvar"].isna().all()
