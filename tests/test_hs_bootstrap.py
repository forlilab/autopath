import numpy as np
import pandas as pd
import pytest

from autopath.pulling.hummer_szabo import (
    F_SOURCES, bootstrap_g0, estimate, f_jarzynski, slice_diagnostics, slice_table,
)

from .hs_toys import flat, harmonic, protocol_grids, simulate_pulls

K = 3765.6
BETA = 1.0 / (0.008314462618 * 300.0)


def _boot(rows, n_boot, f_source="jarzynski", seed=0, **kw):
    return bootstrap_g0(rows, protocol_grids(rows), K, BETA, f_source, n_boot=n_boot, seed=seed, **kw)


def _diag(rows):
    sl = slice_table(rows, protocol_grids(rows), BETA)
    return slice_diagnostics(rows, sl, {"jarzynski": f_jarzynski(sl, BETA)}, K, BETA)


def test_bootstrap_sd_shrinks_with_more_pulls():
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    small = simulate_pulls(G0, dG0, n_pulls=15, speed=0.005, r_start=0.9, length_nm=0.2, seed=6)
    big = simulate_pulls(G0, dG0, n_pulls=120, speed=0.005, r_start=0.9, length_nm=0.2, seed=7)
    s = _boot(small, n_boot=50)
    b = _boot(big, n_boot=50)
    assert np.nanmedian(b.G0_sd) < np.nanmedian(s.G0_sd)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harmonic_within_three_sigma(seed):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=60, speed=0.005, r_start=0.9, length_nm=0.2, seed=seed)
    out = _boot(rows, n_boot=100)
    m = (out.ess_pulls >= 5).to_numpy() & np.isfinite(out.G0_sd.to_numpy())
    assert m.sum() > 3
    d = out.G0.to_numpy()[m] - G0(out.r_bin.to_numpy()[m])
    d = d - d.mean()
    assert np.all(np.abs(d) <= 3 * out.G0_sd.to_numpy()[m] + 0.3)


def test_lag_diagnostics_match_flat_equilibrium():
    rows = simulate_pulls(*flat(), n_pulls=40, speed=0.001, length_nm=0.1, seed=8)
    d = _diag(rows)
    assert abs(d.lag_mean.mean()) < 0.005
    assert d.lag_sd.mean() == pytest.approx(d.lag_sd_eq.iloc[0], rel=0.15)


def test_bootstrap_zero_returns_point_estimate():
    G0, dG0 = harmonic()
    rows = simulate_pulls(G0, dG0, n_pulls=10, speed=0.005, r_start=0.9, length_nm=0.1, seed=1)
    out = _boot(rows, n_boot=0)
    g0, _, _ = estimate(rows, protocol_grids(rows), K, BETA, "jarzynski")
    assert out.G0_sd.isna().all()
    np.testing.assert_allclose(out.G0.to_numpy(), g0.G0.to_numpy())


def test_force_nan_slices_excluded():
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = pd.concat([
        simulate_pulls(G0, dG0, n_pulls=30, speed=0.005, r_start=0.9, length_nm=0.2, seed=1),
        simulate_pulls(G0, dG0, n_pulls=30, speed=0.01, r_start=0.9, length_nm=0.1, seed=2),
    ], ignore_index=True)
    g0, logw, F = estimate(rows, protocol_grids(rows), K, BETA, "force",
                           realized_speed_map={0.005: 0.005, 0.01: 0.01})
    sl = slice_table(rows, protocol_grids(rows), BETA)
    nan_slice = pd.MultiIndex.from_frame(sl.loc[np.isnan(F), ["speed", "step"]])
    in_nan = pd.MultiIndex.from_frame(rows[["speed", "step"]]).isin(nan_slice)
    assert in_nan.any() and (~in_nan).any()
    assert np.isfinite(logw[~in_nan]).all()
    assert np.all(np.isneginf(logw[in_nan]))
    assert np.isfinite(g0.G0).all() and len(g0) > 3


@pytest.mark.parametrize("src", F_SOURCES)
def test_estimate_dispatches_all_sources(src):
    G0, dG0 = harmonic()
    rows = pd.concat([simulate_pulls(G0, dG0, n_pulls=10, speed=v, r_start=0.9, length_nm=0.1, seed=i)
                      for i, v in enumerate((0.005, 0.01))], ignore_index=True)
    g0, logw, F = estimate(rows, protocol_grids(rows), K, BETA, src,
                           realized_speed_map={0.005: 0.005, 0.01: 0.01})
    assert len(logw) == len(rows) and np.isfinite(g0.G0).all()


def test_estimate_unknown_source():
    rows = simulate_pulls(*flat(), n_pulls=3, speed=0.01, length_nm=0.02)
    with pytest.raises(ValueError, match="not supported"):
        estimate(rows, protocol_grids(rows), K, BETA, "bogus")


def test_bootstrap_relabels_duplicates(monkeypatch):
    # Resamples must keep one distinct trajname per resampled pull, per speed.
    from autopath.pulling import hummer_szabo as hs
    G0, dG0 = harmonic()
    rows = pd.concat([simulate_pulls(G0, dG0, n_pulls=n, speed=v, r_start=0.9, length_nm=0.05, seed=n)
                      for n, v in ((6, 0.005), (9, 0.01))], ignore_index=True)
    seen, real = [], hs.estimate

    def spy(r, *a, **kw):
        seen.append(r)
        return real(r, *a, **kw)

    monkeypatch.setattr(hs, "estimate", spy)
    _boot(rows, n_boot=3, seed=5)
    assert len(seen) == 4
    want = rows.groupby("speed").trajname.nunique()
    for r in seen[1:]:
        pd.testing.assert_series_equal(r.groupby("speed").trajname.nunique(), want)


def test_lag_eq_tracks_mean_lag_on_harmonic():
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=40, speed=0.001, r_start=0.9, length_nm=0.2, seed=9)
    d = _diag(rows)
    d = d[d.n_pulls >= 30]
    assert np.corrcoef(d.lag_mean, d.lag_eq)[0, 1] > 0.8
    assert np.mean(np.sign(d.lag_eq) == np.sign(d.lag_mean)) > 0.8


def test_slice_diagnostics_columns_and_spread():
    G0, dG0 = harmonic()
    rows = simulate_pulls(G0, dG0, n_pulls=10, speed=0.005, r_start=0.9, length_nm=0.1, seed=1)
    sl = slice_table(rows, protocol_grids(rows), BETA)
    Fj = f_jarzynski(sl, BETA)
    d = slice_diagnostics(rows, sl, {"jarzynski": Fj, "other": Fj + 3.0}, K, BETA)
    assert {"F_jarzynski", "F_other", "F_spread", "lag_mean", "lag_eq", "lag_sd", "lag_sd_eq"} <= set(d.columns)
    np.testing.assert_allclose(d.F_spread, 0.0, atol=1e-9)
