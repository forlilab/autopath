import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from autopath.pulling.hummer_szabo import (
    f_cumulant, f_force_v0, f_jarzynski, f_self_consistent, g0_profile, hs_log_weights,
    slice_table, slice_weights, wham_density,
)

from .hs_toys import double_well, flat, harmonic, protocol_grids, simulate_pulls

K = 3765.6
BETA = 1.0 / (0.008314462618 * 300.0)


def _F(src, rows, slices, c):
    if src == "jarzynski":
        return f_jarzynski(slices, BETA)
    if src == "cumulant":
        return f_cumulant(slices, BETA)
    if src == "force":
        return f_force_v0(slices, {v: v for v in slices.speed.unique()})
    if src == "self_consistent":
        return f_self_consistent(rows, slices, c, K, BETA)
    raise AssertionError(src)


def _g0(rows, F, c):
    slices = slice_table(rows, protocol_grids(rows), BETA)
    cv = slice_weights(slices, c)
    Fv = _F(F, rows, slices, cv)
    keep = np.isfinite(Fv)          # force: steps without a v->0 intercept
    sub = rows.merge(slices.loc[keep, ["speed", "step"]], on=["speed", "step"])
    slices = slices[keep].reset_index(drop=True)
    logw = hs_log_weights(sub, slices, Fv[keep], cv[keep], K, BETA)
    return g0_profile(sub.r_after.to_numpy(), logw, sub.trajname.to_numpy(), BETA)


def _max_dev(out, G0):
    m = (out.ess_pulls >= 30).to_numpy()
    assert m.sum() > 3
    d = out.G0.to_numpy()[m] - G0(out.r_bin.to_numpy()[m])
    return np.max(np.abs(d - d.mean()))


@pytest.fixture(scope="module")
def dw():   # double well, three speeds, common lambda0
    G0, dG0 = double_well(r_a=1.0, r_b=1.6, barrier=10.0)
    rows = pd.concat([simulate_pulls(G0, dG0, n_pulls=60, speed=v, r_start=0.95, length_nm=0.8, seed=i)
                      for i, v in enumerate((0.0025, 0.005, 0.01))], ignore_index=True)
    return G0, rows


@pytest.mark.parametrize("src", ["jarzynski", "force", "self_consistent"])
def test_double_well_recovered_by(src, dw):
    G0, rows = dw
    assert _max_dev(_g0(rows, F=src, c="ess"), G0) < 1.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cumulant_exact_on_harmonic(seed):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=60, speed=0.005, r_start=0.9, length_nm=0.2, seed=seed)
    assert _max_dev(_g0(rows, F="cumulant", c="ess"), G0) < 1.0


def test_wham_with_jarzynski_F_held_is_classic_hs(dw):
    # One rho step at F = f_jarzynski, c = uniform equals the HS-weight histogram on bin centres.
    _, rows = dw
    bin_nm = 0.002
    rows = rows.assign(r_after=(np.floor(rows.r_after / bin_nm) + 0.5) * bin_nm)
    slices = slice_table(rows, protocol_grids(rows), BETA)
    F, c = f_jarzynski(slices, BETA), slice_weights(slices, "uniform")
    edges, rho = wham_density(rows, slices, F, c, K, BETA, bin_nm)
    lw = hs_log_weights(rows, slices, F, c, K, BETA, dr=bin_nm)
    h, _ = np.histogram(rows.r_after, edges, weights=np.exp(lw - lw.max()))
    np.testing.assert_allclose(rho / rho.sum(), h / h.sum(), rtol=0, atol=1e-8)


def test_self_consistent_stride_matches_full(dw):
    _, rows = dw
    slices = slice_table(rows, protocol_grids(rows), BETA)
    c = slice_weights(slices, "ess")
    full = f_self_consistent(rows, slices, c, K, BETA)
    strided = f_self_consistent(rows, slices, c, K, BETA, stride=10)
    assert np.isfinite(strided).all()
    d = strided - full
    assert np.max(np.abs(d - d.mean())) < 0.5


def test_wham_not_converged_is_not_supported():
    rows = simulate_pulls(*flat(), n_pulls=5, speed=0.01, length_nm=0.1)
    slices = slice_table(rows, protocol_grids(rows), BETA)
    with pytest.raises(ValueError, match="not supported: WHAM did not converge"):
        f_self_consistent(rows, slices, slice_weights(slices, "uniform"), K, BETA, max_iter=1)


def test_force_needs_two_speeds():
    rows = simulate_pulls(*flat(), n_pulls=5, speed=0.01, length_nm=0.1)
    with pytest.raises(ValueError, match="not supported"):
        f_force_v0(slice_table(rows, protocol_grids(rows), BETA), {0.01: 0.01})


def test_cumulant_needs_two_pulls_per_slice():
    rows = simulate_pulls(*flat(), n_pulls=1, speed=0.01, length_nm=0.1)
    with pytest.raises(ValueError, match="not supported"):
        f_cumulant(slice_table(rows, protocol_grids(rows), BETA), BETA)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unequal_pull_counts_across_speeds(seed):
    # Unequal coverage: a slice weight that scales with n_pulls biases where the fast windows end.
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = pd.concat([
        simulate_pulls(G0, dG0, n_pulls=30, speed=0.0025, r_start=0.9, length_nm=0.2, seed=seed),
        simulate_pulls(G0, dG0, n_pulls=90, speed=0.005, r_start=0.9, length_nm=0.1, seed=seed + 10),
    ], ignore_index=True)
    assert _max_dev(_g0(rows, F="jarzynski", c="ess"), G0) < 1.0


def test_cumulant_formula():
    rows = simulate_pulls(*flat(), n_pulls=4, speed=0.01, length_nm=0.02, seed=7)
    slices = slice_table(rows, protocol_grids(rows), BETA)
    g = rows.groupby(["speed", "step"])["work"]
    ref = (g.mean() - 0.5 * BETA * g.var(ddof=1)).to_numpy()
    np.testing.assert_allclose(f_cumulant(slices, BETA), ref, rtol=1e-12)


def test_force_v0_recovers_intercept():
    speeds, b = (0.001, 0.002, 0.004), 50.0
    a = lambda s: 3.0 * np.sin(0.7 * s)
    recs = [(v, s, 0.9 + 0.001 * s, a(s) + b * v) for v in speeds for s in range(10)]
    recs += [(v, 10, 0.91, a(10) + b * v) for v in speeds[:2]]     # step at fewer than all speeds
    slices = pd.DataFrame(recs, columns=["speed", "step", "lam", "Wmean"])
    F = f_force_v0(slices, {v: v for v in speeds})
    full = (slices.step < 10).to_numpy()
    np.testing.assert_allclose(F[full], a(slices.step[full].to_numpy()), rtol=0, atol=1e-8)
    assert np.isnan(F[~full]).all()


def test_force_v0_needs_every_realized_speed():
    slices = pd.DataFrame({"speed": [0.001, 0.002], "step": [0, 0], "lam": [0.9, 0.9], "Wmean": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not supported"):
        f_force_v0(slices, {0.001: 0.001})


def test_self_consistent_is_fixed_point(dw):
    _, rows = dw
    bin_nm, tol = 0.002, 1e-6
    slices = slice_table(rows, protocol_grids(rows), BETA)
    c = slice_weights(slices, "ess")
    F = f_self_consistent(rows, slices, c, K, BETA, bin_nm=bin_nm, tol=tol)
    edges, rho = wham_density(rows, slices, F, c, K, BETA, bin_nm)
    occ = rho > 0
    centres = 0.5 * (edges[:-1] + edges[1:])[occ]
    u = 0.5 * K * (centres[None, :] - slices.lam.to_numpy()[:, None]) ** 2
    Fn = -logsumexp(np.log(rho[occ])[None, :] - BETA * u, axis=1) / BETA
    np.testing.assert_allclose(Fn - Fn[0], F - F[0], rtol=0, atol=10 * tol)
    Fj = f_jarzynski(slices, BETA)
    assert np.max(np.abs((F - F[0]) - (Fj - Fj[0]))) > tol


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_self_consistent_immune_to_staggered_start(seed):
    # Per-speed Jarzynski F is referenced to each speed's own lambda0; WHAM solves one common F.
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = pd.concat([
        simulate_pulls(G0, dG0, n_pulls=30, speed=0.0025, r_start=0.9, length_nm=0.2, seed=seed),
        simulate_pulls(G0, dG0, n_pulls=90, speed=0.005, r_start=1.0, length_nm=0.2, seed=seed + 10),
    ], ignore_index=True)
    sc = _max_dev(_g0(rows, F="self_consistent", c="ess"), G0)
    jz = _max_dev(_g0(rows, F="jarzynski", c="ess"), G0)
    assert sc < 1.0
    assert jz > sc


def test_self_consistent_needs_increasing_lambda():
    rows = simulate_pulls(*flat(), n_pulls=3, speed=0.01, length_nm=0.02, seed=2)
    slices = slice_table(rows, protocol_grids(rows), BETA)
    slices["lam"] = slices["lam"].to_numpy()[::-1]
    with pytest.raises(ValueError, match="not supported"):
        f_self_consistent(rows, slices, slice_weights(slices, "uniform"), K, BETA)
