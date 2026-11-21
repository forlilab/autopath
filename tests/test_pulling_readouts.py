import numpy as np
import pandas as pd
import pytest

from autopath.pulling.readouts import profile_readouts, rupture_displacement, rupture_readouts


def _pmfs(peak_at=0.5):
    r = np.linspace(0.3, 1.3, 101)
    d = r - r[0]
    bump = 20 * np.exp(-((d - peak_at) / 0.05) ** 2)
    rise = 30 * (1 - np.exp(-d / 0.2))
    rows = []
    for est, sp, dG in [("cumulant", 0.01, bump + 5 * d), ("jarzynski", 0.01, rise),
                        ("force", 0.01, rise + 99), ("force", 0.0, rise + 1)]:
        rows.append(pd.DataFrame(dict(estimator=est, speed=sp, r_coord=r, dG=dG)))
    return pd.concat(rows)


def _processed(peak_step, n_pulls=4, n_steps=200, speed=0.01):
    rows = []
    for k in range(n_pulls):
        steps = np.arange(n_steps)
        force = 100 * np.exp(-((steps - peak_step) / 10.0) ** 2) + k
        rows.append(pd.DataFrame(dict(trajname=f"p{k}", speed=speed, step=steps,
                                      r_coord=0.3 + 0.001 * steps, force=force)))
    return pd.concat(rows)


def test_profile_readout_policy():
    out = profile_readouts(_pmfs()).set_index("estimator")
    assert set(out.index) == {"cumulant", "jarzynski", "force"}      # per-speed force is not a free energy
    assert out.loc["force", "speed"] == 0.0
    assert out.loc["cumulant", "dG_profile"] == pytest.approx(20 + 5 * 0.5, abs=0.5)  # the peak, not the end
    assert out.loc["jarzynski", "dG_profile"] == pytest.approx(30 * (1 - np.exp(-5)), rel=1e-6)
    assert not out.loc["cumulant", "peak_fell_back"]


def test_rupture_readout_uses_median_displacement():
    rup = {f"s{i}": rupture_displacement(_processed(p)) for i, p in enumerate((80, 100, 140))}
    assert rup["s1"][0.01] == pytest.approx(0.100)
    out = rupture_readouts({s: _pmfs() for s in rup}, rup)
    assert out.d_star.unique() == pytest.approx([0.100])               # median of 0.08, 0.10, 0.14
    jar = out[out.estimator == "jarzynski"].dG_rupture.iloc[0]
    assert jar == pytest.approx(30 * (1 - np.exp(-0.1 / 0.2)), rel=1e-3)


def test_rupture_readout_nan_when_profile_too_short():
    rup = {"s0": {0.01: 5.0}}
    out = rupture_readouts({"s0": _pmfs()}, rup)
    assert out.dG_rupture.isna().all()
