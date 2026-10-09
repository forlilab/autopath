import numpy as np
import pandas as pd
import pytest

from autopath.metadynamics.learned.select import (fisher, hidden_variable_p, labels_from_r,
                                                   prune, pull_frames)


def test_pull_frames_uses_own_frame_count(tmp_path):
    p = tmp_path / "sMD_replica-1_v0.01_forward.dat"
    t = np.linspace(0, 100, 201)
    pd.DataFrame({"step": np.arange(201), "time": t, "r_after": 0.8 + 0.01 * t,
                  "dW_protocol": np.full(201, 0.5)}).to_csv(p, index=False)
    a, b = pull_frames(str(p), 11), pull_frames(str(p), 6)
    assert a.t_ps.iloc[-1] == pytest.approx(100) and b.t_ps.iloc[-1] == pytest.approx(100)
    assert a.r.iloc[5] == pytest.approx(0.8 + 0.01 * 50)
    assert a.W.iloc[-1] == pytest.approx(0.5 * 201)


def test_labels_from_r():
    first = np.array([0.79, 0.80, 0.81])
    lab = labels_from_r(np.array([0.80, 1.2, 2.0, 2.2]), first, r_max=2.2)
    assert list(lab) == ["bound", "vestibule", "bulk", "bulk"]


def test_prune_keeps_best_of_correlated_pair():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    V = np.c_[a, a + 0.01 * rng.normal(size=500), rng.normal(size=500)]
    kept = prune(V, ["a", "a2", "b"], score=np.array([1.0, 2.0, 0.5]))
    assert sorted(kept) == [1, 2]


def test_fisher_separated_vs_overlapping():
    y = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    assert fisher(np.r_[np.zeros(100), np.ones(100) * 5], y) > 1e3
    rng = np.random.default_rng(0)
    assert fisher(rng.normal(size=200), y) < 0.1


def _pulls(n_pull=60, n_frame=40, planted=True, speeds=(0.01, 0.02), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in speeds:
        for p in range(n_pull):
            hidden = rng.normal()
            r = np.linspace(0.8, 2.0, n_frame)
            W = np.cumsum(np.ones(n_frame)) + (3.0 * hidden * (r > 1.0) if planted else 0)
            W = W + rng.normal(scale=0.5, size=n_frame)
            x = hidden + rng.normal(scale=0.3, size=n_frame)
            rows.append(pd.DataFrame({"r": r, "W": W, "x": x, "pull": f"{v}_{p}", "speed": v}))
    return pd.concat(rows, ignore_index=True)


def test_hidden_variable_finds_planted_feature():
    df = _pulls(planted=True)
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=200, rng=np.random.default_rng(1))
    assert p < 0.05 and rho > 0


def test_hidden_variable_noise_rate_is_nominal():
    hits = 0
    for s in range(20):
        df = _pulls(planted=False, seed=100 + s)
        p, _ = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                                 df.speed.values, n_perm=100, rng=np.random.default_rng(s))
        hits += p < 0.05
    assert hits <= 4                              # nominal 5% of 20 = 1; generous bound


def test_hidden_variable_single_speed_uses_split_halves():
    df = _pulls(planted=True, speeds=(0.01,), n_pull=80)
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=200, rng=np.random.default_rng(2))
    assert p < 0.05 and rho > 0


def test_hidden_variable_skips_bins_where_feature_is_constant():
    df = _pulls(planted=True)
    df.loc[df.r > 1.6, "x"] = 0.0
    p, rho = hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                               df.speed.values, n_perm=200, rng=np.random.default_rng(3))
    assert p < 0.05 and rho > 0


def test_hidden_variable_constant_feature_is_null():
    df = _pulls(planted=True)
    df["x"] = 0.0
    assert hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                             df.speed.values, n_perm=50, rng=np.random.default_rng(4)) == (1.0, 0.0)


def test_hidden_variable_sign_disagreement_across_speeds_is_null():
    df = _pulls(planted=True)
    df.loc[df.speed == 0.02, "x"] *= -1.0
    assert hidden_variable_p(df.x.values, df.W.values, df.r.values, df.pull.values,
                             df.speed.values, n_perm=200, rng=np.random.default_rng(5)) == (1.0, 0.0)
