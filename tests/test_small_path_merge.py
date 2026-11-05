"""Small DTW paths are merged into surviving paths, never dropped."""
import logging

import numpy as np
import pandas as pd
import pytest

from autopath.pulling.Estimators import JarzynskiEstimator
from autopath.pulling.PathModel import DTWPathModel
from autopath.pulling.SMDData import SMDData
from tests.test_path_combination import BETA, SPEED, _Stub, _weights
from autopath.pulling.Estimators import calculate_weighted_pmf

STEPS = list(range(8))


def _features(groups, speed=SPEED, seed=0):
    """Feature table; ``groups`` maps a route offset to its pull count."""
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for offset, n in groups.items():
        for _ in range(n):
            for s in STEPS:
                rows.append({"trajname": f"t{k}", "speed": speed, "step": s,
                             "time": float(s), "f1": offset + 0.05 * rng.normal(),
                             "f2": offset * s / 8 + 0.05 * rng.normal()})
            k += 1
    return pd.DataFrame(rows)


def _cluster(tmp_path, feat, n_paths):
    m = DTWPathModel(do_plots=False, outdir=str(tmp_path), n_geom_pcs=None)
    maps = m.fit_transform(feat, n_paths=n_paths, cluster_across_speeds=False)
    return m, maps


def test_small_path_merged_into_nearest(tmp_path, caplog):
    feat = _features({0.0: 30, 5.0: 3})
    m, maps = _cluster(tmp_path, feat, 2)
    before = pd.Series(maps).value_counts()
    assert before.min() == 3
    with caplog.at_level(logging.WARNING):
        new, merges = m.merge_small_paths(maps, 5)
    assert len(new) == len(maps)
    assert pd.Series(new).nunique() == 1
    assert len(merges) == 1 and merges[0]["n_pulls"] == 3
    assert merges[0]["from_path"] != merges[0]["to_path"]
    assert any("Merged small path" in r.message for r in caplog.records)


def test_all_small_paths_collapse_to_one(tmp_path):
    feat = _features({0.0: 3, 5.0: 2, 10.0: 3})
    m, maps = _cluster(tmp_path, feat, 3)
    new, merges = m.merge_small_paths(maps, 5)
    assert len(new) == len(maps)
    assert pd.Series(new).nunique() == 1
    assert sum(r["n_pulls"] for r in merges) == 5


def test_no_merge_when_all_paths_large(tmp_path):
    feat = _features({0.0: 8, 5.0: 8})
    m, maps = _cluster(tmp_path, feat, 2)
    new, merges = m.merge_small_paths(maps, 5)
    assert new == maps and merges == []


def test_eq21_after_merge_equals_pooled_jarzynski(tmp_path):
    feat = _features({0.0: 30, 5.0: 3})
    m, maps = _cluster(tmp_path, feat, 2)
    new, _ = m.merge_small_paths(maps, 5)
    rng = np.random.default_rng(1)
    rows = [{"trajname": t, "speed": SPEED, "step": s, "path": p,
             "work": (s + 1) * (10.0 + 4.0 * (i % 2)) + rng.normal(0, 2.0)}
            for i, (t, p) in enumerate(new.items()) for s in STEPS]
    raw = pd.DataFrame(rows)
    smd = _Stub(raw, STEPS)
    JarzynskiEstimator().fit_transform(smd)
    p_neq = SMDData._compute_p_neq(raw.groupby("path")["trajname"].nunique().to_dict())
    out = calculate_weighted_pmf(smd, weights_by_estimator=_weights(p_neq),
                                 path_combination="eq21").set_index("step")
    for s in STEPS:
        pooled = JarzynskiEstimator.estimate_dG(
            raw.loc[raw["step"] == s, "work"].to_numpy(), BETA)
        assert out.loc[s, "dG"] == pytest.approx(pooled["dG"], abs=1e-9)


def test_cross_speed_merge_counts_per_speed(tmp_path):
    # Both paths have >=5 pulls in total but <5 per speed; the per-speed support
    # trim would delete everything, so they must collapse into one path.
    speeds = [0.001, 0.005, 0.01, 0.015]
    feat = pd.concat([
        _features({0.0: 4, 5.0: 2}, speed=v, seed=i).assign(
            trajname=lambda d, i=i: f"s{i}_" + d["trajname"])
        for i, v in enumerate(speeds)], ignore_index=True)
    m = DTWPathModel(do_plots=False, outdir=str(tmp_path), n_geom_pcs=None)
    maps = m.fit_transform(feat, n_paths=2, cluster_across_speeds=True)
    assert pd.Series(maps).value_counts().min() == 8
    sp = feat.drop_duplicates("trajname").set_index("trajname")["speed"].to_dict()
    new, merges = m.merge_small_paths(maps, 5, speeds=sp)
    assert len(new) == len(maps)
    assert pd.Series(new).nunique() == 1
    assert sorted(r["speed"] for r in merges) == sorted(speeds)
