"""Per-speed persistence of medoid_info.json and pocket-distance caches.

Regression tests for the bug where the medoid_info.json file and the pocket
distance cache each retained only the *last* speed, because check_convergence
calls the clustering / distance code once per speed (each with a per-speed
SMDData sharing one outdir).
"""
import json

import numpy as np
import pandas as pd


def _toy_feat_df(speed, trajtag, n_traj=4, n_steps=25, seed=0):
    """Trace features for a few trajectories forming two rough clusters."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_traj):
        group = t % 2
        name = f"sMD_replica-{trajtag}{t}_v{speed}_forward"
        for s in range(n_steps):
            r = s / n_steps
            rows.append({
                "trajname": name, "speed": speed, "step": s, "time": float(s),
                "lag": 0.05 * rng.standard_normal(),
                "r_before": r + 0.01 * rng.standard_normal(),
                "work": (2.0 if group else -2.0) + 0.1 * rng.standard_normal() + 5 * r,
            })
    return pd.DataFrame(rows)


def _load_medoid_info(tmp_path):
    with open(tmp_path / "path_analysis" / "medoid_info.json") as f:
        return json.load(f)


def test_medoid_info_accumulates_per_speed(tmp_path):
    """Two per-speed fit_transform calls must both appear in medoid_info.json."""
    from autopath.pulling.PathModel import DTWPathModel
    (tmp_path / "path_analysis").mkdir()

    m = DTWPathModel(do_plots=False, outdir=str(tmp_path))
    m.fit_transform(_toy_feat_df(0.005, "a", seed=1), n_paths=2)
    m2 = DTWPathModel(do_plots=False, outdir=str(tmp_path))
    m2.fit_transform(_toy_feat_df(0.015, "b", seed=2), n_paths=2)

    info = _load_medoid_info(tmp_path)
    # Both speeds are present (the previous behaviour kept only the last).
    assert set(info["by_speed"]) == {"0.005", "0.015"}
    # by_speed maps speed -> {medoid_name: path_id}; no redundant name lists / union.
    assert set(info) == {"by_speed"}
    assert any("_v0.005_" in n for n in info["by_speed"]["0.005"])
    assert any("_v0.015_" in n for n in info["by_speed"]["0.015"])
    # Path ids carry the speed of their group.
    assert all("v0.015" in p for p in info["by_speed"]["0.015"].values())
    # Instance attributes expose the per-speed maps and the derived flat union.
    assert set(m2.medoids_by_speed) == {"0.005", "0.015"}
    union = {}
    for mapping in info["by_speed"].values():
        union.update(mapping)
    assert set(m2.medoid_to_path) == set(union)
    assert m2.all_medoid_names == sorted(union)


def test_medoid_info_single_call_multi_speed(tmp_path):
    """A single fit_transform over multiple speeds records each speed."""
    from autopath.pulling.PathModel import DTWPathModel
    (tmp_path / "path_analysis").mkdir()

    df = pd.concat([_toy_feat_df(0.005, "a", seed=1),
                    _toy_feat_df(0.015, "b", seed=2)], ignore_index=True)
    m = DTWPathModel(do_plots=False, outdir=str(tmp_path))
    m.fit_transform(df, n_paths=2)

    info = _load_medoid_info(tmp_path)
    assert set(info["by_speed"]) == {"0.005", "0.015"}


def test_pocket_distances_per_speed_files(tmp_path, monkeypatch):
    """calculate_pocket_distances writes/loads one cache file per speed."""
    from autopath.pulling.SMDData import SMDData

    trajs = [
        "sMD_replica-1_v0.005_forward.dcd",
        "sMD_replica-2_v0.005_forward.dcd",
        "sMD_replica-3_v0.015_forward.dcd",
    ]

    smd = SMDData.__new__(SMDData)          # bypass log parsing
    smd.outdir = str(tmp_path)
    smd.sysname = "sys"
    smd.reference_pdb = "ref.pdb"
    smd.traj_files = trajs

    # Fake the trajectory-reading step: one row per trajectory, no MDAnalysis.
    calls = {"n": 0}

    def fake_compute(self, trajs_, group_A, group_B, stride):
        calls["n"] += 1
        rows = []
        for tr in trajs_:
            name = SMDData._traj_to_log_name(tr)
            speed = float(name.split("_")[-2].strip("v"))
            rows.append({"trajname": name, "speed": speed, "step": 0,
                         "time": 0.0, "dist_0": 1.0})
        return pd.DataFrame(rows)

    monkeypatch.setattr(SMDData, "_pocket_distances_for_trajs", fake_compute)

    df = smd.calculate_pocket_distances(group_A="a", group_B="b")
    # One per-speed file each; both speeds returned.
    assert (tmp_path / "sys_pocketDistances_v0.005.csv").exists()
    assert (tmp_path / "sys_pocketDistances_v0.015.csv").exists()
    assert sorted(df["speed"].unique()) == [0.005, 0.015]
    first_calls = calls["n"]

    # Second call with recompute=False must hit the cache (no new compute).
    df2 = smd.calculate_pocket_distances(group_A="a", group_B="b")
    assert calls["n"] == first_calls
    assert sorted(df2["speed"].unique()) == [0.005, 0.015]

    # recompute=True recomputes every speed.
    smd.calculate_pocket_distances(group_A="a", group_B="b", recompute=True)
    assert calls["n"] > first_calls


def test_pocket_distances_migrates_legacy_cache(tmp_path, monkeypatch):
    """A pre-existing single-file cache is reused per speed without recomputing."""
    from autopath.pulling.SMDData import SMDData

    smd = SMDData.__new__(SMDData)
    smd.outdir = str(tmp_path)
    smd.sysname = "sys"
    smd.reference_pdb = "ref.pdb"
    smd.traj_files = ["sMD_replica-3_v0.015_forward.dcd"]

    # Legacy combined cache holds v0.015 rows (as the old single-file scheme did).
    pd.DataFrame([{"trajname": "sMD_replica-3_v0.015_forward", "speed": 0.015,
                   "step": 0, "time": 0.0, "dist_0": 2.0}]).to_csv(
        tmp_path / "sys_pocketDistances.csv", index=False)

    def boom(*a, **k):
        raise AssertionError("should not recompute when legacy cache covers the speed")
    monkeypatch.setattr(SMDData, "_pocket_distances_for_trajs", boom)

    df = smd.calculate_pocket_distances(group_A="a", group_B="b")
    assert (df["speed"] == 0.015).all() and len(df) == 1
    # Promoted to a per-speed file for future loads.
    assert (tmp_path / "sys_pocketDistances_v0.015.csv").exists()
