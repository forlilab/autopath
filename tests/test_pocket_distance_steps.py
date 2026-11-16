"""Per-frame features (pocket distances, ligand shape) carry the log step/time of their frame."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autopath.pulling import SMDData
from autopath.pulling.AnalysisSMD import SMDAnalysis

from .hs_toys import flat, simulate_pulls, write_dat

SAVE_FREQ = 5
POCKET = "resname PRT"
LIGAND = "resname LIG"


def _write_pdb(fn):
    lines = [
        "ATOM      1  CA  PRT A   1       0.000   0.000   0.000  1.00  0.00           C",
        "HETATM    2  C1  LIG B   2      10.000   0.000   0.000  1.00  0.00           C",
        "END",
    ]
    with open(fn, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_dcd(out, n_frames):
    """Ligand at x = 1 + frame/100 nm from a pocket atom at the origin."""
    from MDAnalysis.lib.formats.libdcd import DCDFile

    with DCDFile(out, "w") as f:
        f.write_header(remarks="t", natoms=2, istart=0, nsavc=1, delta=1.0, is_periodic=True)
        for j in range(n_frames):
            xyz = np.zeros((2, 3), dtype=np.float32)
            xyz[1, 0] = 10.0 + 0.1 * j
            f.write(xyz, box=np.array([50.0, 90.0, 50.0, 90.0, 90.0, 50.0]))
    return out


def _toy(tmp_path, n_pulls=3, length_nm=0.1, extra_frames=0):
    rows = simulate_pulls(*flat(), n_pulls=n_pulls, speed=0.01, length_nm=length_nm, seed=7)
    files = write_dat(rows, tmp_path / "pulls")
    pdb = str(tmp_path / "ref.pdb")
    _write_pdb(pdb)
    smd = SMDData(files, temperature=300.0, completion_threshold_nm=0.0, reference_pdb=pdb)
    for fn, dcd in zip(files, smd.traj_files):
        n_rows = len(pd.read_csv(fn, comment="#"))
        _write_dcd(dcd, n_rows // SAVE_FREQ + extra_frames)
    return smd, files


def test_pocket_distance_rows_carry_frame_step_and_time(tmp_path):
    smd, files = _toy(tmp_path)
    df = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    raw = smd.raw_data
    for name, g in df.groupby("trajname"):
        g = g.sort_values("step")
        j = np.arange(len(g))
        np.testing.assert_array_equal(g["step"].to_numpy(), (j + 1) * SAVE_FREQ - 1)
        np.testing.assert_allclose(g["dist_0"].to_numpy(), 1.0 + 0.01 * j, atol=1e-5)
        log_t = raw[raw.trajname == name].set_index("step")["time"]
        np.testing.assert_allclose(g["time"].to_numpy(), log_t.loc[g["step"]].to_numpy())
    assert "frame" not in df.columns


def test_stride_keeps_frame_steps(tmp_path):
    smd, _ = _toy(tmp_path)
    df = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=2)
    g = df[df.trajname == df.trajname.iloc[0]].sort_values("step")
    np.testing.assert_array_equal(g["step"].to_numpy()[:3], [4, 14, 24])


def test_inconsistent_frame_count_not_supported(tmp_path):
    smd, _ = _toy(tmp_path, extra_frames=3)
    with pytest.raises(ValueError, match="not supported"):
        smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)


def test_old_convention_cache_is_recomputed(tmp_path, caplog):
    smd, _ = _toy(tmp_path)
    new = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    cache = Path(smd.outdir) / f"{smd.sysname}_pocketDistances_v0.01.csv"
    assert cache.exists()
    # Rewrite the cache as the frame-index convention wrote it (no frame column).
    old = new.copy()
    old["step"] = old.groupby("trajname").cumcount()
    old.to_csv(cache, index=False)
    with caplog.at_level(logging.WARNING, logger="autopath.pulling.SMDData"):
        again = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    assert any("frame-index" in r.getMessage() for r in caplog.records)
    pd.testing.assert_frame_equal(
        again.sort_values(["trajname", "step"]).reset_index(drop=True),
        new.sort_values(["trajname", "step"]).reset_index(drop=True))


def test_new_cache_is_reused(tmp_path, monkeypatch):
    smd, _ = _toy(tmp_path)
    first = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)

    def boom(*a, **k):
        raise AssertionError("cache in the log-step convention must be reused")
    monkeypatch.setattr(SMDData, "_pocket_distances_for_trajs", boom)
    again = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    pd.testing.assert_frame_equal(again, first)


def test_boundary_cut_removes_rows_beyond_cap(tmp_path):
    smd, _ = _toy(tmp_path, length_nm=0.2)
    feat = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    raw = smd.raw_data
    cap = float(raw["r_coord"].quantile(0.5))
    cut = SMDAnalysis._cut_to_boundary(feat, raw, cap)
    r = feat.merge(raw[["trajname", "step", "r_coord"]], on=["trajname", "step"], how="left")
    assert r["r_coord"].notna().all()
    expected = int((r["r_coord"] <= cap).sum())
    assert 0 < expected < len(feat)
    assert len(cut) == expected
    assert cut["step"].max() < feat["step"].max()


def test_merged_features_align_on_frame_rows(tmp_path):
    smd, _ = _toy(tmp_path)
    dist = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    trace = smd.get_trace_features(features=["work"])
    merged = SMDData.merge_feature_sets(trace, dist)
    # Each frame's distance lands on its own log row, not the last frame.
    on_frames = merged.merge(dist[["trajname", "step", "dist_0"]], on=["trajname", "step"],
                             suffixes=("", "_frame"))
    assert len(on_frames) == len(dist)
    np.testing.assert_allclose(on_frames["dist_0"], on_frames["dist_0_frame"])


def test_ligand_features_carry_frame_step_and_time(tmp_path):
    from autopath.pulling.LigandFeatures import LigandTrajectoryFeatures

    smd, _ = _toy(tmp_path)
    lf = LigandTrajectoryFeatures(lig_resname="LIG", features=["rog"], stride=1).compute(
        smd.traj_files, smd.reference_pdb, smd.log_files)
    raw = smd.raw_data
    for name, g in lf.groupby("trajname"):
        j = np.arange(len(g))
        np.testing.assert_array_equal(g["step"].to_numpy(), (j + 1) * SAVE_FREQ - 1)
        log_t = raw[raw.trajname == name].set_index("step")["time"]
        np.testing.assert_allclose(g["time"].to_numpy(), log_t.loc[g["step"]].to_numpy())


def _write_log(fn, n_rows):
    pd.DataFrame({"step": np.arange(n_rows), "time": 0.1 * np.arange(n_rows)}).to_csv(fn, index=False)


@pytest.mark.parametrize("n_rows, n_frames, last", [(1003, 200, 999), (1377, 275, 1374)])
def test_frame_log_index_mapping(tmp_path, n_rows, n_frames, last):
    from autopath.pulling.SMDData import frame_log_index

    fn = tmp_path / "pull.dat"
    _write_log(fn, n_rows)
    steps, times = frame_log_index(str(fn), n_frames)
    assert len(steps) == n_frames and steps[-1] == last
    np.testing.assert_allclose(times, 0.1 * steps)


@pytest.mark.parametrize("n_rows, n_frames", [(24, 4), (10, 0), (10, 11), (1003, 199)])
def test_frame_log_index_refuses(tmp_path, n_rows, n_frames):
    from autopath.pulling.SMDData import frame_log_index

    fn = tmp_path / "pull.dat"
    _write_log(fn, n_rows)
    with pytest.raises(ValueError, match="not supported"):
        frame_log_index(str(fn), n_frames)


def test_cache_covering_fewer_trajectories_is_recomputed(tmp_path, caplog):
    smd, files = _toy(tmp_path)
    sub = SMDData(files[:2], temperature=300.0, completion_threshold_nm=0.0,
                  reference_pdb=smd.reference_pdb)
    sub.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    with caplog.at_level(logging.WARNING, logger="autopath.pulling.SMDData"):
        df = smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    assert any("different set of trajectories" in r.getMessage() for r in caplog.records)
    assert df["trajname"].nunique() == 3


@pytest.mark.parametrize("kw, word", [(dict(stride=2), "stride"),
                                      (dict(group_A="resname PRT and name CA"), "group_A")])
def test_cache_with_other_parameters_is_recomputed(tmp_path, caplog, kw, word):
    smd, _ = _toy(tmp_path)
    smd.calculate_pocket_distances(group_A=POCKET, group_B=LIGAND, stride=1)
    args = dict(group_A=POCKET, group_B=LIGAND, stride=1) | kw
    with caplog.at_level(logging.WARNING, logger="autopath.pulling.SMDData"):
        df = smd.calculate_pocket_distances(**args)
    assert any(word in r.getMessage() for r in caplog.records)
    if "stride" in kw:
        g = df[df.trajname == df.trajname.iloc[0]].sort_values("step")
        np.testing.assert_array_equal(g["step"].to_numpy()[:2], [4, 14])
