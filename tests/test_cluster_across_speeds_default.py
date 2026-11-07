"""Clustering pools speeds by default: one path label per pathway, shared across speeds."""
import json
import logging
import os

import pandas as pd

from autopath.config import Config
from autopath.pulling import SMDData
from autopath.pulling.AnalysisSMD import SMDAnalysis
from autopath.pulling.PathModel import DTWPathModel

from .hs_toys import double_well, simulate_pulls, write_dat
from .test_pathmodel_cross_speed_scaling import _make_df


def _two_speed_smd(tmp_path):
    parts = []
    for i, (speed, seed) in enumerate([(0.01, 1), (0.02, 2)]):
        r = simulate_pulls(*double_well(), n_pulls=12, speed=speed, length_nm=0.8, seed=seed)
        r["trajname"] = r["trajname"].str.replace("replica-", f"replica-{i + 1}0")
        parts.append(r)
    rows = pd.concat(parts, ignore_index=True)
    rows.attrs.update(parts[0].attrs)
    files = write_dat(rows, tmp_path / "pulls")
    return SMDData(files, temperature=300.0, completion_threshold_nm=0.0)


def test_path_model_default_pools_speeds(tmp_path):
    feat = _make_df().drop(columns=["true_path"])
    model = DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None)
    mapping = model.fit_transform(feat, n_paths=2)
    assert not any("_v" in p for p in mapping.values()), mapping


def test_config_default_pools_speeds():
    assert Config().cluster_across_speeds is True


def test_run_default_labels_without_speed_and_builds_meanforce_ti(tmp_path):
    smd = _two_speed_smd(tmp_path)
    out = tmp_path / "out"
    an = SMDAnalysis(do_plots=False, outdir=str(out), estimators=["cumulant", "force"], n_paths=2)
    res = an.run(smd, cluster_to_boundary=False)
    labels = set(res.raw_data["path"].dropna())
    assert labels and not any("_v" in p for p in labels), labels
    ti = pd.read_csv(out / "force_meanforce_ti.csv")
    assert len(ti) > 0 and ti["dG_z"].notna().all()
    assert os.path.exists(out / "friction.csv")


def test_cross_speed_fit_replaces_per_speed_medoids(tmp_path):
    feat = _make_df().drop(columns=["true_path"])
    for sp, g in feat.groupby("speed"):
        DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None).fit_transform(
            g, n_paths=2, cluster_across_speeds=False)
    m = DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None)
    m.fit_transform(feat, n_paths=2)
    with open(tmp_path / "path_analysis" / "medoid_info.json") as f:
        info = json.load(f)
    assert set(info["by_speed"]) == {"all"}
    assert len(m.all_medoid_names) == 2 and set(m.medoid_to_path.values()) == {"path-0", "path-1"}


def test_convergence_fits_do_not_persist_medoids(tmp_path):
    feat = _make_df().drop(columns=["true_path"])
    m = DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None)
    m.fit_transform(feat, n_paths=2)
    path = tmp_path / "path_analysis" / "medoid_info.json"
    before = path.read_text()
    g = feat[feat.speed == 0.001]
    c = DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None)
    c.fit_transform(g, n_paths=2, cluster_across_speeds=False, persist_medoids=False)
    assert path.read_text() == before
    assert set(c.medoids_by_speed) == {"0.001"}


def test_check_convergence_keeps_run_medoids(tmp_path):
    smd = _two_speed_smd(tmp_path)
    out = tmp_path / "out"
    an = SMDAnalysis(do_plots=False, outdir=str(out), estimators=["cumulant"], n_paths=2)
    an.run(smd, cluster_to_boundary=False)
    path = out / "path_analysis" / "medoid_info.json"
    before = json.loads(path.read_text())
    an.check_convergence(logs=smd.log_files, min_replicas=5, cluster_to_boundary=False)
    assert json.loads(path.read_text()) == before == {"by_speed": before["by_speed"]}
    assert set(before["by_speed"]) == {"all"}


def test_unsuffixed_path_friction_speed_is_logged(caplog):
    from autopath.pulling.Diagnostics import _get_friction_profile_for_path

    prof = {0.01: pd.Series([1.0]), 0.02: pd.Series([2.0])}
    with caplog.at_level(logging.WARNING, logger="autopath.pulling.Diagnostics"):
        got = _get_friction_profile_for_path("path-0", prof, None)
    assert got.iloc[0] == 2.0
    assert any("speed=0.02" in r.getMessage() and "path-0" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="autopath.pulling.Diagnostics"):
        _get_friction_profile_for_path("path-0_v0.01", prof, None)
    assert not caplog.records


def test_mode_change_logs_dropped_medoid_entries(tmp_path, caplog):
    feat = _make_df().drop(columns=["true_path"])
    DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None).fit_transform(feat, n_paths=2)
    g = feat[feat.speed == feat.speed.iloc[0]]
    with caplog.at_level(logging.INFO, logger="autopath"):
        DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None).fit_transform(
            g, n_paths=2, cluster_across_speeds=False)
    assert any("dropped stored medoid entries ['all']" in r.getMessage() for r in caplog.records)
