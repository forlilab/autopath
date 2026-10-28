"""Clustering pools speeds by default: one path label per pathway, shared across speeds."""
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
