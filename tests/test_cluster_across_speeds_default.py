"""Clustering pools speeds by default: one path label per pathway, shared across speeds."""
from autopath.config import Config
from autopath.pulling.PathModel import DTWPathModel

from .test_pathmodel_cross_speed_scaling import _make_df


def test_path_model_default_pools_speeds(tmp_path):
    feat = _make_df().drop(columns=["true_path"])
    model = DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path), n_geom_pcs=None)
    mapping = model.fit_transform(feat, n_paths=2)
    assert not any("_v" in p for p in mapping.values()), mapping


def test_config_default_pools_speeds():
    assert Config().cluster_across_speeds is True
