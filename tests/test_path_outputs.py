import json

import numpy as np
import pandas as pd
import pytest

from .smd_toy_run import analysed_dir
from .test_pathmodel_pca import _toy_feat_df


def _fit(tmp_path, **kw):
    from autopath.pulling.PathModel import DTWPathModel
    (tmp_path / "path_analysis").mkdir(exist_ok=True)
    df = _toy_feat_df(n_traj=6)
    m = DTWPathModel(do_plots=False, outdir=str(tmp_path))
    m.fit_transform(df, n_paths=kw.pop("n_paths", None), **kw)
    return tmp_path / "path_analysis", df


def test_fit_transform_saves_path_data(tmp_path):
    d, df = _fit(tmp_path)
    D = np.load(d / "dtw_distance_vGlobal.npy")
    assert D.shape == (6, 6) and np.allclose(D, D.T) and np.allclose(np.diag(D), 0)
    lab = pd.read_csv(d / "dtw_labels_vGlobal.csv")
    meta = json.loads((d / "path_model_vGlobal.json").read_text())
    assert len(lab) == 6 and lab["is_medoid"].sum() == meta["K"]
    assert set(meta["silhouette"]) == {"2", "3", "4", "5"}
    assert len(meta["explained_variance_ratio"]) == 2
    sc = pd.read_csv(d / "pca_scores_vGlobal.csv")
    assert list(sc.columns) == ["trajname", "path", "frame", "PC1", "PC2", "is_medoid"]
    # no PCA of a PCA: loadings are over the original features
    ld = pd.read_csv(d / "pca_loadings_vGlobal.csv")
    expected = [c for c in df.columns if c not in ("trajname", "time", "step", "speed", "path")]
    assert list(ld["feature"]) == expected


def test_fixed_k_has_empty_silhouette(tmp_path):
    d, _ = _fit(tmp_path, n_paths=2)
    meta = json.loads((d / "path_model_vGlobal.json").read_text())
    assert meta["K"] == 2 and meta["silhouette"] == {}


def test_convergence_fit_does_not_save_path_data(tmp_path):
    d, _ = _fit(tmp_path, persist_medoids=False)
    assert not list(d.glob("dtw_distance_*.npy"))


def test_subclass_hooks_still_called(tmp_path):
    from autopath.pulling.PathModel import DTWPathModel
    seen = []

    class Cap(DTWPathModel):
        def plot_clusters_PCA(self, feature_df=None, path_mapping_dic=None, **kw):
            seen.append(path_mapping_dic)

    (tmp_path / "path_analysis").mkdir()
    Cap(do_plots=False, outdir=str(tmp_path)).fit_transform(_toy_feat_df(n_traj=4), n_paths=2)
    assert len(seen) == 1 and len(seen[0]) == 4


def test_viz_default_is_pml():
    from autopath.pulling.PathModel import DTWPathModel
    assert DTWPathModel(do_plots=False, outdir="/tmp/_pm_viz_default").viz_output_format == "pml"


@pytest.fixture(scope="module")
def null_dir(tmp_path_factory):
    return analysed_dir(tmp_path_factory.mktemp("null"), path_model="null", n_paths=None)


def test_null_model_writes_no_path_data(null_dir):
    assert not list((null_dir / "path_analysis").glob("*.npy"))


def test_path_pmfs_single_path_matches_mixture(null_dir):
    pp = pd.read_csv(null_dir / "path_pmfs.csv")
    mp = pd.read_csv(null_dir / "mixture_pmfs.csv")
    a = pp[(pp.estimator == "cumulant") & (pp.speed > 0)].groupby(["speed", "step"]).dG.mean()
    b = mp[(mp.estimator == "cumulant") & (mp.speed > 0)].set_index(["speed", "step"]).dG
    a, b = a.align(b, join="inner")
    assert len(a) > 10
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)


def test_path_friction_single_path_matches_whole_set(null_dir):
    pf = pd.read_csv(null_dir / "path_friction.csv")
    fr = pd.read_csv(null_dir / "friction.csv")
    f_pf = pf[(pf.estimator == "force") & (pf.method == "regression")].sort_values("r_coord")
    f_fr = fr[(fr.estimator == "force") & (fr.method == "regression")].sort_values("r_coord")
    assert len(f_pf) > 10
    np.testing.assert_allclose(f_pf.Gamma.values, f_fr.Gamma.values, rtol=1e-9)
    c_pf = pf[pf.estimator == "cumulant"].sort_values(["speed", "r_coord"])
    c_fr = fr[(fr.estimator == "cumulant") & (fr.method == "derivative")].sort_values(["speed", "r_coord"])
    assert len(c_pf) == len(c_fr) > 0
    np.testing.assert_allclose(c_pf.Gamma.values, c_fr.Gamma.values, rtol=1e-9)
