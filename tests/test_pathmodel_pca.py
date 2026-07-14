import numpy as np
import pandas as pd
import pytest


def _toy_feat_df(n_traj=4, n_steps=25, seed=0):
    """Trace + geom_ columns for a few trajectories with two rough clusters."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_traj):
        group = t % 2                      # two latent path groups
        for s in range(n_steps):
            r = s / n_steps
            rows.append({
                "trajname": f"t{t}", "speed": 0.01, "step": s, "time": float(s),
                "lag": 0.05 * rng.standard_normal(),
                "r_before": r + 0.01 * rng.standard_normal(),
                "geom_rog": 0.30 + 0.001 * rng.standard_normal(),      # ~constant
                "geom_npr1": 0.15 + 0.001 * rng.standard_normal(),     # ~constant
                "geom_npr2": 0.95 + 0.001 * rng.standard_normal(),     # ~constant
                "geom_exit_1": (0.6 if group else -0.6) + 0.05 * rng.standard_normal(),
                "geom_exit_2": (0.3 if group else -0.3) + 0.05 * rng.standard_normal(),
                "geom_exit_3": 0.1 * rng.standard_normal(),
            })
    return pd.DataFrame(rows)


def test_dtwpathmodel_defaults_pca_all():
    from autopath.pulling.PathModel import DTWPathModel
    m = DTWPathModel(do_plots=False, outdir="/tmp/_pm_defaults_test")
    assert m.pca_all_features is True
    assert m.n_geom_pcs == 4


def test_fit_transform_pca_all_reduces_and_clusters(tmp_path):
    from autopath.pulling.PathModel import DTWPathModel
    (tmp_path / "path_analysis").mkdir()      # SMDAnalysis makes this in real use
    df = _toy_feat_df()
    m = DTWPathModel(do_plots=False, outdir=str(tmp_path))   # pca_all_features=True
    mapping = m.fit_transform(df, n_paths=2)
    # one label per trajectory, exactly the requested number of paths
    assert set(mapping.keys()) == set(df["trajname"].unique())
    assert len({v.split("_")[0] for v in mapping.values()}) == 2


def test_fit_transform_pca_all_handles_all_geom_prefixes(tmp_path):
    """pca_all ignores the prefix — even non-geom trace columns get PCA'd."""
    from autopath.pulling.PathModel import DTWPathModel
    (tmp_path / "path_analysis").mkdir()
    df = _toy_feat_df(n_traj=3, n_steps=20)
    m = DTWPathModel(do_plots=False, outdir=str(tmp_path))
    mapping = m.fit_transform(df, n_paths=2)
    assert len(mapping) == 3
