"""Cross-speed clustering must standardize features WITHIN each speed group.

Speed-dependent features (work, lag, force) carry a large systematic offset
between pulling speeds -- on HSP90, `work` differs ~11x more between speeds
(median 272 kJ/mol) than between pathways (median 23 kJ/mol). With a single
global scaler, k-medoids partitions by pulling speed instead of by unbinding
pathway. Standardizing per speed removes that offset while preserving the
within-speed pathway signal.
"""
import numpy as np
import pandas as pd
import pytest

from autopath.pulling.PathModel import DTWPathModel


def _make_df(n_steps=40, n_rep=3, seed=0):
    """Synthetic features where speed and pathway signals are separable.

    - ``work``/``lag``: speed-dependent (huge offset between speeds, no path info)
    - ``geom_exit_1``: pathway-dependent, consistent ACROSS speeds (the truth)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for speed, w_off, lag_off in [(0.001, 100.0, 0.02), (0.01, 1000.0, 0.20)]:
        for path in (0, 1):
            for rep in range(n_rep):
                name = f"sMD_replica-{path}{rep}_v{speed}_forward"
                step = np.arange(n_steps)
                rows.append(pd.DataFrame({
                    "trajname": name,
                    "step": step,
                    "speed": speed,
                    "r_coord": 1.0 + 0.01 * step,
                    # speed-dependent, pathway-blind
                    "work": w_off + 0.5 * step + rng.normal(0, 0.5, n_steps),
                    "lag": lag_off + rng.normal(0, 0.001, n_steps),
                    # pathway-dependent, speed-blind  <- the real signal
                    "geom_exit_1": (1.0 if path == 0 else -1.0)
                                   + rng.normal(0, 0.05, n_steps),
                    "true_path": path,
                }))
    return pd.concat(rows, ignore_index=True)


def _truth(df):
    return df.groupby("trajname")["true_path"].first().to_dict()


def _speed_of(df):
    return df.groupby("trajname")["speed"].first().to_dict()


def _purity(mapping, keyed):
    """Fraction of clusters that are pure w.r.t. `keyed` (dict traj->label)."""
    inv = {}
    for traj, cid in mapping.items():
        inv.setdefault(cid, []).append(keyed[traj])
    return np.mean([len(set(v)) == 1 for v in inv.values()])


def test_cross_speed_clusters_by_pathway_not_speed(tmp_path):
    df = _make_df()
    feat = df.drop(columns=["true_path"])
    model = DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path),
                         n_geom_pcs=None)  # isolate scaling from PCA
    mapping = model.fit_transform(feat, n_paths=2, cluster_across_speeds=True)

    truth, speeds = _truth(df), _speed_of(df)
    assert len(mapping) == df.trajname.nunique()

    # Clusters must be pure in PATHWAY, not in SPEED.
    assert _purity(mapping, truth) == 1.0, (
        "clusters are not pathway-pure -> features were standardized globally, "
        "so the speed offset in work/lag dominated the DTW metric"
    )
    # Each cluster should contain BOTH speeds (i.e. it did not split by speed).
    inv = {}
    for traj, cid in mapping.items():
        inv.setdefault(cid, set()).add(speeds[traj])
    assert all(len(v) == 2 for v in inv.values()), \
        f"clusters split by pulling speed: {inv}"


def test_per_speed_clustering_unchanged(tmp_path):
    """Default path (cluster_across_speeds=False) keeps speed-suffixed labels."""
    df = _make_df()
    feat = df.drop(columns=["true_path"])
    model = DTWPathModel(seed=42, do_plots=False, outdir=str(tmp_path),
                         n_geom_pcs=None)
    mapping = model.fit_transform(feat, n_paths=2, cluster_across_speeds=False)
    assert len(mapping) == df.trajname.nunique()
    assert all("_v" in p for p in mapping.values()), mapping
    # within a single speed the pathway signal must still be recovered
    truth = _truth(df)
    for sp in ("0.001", "0.01"):
        sub = {t: c for t, c in mapping.items() if f"_v{sp}_" in t}
        assert _purity(sub, truth) == 1.0, f"speed {sp} not pathway-pure: {sub}"
