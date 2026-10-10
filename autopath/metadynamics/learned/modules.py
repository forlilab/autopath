"""MoSAIC-style feature modules (Leiden CPM) on an r-controlled similarity; resolution by pull-grouped CV."""
import threading

import numpy as np
import pandas as pd

BIN_NM, MIN_FRAMES = 0.05, 50
RES_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# leidenalg seeds a process-global igraph RNG, then optimises in a separate call
_LEIDEN_LOCK = threading.Lock()


def binned_similarity(V, r, bin_nm=BIN_NM, min_frames=MIN_FRAMES, idx=None):
    """|corr| of features after per-r-bin z-scoring: frame-weighted mean of the bin correlation matrices.
    Bins are anchored on the global r.min(); a subset idx uses threshold max(10, min_frames*n_sub/n_all)."""
    Z = np.nan_to_num(np.asarray(V, float))
    r = np.asarray(r, float)
    r0, n_all = r.min(), len(r)
    if idx is not None:
        idx = np.asarray(idx)
        Z, r = Z[idx], r[idx]
    thr = max(10, min_frames * len(r) / n_all) if len(r) < n_all else min_frames
    bins = np.floor((r - r0) / bin_nm).astype(int)
    acc, w = np.zeros((Z.shape[1],) * 2), 0
    for b in np.unique(bins):
        m = bins == b
        if m.sum() < thr:
            continue
        X = Z[m]
        s = X.std(0)
        good = s > 1e-9
        X = (X - X.mean(0)) / np.where(good, s, 1)
        C = X.T @ X / len(X)
        C[~good, :] = 0
        C[:, ~good] = 0
        acc += m.sum() * C
        w += m.sum()
    if w == 0:
        raise ValueError(f"no r bin ({bin_nm} nm) has >= {thr:.0f} frames; cannot build the r-controlled similarity")
    S = np.clip(np.nan_to_num(np.abs(acc / w)), 0.0, 1.0)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S


def fit_modules(S, resolution, seed=0):
    """Leiden/CPM module label per feature; mirrors mosaic.Clustering(mode='CPM') 0.5.0 (weighted full graph,
    zeros and diagonal dropped, n_iterations=-1). Labels are partition indices (mosaic re-sorts them)."""
    import igraph as ig
    import leidenalg as la
    mat = np.array(S, dtype=float)
    mat[mat == 0] = np.nan
    mat[np.diag_indices_from(mat)] = np.nan
    mat[np.isnan(mat)] = 0
    graph = ig.Graph.Weighted_Adjacency(list(mat.astype(np.float64)), loops=False)
    with _LEIDEN_LOCK:
        part = la.find_partition(graph, la.CPMVertexPartition, n_iterations=-1,
                                 resolution_parameter=float(resolution), weights=graph.es["weight"], seed=int(seed))
    return np.asarray(part.membership, int)


def module_score(S, labels):
    """mosaic Clustering.score: silhouette of the similarity rows under labels; -1 if 1 or n modules."""
    from sklearn.metrics import silhouette_score
    n = len(np.unique(labels))
    if n in (1, len(labels)):
        return -1.0
    return float(silhouette_score(S, labels=labels))


def _cv_fold_score(St, Se, resolution, seed):
    lab = fit_modules(St, resolution, seed)
    return module_score(Se, lab), int((np.bincount(lab) > 1).sum())


def pull_folds(pull, speed, n_folds=5, seed=0):
    """Fold per frame: pulls shuffled within each speed, dealt round-robin across folds."""
    rng = np.random.default_rng(seed)
    pull, speed = np.asarray(pull), np.asarray(speed)
    fold_of, k = {}, 0
    for sp in np.unique(speed):
        ps = np.unique(pull[speed == sp])
        rng.shuffle(ps)
        for p in ps:
            fold_of[p] = k % n_folds
            k += 1
    return np.array([fold_of[p] for p in pull])


def choose_resolution(V, r, pull, speed, grid=RES_GRID, n_folds=5, seeds=(0, 1, 2), n_jobs=-1, backend="threading"):
    """Resolution maximising the held-out silhouette of training-fold modules (pull-grouped folds
    stratified by speed; r-binned similarity built separately on train and held-out pulls).
    Returns (best, table) with table columns resolution, mean, sd (across folds), n_modules.
    Threads by default: process workers would re-import autopath (slow, noisy)."""
    from joblib import Parallel, delayed
    fold = pull_folds(pull, speed, n_folds)
    mats = [(binned_similarity(V, r, idx=np.flatnonzero(fold != f)),
             binned_similarity(V, r, idx=np.flatnonzero(fold == f))) for f in range(n_folds)]
    jobs = [(res, s, f) for res in grid for s in seeds for f in range(n_folds)]
    out = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_cv_fold_score)(*mats[f], res, s) for res, s, f in jobs)
    df = pd.DataFrame([(res, s, f, sc, nm) for (res, s, f), (sc, nm) in zip(jobs, out)],
                      columns=["resolution", "seed", "fold", "score", "n_modules"])
    per_fold = df.groupby(["resolution", "fold"]).agg(score=("score", "mean"), n_modules=("n_modules", "mean"))
    table = per_fold.groupby(level=0).agg(mean=("score", "mean"), sd=("score", "std"),
                                          n_modules=("n_modules", "mean")).reset_index()
    best = float(table.resolution[table["mean"].idxmax()])
    return best, table
