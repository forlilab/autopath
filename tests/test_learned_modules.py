import numpy as np
import pandas as pd
import pytest

from autopath.metadynamics.learned.modules import binned_similarity, choose_resolution, fit_modules
from autopath.metadynamics.learned.select import select


def _blocks(n_pull=30, n_frame=40, sizes=(4, 3, 3), seed=0, noise=0.3):
    """Pulls along r 0.8-2.0; each block shares a latent within every r bin; all features carry an r trend."""
    rng = np.random.default_rng(seed)
    rows, V = [], []
    for v in (0.01, 0.02):
        for p in range(n_pull):
            r = np.linspace(0.8, 2.0, n_frame)
            lat = rng.normal(size=(n_frame, len(sizes)))
            cols = [5 * r + lat[:, b] + noise * rng.normal(size=n_frame) for b, s in enumerate(sizes) for _ in range(s)]
            V.append(np.c_[tuple(cols)])
            rows.append(pd.DataFrame({"r": r, "pull": f"{v}_{p}", "speed": v}))
    return np.vstack(V), pd.concat(rows, ignore_index=True), np.repeat(np.arange(len(sizes)), sizes)


def test_binned_similarity_removes_shared_r_trend():
    rng = np.random.default_rng(0)
    r = np.repeat(np.linspace(0.8, 2.0, 25), 80)
    h = rng.normal(size=len(r))
    V = np.c_[r + 0.05 * rng.normal(size=len(r)), r + 0.05 * rng.normal(size=len(r)),
              h + 0.1 * rng.normal(size=len(r)), h + 0.1 * rng.normal(size=len(r))]
    assert abs(np.corrcoef(V[:, 0], V[:, 1])[0, 1]) > 0.95      # raw: co-vary through r
    S = binned_similarity(V, r)
    assert S[0, 1] < 0.1 and S[2, 3] > 0.9
    assert np.allclose(np.diag(S), 1) and np.allclose(S, S.T)


def test_binned_similarity_raises_without_populated_bins():
    with pytest.raises(ValueError, match="no r bin"):
        binned_similarity(np.random.default_rng(0).normal(size=(40, 3)), np.linspace(0, 4, 40))


def test_fit_modules_groups_planted_blocks():
    V, fr, truth = _blocks()
    lab = fit_modules(binned_similarity(V, fr.r.values), 0.5)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(truth, lab) == pytest.approx(1.0)


def test_select_mosaic_keeps_highest_fisher_per_module_and_f3():
    V, fr, truth = _blocks()
    n = len(V)
    lab = np.where(fr.r.values < 1.2, "bound", "vestibule").astype(object)
    rng = np.random.default_rng(1)
    shift = (lab == "bound")[:, None] * np.array([2.0, 3.0, 2.5, 2.2, 2.0, 2.4, 4.0, 2.1, 2.5, 2.3])
    V = V - 5 * fr.r.values[:, None] + shift          # drop the r trend so Fisher reflects the class shift
    V = np.c_[V, rng.normal(size=n)]                    # F3, irrelevant by Fisher
    fams = ["F7"] * 10 + ["F3"]
    out = select(V, [f"f{i}" for i in range(11)], fams, fr, lab, hidden=False)
    fis = out["fisher"]
    expect = sorted(int(np.flatnonzero(truth == b)[np.argmax(fis[:10][truth == b])]) for b in range(3))
    assert out["kept"] == expect + [10]
    assert (out["modules"][:10] >= 0).all()
    assert len({out["modules"][k] for k in expect}) == 3


def test_select_rejects_unknown_redundancy():
    V, fr, _ = _blocks(n_pull=5)
    with pytest.raises(ValueError, match="redundancy"):
        select(V, list("abcdefghij"), ["F7"] * 10, fr, np.array(["bound"] * len(V)), hidden=False,
               redundancy="x")


def test_choose_resolution_returns_grid_value():
    V, fr, _ = _blocks(n_pull=10, n_frame=100)
    best, tab = choose_resolution(V, fr.r.values, fr.pull.values, fr.speed.values, grid=(0.3, 0.6),
                                  n_folds=3, seeds=(0,), n_jobs=1)
    assert best in (0.3, 0.6)
    assert list(tab.resolution) == [0.3, 0.6] and np.isfinite(tab["mean"]).all()


def test_choose_resolution_picks_the_resolution_that_recovers_the_blocks():
    V, fr, truth = _blocks(n_pull=10, n_frame=100, sizes=(4, 4, 4), noise=1.0)   # within-block |corr| ~0.5
    S = binned_similarity(V, fr.r.values)
    from sklearn.metrics import adjusted_rand_score
    grid = (0.01, 0.3, 0.8)
    assert adjusted_rand_score(truth, fit_modules(S, 0.3)) == pytest.approx(1.0)
    assert adjusted_rand_score(truth, fit_modules(S, 0.01)) < 1 and adjusted_rand_score(truth, fit_modules(S, 0.8)) < 1
    best, tab = choose_resolution(V, fr.r.values, fr.pull.values, fr.speed.values, grid=grid, n_folds=3,
                                  seeds=(0,), n_jobs=1)
    assert best == 0.3
    assert tab["mean"].iloc[1] > tab["mean"].drop(1).max() + 0.1


def test_pull_folds_stratified_and_pulls_not_split():
    from autopath.metadynamics.learned.modules import pull_folds
    pull = np.repeat([f"{v}_{p}" for v in (1, 2, 3) for p in range(7)], 5)
    speed = np.repeat([v for v in (1, 2, 3) for _ in range(7)], 5)
    fold = pull_folds(pull, speed, n_folds=5)
    for p in np.unique(pull):
        assert len(set(fold[pull == p])) == 1
    for f in range(5):
        assert set(speed[fold == f]) == {1, 2, 3}


def test_choose_resolution_same_with_one_or_two_jobs():
    V, fr, _ = _blocks(n_pull=10, n_frame=100)
    args = (V, fr.r.values, fr.pull.values, fr.speed.values)
    kw = dict(grid=(0.3, 0.6), n_folds=3, seeds=(0, 1))
    a, ta = choose_resolution(*args, n_jobs=1, **kw)
    b, tb = choose_resolution(*args, n_jobs=2, **kw)
    assert a == b
    pd.testing.assert_frame_equal(ta, tb)


@pytest.mark.parametrize("bad", [True, False, 0, 0.0, -0.5, 1.5, "0.5", None])
def test_learn_cv_rejects_bad_mosaic_resolution(bad, tmp_path):
    from autopath.metadynamics.learned.pipeline import learn_cv
    with pytest.raises(ValueError, match="mosaic_resolution"):
        learn_cv("t", "l", [0], [1], "d", str(tmp_path), mosaic_resolution=bad)
