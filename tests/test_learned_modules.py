import numpy as np
import pandas as pd
import pytest

from autopath.metadynamics.learned.modules import binned_similarity, choose_resolution, fit_modules
from autopath.metadynamics.learned.select import select


def _blocks(n_pull=30, n_frame=40, sizes=(4, 3, 3), seed=0):
    """Pulls along r 0.8-2.0; each block shares a latent within every r bin; all features carry an r trend."""
    rng = np.random.default_rng(seed)
    rows, V = [], []
    for v in (0.01, 0.02):
        for p in range(n_pull):
            r = np.linspace(0.8, 2.0, n_frame)
            lat = rng.normal(size=(n_frame, len(sizes)))
            cols = [5 * r + lat[:, b] + 0.3 * rng.normal(size=n_frame) for b, s in enumerate(sizes) for _ in range(s)]
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
