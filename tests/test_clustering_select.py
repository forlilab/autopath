import numpy as np, pytest, MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD
from autopath.Clustering import Clustering, PairwiseDistances


@pytest.fixture
def c():
    u = mda.Universe(PSF, DCD)
    return Clustering(PairwiseDistances("resid 1-5 and name CA",
                                        "resid 20-30 and name CA")).fit(u)


def test_single_feature_featurizer_does_not_explode():
    """CoordinateRMSD returns one feature; the default n_components=2 would
    make PCA raise. fit() must clamp instead."""
    from autopath.Clustering import CoordinateRMSD
    u = mda.Universe(PSF, DCD)
    c = Clustering(CoordinateRMSD("name CA")).fit(u)
    assert c.X_reduced.shape == (len(u.trajectory), 1)


def test_select_before_fit_raises():
    """_require_fit guard; select() lives in this task so the test does too."""
    c2 = Clustering(PairwiseDistances("a", "b"))
    with pytest.raises(RuntimeError, match="fit"):
        c2.select(3, strategy="centroid")


def test_unknown_strategy_raises(c):
    """No default: a typo must fail loudly, not silently seed the wrong way."""
    with pytest.raises(ValueError, match="strategy"):
        c.select(3, strategy="centroids")


def test_strategy_is_required(c):
    with pytest.raises(TypeError):
        c.select(3)


def test_time_spaced_is_evenly_spaced(c):
    f = c.select(5, strategy="time_spaced")
    assert len(f) == 5
    gaps = np.diff(f)
    assert gaps.max() - gaps.min() <= 1          # even to within rounding


def test_centroid_and_time_spaced_differ(c):
    a = c.select(5, strategy="centroid")
    b = c.select(5, strategy="time_spaced")
    assert not np.array_equal(a, b)


def test_no_duplicate_frames_either_strategy(c):
    for s in ("centroid", "time_spaced"):
        f = c.select(5, strategy=s)
        assert len(set(f.tolist())) == len(f), s


def test_min_frame_separation_is_honoured(c):
    f = c.select(4, strategy="centroid", min_frame_separation=10)
    assert np.diff(np.sort(f)).min() >= 10


def test_n_exceeding_frames_raises(c):
    with pytest.raises(ValueError, match="frames"):
        c.select(10_000, strategy="time_spaced")


def test_labels_reset_when_switching_to_time_spaced(c):
    """Stale clustering state would misrepresent a time_spaced selection."""
    c.select(4, strategy="centroid")
    assert c.labels_ is not None
    c.select(4, strategy="time_spaced")
    assert c.labels_ is None and c.centers_ is None


def test_time_spaced_honours_min_frame_separation(c):
    """The signature accepts it for both strategies; ignoring it silently
    would be a contract violation."""
    with pytest.raises(ValueError, match="min_frame_separation"):
        c.select(50, strategy="time_spaced", min_frame_separation=10)


def test_degenerate_pc1_raises_and_names_the_alternative():
    """A converged trajectory has no PC1 spread; RegularSpace then returns one
    centre regardless of n, which would silently under-deliver frames."""
    import numpy as _np
    from autopath.Clustering import Clustering as _C, Featurizer as _F

    class _Flat(_F):
        def compute(self, u):
            return _np.ones((60, 4))

    cc = _C(_Flat()).fit(mda.Universe(PSF, DCD))
    with pytest.raises(ValueError, match="time_spaced"):
        cc.select(5, strategy="centroid")


def test_centroid_uses_pc1_not_full_matrix(c, monkeypatch):
    """Review Focus #1: selection runs on the 1-D PC1 column.

    Pins the argument actually handed to match_cluster_centroids_unique;
    asserting on centers_ alone passes even if the full matrix is passed.
    """
    import autopath.utils as U
    real = U.match_cluster_centroids_unique
    seen = {}

    def spy(X, centers, **kw):
        seen["X"] = np.asarray(X)
        return real(X, centers, **kw)

    monkeypatch.setattr(U, "match_cluster_centroids_unique", spy)
    f = c.select(4, strategy="centroid")
    assert seen["X"].shape == (c.X_reduced.shape[0], 1)
    np.testing.assert_allclose(seen["X"][:, 0], c.X_reduced[:, 0])
    assert c.centers_.shape[1] == 1
    assert len(f) <= 4


def test_centroid_frames_are_returned_in_pc1_order(c):
    """B2: centroids are sorted bound -> unbound along PC1, and select() must
    preserve that order; re-sorting by frame index scrambles the path CV."""
    f = c.select(5, strategy="centroid")
    pc1 = c.X_reduced[f, 0]
    assert np.all(np.diff(pc1) > 0), pc1


def test_pc1_order_differs_from_frame_order():
    """The regression only bites when PC1 is non-monotone in frame index,
    which is the normal case for pooled medoid trajectories."""
    from autopath.Clustering import Clustering as _C, Featurizer as _F

    rng = np.random.default_rng(0)
    vals = rng.permutation(60).astype(float)

    class _Perm(_F):
        def compute(self, u):
            return np.column_stack([vals, 0.01 * np.arange(60.0)])

    cc = _C(_Perm(), scale=False).fit(mda.Universe(PSF, DCD))
    f = cc.select(6, strategy="centroid")
    assert not np.array_equal(f, np.sort(f)), "PC1 order collapsed to frame order"
    assert np.all(np.diff(cc.X_reduced[f, 0]) > 0)


def test_stride_maps_selected_indices_back_to_trajectory_frames():
    """I4: select() indexes the strided feature matrix; the writers index the
    trajectory. Without the *stride mapping every written frame is wrong."""
    u = mda.Universe(PSF, DCD)
    lig, poc = "resid 1-5 and name CA", "resid 20-30 and name CA"
    c1 = Clustering(PairwiseDistances(lig, poc)).fit(u)
    c3 = Clustering(PairwiseDistances(lig, poc, stride=3)).fit(u)

    for strategy in ("time_spaced", "centroid"):
        f = c3.select(4, strategy=strategy)
        assert all(int(x) % 3 == 0 for x in f), (strategy, f)
        assert f.max() < len(u.trajectory)
        rows = (f // 3)
        # feature row r must describe trajectory frame 3*r
        for r, fr in zip(rows, f):
            np.testing.assert_allclose(c3.X[int(r)], c1.X[int(fr)],
                                       rtol=0, atol=1e-8)


def test_plot_after_time_spaced_has_no_labels(c, tmp_path):
    """M2: select() clears labels_; plot() must still render."""
    import os
    f = c.select(4, strategy="time_spaced")
    assert c.labels_ is None
    png = c.plot(str(tmp_path), "ms", frames=f)
    assert os.path.exists(png)
