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


def test_centroid_uses_pc1_not_full_matrix(c):
    """Review Focus #1: selection runs on the 1-D PC1 column."""
    f = c.select(4, strategy="centroid")
    assert c.centers_.shape[1] == 1
    assert len(f) <= 4
