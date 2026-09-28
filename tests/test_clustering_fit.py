import numpy as np, pytest, MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD
from autopath.Clustering import Clustering, PairwiseDistances, CoordinateRMSD


@pytest.fixture
def clu():
    u = mda.Universe(PSF, DCD)
    c = Clustering(PairwiseDistances("resid 1-5 and name CA",
                                     "resid 20-30 and name CA"))
    return c.fit(u), u


def test_fit_returns_self_and_sets_attributes(clu):
    c, u = clu
    assert isinstance(c, Clustering)
    assert c.X.shape[0] == len(u.trajectory)
    assert c.X_reduced.shape == (len(u.trajectory), 2)
    assert len(c.explained_variance_ratio_) == 2


def test_n_components_is_honoured():
    u = mda.Universe(PSF, DCD)
    c = Clustering(PairwiseDistances("resid 1-5 and name CA",
                                     "resid 20-30 and name CA"),
                   n_components=3).fit(u)
    assert c.X_reduced.shape[1] == 3


def test_single_feature_featurizer_does_not_explode():
    u = mda.Universe(PSF, DCD)
    c = Clustering(CoordinateRMSD("name CA")).fit(u)
    assert c.X_reduced.shape == (len(u.trajectory), 1)
