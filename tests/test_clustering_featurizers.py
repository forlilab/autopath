import numpy as np, pytest, MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD
from autopath.Clustering import Featurizer, PairwiseDistances, CoordinateRMSD
from autopath.utils import compute_distance_features


@pytest.fixture
def u():
    return mda.Universe(PSF, DCD)


def test_pairwise_distances_matches_hand_computed_reference(u):
    """Independent reference: PairwiseDistances must equal distances computed
    directly from coordinates. Comparing it to compute_distance_features would
    be vacuous, because the implementation delegates to that function."""
    lig, pock = "resid 1-5 and name CA", "resid 20-30 and name CA"
    X = PairwiseDistances(lig, pock).compute(u)
    lg, pk = u.select_atoms(lig), u.select_atoms(pock)
    u.trajectory[0]
    expected = np.linalg.norm(
        lg.positions[:, None, :] - pk.positions[None, :, :], axis=-1).ravel()
    assert X.shape == (len(u.trajectory), lg.n_atoms * pk.n_atoms)
    np.testing.assert_allclose(X[0], expected, rtol=1e-6, atol=1e-6)
    assert (X >= 0).all()


def test_pairwise_distances_respects_stride(u):
    lig, pock = "resid 1-5 and name CA", "resid 20-30 and name CA"
    full = PairwiseDistances(lig, pock, stride=1).compute(u)
    strided = PairwiseDistances(lig, pock, stride=3).compute(u)
    assert strided.shape[0] == len(range(0, full.shape[0], 3))
    assert strided.shape[1] == full.shape[1]


def test_coordinate_rmsd_zero_against_own_first_frame(u):
    X = CoordinateRMSD("name CA").compute(u)
    assert X.shape == (len(u.trajectory), 1)
    assert X[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert (X >= 0).all()


def test_featurizer_is_abstract():
    with pytest.raises(TypeError):
        Featurizer()
