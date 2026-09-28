import inspect
import autopath.utils as U


def test_superseded_functions_are_still_importable():
    for name in ("compute_distance_features", "extract_milestones",
                 "match_cluster_centroids_unique", "_plot_pca_milestones"):
        assert callable(getattr(U, name)), name


def test_deprecated_banner_present_and_functions_are_below_it():
    src = inspect.getsource(U)
    assert "### DEPRECATED ###" in src
    banner = src.index("### DEPRECATED ###")
    for name in ("def extract_milestones", "def compute_distance_features",
                 "def match_cluster_centroids_unique", "def _plot_pca_milestones"):
        assert src.index(name) > banner, name
