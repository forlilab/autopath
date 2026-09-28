import inspect
import autopath.autopath_core as core


def test_core_imports_clustering():
    src = inspect.getsource(core)
    assert "from autopath.Clustering import" in src or "Clustering(" in src


def test_core_no_longer_calls_extract_milestones():
    src = inspect.getsource(core)
    assert "extract_milestones(" not in src


def test_milestone_filter_matches_xml_not_pdb():
    """The stale '_relax.pdb' filter would drop every state-based milestone."""
    src = inspect.getsource(core)
    assert "_relax.pdb" not in src
