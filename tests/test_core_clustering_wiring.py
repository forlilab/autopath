import inspect
import autopath.autopath_core as core


def test_core_imports_clustering():
    src = inspect.getsource(core)
    assert "from autopath.clustering import" in src or "Clustering(" in src


def test_core_no_longer_calls_extract_milestones():
    src = inspect.getsource(core)
    assert "extract_milestones(" not in src


def test_milestone_filter_matches_xml_not_pdb():
    """Milestone filtering no longer looks for '_relax.pdb'."""
    src = inspect.getsource(core)
    assert "_relax.pdb" not in src


def test_milestone_glob_excludes_relax_outputs():
    """The milestone glob skips RelaxMD's _relax*.xml outputs."""
    src = inspect.getsource(core)
    assert "milestone_*_frame_*.xml" in src
    assert "'_relax' not in os.path.basename(m)" in src
    assert "_relax_state.xml'" not in src


def test_write_states_call_sites_pass_ligand_sel():
    """Both write_states call sites pass ligand_sel for re-imaging."""
    src = inspect.getsource(core)
    assert src.count("ligand_sel=ligand_sel") == 2
