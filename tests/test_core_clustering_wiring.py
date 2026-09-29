import inspect
import autopath.autopath_core as core


def test_core_imports_clustering():
    src = inspect.getsource(core)
    assert "from autopath.clustering import" in src or "Clustering(" in src


def test_core_no_longer_calls_extract_milestones():
    src = inspect.getsource(core)
    assert "extract_milestones(" not in src


def test_milestone_filter_matches_xml_not_pdb():
    """The stale '_relax.pdb' filter would drop every state-based milestone."""
    src = inspect.getsource(core)
    assert "_relax.pdb" not in src


def test_milestone_glob_excludes_relax_outputs():
    """B3: RelaxMD writes _relax_system.xml too; feeding that to read_positions
    raises AttributeError on any resumed run."""
    src = inspect.getsource(core)
    assert "milestone_*_frame_*.xml" in src
    assert "'_relax' not in os.path.basename(m)" in src
    assert "_relax_state.xml'" not in src


def test_write_states_call_sites_pass_ligand_sel():
    """I1: re-imaging was dropped in the migration; both call sites restore it."""
    src = inspect.getsource(core)
    assert src.count("ligand_sel=ligand_sel") == 2
