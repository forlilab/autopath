import inspect
from autopath.relax_md import RelaxMD


def test_run_accepts_state_xml_file():
    assert "state_xml_file" in inspect.signature(RelaxMD.run).parameters


def test_save_outputs_writes_a_state_xml():
    src = inspect.getsource(RelaxMD._save_outputs)
    assert "_relax_state.xml" in src
    assert "XmlSerializer.serialize" in src


def test_run_does_not_use_loadState():
    """loadState fails on stripped systems carrying stale global parameters."""
    assert "loadState" not in inspect.getsource(RelaxMD.run)
