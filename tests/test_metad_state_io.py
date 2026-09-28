import inspect
from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD


def test_run_accepts_state_xml_file():
    """The input-resolution chain lives in run(), not __init__: __init__ spans
    lines 94-141 and never builds a Simulation, so it has no Context to set
    positions on."""
    assert "state_xml_file" in inspect.signature(MetadynamicsMD.run).parameters


def test_no_input_still_raises():
    src = inspect.getsource(MetadynamicsMD.run)
    assert "state_xml_file" in src and "raise ValueError" in src


def test_state_path_sets_box_positions_velocities_individually():
    src = inspect.getsource(MetadynamicsMD.run)
    assert "setPeriodicBoxVectors" in src
    assert "setVelocities" in src
    assert "loadState" not in src
