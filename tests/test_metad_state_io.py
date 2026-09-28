import inspect
from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD


def test_init_accepts_state_xml_file():
    assert "state_xml_file" in inspect.signature(MetadynamicsMD.__init__).parameters


def test_no_input_still_raises():
    src = inspect.getsource(MetadynamicsMD.__init__)
    assert "state_xml_file" in src and "raise ValueError" in src


def test_state_path_sets_box_positions_velocities_individually():
    src = inspect.getsource(MetadynamicsMD.__init__)
    assert "setPeriodicBoxVectors" in src
    assert "setVelocities" in src
