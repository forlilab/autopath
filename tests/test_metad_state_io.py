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


import os
import pytest
from autopath.metadynamics.CV import path_rmsd_cv, rmsd_states_cv


def test_path_rmsd_cv_rejects_state_xml_milestones():
    """I2: _build_reference_dict reads milestones with PDBFile, which fails
    with an opaque IndexError on a serialized State."""
    with pytest.raises(ValueError, match=r"\.pdb"):
        path_rmsd_cv(topology=None,
                     milestones=["milestone_1_frame_0.xml"],
                     grid_min=0.0, grid_max=1.0)


def test_path_rmsd_cv_still_accepts_pdb_milestones():
    spec = path_rmsd_cv(topology=None,
                        milestones=["milestone_2_frame_9.pdb",
                                    "milestone_1_frame_0.pdb"],
                        grid_min=0.0, grid_max=1.0)
    assert spec.name == "path_rmsd"


def test_rmsd_states_cv_ignores_relax_pdbs(tmp_path):
    """I3: RelaxMD writes <milestone>_relax.pdb next to the milestones; the
    guard must not accept those as the path."""
    from openmm import app
    (tmp_path / "milestone_1_frame_0_relax.pdb").write_text("")
    spec = rmsd_states_cv(topology=None, grid_min=0.0, grid_max=1.0,
                          milestone_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="milestone_.*_frame_"):
        spec._factory(None, 0, app.Topology())
