import inspect
from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD


def test_run_accepts_state_xml_file():
    """run() accepts a state_xml_file input."""
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
    """path_rmsd_cv rejects non-PDB milestones with a clear ValueError."""
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
    """RelaxMD's *_relax.pdb outputs are not accepted as milestones."""
    from openmm import app
    (tmp_path / "milestone_1_frame_0_relax.pdb").write_text("")
    spec = rmsd_states_cv(topology=None, grid_min=0.0, grid_max=1.0,
                          milestone_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="milestone_.*_frame_"):
        spec._factory(None, 0, app.Topology())
