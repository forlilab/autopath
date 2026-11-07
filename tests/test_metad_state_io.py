import numpy as np
import openmm as mm
import openmm.unit as u
import pytest

from autopath.metadynamics.CV import path_rmsd_cv, rmsd_states_cv
from autopath.metadynamics.MetadynamicsMD import MetadynamicsMD

from ._metad_toy import toy, x_spec


def _md(tmp_path, top):
    return MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[0],
                          platform="Reference", out_dir=str(tmp_path))


def test_run_starts_from_the_state_xml_not_the_pdb(tmp_path):
    """Positions come from state_xml_file even when the PDB holds a different frame."""
    top, s, pdb = toy(tmp_path)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[1.2, 1.5, 1.5]] * u.nanometer)
    ctx.setVelocitiesToTemperature(300 * u.kelvin)
    xml = tmp_path / "start.xml"
    xml.write_text(mm.XmlSerializer.serialize(ctx.getState(getPositions=True, getVelocities=True)))
    del ctx

    _md(tmp_path, top).run(pdb_file=pdb, state_xml_file=str(xml), system=s, run_id="w",
                           cv_specs=[x_spec()], mMD_time=0.004, bias_factor=10, hill_height=1.0,
                           biasFrequency=2, bias_dir=str(tmp_path / "b"))
    colvar = np.load(tmp_path / "COLVAR_w.npy")[:, 0]
    assert colvar[0] == pytest.approx(1.2, abs=1e-4)


def test_run_without_a_starting_structure_raises(tmp_path):
    top, s, _ = toy(tmp_path)
    with pytest.raises(ValueError, match="must be provided"):
        _md(tmp_path, top).run(system=s, run_id="w", cv_specs=[x_spec()], mMD_time=0.004,
                               bias_factor=10, hill_height=1.0, biasFrequency=2,
                               bias_dir=str(tmp_path / "b"))


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
