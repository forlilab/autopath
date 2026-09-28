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


# --- real execution: the signature/source assertions above cannot catch a
# --- NameError inside run(), which is how the state_xml_file branch shipped
# --- broken (initial_positions was bound only in the pdb_file branch).

import numpy as np
import openmm as mm
import pytest
from openmm import app as _app
from openmm import unit as ommunit


def _tiny_state_and_system(tmp_path, n=6):
    system = mm.System()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * ommunit.nanometer)
    for _ in range(n):
        system.addParticle(12.0)
        nb.addParticle(0.0, 0.3, 0.1)
    # The barostat refuses a non-periodic system, so the toy needs a periodic force.
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0) * ommunit.nanometer,
                                        mm.Vec3(0, 3, 0) * ommunit.nanometer,
                                        mm.Vec3(0, 0, 3) * ommunit.nanometer)
    top = _app.Topology()
    ch = top.addChain()
    for _ in range(n):
        r = top.addResidue("UNK", ch)
        top.addAtom("C", _app.Element.getBySymbol("C"), r)

    integ = mm.VerletIntegrator(0.001 * ommunit.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    pos = np.array([[0.5, 0.5, 0.5], [0.8, 0.5, 0.5], [0.5, 0.8, 0.5],
                    [1.6, 1.6, 1.6], [1.9, 1.6, 1.6], [1.6, 1.9, 1.6]])
    ctx.setPeriodicBoxVectors(mm.Vec3(3.3, 0, 0) * ommunit.nanometer,
                              mm.Vec3(0, 3.3, 0) * ommunit.nanometer,
                              mm.Vec3(0, 0, 3.3) * ommunit.nanometer)
    ctx.setPositions(pos * ommunit.nanometer)
    ctx.setVelocitiesToTemperature(300 * ommunit.kelvin)
    p = tmp_path / "milestone_1_frame_0.xml"
    p.write_text(mm.XmlSerializer.serialize(
        ctx.getState(getPositions=True, getVelocities=True)))
    del ctx, integ
    return str(p), system, top, pos


def test_run_from_state_xml_executes(tmp_path, monkeypatch):
    """Calls run(state_xml_file=...) for real; it used to raise
    UnboundLocalError on initial_positions before the restraints were added."""
    import autopath.relax_md as rmd

    state_xml, system, top, pos = _tiny_state_and_system(tmp_path)
    seen = {}
    real = rmd.add_harmonic_restraints

    def spy(system, positions, *a, **kw):
        seen["positions"] = np.asarray(
            [[c.value_in_unit(ommunit.nanometer) for c in xyz] for xyz in positions])
        return real(system, positions, *a, **kw)

    monkeypatch.setattr(rmd, "add_harmonic_restraints", spy)

    relax = rmd.RelaxMD(topology=top, ligand_atoms=[0, 1, 2],
                        pocket_atoms=[3, 4, 5], out_dir=str(tmp_path / "out"),
                        timestep=0.002, temp=300, platform="CPU")
    out_system = relax.run(system=system, state_xml_file=state_xml,
                           run_id="milestone_1_frame_0", npt_steps=10)

    # The restraint reference must be the state's coordinates, not a PDB's.
    assert "positions" in seen, "add_harmonic_restraints was never reached"
    np.testing.assert_allclose(seen["positions"], pos, rtol=0, atol=1e-4)
    assert out_system.getNumParticles() == 6
    assert (tmp_path / "out" / "milestone_1_frame_0_relax_state.xml").exists()
