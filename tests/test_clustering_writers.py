import os, numpy as np, pytest
import openmm as mm
from openmm import unit as ommunit
from openmm import app
from autopath.Clustering import read_positions


def _tiny_system(n=6):
    system = mm.System()
    for _ in range(n):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(3, 0, 0) * ommunit.nanometer,
        mm.Vec3(0, 3, 0) * ommunit.nanometer,
        mm.Vec3(0, 0, 3) * ommunit.nanometer)
    return system


def test_written_state_roundtrips_positions_and_box(tmp_path):
    """Review Focus #3: box vectors must survive, else the restart silently
    adopts the System default box."""
    system = _tiny_system()
    integ = mm.LangevinMiddleIntegrator(300 * ommunit.kelvin,
                                        1 / ommunit.picosecond,
                                        0.002 * ommunit.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    pos = np.arange(18, dtype=float).reshape(6, 3) * 0.1
    box = [mm.Vec3(4.1, 0, 0), mm.Vec3(0, 4.2, 0), mm.Vec3(0, 0, 4.3)]
    ctx.setPeriodicBoxVectors(*[b * ommunit.nanometer for b in box])
    ctx.setPositions(pos * ommunit.nanometer)
    ctx.setVelocitiesToTemperature(300 * ommunit.kelvin)
    state = ctx.getState(getPositions=True, getVelocities=True)
    p = tmp_path / "s.xml"
    p.write_text(mm.XmlSerializer.serialize(state))

    back = mm.XmlSerializer.deserialize(p.read_text())
    np.testing.assert_allclose(
        back.getPositions(asNumpy=True).value_in_unit(ommunit.nanometer), pos,
        rtol=0, atol=1e-6)
    bv = back.getPeriodicBoxVectors()
    assert bv[0][0].value_in_unit(ommunit.nanometer) == pytest.approx(4.1)
    assert bv[2][2].value_in_unit(ommunit.nanometer) == pytest.approx(4.3)
    v = back.getVelocities(asNumpy=True).value_in_unit(
        ommunit.nanometer / ommunit.picosecond)
    assert np.abs(v).sum() > 0                      # velocities present


def test_read_positions_accepts_xml(tmp_path):
    system = _tiny_system()
    integ = mm.VerletIntegrator(0.001 * ommunit.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    pos = np.arange(18, dtype=float).reshape(6, 3) * 0.1
    ctx.setPositions(pos * ommunit.nanometer)
    p = tmp_path / "s.xml"
    p.write_text(mm.XmlSerializer.serialize(ctx.getState(getPositions=True)))
    np.testing.assert_allclose(read_positions(str(p)), pos, rtol=0, atol=1e-6)


def test_read_positions_rejects_unknown_extension(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("nope")
    with pytest.raises(ValueError, match="extension"):
        read_positions(str(p))
