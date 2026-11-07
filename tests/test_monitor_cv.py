import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import Element, PDBFile, Topology

from autopath.metadynamics import CVSpec, MetadynamicsMD
from autopath.metadynamics.MetadynamicsMD import add_monitor


def _two_particle_system():
    s = mm.System()
    s.addParticle(12.0)
    s.addParticle(12.0)
    b = mm.HarmonicBondForce()
    b.addBond(0, 1, 0.3, 1000.0)
    s.addForce(b)
    return s


def _state(system, X):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(X)
    st = ctx.getState(getEnergy=True, getForces=True)
    return ctx, st.getPotentialEnergy()._value, st.getForces(asNumpy=True)._value


def test_monitor_adds_no_energy_or_force_and_reports_value():
    X = np.array([[0, 0, 0], [0.42, 0, 0]])
    s0 = _two_particle_system()
    _, e0, f0 = _state(s0, X)
    s1 = _two_particle_system()
    d = mm.CustomBondForce("r")
    d.addBond(0, 1, [])
    mon = add_monitor(s1, d, "dist")
    ctx, e1, f1 = _state(s1, X)
    assert e1 == pytest.approx(e0, abs=1e-9)
    np.testing.assert_allclose(f1, f0, atol=1e-9)
    assert mon.getCollectiveVariableValues(ctx)[0] == pytest.approx(0.42)
    assert mon.getForceGroup() == 31


def _toy(tmp_path):
    top = Topology()
    top.addAtom("C", Element.getBySymbol("C"), top.addResidue("UNK", top.addChain()))
    top.setPeriodicBoxVectors(np.eye(3) * 3.0 * u.nanometer)
    pdb = str(tmp_path / "one.pdb")
    with open(pdb, "w") as fh:
        PDBFile.writeFile(top, [[1.0, 1.5, 1.5]] * u.nanometer, fh)
    s = mm.System()
    s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.0))
    f = mm.CustomExternalForce("0.5*41840*(step(0.6-x)*(0.6-x)^2 + step(x-1.5)*(x-1.5)^2)"
                               " + 1000*((y-1.5)^2+(z-1.5)^2)")
    f.addParticle(0, [])
    s.addForce(f)
    return top, s, pdb


def _x_spec():
    cv = mm.CustomExternalForce("x")
    cv.addParticle(0, [])
    return CVSpec(cv=cv, grid_min=0.5, grid_max=1.6, hill_width=0.03, grid_points=111, name="x")


def _run(tmp_path, monitor_cvs):
    top, s, pdb = _toy(tmp_path)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[0],
                        platform="Reference", out_dir=str(tmp_path))
    md.run(pdb_file=pdb, system=s, run_id="w", cv_specs=[_x_spec()], mMD_time=0.02,
           bias_factor=10, hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"),
           reweight=True, rct_ustride=1, dcd_interval=10.0, monitor_cvs=monitor_cvs)
    return np.load(tmp_path / "REWEIGHT_w.npz")


def test_monitor_is_logged_in_reweight_and_tracks_the_coordinate(tmp_path):
    y = mm.CustomExternalForce("y")
    y.addParticle(0, [])
    rw = _run(tmp_path, {"y": y})
    assert len(rw["monitor_y"]) == len(rw["bias_kj"])
    np.testing.assert_allclose(rw["monitor_y"], 1.5, atol=0.15)
