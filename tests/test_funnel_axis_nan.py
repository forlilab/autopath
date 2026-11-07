import numpy as np
import openmm as mm
import pytest
from openmm import unit

from autopath.metadynamics.Funnel import create_funnel_force_from_trajectory_analysis

AXIS = np.array([-0.549, 0.834, 0.048])
AXIS = AXIS / np.linalg.norm(AXIS)


def _params():
    A = unit.angstrom
    return {"host_index": [0], "guest_index": [1],
            "k_xy": 10.0 * unit.kilocalorie_per_mole / A**2,
            "z_cc": 14.16 * A, "z_max": 26.5 * A, "alpha": 45.0 * unit.degrees,
            "R_cylinder": 4.37 * A, "unbinding_axis": AXIS}


def _context(expression=None):
    s = mm.System()
    s.addParticle(100.0)
    s.addParticle(100.0)
    f = create_funnel_force_from_trajectory_analysis(_params())
    if expression is not None:
        f.setEnergyFunction(expression)
    s.addForce(f)
    return mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))


def _state(ctx, guest_nm):
    ctx.setPositions(np.array([[0.0, 0.0, 0.0], guest_nm]))
    st = ctx.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(st.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))


@pytest.mark.parametrize("z_ang", [2.0, 15.0, 26.0, 28.0])
def test_force_is_finite_exactly_on_the_axis(z_ang):
    """Energy and force stay finite on the axis, where d(r_xy)/dr diverges."""
    E, F = _state(_context(), AXIS * z_ang / 10.0)
    assert np.isfinite(E) and np.isfinite(F).all()


def test_off_axis_energy_and_force_are_unchanged():
    rng = np.random.default_rng(0)
    legacy = (
        "U_funnel + U_cylinder + U_axial + U_lower;"
        "U_funnel = step(z_cc - r_z)*step(r_xy - R_funnel)*Wall_funnel;"
        "U_cylinder = step(r_z - z_cc)*step(r_xy - R_cylinder)*Wall_cylinder;"
        "U_axial = step(r_z - z_max)*0.5*k_z*(r_z - z_max)^2;"
        "U_lower = step(z_low - r_z)*0.5*k_z*(z_low - r_z)^2;"
        "Wall_funnel = 0.5 * k_xy * (r_xy - R_funnel)^2;"
        "Wall_cylinder = 0.5 * k_xy * (r_xy - R_cylinder)^2;"
        "R_funnel = max(z_cc-r_z, 0)*tan(alpha) + R_cylinder;"
        "r_xy = sqrt(max(0, r2 - r_z*r_z));"
        "r_z = nx*(x2-x1) + ny*(y2-y1) + nz*(z2-z1);"
        "r2 = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2;")
    new_ctx, old_ctx = _context(), _context(legacy)
    for _ in range(200):
        g = rng.normal(0, 1.2, 3)
        e_new, f_new = _state(new_ctx, g)
        e_old, f_old = _state(old_ctx, g)
        assert e_new == pytest.approx(e_old, rel=1e-9, abs=1e-6)
        np.testing.assert_allclose(f_new, f_old, rtol=1e-7, atol=1e-5)
