import numpy as np
import openmm as mm
import pytest
from openmm import unit

from autopath.metadynamics.Funnel import (create_funnel_force_from_trajectory_analysis,
                                          funnel_lower_wall_ang, funnel_z_grid_min_ang)

A = unit.angstrom
AXIS = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
K_KJ_PER_A2 = 10.0 * 4.184


def _params(**kw):
    p = {"host_index": [0], "guest_index": [1],
         "k_xy": 10.0 * unit.kilocalorie_per_mole / A**2,
         "z_cc": 14.0 * A, "z_max": 26.0 * A, "alpha": 45.0 * unit.degrees,
         "R_cylinder": 4.0 * A, "unbinding_axis": AXIS,
         "axial_distances": np.r_[np.full(100, 2.0), np.linspace(2.0, 20.0, 900)]}
    p.update(kw)
    return p


def _energy(params, z_ang, r_xy_ang=0.0):
    s = mm.System()
    s.addParticle(100.0)
    s.addParticle(100.0)
    s.addForce(create_funnel_force_from_trajectory_analysis(params))
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    perp = np.cross(AXIS, [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    guest = (z_ang * AXIS + r_xy_ang * perp) / 10.0
    ctx.setPositions(np.array([[0.0, 0.0, 0.0], guest]))
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_lower_wall_sits_below_the_smd_site_and_the_grid_below_it():
    p = _params()
    assert funnel_lower_wall_ang(p) == pytest.approx(2.0 - 6.0)
    assert funnel_z_grid_min_ang(p) == pytest.approx(2.0 - 6.0 - 3.0)
    assert funnel_lower_wall_ang(_params(z_low=-1.5 * A)) == pytest.approx(-1.5)


def test_guest_is_free_between_the_wall_and_the_site():
    assert _energy(_params(), -3.0) == pytest.approx(0.0, abs=1e-6)
    assert _energy(_params(), -1.0, r_xy_ang=5.0) == pytest.approx(0.0, abs=1e-6)


def test_lower_wall_is_harmonic_on_z():
    z_low = funnel_lower_wall_ang(_params())
    assert _energy(_params(), z_low - 1.0) == pytest.approx(0.5 * K_KJ_PER_A2 * 1.0, rel=1e-4)


def test_below_the_wall_only_the_wall_acts_inside_the_cylinder():
    """Far below the host the restraint is the lower wall, with no z_max-like mirror."""
    p = _params()
    z = -30.0                                     # past the mirror of z_max
    expected = 0.5 * K_KJ_PER_A2 * (funnel_lower_wall_ang(p) - z) ** 2
    assert _energy(p, z, r_xy_ang=3.0) == pytest.approx(expected, rel=1e-4)


def test_cone_is_unchanged_above_the_wall():
    """Between the wall and z_cc the cone radius is (z_cc - |z|) tan(alpha) + R_cylinder."""
    p = _params()
    for z, r_extra in [(5.0, 1.0), (-2.0, 1.5)]:
        r = (14.0 - abs(z)) + 4.0 + r_extra
        assert _energy(p, z, r_xy_ang=r) == pytest.approx(0.5 * K_KJ_PER_A2 * r_extra ** 2, rel=1e-4)


def test_upper_side_is_unchanged():
    p = _params()
    assert _energy(p, 20.0, r_xy_ang=6.0) == pytest.approx(0.5 * K_KJ_PER_A2 * 2.0 ** 2, rel=1e-4)
    assert _energy(p, 27.0) == pytest.approx(0.5 * K_KJ_PER_A2 * 1.0, rel=1e-4)
