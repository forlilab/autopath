import numpy as np
import openmm as mm
import pytest
from openmm import unit

from autopath.metadynamics.CV import axis_cos_cv, membrane_z_cv

LZ = 7.0
SOLUTE, CENTER = [0, 1], [2]


def _context(spec, pos_nm, lz=LZ):
    s = mm.System()
    for _ in range(len(pos_nm)):
        s.addParticle(1.0)
    s.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, lz))
    cv = spec.resolve(pos_nm, len(pos_nm), None).cv
    cv.addToSystem(s)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos_nm)
    return cv, ctx


def _positions(z_solute):
    return np.array([[0.1, 0.2, z_solute - 0.05], [0.3, 0.1, z_solute + 0.05], [0.0, 0.0, 0.0]])


def _z(cv, ctx):
    return cv.getValue(ctx).value_in_unit(unit.nanometers)


@pytest.mark.parametrize("z", [-3.2, -1.3, -0.4, 0.4, 1.3, 3.2])
def test_membrane_z_is_the_signed_offset_inside_the_box(z):
    cv, ctx = _context(membrane_z_cv(SOLUTE, CENTER, half_width=LZ / 2), _positions(z))
    assert _z(cv, ctx) == pytest.approx(z, abs=1e-6)


@pytest.mark.parametrize("z, wrapped", [(4.0, -3.0), (-4.5, 2.5), (9.0, 2.0), (-12.0, 2.0)])
def test_membrane_z_takes_the_minimum_image_across_the_box(z, wrapped):
    cv, ctx = _context(membrane_z_cv(SOLUTE, CENTER, half_width=LZ / 2), _positions(z))
    assert _z(cv, ctx) == pytest.approx(wrapped, abs=1e-6)


def test_membrane_z_follows_a_box_change_without_reparametrising():
    cv, ctx = _context(membrane_z_cv(SOLUTE, CENTER, half_width=4.0), _positions(3.2))
    assert _z(cv, ctx) == pytest.approx(3.2, abs=1e-6)
    ctx.setPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 6.0))
    assert _z(cv, ctx) == pytest.approx(-2.8, abs=1e-6)


def test_membrane_z_is_clamped_to_the_grid():
    spec = membrane_z_cv(SOLUTE, CENTER, half_width=2.0)
    cv, ctx = _context(spec, _positions(3.0))
    assert _z(cv, ctx) == pytest.approx(2.0, abs=1e-6)
    ctx.setPositions(_positions(-3.0))
    assert _z(cv, ctx) == pytest.approx(-2.0, abs=1e-6)


def test_membrane_z_spec_grid_spacing_is_a_fifth_of_the_hill_width():
    spec = membrane_z_cv(SOLUTE, CENTER, half_width=3.6, hill_width=0.2)
    assert (spec.grid_min, spec.grid_max, spec.grid_points) == (-3.6, 3.6, 181)
    assert spec.cv_length_per_unit_ang == 10.0


@pytest.mark.parametrize("head, cos", [((0, 0, 1), 1.0), ((0, 0, -1), -1.0), ((1, 0, 0), 0.0),
                                       ((1, 0, 1), np.sqrt(0.5))])
def test_axis_cos_is_the_cosine_to_z(head, cos):
    pos = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) + np.array([[0, 0, 0], head]) * 0.15
    cv, ctx = _context(axis_cos_cv(0, 1), pos)
    assert cv.getValue(ctx).value_in_unit(unit.dimensionless) == pytest.approx(cos, abs=1e-6)
