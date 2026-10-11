"""Tilt cos in OpenMM equals the numpy evaluator under periodic boxes, far-out ligands included."""
import numpy as np
import openmm as mm
import openmm.unit as u
import pytest

from autopath.metadynamics.learned.features import Feature, evaluate
from autopath.metadynamics.learned.forces import feature_cv, fused_sw_forces

REF = mm.Platform.getPlatformByName("Reference")
A = 6.66
BOXES = {
    "cubic": np.eye(3) * A,
    "rhombic_dodecahedron": np.array([[A, 0, 0], [0, A, 0], [A / 2, A / 2, A * np.sqrt(2) / 2]]),
    "truncated_octahedron": np.array([[A, 0, 0], [A / 3, 2 * np.sqrt(2) * A / 3, 0],
                                      [-A / 3, np.sqrt(2) * A / 3, np.sqrt(6) * A / 3]]),
}
FEAT = Feature("tilt", "cos", "F8", groups=[[0], [1], [2, 3, 4, 5], [3, 4, 5]])


def _system(box):
    s = mm.System()
    for _ in range(6):
        s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(*[mm.Vec3(*b) for b in box])
    return s


def _positions(dz):
    p0 = np.array([1.0, 1.0, 1.0])
    return np.array([p0 + [0, 0, dz], p0 + [0.3, 0.2, dz + 0.1],
                     p0, p0 + [0.4, 0, 0], p0 + [0, 0.4, 0], p0 + [0.25, 0.1, 0.2]])


def _context(build, box):
    s = _system(box)
    if build == "feature_cv":
        force = feature_cv(FEAT, s, "f")
    else:
        forces, const = fused_sw_forces([FEAT], [1.0], [0.0], [1.0], s)
        assert const == 0.0 and len(forces) == 1
        force = forces[0]
    force.addToSystem(s)
    return mm.Context(s, mm.VerletIntegrator(0.001), REF)


def _ef(ctx, pos):
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))


@pytest.mark.parametrize("dz", [1.5, 2.2, 2.4, 3.0])
@pytest.mark.parametrize("shape", list(BOXES))
@pytest.mark.parametrize("build", ["feature_cv", "fused"])
def test_cos_matches_numpy_evaluator(shape, dz, build):
    pos = _positions(dz)
    e, f = _ef(_context(build, BOXES[shape]), pos)
    assert e == pytest.approx(evaluate([FEAT], pos[None])[0, 0], abs=1e-9)
    assert np.isfinite(f).all()


@pytest.mark.parametrize("build", ["feature_cv", "fused"])
def test_cos_force_matches_finite_difference_far_out(build):
    pos = _positions(3.0)
    ctx = _context(build, BOXES["rhombic_dodecahedron"])
    _, f = _ef(ctx, pos)
    h = 1e-5
    fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for k in range(3):
            p, m = pos.copy(), pos.copy()
            p[i, k] += h
            m[i, k] -= h
            fd[i, k] = -(_ef(ctx, p)[0] - _ef(ctx, m)[0]) / (2 * h)
    assert np.abs(f).max() > 0
    assert np.allclose(f, fd, atol=1e-6)
