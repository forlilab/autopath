"""Protein-frame projection CV, spring and lateral restraint for projection pulls."""
import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from scipy.spatial.transform import Rotation

from autopath.metadynamics.Funnel import create_funnel_force_from_trajectory_analysis, frame_basis
from autopath.pulling.projection import (check_projection_params, projection_cv, projection_spring,
                                         projection_z_nm)

CPU = mm.Platform.getPlatformByName("CPU")
POCKET, LIGAND, FRAME_B, FRAME_C = [0, 1], [2], [3, 4], [5, 6]
MASSES = np.array([12.0, 14.0, 16.0, 12.0, 15.0, 13.0, 14.0])
AXIS = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
BOX_NM = 6.0


def _positions():
    p = np.array([[3.0, 3.0, 3.0], [3.1, 2.9, 3.05],      # pocket
                  [0.0, 0.0, 0.0],                        # ligand, placed below
                  [4.0, 3.2, 2.9], [4.1, 3.3, 3.0],        # frame b
                  [3.1, 4.1, 3.2], [3.0, 4.0, 3.3]])       # frame c
    com = lambda idx: np.average(p[idx], axis=0, weights=MASSES[idx])
    p[2] = com(POCKET) + 0.4 * AXIS + np.array([0.05, 0.02, -0.03])
    return p


def _params(p=None):
    p = _positions() if p is None else p
    com = lambda idx: np.average(p[idx], axis=0, weights=MASSES[idx])
    coeffs = frame_basis(com(POCKET), com(FRAME_B), com(FRAME_C)) @ AXIS
    return {"unbinding_axis": AXIS, "host_index": POCKET, "guest_index": LIGAND,
            "frame_b_index": FRAME_B, "frame_c_index": FRAME_C, "axis_coeffs": coeffs,
            "k_xy": 10.0 * u.kilocalorie_per_mole / u.angstrom**2, "z_cc": 6.0 * u.angstrom,
            "z_max": 12.0 * u.angstrom, "alpha": 45.0 * u.degrees, "R_cylinder": 2.0 * u.angstrom,
            "z_low": -2.0 * u.angstrom}


def _system():
    s = mm.System()
    s.setDefaultPeriodicBoxVectors(*(mm.Vec3(*(BOX_NM * np.eye(3)[i])) for i in range(3)))
    for m in MASSES:
        s.addParticle(float(m))
    bonds = mm.HarmonicBondForce()
    p = _positions()
    protein = POCKET + FRAME_B + FRAME_C
    for i in protein:
        for j in protein:
            if i < j:
                bonds.addBond(i, j, float(np.linalg.norm(p[i] - p[j])), 1.0e4)
    s.addForce(bonds)
    return s


def _context(system, pos):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), CPU)
    ctx.setPositions(pos)
    return ctx


def _cv_value(pos, params=None):
    s = _system()
    cv = projection_cv(_params() if params is None else params)
    cv.addToSystem(s)
    return cv.getValue(_context(s, pos)).value_in_unit(u.nanometer)


def _rotate(pos, seed=7):
    R = Rotation.random(random_state=seed).as_matrix()
    c = pos.mean(axis=0)
    return (pos - c) @ R.T + c


def _lab_z(pos):
    com = lambda idx: np.average(pos[idx], axis=0, weights=MASSES[idx])
    return float(AXIS @ (com(LIGAND) - com(POCKET)))


def test_projection_cv_equals_n_dot_dcom():
    pos = _positions()
    z = _cv_value(pos)
    assert z == pytest.approx(projection_z_nm(_params(), pos, MASSES), abs=1e-6)
    assert z == pytest.approx(_lab_z(pos), abs=1e-6)


def test_projection_cv_rotation_invariant():
    pos = _positions()
    assert _cv_value(_rotate(pos)) == pytest.approx(_cv_value(pos), abs=1e-6)


def test_projection_cv_is_not_clamped():
    pos = _positions()
    far, deep = pos.copy(), pos.copy()
    far[LIGAND] += 2.5 * AXIS                     # beyond z_max
    deep[LIGAND] -= 1.5 * AXIS                    # below z_low
    for p in (far, deep):
        assert _cv_value(p) == pytest.approx(_lab_z(p), abs=1e-6)


def test_lab_axis_is_not_rotation_invariant():
    pos = _positions()
    assert abs(_lab_z(_rotate(pos)) - _lab_z(pos)) > 0.05


def test_spring_energy():
    pos = _positions()
    z = _cv_value(pos)
    k = 1000.0
    s = _system()
    spring = projection_spring(_params(), k)
    spring.setForceGroup(1)
    spring.addToSystem(s)
    ctx = _context(s, pos)
    ctx.setParameter("r0_smd", z + 0.1)
    assert ctx.getParameter("fc_pull") == pytest.approx(k)
    e = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    assert e == pytest.approx(0.5 * k * 0.1**2, rel=1e-5)


def _funnel_energy(pos, axial_walls):
    s = _system()
    f = create_funnel_force_from_trajectory_analysis(_params(), axial_walls=axial_walls)
    f.setForceGroup(10)
    s.addForce(f)
    ctx = _context(s, pos)
    return ctx.getState(getEnergy=True, groups={10}).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)


@pytest.mark.parametrize("z_nm", [1.2 + 1.0, -0.5])
def test_lateral_force_has_no_axial_walls(z_nm):
    pos = _positions()
    com = lambda idx: np.average(pos[idx], axis=0, weights=MASSES[idx])
    pos[LIGAND] = com(POCKET) + z_nm * AXIS        # on the axis: no lateral energy
    assert _funnel_energy(pos, axial_walls=False) == pytest.approx(0.0, abs=1e-6)
    assert _funnel_energy(pos, axial_walls=True) > 1.0


def test_lateral_force_without_walls_keeps_the_cone():
    pos = _positions()
    pos[LIGAND] += 0.8 * np.cross(AXIS, [1.0, 0.0, 0.0])   # well off the axis
    assert _funnel_energy(pos, axial_walls=False) > 1.0


def test_projection_params_accepted():
    check_projection_params(_params(), LIGAND, POCKET)
    check_projection_params(_params(), [2], [1, 0])        # group order does not matter


@pytest.mark.parametrize("ligand, pocket", [([2, 3], POCKET), (LIGAND, [0]), (POCKET, LIGAND)])
def test_group_mismatch_not_supported(ligand, pocket):
    with pytest.raises(ValueError, match="^not supported:"):
        check_projection_params(_params(), ligand, pocket)


def test_lab_frame_params_not_supported():
    lab = {k: v for k, v in _params().items() if k not in ("frame_b_index", "frame_c_index", "axis_coeffs")}
    with pytest.raises(ValueError, match="^not supported:"):
        check_projection_params(lab, LIGAND, POCKET)
    with pytest.raises(ValueError, match="^not supported:"):
        projection_cv(lab)


@pytest.mark.parametrize("drop", ["frame_b_index", "frame_c_index"])
def test_partial_protein_frame_not_supported(drop):
    partial = {k: v for k, v in _params().items() if k != drop}
    with pytest.raises(ValueError, match=f"^not supported:.*{drop}"):
        check_projection_params(partial, LIGAND, POCKET)


def test_mixed_frame_params_not_supported():
    mixed = dict(_params(), orient_index=[0, 1])
    with pytest.raises(ValueError, match="^not supported:"):
        check_projection_params(mixed, LIGAND, POCKET)
