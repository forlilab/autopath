import numpy as np
import openmm as mm
import pytest
from openmm import unit

from autopath.metadynamics.CV import funnel_z_cv

AXIS = np.array([0.3, -0.5, 0.8])
MASSES = [12.0, 16.0, 1.0, 35.0]          # host 0,1 ; guest 2,3
HOST, GUEST = [0, 1], [2, 3]


def _context(pos_nm):
    s = mm.System()
    for m in MASSES:
        s.addParticle(m)
    # Wide grid so the value tests never touch the clamp
    spec = funnel_z_cv(HOST, GUEST, AXIS, grid_min=-10.0, grid_max=10.0)
    cv = spec.resolve(pos_nm, len(MASSES), None).cv
    cv.addToSystem(s)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos_nm)
    return cv, ctx


def _reference_z(pos_nm):
    m = np.asarray(MASSES)
    host = np.average(pos_nm[HOST], axis=0, weights=m[HOST])
    guest = np.average(pos_nm[GUEST], axis=0, weights=m[GUEST])
    n = AXIS / np.linalg.norm(AXIS)
    return float(np.dot(guest - host, n))


def test_value_is_the_mass_weighted_projection_on_the_funnel_axis():
    rng = np.random.default_rng(1)
    pos = rng.normal(0, 1.0, (4, 3))
    cv, ctx = _context(pos)
    assert cv.getValue(ctx).value_in_unit(unit.nanometers) == pytest.approx(_reference_z(pos), abs=1e-6)


def test_moving_the_guest_along_the_axis_changes_z_by_the_same_amount():
    """Translating the guest along the axis shifts z by the same distance."""
    rng = np.random.default_rng(2)
    pos = rng.normal(0, 1.0, (4, 3))
    n = AXIS / np.linalg.norm(AXIS)
    cv, ctx = _context(pos)
    z0 = cv.getValue(ctx).value_in_unit(unit.nanometers)
    moved = pos.copy()
    moved[GUEST] += 3.0 * n
    ctx.setPositions(moved)
    assert cv.getValue(ctx).value_in_unit(unit.nanometers) - z0 == pytest.approx(3.0, abs=1e-6)


def test_spec_is_a_length_in_nm():
    spec = funnel_z_cv(HOST, GUEST, AXIS, grid_min=0.0, grid_max=2.77)
    assert spec.name == "funnel_z"
    assert spec.cv_length_per_unit_ang == 10.0
    assert (spec.grid_min, spec.grid_max) == (0.0, 2.77)


def test_round_trips_are_counted_in_fractions_of_the_grid():
    """Round-trip thresholds are grid fractions, so a rescaled CV counts the same trips."""
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis
    s = np.r_[np.full(50, 0.05), np.full(50, 0.95), np.full(50, 0.05)]
    z = s * 2.77                                   # same trajectory in nm on [0, 2.77]
    a = MetadynamicsAnalysis.__new__(MetadynamicsAnalysis)
    ref = a.analyze_colvar_transitions(s[:, None])
    got = a.analyze_colvar_transitions(z[:, None], grid_min=0.0, grid_max=2.77)
    assert got["n_round_trips"] == ref["n_round_trips"] == 1
    assert got["frac_bound"] == pytest.approx(ref["frac_bound"])


@pytest.mark.parametrize("kw, ok", [
    (dict(mMD_cv="bad"), False),
    (dict(mMD_cv="funnel_z"), False),                                   # needs the funnel
    (dict(mMD_cv="funnel_z", mMD_use_funnel_potential=True, mMD_preseed_bias=True), False),
    (dict(mMD_cv="funnel_z", mMD_use_funnel_potential=True), True),
])
def test_mmd_cv_option_is_validated(kw, ok):
    from autopath.autopath_core import AutoPath
    if ok:
        assert AutoPath(pdb_path="examples/data/3ptb.pdb", **kw).mMD_cv == "funnel_z"
    else:
        with pytest.raises(ValueError):
            AutoPath(pdb_path="examples/data/3ptb.pdb", **kw)


def test_z_is_clamped_to_the_grid():
    """z is clamped to [grid_min, grid_max], outside which the bias would vanish."""
    rng = np.random.default_rng(3)
    pos = rng.normal(0, 0.1, (4, 3))
    n = AXIS / np.linalg.norm(AXIS)
    s = mm.System()
    for m in MASSES:
        s.addParticle(m)
    cv = funnel_z_cv(HOST, GUEST, AXIS, grid_min=0.0, grid_max=2.0).resolve(pos, 4, None).cv
    cv.addToSystem(s)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    for shift, expected in [(5.0, 2.0), (-5.0, 0.0)]:
        moved = pos.copy()
        moved[GUEST] += shift * n
        ctx.setPositions(moved)
        assert cv.getValue(ctx).value_in_unit(unit.nanometers) == pytest.approx(expected, abs=1e-6)
