import numpy as np
import openmm as mm
import pytest

from autopath.metadynamics.learned.features import Feature, cn_switch, evaluate
from autopath.metadynamics.learned.forces import cn_expression, feature_cv, fused_sw_forces

N = 14
WATER = list(range(7, 14))


def _system(box, linked_exclusion=False):
    s = mm.System()
    for _ in range(N):
        s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in box])
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0)
    for _ in range(N):
        nb.addParticle(0.0, 0.3, 0.0)
    nb.addException(0, 1, 0.0, 0.3, 0.0)          # excluded pair, no term links it
    if linked_exclusion:
        nb.addException(3, 0, 0.0, 0.3, 0.0)      # excluded pair linked by a W3_p term
    s.addForce(nb)
    return s


BOXES = {"cubic": np.diag([4.0, 4.0, 4.0]),
         "rhombic": np.array([[4.0, 0, 0], [0, 4.0, 0], [2.0, 2.0, 4.0 * np.sqrt(2) / 2]])}


def _feats(linked_exclusion=False):
    # atoms 0-2: patch (0,1 one residue; 2 another); 3-6: other protein; 7-13: water
    return [Feature("W1_x", "cn", "W1", terms=[[[0], WATER, 1.0]]),
            Feature("W3_p", "cn", "W3", terms=[[[0, 1], WATER, 1.0], [[0, 1], [2, 3, 4, 5, 6], 0.63],
                                               [[2], WATER, 1.0], [[2], [0, 1, 3, 4, 5, 6], 0.63]],
                    exclude=[[0, 3]] if linked_exclusion else None)]


def _energy(force, X, box, linked_exclusion=False):
    s = _system(box, linked_exclusion)
    force.setForceGroup(1)
    s.addForce(force)
    c = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    c.setPeriodicBoxVectors(*[mm.Vec3(*v) for v in box])
    c.setPositions(X)
    st = c.getState(getEnergy=True, getForces=True, groups={1})
    return st.getPotentialEnergy()._value, st.getForces(asNumpy=True)._value


def _positions(seed):
    X = np.random.default_rng(seed).uniform(0, 3.5, size=(N, 3))
    X[7] = X[0] + [0.35 * (1 + 4e-4), 0, 0]            # inside the r0 expansion window
    X[9] = X[0] + [0.0, 0.35, 0.0]                      # exactly r0
    X[3] = X[0] + [0.0, 0.0, 0.3]                       # close, linked to 0 by W3_p
    return X


@pytest.mark.parametrize("linked", [False, True], ids=["unlinked_excl", "linked_excl"])
@pytest.mark.parametrize("box", BOXES.values(), ids=BOXES.keys())
@pytest.mark.parametrize("seed", range(3))
def test_cn_openmm_equals_numpy(box, seed, linked):
    X = _positions(seed)
    feats = _feats(linked)
    ref = evaluate(feats, X[None], box=box[None])[0]
    assert np.all(ref > 0)
    for k, f in enumerate(feats):
        e, _ = _energy(feature_cv(f, _system(box, linked), "f"), X, box, linked)
        assert e == pytest.approx(ref[k], rel=1e-10, abs=1e-12)
    forces, const = fused_sw_forces(feats, np.array([0.7, -1.3]), np.zeros(2), np.ones(2), _system(box, linked))
    assert len(forces) == 1 and const == 0.0
    e, _ = _energy(forces[0], X, box, linked)
    assert e == pytest.approx(0.7 * ref[0] - 1.3 * ref[1], rel=1e-10, abs=1e-12)


def test_linked_exclusion_changes_the_value():
    box, X = BOXES["cubic"], _positions(0)
    with_ex = evaluate(_feats(True), X[None], box=box[None])[0, 1]
    without = evaluate(_feats(False), X[None], box=box[None])[0, 1]
    assert without - with_ex == pytest.approx(0.63 * cn_switch(0.3), rel=1e-12)


def test_cn_with_overlapping_term_sides_equals_numpy():
    box, X = BOXES["rhombic"], _positions(4)
    X[2] = X[1] + [0.4, 0.1, 0]                       # intra-class pair inside the cutoff
    f = Feature("ov", "cn", "W3", terms=[[[0, 1, 2], [1, 2] + WATER, 0.5]], exclude=[[0, 1]])
    ref = evaluate([f], X[None], box=box[None])[0, 0]
    e, _ = _energy(feature_cv(f, _system(box), "f"), X, box)
    assert e == pytest.approx(ref, rel=1e-10)


def test_cn_forces_match_finite_differences():
    box = BOXES["rhombic"]
    X = np.random.default_rng(5).uniform(0, 3.5, size=(N, 3))
    feats = _feats()
    forces, _ = fused_sw_forces(feats, np.ones(2), np.zeros(2), np.ones(2), _system(box))
    _, F = _energy(forces[0], X, box)
    h = 1e-6
    for i in (0, 2, 9):
        for ax in range(3):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, ax] += h
            Xm[i, ax] -= h
            num = -(evaluate(feats, Xp[None], box=box[None])[0].sum()
                    - evaluate(feats, Xm[None], box=box[None])[0].sum()) / (2 * h)
            assert F[i, ax] == pytest.approx(num, rel=1e-5, abs=1e-7)


def test_cn_expression_matches_cn_switch_piecewise():
    r = np.array([0.1, 0.3, 0.35 * (1 - 5e-4), 0.35, 0.35 * (1 + 9e-4), 0.35 * (1 + 2e-3), 0.8, 1.1999, 1.2, 1.5])
    s = mm.System()
    s.addParticle(1.0)
    s.addParticle(1.0)
    b = mm.CustomBondForce(cn_expression())
    b.addBond(0, 1, [])
    s.addForce(b)
    c = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    for ri, ref in zip(r, cn_switch(r)):
        c.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(ri, 0, 0)])
        e = c.getState(getEnergy=True).getPotentialEnergy()._value
        assert e == pytest.approx(ref, rel=1e-12, abs=1e-14)


def test_cn_exclude_inconsistent_with_the_system_is_not_supported():
    box = BOXES["cubic"]
    with pytest.raises(ValueError, match="not supported"):
        fused_sw_forces(_feats(False), np.ones(2), np.zeros(2), np.ones(2), _system(box, True))
    with pytest.raises(ValueError, match="not supported"):
        feature_cv(_feats(True)[1], _system(box, False), "f")


def test_cn_cutoff_too_large_for_the_box_is_not_supported():
    small = np.diag([2.4, 2.4, 2.4])
    with pytest.raises(ValueError, match="not supported"):
        fused_sw_forces(_feats(), np.ones(2), np.zeros(2), np.ones(2), _system(small))
    with pytest.raises(ValueError, match="not supported"):
        feature_cv(_feats()[0], _system(small), "f")


def test_repeated_index_inside_a_term_side_is_not_supported():
    with pytest.raises(ValueError, match="not supported"):
        Feature("dup", "cn", "W3", terms=[[[0, 0, 1], WATER, 1.0]])
    f = Feature("dup", "cn", "W3", terms=[[[0, 1], WATER, 1.0]])
    f.terms[0][1] = [7, 7, 8]                         # bypass construction-time validation
    with pytest.raises(ValueError, match="not supported"):
        feature_cv(f, _system(BOXES["cubic"]), "f")
