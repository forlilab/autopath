from types import SimpleNamespace

import numpy as np
import openmm
import openmm.app
import openmm.unit as openmmunit
import pytest

from autopath.pulling.pbc import reduction_identity_limit, triclinic_minimum_image
from autopath.pulling.steered_md import SteeredMD

# OpenMM square-xy rhombic dodecahedron, reduced form (nm).
_DODEC = np.array([[8.4773, 0.0, 0.0], [0.0, 8.4773, 0.0], [4.2386, 4.2386, 5.9943]])


def _old_cap(v, L):
    abs_v = np.abs(v)
    with np.errstate(divide="ignore"):
        per = np.where(abs_v > 1e-6, (L / 2.0) * np.linalg.norm(v) / abs_v, np.inf)
    return float(per.min())


def test_orthorhombic_cap_equals_old_formula():
    rng = np.random.default_rng(0)
    for _ in range(200):
        L = rng.uniform(4.0, 15.0, 3)
        v = rng.normal(size=3)
        v[rng.integers(0, 3)] *= rng.choice([1.0, 0.0])
        lim, _ = reduction_identity_limit(v / np.linalg.norm(v), np.diag(L))
        assert lim == pytest.approx(_old_cap(v, L), rel=1e-12)


def test_orthorhombic_minimum_image_matches_old():
    rng = np.random.default_rng(1)
    L = np.array([7.3, 9.1, 11.7])
    d = rng.uniform(-30, 30, size=(500, 3))
    np.testing.assert_allclose(triclinic_minimum_image(d, np.diag(L)), d - np.round(d / L) * L, atol=1e-12)


def test_dodecahedron_reduction_recovers_in_brick_vectors():
    half = np.diag(_DODEC) / 2
    rng = np.random.default_rng(2)
    for _ in range(200):
        v = rng.uniform(-0.99, 0.99, 3) * half
        far = v + np.array(rng.integers(-2, 3, 3), float) @ _DODEC
        np.testing.assert_allclose(triclinic_minimum_image(far, _DODEC), v, atol=1e-9)


def _openmm_distance(box, p0, p1):
    system = openmm.System()
    for _ in range(2):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box])
    f = openmm.CustomCentroidBondForce(2, "distance(g1,g2)")
    f.addGroup([0])
    f.addGroup([1])
    f.addBond([0, 1], [])
    f.setUsesPeriodicBoundaryConditions(True)
    system.addForce(f)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([openmm.Vec3(*p0), openmm.Vec3(*p1)] * openmmunit.nanometers)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmmunit.kilojoules_per_mole)


@pytest.mark.parametrize("n", [np.array([0.0, 0.0, 1.0]), np.random.default_rng(3).normal(size=3)])
def test_cap_is_identity_region_of_openmm_pbc_distance(n):
    n = n / np.linalg.norm(n)
    cap, _ = reduction_identity_limit(n, _DODEC)
    p0 = np.array([1.0, 2.0, 3.0])
    for r, same in ((cap - 0.01, True), (cap + 0.02, False)):
        d = _openmm_distance(_DODEC, p0 + r * n, p0)
        if same:
            assert d == pytest.approx(r, abs=1e-5)
        else:
            assert abs(d - r) > 1e-3


def _smd(lig, prot, anchor=None):
    smd = SteeredMD.__new__(SteeredMD)
    smd.ligand_atoms, smd.groupA_atoms = lig, lig
    smd.groupB_atoms = anchor if anchor is not None else prot
    smd._protein_heavy = np.asarray(prot)
    return smd


def test_image_distance_sees_protein_across_the_boundary():
    # ligand 0.3 nm inside the top face; the protein image one c-vector down sits 0.3+0.2 nm away
    c = _DODEC[2]
    pos = np.array([[0.0, 0.0, 0.0],              # protein atom
                    c - np.array([0.0, 0.0, 0.5])])  # ligand atom, 0.5 nm below the protein's +c image
    d = _smd([1], [0])._min_ligand_protein_image_nm(pos, _DODEC)
    assert d == pytest.approx(0.5, abs=1e-9)
    # same-image distance is not counted: only the 26 neighbouring images
    assert np.linalg.norm(pos[1] - pos[0]) > 5.0


def test_co_image_ligand_shifts_the_whole_ligand_once():
    # topology: residue 0 = protein (atom 0), residue 1 = ligand (atoms 1, 2)
    top = openmm.app.Topology()
    ch = top.addChain()
    r0, r1 = top.addResidue("ALA", ch), top.addResidue("UNK", ch)
    for res, n in ((r0, 1), (r1, 2)):
        for _ in range(n):
            top.addAtom("C", openmm.app.element.carbon, res)
    lig = np.array([[0.1, 0.2, 1.5], [0.2, 0.2, 1.5]]) + _DODEC[2] + _DODEC[1]  # wrapped one c+b away
    pos = np.vstack([[0.1, 0.2, 0.0], lig])
    box = [openmm.Vec3(*row) * openmmunit.nanometers for row in _DODEC]
    stored = {}
    state = SimpleNamespace(getPositions=lambda asNumpy=True: pos * openmmunit.nanometers,
                            getPeriodicBoxVectors=lambda asNumpy=True: np.asarray(_DODEC) * openmmunit.nanometers)
    ctx = SimpleNamespace(getState=lambda **kw: state, setPositions=lambda p: stored.setdefault("p", p))
    smd = _smd([1, 2], [0])
    smd.topology = top
    SteeredMD._co_image_ligand(smd, SimpleNamespace(context=ctx))
    new = stored["p"].value_in_unit(openmmunit.nanometers)
    np.testing.assert_allclose(new[0], pos[0])                                   # protein untouched
    np.testing.assert_allclose(new[1:], lig - _DODEC[2] - _DODEC[1], atol=1e-9)  # both ligand atoms moved back
    assert np.linalg.norm(new[1] - new[0]) == pytest.approx(1.5, abs=1e-9)


def test_com_reduction_margin_follows_current_direction():
    from autopath.pulling.pbc import com_reduction_margin
    box = _DODEC
    w = np.array([1.0])
    # 2.0 nm along x: limit a_x/2 = 4.24 nm; same length along z: limit c_z/2 = 3.00 nm
    pos_x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pos_z = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    d, lim_x = com_reduction_margin(pos_x, box, [1], w, [0], w)
    assert d == pytest.approx(2.0) and lim_x == pytest.approx(box[0, 0] / 2)
    d, lim_z = com_reduction_margin(pos_z, box, [1], w, [0], w)
    assert d == pytest.approx(2.0) and lim_z == pytest.approx(box[2, 2] / 2)
    assert lim_z < lim_x   # a pull that tilts toward z loses margin the t=0 cap did not see


def test_com_reduction_margin_matches_openmm_cv_inside_limit():
    from autopath.pulling.pbc import com_reduction_margin
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        lim, _ = reduction_identity_limit(n, _DODEC)
        r = rng.uniform(0.1, 0.95) * lim
        pos = np.array([[0.0, 0.0, 0.0], r * n])
        d, lim2 = com_reduction_margin(pos, _DODEC, [1], [1.0], [0], [1.0])
        assert d == pytest.approx(r) and lim2 == pytest.approx(lim)
        assert np.linalg.norm(triclinic_minimum_image(r * n, _DODEC)) == pytest.approx(r)
