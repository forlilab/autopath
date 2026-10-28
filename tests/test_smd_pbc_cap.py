import logging
import re
from types import SimpleNamespace

import numpy as np
import openmm
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


class _Stop(Exception):
    pass


class _FakeSMD(SimpleNamespace):
    @property
    def autostop_nc_window(self):
        # Halts pull_single_direction right after the move cap.
        raise _Stop


def _run_cap(tmp_path, box, pos, n_moves, monkeypatch, caplog):
    monkeypatch.setattr("autopath.pulling.steered_md.add_reporters", lambda *a, **k: None)
    box_vecs = [[x * openmmunit.nanometers for x in row] for row in box]
    positions = np.asarray(pos) * openmmunit.nanometers
    r = float(np.linalg.norm(triclinic_minimum_image(np.subtract(*pos), box)))
    state = SimpleNamespace(getPeriodicBoxVectors=lambda: box_vecs, getPositions=lambda **kw: positions)
    sim = SimpleNamespace(context=SimpleNamespace(setParameter=lambda *a, **k: None,
                                                  getState=lambda **kw: state))
    smd = _FakeSMD(out_dir=str(tmp_path), use_GReweighting=False,
                   com_dist=SimpleNamespace(getValue=lambda *a, **k: r * openmmunit.nanometers),
                   sMD_moves=n_moves, steps_per_move=1, save_freq=1, sMD_max_r_offset=100.0,
                   pbc_safety_nm=0.5, groupA_atoms=[0], groupB_atoms=[1],
                   dx_per_move=0.001 * openmmunit.nanometers)
    with caplog.at_level(logging.WARNING, logger="autopath"), pytest.raises(_Stop):
        SteeredMD.pull_single_direction(smd, sim, run_id=1, direction="forward",
                                        lambda0=r * openmmunit.nanometers)
    return r


@pytest.mark.parametrize("wrapped", [False, True])
def test_pull_cap_in_dodecahedron(tmp_path, monkeypatch, caplog, wrapped):
    # COM-COM vector along z; wrapped stores it across the c+b boundary to exercise the image.
    pos = [[0.1, 0.2, 1.5], [0.1, 0.2, 0.0]]
    if wrapped:
        pos[0] = list(np.asarray(pos[0]) + _DODEC[2] + _DODEC[1])
    r = _run_cap(tmp_path, _DODEC, pos, 10000, monkeypatch, caplog)
    assert r == pytest.approx(1.5)
    want = int((_old_cap(np.array([0.0, 0.0, r]), np.diag(_DODEC)) - 0.5 - r) / 0.001)
    m = re.search(r"Capping sMD_moves 10000 → (\d+)", caplog.text)
    assert m and int(m.group(1)) == want
    assert "non-orthorhombic" not in caplog.text


def test_pull_cap_unchanged_in_orthorhombic_box(tmp_path, monkeypatch, caplog):
    L = np.array([7.0, 9.0, 11.0])
    v = np.array([1.0, 0.6, 0.3])
    r = _run_cap(tmp_path, np.diag(L), [list(v), [0.0, 0.0, 0.0]], 10000, monkeypatch, caplog)
    want = int((_old_cap(v, L) - 0.5 - r) / 0.001)
    m = re.search(r"Capping sMD_moves 10000 → (\d+)", caplog.text)
    assert m and int(m.group(1)) == want
