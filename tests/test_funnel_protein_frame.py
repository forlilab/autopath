"""Protein-frame funnel: axis rebuilt from host + two protein CA centroids at run time."""
import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import Element, Topology
from scipy.spatial.transform import Rotation

from autopath.metadynamics.CV import funnel_z_cv
from autopath.metadynamics.Funnel import (FRAME_MIN_ATOMS, FRAME_MIN_DIST_NM,
                                          create_funnel_force_from_trajectory_analysis, funnel_axis_at,
                                          funnel_frame, funnel_z_ang, load_funnel_params, save_funnel_params,
                                          select_frame_groups, upgrade_funnel_params)
from autopath.metadynamics.learned.forces import check_funnel_match, funnel_z_force

REF = mm.Platform.getPlatformByName("Reference")
A = u.angstrom
KCAL_A2 = u.kilocalorie_per_mole / A**2
AXIS = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
KJ_NM = u.kilojoule_per_mole / u.nanometer


def _toy(n_res=90, shape="ball", seed=3):
    """(topology, positions nm, masses, host, guest, protein atoms): one-chain protein, ligand, waters."""
    rng = np.random.default_rng(seed)
    if shape == "ball":
        d = rng.normal(size=(n_res, 3))
        ca = d / np.linalg.norm(d, axis=1)[:, None] * 2.4 * rng.uniform(0, 1, (n_res, 1)) ** (1 / 3)
        pocket = np.array([0.0, 0.0, 1.9])
    else:                                              # a rod through the pocket
        ca = np.c_[np.linspace(-3.0, 3.0, n_res), rng.normal(0, 0.05, (n_res, 2))]
        pocket = np.zeros(3)
    top = Topology()
    chain = top.addChain()
    pos, masses, prev_c, protein, ca_idx = [], [], None, [], []
    for k, x in enumerate(ca):
        res = top.addResidue("ALA", chain)
        atoms = []
        for name, el, off, m in (("N", "N", (-0.1, 0.05, 0.0), 14.007),
                                 ("CA", "C", (0.0, 0.0, 0.0), 12.011 if k % 2 else 10.0),
                                 ("C", "C", (0.1, 0.05, 0.0), 12.011)):
            atoms.append(top.addAtom(name, Element.getBySymbol(el), res))
            pos.append(x + np.array(off))
            masses.append(m)
        top.addBond(atoms[0], atoms[1])
        top.addBond(atoms[1], atoms[2])
        if prev_c is not None:
            top.addBond(prev_c, atoms[0])
        prev_c = atoms[2]
        protein += [a.index for a in atoms]
        ca_idx.append(atoms[1].index)
    ca_idx = np.array(ca_idx)
    host = sorted(int(i) for i in ca_idx[np.argsort(np.linalg.norm(ca - pocket, axis=1))[:6]])
    m = np.array(masses)
    hc = np.average(np.array(pos)[host], axis=0, weights=m[host])
    perp = np.cross(AXIS, [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    lig = top.addResidue("UNK", top.addChain())
    guest = []
    for j, (el, mm_) in enumerate((("C", 12.0), ("N", 14.0), ("O", 16.0), ("H", 1.5))):
        guest.append(top.addAtom(f"X{j}", Element.getBySymbol(el), lig).index)
        pos.append(hc + 0.4 * AXIS + 0.25 * perp + rng.normal(0, 0.08, 3))
        masses.append(mm_)
    for _ in range(30):
        w = top.addResidue("HOH", top.addChain())
        top.addAtom("O", Element.getBySymbol("O"), w)
        pos.append(rng.uniform(-3.0, 3.0, 3))
        masses.append(18.0)
    return top, np.array(pos), np.array(masses), host, guest, np.array(protein)


def _lab_params(host, guest, axis=AXIS):
    return {"host_index": list(host), "guest_index": list(guest), "k_xy": 10.0 * KCAL_A2,
            "k_z": 50.0 * KCAL_A2, "z_cc": 8.0 * A, "z_max": 14.0 * A, "alpha": 10.0 * u.degrees,
            "R_cylinder": 0.5 * A, "unbinding_axis": np.array(axis),
            "axial_distances": np.r_[np.full(100, 2.0), np.linspace(2.0, 20.0, 900)],
            "com_trajectory": np.zeros((2, 3)), "host_com_trajectory": np.zeros((2, 3)),
            "guest_com_trajectory": np.zeros((2, 3)), "radial_distances": np.zeros(2),
            "trajectory_length": 2, "force_group": 10}


TOY = _toy()
TOP, POS, MASSES, HOST, GUEST, PROTEIN = TOY
LAB = _lab_params(HOST, GUEST)
NEW = upgrade_funnel_params(LAB, POS, TOP, MASSES)
SIDE = np.cross(AXIS, [0.0, 0.0, 1.0]) / np.linalg.norm(np.cross(AXIS, [0.0, 0.0, 1.0]))


def _system():
    s = mm.System()
    for m in MASSES:
        s.addParticle(float(m))
    return s


def _restraint(params, pos):
    s = _system()
    s.addForce(create_funnel_force_from_trajectory_analysis(params))
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            st.getForces(asNumpy=True).value_in_unit(KJ_NM), ctx)


def _cv_value(cv, pos):
    s = _system()
    cv.addToSystem(s)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    ctx.setPositions(pos)
    return cv.getValue(ctx).value_in_unit(u.nanometer)


def _z_cv(params, pos):
    spec = funnel_z_cv(params["host_index"], params["guest_index"], params["unbinding_axis"],
                       grid_min=-10.0, grid_max=10.0, frame=funnel_frame(params))
    return _cv_value(spec.resolve(pos, len(MASSES), None).cv, pos)


def _z_force(params, pos):
    return _cv_value(funnel_z_force(params["host_index"], params["guest_index"], params["unbinding_axis"],
                                    frame=funnel_frame(params)), pos)


def _z_ang(params, pos):
    return funnel_z_ang(pos, params["host_index"], params["guest_index"], MASSES, params["unbinding_axis"],
                        frame=funnel_frame(params))


def _com(pos, idx):
    return np.average(pos[idx], axis=0, weights=MASSES[idx])


def _rotate(pos, idx, R, centre):
    out = pos.copy()
    out[idx] = (pos[idx] - centre) @ R.T + centre
    return out


def _random_rotation(seed):
    return Rotation.random(random_state=seed).as_matrix()


# ---- frame groups --------------------------------------------------------------------------------

def test_frame_groups_are_deterministic_large_and_far_from_the_host():
    b, c = select_frame_groups(POS, TOP, HOST, MASSES)
    assert (b, c) == select_frame_groups(POS, TOP, HOST, MASSES)
    assert len(b) >= FRAME_MIN_ATOMS and len(c) >= FRAME_MIN_ATOMS and not set(b) & set(c)
    names = {a.index: a.name for a in TOP.atoms()}
    assert all(names[i] == "CA" for i in b + c)
    hc = _com(POS, HOST)
    assert np.linalg.norm(POS[b + c] - hc, axis=1).min() > FRAME_MIN_DIST_NM
    assert min(b) < min(c)
    assert NEW["frame_b_index"] == b and NEW["frame_c_index"] == c


def test_ill_conditioned_frames_raise():
    top, pos, masses, host, _, _ = _toy(shape="rod")
    with pytest.raises(ValueError, match="ill-conditioned"):
        select_frame_groups(pos, top, host, masses)
    top, pos, masses, host, _, _ = _toy(n_res=30)
    with pytest.raises(ValueError, match=f"fewer than {FRAME_MIN_ATOMS}"):
        select_frame_groups(pos, top, host, masses)


def test_host_spanning_two_molecules_is_not_supported():
    water = len(MASSES) - 1
    with pytest.raises(ValueError, match="not supported"):
        select_frame_groups(POS, TOP, HOST + [water], MASSES)


# ---- reference parity, rotation invariance -------------------------------------------------------

def test_reference_orientation_reproduces_the_lab_frame():
    assert funnel_frame(LAB) is None and "axis_coeffs" not in LAB
    assert funnel_axis_at(NEW, POS, MASSES) == pytest.approx(AXIS, abs=1e-12)
    e_lab, _, _ = _restraint(LAB, POS)
    e_new, _, _ = _restraint(NEW, POS)
    assert e_lab > 1.0
    assert e_new == pytest.approx(e_lab, abs=1e-6)
    for f in (_z_cv, _z_force, _z_ang):
        assert f(NEW, POS) == pytest.approx(f(LAB, POS), abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rigid_rotation_of_the_whole_system_changes_nothing(seed):
    R = _random_rotation(seed)
    pos = _rotate(POS, np.arange(len(POS)), R, _com(POS, HOST)) + np.array([0.7, -1.1, 0.4])
    e0, f0, _ = _restraint(NEW, POS)
    e1, f1, _ = _restraint(NEW, pos)
    assert e1 == pytest.approx(e0, abs=1e-8)
    assert np.allclose(f1, f0 @ R.T, atol=1e-8)
    for f in (_z_cv, _z_force):
        assert f(NEW, pos) == pytest.approx(f(NEW, POS), abs=1e-8)
    assert _z_ang(NEW, pos) == pytest.approx(_z_ang(NEW, POS), abs=1e-7)       # Å
    assert funnel_axis_at(NEW, pos, MASSES) == pytest.approx(R @ AXIS, abs=1e-12)
    assert abs(_restraint(LAB, pos)[0] - e0) > 1e-3            # the lab frame is not invariant


def test_frame_follows_the_protein_not_the_ligand():
    R = _random_rotation(7)
    hc = _com(POS, HOST)
    g_rel = _com(POS, GUEST) - hc
    lig = _rotate(POS, np.array(GUEST), R, hc)
    assert _z_cv(NEW, lig) == pytest.approx(float(AXIS @ (R @ g_rel)), abs=1e-9)
    prot = _rotate(POS, PROTEIN, R, hc)
    z = _z_cv(NEW, prot)
    assert z == pytest.approx(float((R @ AXIS) @ g_rel), abs=1e-9)
    assert abs(z - _z_cv(NEW, POS)) > 1e-3


def test_restraint_forces_are_finite_and_match_finite_differences():
    rng = np.random.default_rng(5)
    pos = _rotate(POS, np.arange(len(POS)), _random_rotation(3), np.zeros(3)) + rng.normal(0, 0.02, POS.shape)
    e, f, ctx = _restraint(NEW, pos)
    assert e > 1.0 and np.isfinite(f).all()
    b, c = NEW["frame_b_index"], NEW["frame_c_index"]
    atoms = [HOST[0], GUEST[0], *b[:3], *c[:3]]
    assert np.abs(f[b]).max() > 1e-3 and np.abs(f[c]).max() > 1e-3
    h = 1e-5
    for i in atoms:
        for k in range(3):
            p, m = pos.copy(), pos.copy()
            p[i, k] += h
            m[i, k] -= h
            ctx.setPositions(p)
            ep = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            ctx.setPositions(m)
            em = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            assert f[i, k] == pytest.approx(-(ep - em) / (2 * h), rel=1e-5, abs=1e-5)


# ---- old params ----------------------------------------------------------------------------------

def _old_energy(params, pos):
    """The funnel restraint written out in numpy (mass-weighted centroids; frame axis if any)."""
    to = lambda q, un: q.value_in_unit(un)
    nm, kjnm2 = u.nanometer, u.kilojoule_per_mole / u.nanometer**2
    from autopath.metadynamics.Funnel import funnel_axis_at, funnel_lower_wall_ang
    n = funnel_axis_at(params, pos, MASSES)
    com = lambda idx: np.average(pos[idx], axis=0, weights=MASSES[idx])
    r = com(params["guest_index"]) - com(params["host_index"])
    rz, rxy = float(n @ r), float(np.linalg.norm(np.cross(n, r)))
    zcc, zmax, rc = (to(params[k], nm) for k in ("z_cc", "z_max", "R_cylinder"))
    zlow = funnel_lower_wall_ang(params) / 10.0
    kxy, kz = to(params["k_xy"], kjnm2), to(params["k_z"], kjnm2)
    rf = max(zcc - rz, 0.0) * np.tan(np.radians(10.0)) + rc
    return ((rz < zcc) * (rxy > rf) * 0.5 * kxy * (rxy - rf) ** 2
            + (rz > zcc) * (rxy > rc) * 0.5 * kxy * (rxy - rc) ** 2
            + (rz > zmax) * 0.5 * kz * (rz - zmax) ** 2 + (rz < zlow) * 0.5 * kz * (zlow - rz) ** 2)


def test_old_params_keep_the_lab_frame_restraint():
    force = create_funnel_force_from_trajectory_analysis(LAB)
    assert force.getNumGroups() == 2
    names = {force.getGlobalParameterName(i) for i in range(force.getNumGlobalParameters())}
    assert {"nx", "ny", "nz"} <= names
    for seed in range(4):
        pos = _rotate(POS, np.arange(len(POS)), _random_rotation(seed), np.zeros(3))
        for shift in (0.0, 1.0, -0.8):
            p = pos.copy()
            p[GUEST] += shift * AXIS + 0.6 * SIDE
            assert _restraint(LAB, p)[0] == pytest.approx(_old_energy(LAB, p), rel=1e-9, abs=1e-9)



def test_frame_restraint_uses_mass_weighted_centroids():
    for seed in range(3):
        pos = _rotate(POS, np.arange(len(POS)), _random_rotation(seed), np.zeros(3))
        for shift in (0.0, 1.0, -0.8):
            p = pos.copy()
            p[GUEST] += shift * AXIS + 0.6 * SIDE           # off axis: the funnel wall acts
            e = _old_energy(NEW, p)
            assert e > 1.0
            assert _restraint(NEW, p)[0] == pytest.approx(e, rel=1e-7, abs=1e-7)

# ---- training z, upgrade round trip, learned-CV match --------------------------------------------

def test_training_z_equals_the_run_time_z_force():
    from autopath.metadynamics.learned.pipeline import _cache_key, _funnel_geometry, _funnel_z_nm
    g = _funnel_geometry(NEW)
    assert _funnel_geometry(LAB)["frame"] is None
    rng = np.random.default_rng(9)
    X = np.stack([_rotate(POS, np.arange(len(POS)), _random_rotation(s), np.zeros(3))
                  + rng.normal(0, 0.03, POS.shape) for s in range(3)])
    z = _funnel_z_nm(X, g, MASSES)
    assert z == pytest.approx([_z_force(NEW, x) for x in X], abs=1e-10)
    assert not np.allclose(z, _funnel_z_nm(X, _funnel_geometry(LAB), MASSES))
    args = ([], 1, "forward", __file__, __file__, [0], [1])
    assert (_cache_key(*args, code_hash="a", classes="rupture", funnel=g, masses=MASSES)
            != _cache_key(*args, code_hash="a", classes="rupture", funnel=_funnel_geometry(LAB), masses=MASSES))


def test_upgrade_round_trips_through_save_and_load(tmp_path):
    save_funnel_params(NEW, str(tmp_path / "fp"))
    got = load_funnel_params(str(tmp_path / "fp.npz"))
    assert got["frame_b_index"] == NEW["frame_b_index"] and got["frame_c_index"] == NEW["frame_c_index"]
    assert np.array_equal(got["axis_coeffs"], NEW["axis_coeffs"])
    pos = _rotate(POS, PROTEIN, _random_rotation(4), _com(POS, HOST))
    assert _restraint(got, pos)[0] == pytest.approx(_restraint(NEW, pos)[0], abs=1e-12)
    save_funnel_params(LAB, str(tmp_path / "old"))
    old = load_funnel_params(str(tmp_path / "old.npz"))
    assert funnel_frame(old) is None and "axis_coeffs" not in old


def test_upgrade_refuses_params_that_already_have_a_frame():
    with pytest.raises(ValueError, match="already"):
        upgrade_funnel_params(NEW, POS, TOP, MASSES)


def test_check_funnel_match_compares_the_frame(tmp_path):
    def npz(name, frame):
        path = str(tmp_path / name)
        extra = {} if frame is None else dict(funnel_frame_b=np.array(frame["b"]), funnel_frame_c=np.array(frame["c"]),
                                             funnel_axis_coeffs=np.array(frame["coeffs"]))
        np.savez(path, funnel_host=np.array(HOST), funnel_guest=np.array(GUEST), funnel_axis=AXIS, **extra)
        return path
    with_frame, lab = npz("f.npz", funnel_frame(NEW)), npz("l.npz", None)
    check_funnel_match(with_frame, NEW)
    check_funnel_match(lab, LAB)
    for path, fp in ((with_frame, LAB), (lab, NEW)):
        with pytest.raises(ValueError, match="not supported"):
            check_funnel_match(path, fp)


# ---- learned CV: residual z and the written npz --------------------------------------------------

def test_residual_s_w_uses_the_protein_frame_z(tmp_path):
    from types import SimpleNamespace
    from autopath.metadynamics.BoundaryCorrection import composite_bias_force
    from autopath.metadynamics.Funnel import frame_basis
    from autopath.metadynamics.learned.forces import learned_cv_spec
    from . import learned_toys as t
    m = np.array(t.MASSES)
    cen = lambda X, idx: np.average(X[idx], axis=0, weights=m[idx])
    b, c = [4], [0]
    coeffs = frame_basis(cen(t.X0, t.HOST), cen(t.X0, b), cen(t.X0, c)) @ (t.AXIS / np.linalg.norm(t.AXIS))
    path = t.write_npz_r2(str(tmp_path / "r.npz"), True, frame={"b": b, "c": c, "coeffs": coeffs})

    def s_value(X):
        s = t.toy_system()
        comp = learned_cv_spec(path, s, 1, fused=False)[0].resolve(None, 5, None).cv
        force, _ = composite_bias_force([SimpleNamespace(force=comp, periodic=False)])
        force.setEnergyFunction(force.getEnergyFunction().replace("table(s0)", "s0", 1))
        force.setForceGroup(5)
        s.addForce(force)
        ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
        ctx.setPositions(X)
        return ctx.getState(getEnergy=True, groups={5}).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

    X1 = (t.X0 - t.X0.mean(0)) @ _random_rotation(11).T + t.X0.mean(0)
    assert s_value(t.X0) == pytest.approx(t.s_ref(t.X0), abs=1e-9)
    assert s_value(X1) == pytest.approx(t.s_ref(t.X0), abs=1e-9)
    assert abs(t.s_ref(X1) - t.s_ref(t.X0)) > 1e-3             # the lab-frame z would move


def test_learned_npz_stores_the_funnel_frame(monkeypatch, tmp_path):
    from .learned_toys import FUNNEL, rupture_data
    from .test_learned_rupture import _run_learn
    fp = dict(FUNNEL, frame_b_index=[4], frame_c_index=[1], axis_coeffs=np.array([0.6, 0.0, 0.8]))
    path, _ = _run_learn(monkeypatch, tmp_path, rupture_data(), funnel=fp)
    d = np.load(path)
    assert list(d["funnel_frame_b"]) == [4] and list(d["funnel_frame_c"]) == [1]
    assert np.array_equal(d["funnel_axis_coeffs"], [0.6, 0.0, 0.8])
    check_funnel_match(path, fp)
    with pytest.raises(ValueError, match="not supported"):
        check_funnel_match(path, FUNNEL)


def test_orchestrator_z_monitor_uses_the_protein_frame():
    from autopath.autopath_core import funnel_z_monitor
    spec = funnel_z_monitor(object(), NEW, "path")["funnel_z"]
    pos = _rotate(POS, PROTEIN, _random_rotation(8), _com(POS, HOST))
    assert _cv_value(spec.resolve(pos, len(MASSES), None).cv, pos) == pytest.approx(_z_force(NEW, pos), abs=1e-10)
    assert abs(_z_force(NEW, pos) - _z_force(LAB, pos)) > 1e-3


def test_orchestrator_funnel_setup_raises_on_an_unsupported_frame_and_still_swallows_other_errors():
    from autopath.autopath_core import funnel_setup_failed
    water = len(MASSES) - 1
    params = _lab_params(HOST + [water], GUEST)
    try:
        upgrade_funnel_params(params, POS, TOP, MASSES)
    except Exception as e:
        with pytest.raises(ValueError, match="not supported"):
            funnel_setup_failed(e)
    else:
        pytest.fail("upgrade should have raised")
    funnel_setup_failed(RuntimeError("boom"))
    funnel_setup_failed(ValueError("unrelated"))


def test_frame_z_cv_clamps_to_the_grid():
    for lo, hi in ((-0.2, 0.3), (-5.0, 0.1)):
        spec = funnel_z_cv(HOST, GUEST, AXIS, grid_min=lo, grid_max=hi, frame=funnel_frame(NEW))
        z = _cv_value(spec.resolve(POS, len(MASSES), None).cv, POS)
        assert z == pytest.approx(min(max(_z_force(NEW, POS), lo), hi), abs=1e-10)
    far = POS.copy()
    far[GUEST] += 5.0 * AXIS
    near = POS.copy()
    near[GUEST] -= 5.0 * AXIS
    spec = funnel_z_cv(HOST, GUEST, AXIS, grid_min=-0.4, grid_max=0.6, frame=funnel_frame(NEW))
    assert _cv_value(spec.resolve(far, len(MASSES), None).cv, far) == pytest.approx(0.6, abs=1e-10)
    assert _cv_value(spec.resolve(near, len(MASSES), None).cv, near) == pytest.approx(-0.4, abs=1e-10)


def test_cache_key_depends_on_the_axis_coefficients_alone():
    from autopath.metadynamics.learned.pipeline import _cache_key, _funnel_geometry
    g = _funnel_geometry(NEW)
    g2 = _funnel_geometry(dict(NEW, axis_coeffs=np.asarray(NEW["axis_coeffs"]) + 0.01))
    assert g["frame"]["b"] == g2["frame"]["b"] and g["frame"]["c"] == g2["frame"]["c"]
    args = ([], 1, "forward", __file__, __file__, [0], [1])
    kw = dict(code_hash="a", classes="rupture", masses=MASSES)
    assert _cache_key(*args, funnel=g, **kw) != _cache_key(*args, funnel=g2, **kw)
