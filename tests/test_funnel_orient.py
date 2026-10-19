"""Orientation-restrained funnel: core pinned by OrientationRestraintForce, lab axis, origin COM(core) + v."""
import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app import PDBFile
from scipy.spatial.transform import Rotation

from autopath.metadynamics.CV import funnel_z_cv
from autopath.metadynamics.Funnel import (FRAME_MIN_ATOMS, ORIENT_FORCE_GROUP, ORIENT_MAX_START_DEG,
                                          _default_core, check_orient_start,
                                          create_funnel_force_from_trajectory_analysis, funnel_axis_at,
                                          funnel_coords_ang, funnel_frame, funnel_lower_wall_ang, funnel_mode,
                                          funnel_z_ang, load_funnel_params, orient_params_for_state,
                                          orient_theta_deg, orientation_restrained_params,
                                          orientation_restraint_force, save_funnel_params)
from autopath.metadynamics.learned.forces import check_funnel_match, funnel_z_force

from .test_funnel_protein_frame import (AXIS, GUEST, HOST, LAB, MASSES, NEW, POS, PROTEIN, SIDE, TOP, _com,
                                        _cv_value, _lab_params, _random_rotation, _rotate, _system)

REF = mm.Platform.getPlatformByName("Reference")
KJ = u.kilojoule_per_mole
CA = np.array([a.index for a in TOP.atoms() if a.name == "CA"])
CORE = [int(i) for i in CA if i not in HOST
        and np.linalg.norm(POS[GUEST] - POS[i], axis=1).min() > 1.0]
ORI = orientation_restrained_params(NEW, POS, TOP, MASSES, k=1.0e4, core_index=CORE)
ALL = np.arange(len(POS))


def _ctx(forces, pos):
    s = _system()
    for f in forces:
        s.addForce(f)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    ctx.setPositions(pos)
    return ctx


def _energy(params, pos, group=None):
    f = create_funnel_force_from_trajectory_analysis(params)
    st = _ctx([f], pos).getState(getEnergy=True)
    return st.getPotentialEnergy().value_in_unit(KJ)


def _z_cv(params, pos):
    spec = funnel_z_cv(params["host_index"], params["guest_index"], params["unbinding_axis"],
                       grid_min=-10.0, grid_max=10.0, frame=funnel_frame(params))
    return _cv_value(spec.resolve(pos, len(MASSES), None).cv, pos)


def _z_force(params, pos):
    return _cv_value(funnel_z_force(params["host_index"], params["guest_index"], params["unbinding_axis"],
                                    frame=funnel_frame(params)), pos)


def _numpy_energy(params, pos):
    """The one-sided funnel written out in numpy from the Python mirror's (z, r_xy)."""
    nm, kjnm2 = u.nanometer, KJ / u.nanometer**2
    rz, rxy = (c / 10.0 for c in funnel_coords_ang(params, pos, MASSES))
    zcc, zmax, rc = (params[k].value_in_unit(nm) for k in ("z_cc", "z_max", "R_cylinder"))
    zlow = funnel_lower_wall_ang(params) / 10.0
    kxy, kz = params["k_xy"].value_in_unit(kjnm2), params["k_z"].value_in_unit(kjnm2)
    rf = max(zcc - rz, 0.0) * np.tan(params["alpha"].value_in_unit(u.radian)) + rc
    return ((rz < zcc) * (rxy > rf) * 0.5 * kxy * (rxy - rf) ** 2
            + (rz > zcc) * (rxy > rc) * 0.5 * kxy * (rxy - rc) ** 2
            + (rz > zmax) * 0.5 * kz * (rz - zmax) ** 2 + (rz < zlow) * 0.5 * kz * (zlow - rz) ** 2)


def _rigid(pos, seed, shift=(0.7, -1.1, 0.4)):
    R = _random_rotation(seed)
    return _rotate(pos, ALL, R, _com(pos, CORE)) + np.array(shift), R


def _guest_frames():
    """Guest at the site, far up the axis, off axis (wall active) and below the lower wall."""
    out = []
    for shift, side in ((0.0, 0.0), (1.6, 0.0), (0.5, 0.6), (1.6, 0.3), (-1.2, 0.2)):
        p = POS.copy()
        p[GUEST] += shift * AXIS + side * SIDE
        out.append(p)
    return out


# ---- builder ------------------------------------------------------------------------------------

@pytest.mark.parametrize("src", [LAB, NEW])
def test_builder_keeps_z_and_rxy_at_the_reference(src):
    o = orientation_restrained_params(src, POS, TOP, MASSES, core_index=CORE)
    assert funnel_mode(o) == "orient" and not {"axis_coeffs", "frame_b_index", "frame_c_index"} & set(o)
    for p in _guest_frames():
        assert funnel_coords_ang(o, p, MASSES) == pytest.approx(funnel_coords_ang(src, p, MASSES), abs=1e-6)
    assert o["unbinding_axis"] == pytest.approx(funnel_axis_at(src, POS, MASSES), abs=1e-12)
    v = _com(POS, HOST) - _com(POS, CORE)
    assert o["orient_offset"] == pytest.approx(v, abs=1e-12)
    assert np.array_equal(o["orient_ref"], POS[CORE]) and o["orient_k"] == 1.0e5
    for k in ("z_cc", "z_max", "R_cylinder", "alpha"):
        assert o[k] == src[k]


def test_mixed_partial_or_repeated_frames_are_not_supported():
    for bad in (dict(ORI, axis_coeffs=NEW["axis_coeffs"]), dict(ORI, origin_index=[1]),
                {k: v for k, v in ORI.items() if k != "orient_ref"}):
        with pytest.raises(ValueError, match="not supported"):
            funnel_mode(bad)
    with pytest.raises(ValueError, match="not supported"):
        orientation_restrained_params(ORI, POS, TOP, MASSES, core_index=CORE)
    with pytest.raises(ValueError, match="not supported"):
        orientation_restrained_params(LAB, POS, TOP, MASSES, core_index=CORE[:FRAME_MIN_ATOMS - 1])
    water = len(MASSES) - 1
    with pytest.raises(ValueError, match="not supported"):
        orientation_restrained_params(LAB, POS, TOP, MASSES, core_index=CORE + [water])
    with pytest.raises(ValueError, match="not supported"):
        orient_params_for_state(NEW, POS)


def test_default_core_is_dssp_ca_away_from_the_guest_and_host():
    from MDAnalysis.analysis.dssp.pydssp_numpy import assign
    pdb = PDBFile(str(__import__("pathlib").Path(__file__).parent / "data" / "membrane_peptide.pdb"))
    pos = np.array(pdb.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    res = [r for r in pdb.topology.residues() if {"N", "CA", "C", "O"} <= {a.name for a in r.atoms()}]
    idx = lambda r, n: next(a.index for a in r.atoms() if a.name == n)
    ss = np.array(["-", "H", "E"])[assign(pos[[[idx(r, n) for n in ("N", "CA", "C", "O")] for r in res]]
                                          * 10.0).argmax(1)]
    guest = [a.index for a in res[30].atoms()]
    host = [idx(res[29], "CA"), idx(res[31], "CA")]
    core = _default_core(pos, pdb.topology, host, guest)
    want = [idx(r, "CA") for r, s in zip(res, ss) if s != "-" and r is not res[30]
            and np.linalg.norm(pos[guest] - pos[idx(r, "CA")], axis=1).min() > 1.0
            and idx(r, "CA") not in host]
    assert core == want and 0 < len(core) < sum(ss != "-")


# ---- run-time forces vs Python mirrors ----------------------------------------------------------

def test_openmm_parity_on_guest_placements():
    hit = 0
    for p in _guest_frames():
        e = _energy(ORI, p)
        assert e == pytest.approx(_numpy_energy(ORI, p), rel=1e-8, abs=1e-8)
        hit += e > 1.0
        z = funnel_z_ang(p, HOST, GUEST, MASSES, ORI["unbinding_axis"], frame=funnel_frame(ORI)) / 10.0
        assert _z_cv(ORI, p) == pytest.approx(z, abs=1e-10)
        assert _z_force(ORI, p) == pytest.approx(z, abs=1e-10)
    assert hit >= 3                                    # walls and cone act on most placements


def test_restraint_forces_match_finite_differences():
    p = _guest_frames()[2]
    s = _system()
    s.addForce(create_funnel_force_from_trajectory_analysis(ORI))
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    ctx.setPositions(p)
    f = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(KJ / u.nanometer)
    assert np.abs(f[CORE]).max() > 1e-3
    h = 1e-5
    for i in (CORE[0], CORE[5], GUEST[0]):
        for k in range(3):
            pp, pm = p.copy(), p.copy()
            pp[i, k] += h
            pm[i, k] -= h
            ctx.setPositions(pp)
            ep = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
            ctx.setPositions(pm)
            em = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
            assert f[i, k] == pytest.approx(-(ep - em) / (2 * h), rel=1e-5, abs=1e-5)


def test_pocket_breathing_does_not_move_z():
    rng = np.random.default_rng(2)
    p = POS.copy()
    p[HOST] += rng.normal(0, 0.03, (len(HOST), 3)) + 0.03 * AXIS
    for f in (_z_cv, _z_force):
        assert f(ORI, p) == pytest.approx(f(ORI, POS), abs=1e-12)
        assert abs(f(NEW, p) - f(NEW, POS)) > 1e-3        # the host-COM origin moves


def test_orientation_restraint_energy():
    k = float(ORI["orient_k"])
    s = _system()
    s.addForce(orientation_restraint_force(ORI, len(MASSES)))
    assert s.getForce(0).getForceGroup() == ORIENT_FORCE_GROUP
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)
    for seed, ang in ((0, 0.0), (1, 3.0), (2, 25.0)):
        R = Rotation.from_rotvec(np.radians(ang) * np.array([0.2, 0.9, -0.4]) / np.linalg.norm([0.2, 0.9, -0.4]))
        p = _rotate(POS, ALL, R.as_matrix(), _com(POS, CORE)) + 0.3
        ctx.setPositions(p)
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
        th = np.radians(orient_theta_deg(ORI, p))
        assert th == pytest.approx(np.radians(ang), abs=1e-9)
        assert e == pytest.approx(2 * k * np.sin(th / 2) ** 2, rel=1e-9, abs=1e-9)
        assert (e > 1.0) == (ang > 0)


def test_restraint_is_added_with_the_funnel_and_once():
    from autopath.metadynamics.MetadynamicsMD import add_orientation_restraint
    s = _system()
    create_funnel_force_from_trajectory_analysis(ORI, system=s)
    assert sum(isinstance(f, mm.OrientationRestraintForce) for f in s.getForces()) == 1
    with pytest.raises(ValueError, match="not supported"):
        add_orientation_restraint(s, create_funnel_force_from_trajectory_analysis(ORI), ORI, POS)
    s = _system()
    assert add_orientation_restraint(s, create_funnel_force_from_trajectory_analysis(ORI), ORI, POS) == \
        pytest.approx(0.0, abs=1e-6)
    assert [type(f) for f in s.getForces()] == [mm.OrientationRestraintForce]
    assert add_orientation_restraint(_system(), create_funnel_force_from_trajectory_analysis(NEW), NEW, POS) is None
    for force, params in ((create_funnel_force_from_trajectory_analysis(NEW), ORI),
                          (create_funnel_force_from_trajectory_analysis(ORI), NEW), (None, ORI)):
        with pytest.raises(ValueError, match="not supported"):
            add_orientation_restraint(_system(), force, params, POS)


# ---- start guard and re-targeting ---------------------------------------------------------------

def test_guard_refuses_ten_degrees_and_accepts_half_a_degree():
    from autopath.metadynamics.MetadynamicsMD import add_orientation_restraint
    axis = np.array([0.4, 0.1, 0.9]) / np.linalg.norm([0.4, 0.1, 0.9])
    turn = lambda deg: _rotate(POS, ALL, Rotation.from_rotvec(np.radians(deg) * axis).as_matrix(), _com(POS, CORE))
    assert check_orient_start(ORI, turn(0.5)) == pytest.approx(0.5, abs=1e-9)
    assert ORIENT_MAX_START_DEG == 2.0
    with pytest.raises(ValueError, match="re-target with orient_params_for_state"):
        check_orient_start(ORI, turn(10.0))
    with pytest.raises(ValueError, match="not supported"):
        add_orientation_restraint(_system(), create_funnel_force_from_trajectory_analysis(ORI), ORI, turn(10.0))


def test_a_core_split_across_the_box_is_not_supported():
    p = POS.copy()
    p[CORE[3]] += np.array([6.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not supported"):
        orient_params_for_state(ORI, p)
    with pytest.raises(ValueError, match="not supported"):
        check_orient_start(ORI, p)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_retargeted_params_at_a_rotated_system_match_the_master(seed):
    for p in _guest_frames():
        q, R = _rigid(p, seed)
        t = orient_params_for_state(ORI, q)
        assert orient_theta_deg(t, q) == pytest.approx(0.0, abs=1e-6)
        assert t["orient_theta_deg"] == pytest.approx(orient_theta_deg(ORI, q), abs=1e-6)
        assert t["unbinding_axis"] == pytest.approx(t["orient_rotation"] @ t["master_axis"], abs=1e-12)
        assert t["unbinding_axis"] == pytest.approx(R @ ORI["unbinding_axis"], abs=1e-9)
        assert funnel_coords_ang(t, q, MASSES) == pytest.approx(funnel_coords_ang(ORI, p, MASSES), abs=1e-7)
        for f in (_z_cv, _z_force):
            assert f(t, q) == pytest.approx(f(ORI, p), abs=1e-9)
        assert _energy(t, q) == pytest.approx(_energy(ORI, p), rel=1e-8, abs=1e-8)
        assert abs(_z_cv(ORI, q) - _z_cv(ORI, p)) > 1e-3      # the master frame is lab-fixed


def test_two_walkers_rotated_differently_share_one_z():
    q1, _ = _rigid(POS, 3)
    q2, _ = _rigid(POS, 4, shift=(-2.0, 0.5, 1.3))
    t1, t2 = orient_params_for_state(ORI, q1), orient_params_for_state(ORI, q2)
    t2b = orient_params_for_state(orient_params_for_state(ORI, q1), q2)       # chained re-targeting
    assert t2b["unbinding_axis"] == pytest.approx(t2["unbinding_axis"], abs=1e-12)
    assert t2b["orient_rotation"] == pytest.approx(t2["orient_rotation"], abs=1e-12)
    rng = np.random.default_rng(6)
    for _ in range(4):
        d = rng.normal(0, 0.4, 3)                          # guest move, core-relative
        p = POS.copy()
        p[GUEST] += d
        w1, R1 = _rigid(p, 3)
        w2, R2 = _rigid(p, 4, shift=(-2.0, 0.5, 1.3))
        assert _z_cv(t1, w1) == pytest.approx(_z_cv(t2, w2), abs=1e-9)
        assert _z_cv(t1, w1) == pytest.approx(_z_cv(ORI, p), abs=1e-9)


# ---- persistence and learned-CV consumers -------------------------------------------------------

def test_round_trip_through_save_and_load(tmp_path):
    t = orient_params_for_state(ORI, _rigid(POS, 5)[0])
    save_funnel_params(t, str(tmp_path / "fp"))
    got = load_funnel_params(str(tmp_path / "fp.npz"))
    for k in ("orient_index", "orient_k", "orient_theta_deg"):
        assert got[k] == t[k]
    for k in ("orient_ref", "orient_offset", "master_axis", "master_offset", "orient_rotation", "unbinding_axis"):
        assert np.array_equal(got[k], t[k])
    p = _rigid(_guest_frames()[2], 5)[0]
    assert _energy(got, p) == _energy(t, p)
    save_funnel_params(NEW, str(tmp_path / "pf"))
    assert funnel_mode(load_funnel_params(str(tmp_path / "pf.npz"))) == "protein"


def test_training_z_is_the_run_time_z_at_the_reference_orientation():
    from autopath.metadynamics.learned.pipeline import _cache_key, _funnel_geometry, _funnel_z_nm
    g = _funnel_geometry(ORI)
    frames = _guest_frames()
    z = _funnel_z_nm(np.stack(frames), g, MASSES)
    assert z == pytest.approx([_z_force(ORI, p) for p in frames], abs=1e-10)
    rot = np.stack([_rigid(p, s)[0] for s, p in enumerate(frames)])
    assert _funnel_z_nm(rot, g, MASSES) == pytest.approx(z, abs=1e-9)        # each frame in its own core frame
    args = ([], 1, "forward", __file__, __file__, [0], [1])
    keys = {_cache_key(*args, code_hash="a", classes="rupture", funnel=_funnel_geometry(fp), masses=MASSES)
            for fp in (ORI, NEW, orient_params_for_state(ORI, rot[0]))}
    assert len(keys) == 3


def test_check_funnel_match_compares_the_orientation_frame(tmp_path):
    fr = funnel_frame(ORI)
    path = str(tmp_path / "o.npz")
    np.savez(path, funnel_host=np.array(HOST), funnel_guest=np.array(GUEST), funnel_axis=ORI["unbinding_axis"],
             funnel_orient_index=np.array(fr["orient"]), funnel_orient_offset=fr["offset"],
             funnel_orient_ref=fr["ref"])
    check_funnel_match(path, ORI)
    for other in (NEW, orient_params_for_state(ORI, _rigid(POS, 1)[0])):
        with pytest.raises(ValueError, match="not supported"):
            check_funnel_match(path, other)


def test_orchestrator_z_monitor_honours_the_orientation_frame():
    from autopath.autopath_core import funnel_z_monitor
    spec = funnel_z_monitor(object(), ORI, "path")["funnel_z"]
    p = _guest_frames()[3]
    assert _cv_value(spec.resolve(p, len(MASSES), None).cv, p) == pytest.approx(_z_force(ORI, p), abs=1e-10)


def test_lab_and_protein_params_build_the_same_forces_as_before():
    for params, groups, names in ((LAB, 2, {"nx", "ny", "nz"}), (NEW, 5, set())):
        f = create_funnel_force_from_trajectory_analysis(params)
        got = {f.getGlobalParameterName(i) for i in range(f.getNumGlobalParameters())}
        assert f.getNumGroups() == groups and names <= got and not {"fo_vx", "fo_vy", "fo_vz"} & got
        s = _system()
        create_funnel_force_from_trajectory_analysis(params, system=s)
        assert s.getNumForces() == 1
