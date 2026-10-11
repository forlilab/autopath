"""Fused s_w forces: same value and forces as the per-feature composite."""
import sys
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest

from autopath.metadynamics import LinearComposite, MetadynamicsMD
from autopath.metadynamics.BoundaryCorrection import composite_bias_force
from autopath.metadynamics.CV import funnel_z_cv
from autopath.metadynamics.learned.features import Feature
from autopath.metadynamics.learned.forces import composite_cv, fused_sw_composite, fused_sw_forces, learned_cv_spec

from .learned_toys import AXIS, GUEST, HOST, X0, Z0, toy_pdb, toy_system, write_npz_r2

REF = mm.Platform.getPlatformByName("Reference")
BOX = 3.0
N = 12
# two contacts r0 values; pair (0, 5) in c1 and c2; (0, 6) in c1 and, reversed, in c4; (1, 4) at both r0
FEATS = [Feature("p1", "pair", "F1", pairs=[(0, 1), (2, 3)], r0=0.35),
         Feature("p2", "pair", "F1", pairs=[(0, 1)], r0=0.3),
         Feature("c1", "contacts", "F2", group_a=[0, 1], group_b=[4, 5, 6, 7], r0=0.3),
         Feature("c2", "contacts", "F2", group_a=[0], group_b=[5, 8, 9], r0=0.3),
         Feature("c3", "contacts", "F4", group_a=[1, 10], group_b=[4, 5, 8, 9, 11], r0=0.2),
         Feature("c4", "contacts", "F6", group_a=[6, 7], group_b=[0, 10], r0=0.3),
         Feature("d1", "centroid_distance", "F7", groups=[[0, 1], [4, 5]]),
         Feature("d2", "centroid_distance", "F7", groups=[[2], [3, 6]]),
         Feature("o", "cos", "F8", groups=[[0], [1], [2, 3], [4]])]
_RNG = np.random.default_rng(5)
W = _RNG.normal(size=len(FEATS))
MU = _RNG.normal(size=len(FEATS))
SD = _RNG.uniform(0.5, 2.0, size=len(FEATS))


def _system():
    s = mm.System()
    for _ in range(N):
        s.addParticle(12.0)
    s.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0), mm.Vec3(0, 0, BOX))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.2)
    for _ in range(N):
        nb.addParticle(0.0, 0.3, 0.0)
    nb.addException(1, 6, 0.0, 0.3, 0.0)                    # excluded from contacts
    s.addForce(nb)
    return s


def _configs():
    rng = np.random.default_rng(1)
    out = [rng.uniform(0, BOX, size=(N, 3)) for _ in range(3)]
    X = rng.uniform(0, BOX, size=(N, 3))
    X[0] = [2.95, 1.0, 1.0]
    X[5] = [0.05, 1.0, 1.0]                                 # (0, 5) across the x face
    X[6] = X[0] + [0.0, 1.1, 0.0]                           # (0, 6) in the r0 = 0.3 switching range
    X[4] = X[0] + [0.0, 0.0, 1.25]                          # (0, 4) beyond the r0 = 0.3 cutoff
    X[1] = [1.5, 2.9, 2.9]
    X[8] = [1.5, 0.62, 2.9]                                 # (1, 8): 0.72 across y, r0 = 0.2 switching
    X[9] = X[1] + [0.85, 0.0, 0.0]                          # (1, 9) beyond the r0 = 0.2 cutoff
    out.append(X)
    return out


def _energy_context(composite, s):
    force, _ = composite_bias_force([SimpleNamespace(force=composite, periodic=False)])
    force.setEnergyFunction(force.getEnergyFunction().replace("table(s0)", "s0", 1))   # energy = s_w
    force.setForceGroup(5)
    s.addForce(force)
    return mm.Context(s, mm.VerletIntegrator(0.001), REF)


def _ef(ctx, X):
    ctx.setPositions(X)
    st = ctx.getState(getEnergy=True, getForces=True, groups={5})
    return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))


def test_fused_s_w_matches_the_per_feature_composite():
    s1, s2 = _system(), _system()
    ref = _energy_context(composite_cv(FEATS, W, MU, SD, s1), s1)
    fus = _energy_context(fused_sw_composite(FEATS, W, MU, SD, s2), s2)
    for X in _configs():
        e1, f1 = _ef(ref, X)
        e2, f2 = _ef(fus, X)
        assert e2 == pytest.approx(e1, rel=1e-10, abs=1e-12)
        assert np.abs(f1).max() > 0
        assert np.allclose(f2, f1, rtol=1e-8, atol=1e-8 * np.abs(f1).max())


def test_the_constructed_config_hits_switching_cutoff_and_wrap():
    X = _configs()[-1]
    d = X[5] - X[0]
    assert np.linalg.norm(d) > 2.5 and np.linalg.norm(d - BOX * np.round(d / BOX)) == pytest.approx(0.1)
    r06, r04 = np.linalg.norm(X[6] - X[0]), np.linalg.norm(X[4] - X[0])
    assert 3.5 * 0.3 < r06 < 4 * 0.3 < r04
    d = X[8] - X[1]
    assert 3.5 * 0.2 < np.linalg.norm(d - BOX * np.round(d / BOX)) < 4 * 0.2 < np.linalg.norm(X[9] - X[1])


def test_shared_pair_weight_is_the_sum_of_its_features():
    # only (0, 5) inside the cutoff: s = (c1 + c2) * contact(0, 5) + const
    feats = [FEATS[2], FEATS[3]]
    w, mu, sd = np.array([0.7, -1.9]), np.array([0.3, 0.1]), np.array([1.5, 0.4])
    X = np.full((N, 3), 0.0)
    X[:, 0] = np.arange(N) * 0.24                           # 12 points evenly on x: every pair >= 0.24
    X[:, 1] = np.arange(N) * 1.37 % BOX
    X[:, 2] = np.arange(N) * 2.11 % BOX
    X[5] = X[0] + [0.0, 0.0, 0.25]
    others = [(i, j) for i in (0, 1) for j in (4, 5, 6, 7, 8, 9) if (i, j) != (0, 5)]
    for i, j in others:
        dd = X[j] - X[i]
        assert np.linalg.norm(dd - BOX * np.round(dd / BOX)) > 1.2
    s = _system()
    ctx = _energy_context(fused_sw_composite(feats, w, mu, sd, s), s)
    e, _ = _ef(ctx, X)
    c = w / sd
    assert e == pytest.approx((c[0] + c[1]) / (1 + (0.25 / 0.3) ** 6) - np.sum(c * mu), rel=1e-12)


def test_fused_force_count():
    forces, const = fused_sw_forces(FEATS, W, MU, SD, _system())
    # one per kind: pair, contacts (both r0 values), centroid_distance, cos
    assert len(forces) == 4
    assert const == pytest.approx(-np.sum(W / SD * MU), rel=1e-14)
    assert len(fused_sw_forces(FEATS[:2], W[:2], MU[:2], SD[:2], _system())[0]) == 1


def test_centroid_count_and_overlapping_contacts_groups_are_not_supported():
    cc = Feature("w", "centroid_count", "F5", groups=[[0, 1]], group_b=[2, 3, 4], r0=0.5)
    with pytest.raises(ValueError, match="centroid_count.*not supported in the fused s_w"):
        fused_sw_forces([FEATS[0], cc], W[:2], MU[:2], SD[:2], _system())
    ov = Feature("v", "contacts", "F2", group_a=[0, 1], group_b=[1, 2], r0=0.3)
    with pytest.raises(ValueError, match="not supported"):
        fused_sw_forces([ov], W[:1], MU[:1], SD[:1], _system())


def test_learned_cv_spec_fused_default_and_residualised_rejected(tmp_path):
    plain = write_npz_r2(str(tmp_path / "plain.npz"), residualised=False)
    resid = write_npz_r2(str(tmp_path / "resid.npz"), residualised=True)
    fused = learned_cv_spec(plain, toy_system(), 1)[0].resolve(None, 5, None).cv
    per = learned_cv_spec(plain, toy_system(), 1, fused=False)[0].resolve(None, 5, None).cv
    assert isinstance(fused, LinearComposite) and len(fused.forces) == 3 and len(per.forces) == 3
    with pytest.raises(ValueError, match="fused s_w does not support residualised features"):
        learned_cv_spec(resid, toy_system(), 1)
    assert learned_cv_spec(resid, toy_system(), 1, fused=False)[0].name == "learned_sw"


def _seeded(monkeypatch):
    mod = sys.modules[MetadynamicsMD.__module__]
    orig_int, orig_vel = mod.LangevinMiddleIntegrator, mm.Context.setVelocitiesToTemperature

    def integrator(*a):
        i = orig_int(*a)
        i.setRandomNumberSeed(1234)
        return i

    monkeypatch.setattr(mod, "LangevinMiddleIntegrator", integrator)
    monkeypatch.setattr(mm.Context, "setVelocitiesToTemperature", lambda self, T, *a: orig_vel(self, T, 77))


def _run_2d(tmp_path, fused):
    top, pdb = toy_pdb(tmp_path)
    zspec = funnel_z_cv(HOST, GUEST, AXIS, grid_min=Z0 - 1.0, grid_max=Z0 + 1.0, hill_width=0.05,
                        grid_points=81)
    sw = learned_cv_spec(write_npz_r2(str(tmp_path / "r2.npz"), residualised=False), toy_system(), 1,
                         n_plain_cvs=1, fused=fused)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[1], platform="Reference",
                        out_dir=str(tmp_path / "out"))
    md.run(pdb_file=pdb, system=toy_system(), run_id="w", cv_specs=[zspec, *sw], mMD_time=0.02,
           bias_factor=10, hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"), rct_ustride=1)
    return np.load(tmp_path / "out" / "COLVAR_w.npy"), dict(np.load(tmp_path / "out" / "HILLS_w.npz"))


def test_2d_metad_run_records_the_same_hills_and_colvar(tmp_path, monkeypatch):
    _seeded(monkeypatch)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    c1, h1 = _run_2d(tmp_path / "a", fused=False)
    c2, h2 = _run_2d(tmp_path / "b", fused=True)
    assert c1.shape == (11, 2) and np.ptp(c1[:, 1]) > 0
    assert np.allclose(c2, c1, rtol=1e-8, atol=1e-8)
    assert np.allclose(h2["cv"], h1["cv"], rtol=1e-8, atol=1e-8)
    assert np.allclose(h2["height_kj"], h1["height_kj"], rtol=1e-8, atol=1e-8)
