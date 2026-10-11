"""s_w composite clamp: OpenMM == numpy, zero force beyond the edges, spec wiring, COLVAR range."""
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest

from autopath.metadynamics import LinearComposite, MetadynamicsMD, ResidualComposite
from autopath.metadynamics.BoundaryCorrection import composite_bias_force
from autopath.metadynamics.learned.forces import (composite_cv, feature_cv, fused_sw_composite, funnel_z_force,
                                                  learned_cv_spec)

from .learned_toys import (AXIS, FEATS, GUEST, HOST, MU, SD, TABLES, W, X0, ZMAX, ZMIN, s_plain, s_ref,
                           toy_pdb, toy_system, write_npz_r2)

REF = mm.Platform.getPlatformByName("Reference")
X1 = X0 + np.random.default_rng(2).normal(0, 0.03, X0.shape)
EDGES = {"below": (0.3, 0.9), "inside": (-0.4, 0.4), "above": (-0.9, -0.3)}   # offsets from s at X1


def _ctx(composite, s):
    force, _ = composite_bias_force([SimpleNamespace(force=composite, periodic=False)])
    force.setEnergyFunction(force.getEnergyFunction().replace("table(s0)", "s0", 1))
    force.setForceGroup(5)
    s.addForce(force)
    return mm.Context(s, mm.VerletIntegrator(0.001), REF)


def _ef(ctx, X):
    ctx.setPositions(X)
    st = ctx.getState(getEnergy=True, getForces=True, groups={5})
    return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))


def _builders():
    def plain(s, clamp):
        c = composite_cv(FEATS, W, MU, SD, s)
        c.clamp = clamp
        return c

    def fused(s, clamp):
        c = fused_sw_composite(FEATS, W, MU, SD, s)
        c.clamp = clamp
        return c

    def resid(s, clamp):
        forces = [feature_cv(f, s, f"f{k}") for k, f in enumerate(FEATS)]
        return ResidualComposite(forces, W, MU, SD, z_force=funnel_z_force(HOST, GUEST, AXIS), z_min=ZMIN,
                                 z_max=ZMAX, tables=TABLES, clamp=clamp)
    return {"plain": (plain, s_plain), "fused": (fused, s_plain), "resid": (resid, s_ref)}


@pytest.mark.parametrize("path", ["plain", "fused", "resid"])
@pytest.mark.parametrize("kind", ["below", "inside", "above"])
def test_clamped_composite_matches_numpy_and_force_vanishes_beyond(path, kind):
    build, ref = _builders()[path]
    X = X1
    raw = ref(X)
    lo, hi = (raw + o for o in EDGES[kind])
    s1, s2 = toy_system(), toy_system()
    un = _ctx(build(s1, None), s1)
    cl = _ctx(build(s2, (lo, hi)), s2)
    e_un, f_un = _ef(un, X)
    e_cl, f_cl = _ef(cl, X)
    want = min(max(ref(X), lo), hi)
    assert e_cl == pytest.approx(want, abs=1e-9)
    if kind == "inside":
        assert np.allclose(f_cl, f_un, rtol=1e-10, atol=1e-10)
    else:
        assert np.abs(f_cl).max() < 1e-12
        assert np.abs(f_un).max() > 1e-6


def _feat_values(X):
    from autopath.metadynamics.learned.features import evaluate
    return evaluate(FEATS, X[None])[0]


def test_numpy_value_clamps_for_both_composites():
    f = _feat_values(X0)
    raw = float(np.sum(W * (f - MU) / SD))
    plain = LinearComposite([None] * 3, W, MU, SD, clamp=(raw - 1.0, raw - 0.5))
    assert plain.value(f) == raw - 0.5
    assert LinearComposite([None] * 3, W, MU, SD, clamp=(raw + 0.5, raw + 1.0)).value(f) == raw + 0.5
    assert LinearComposite([None] * 3, W, MU, SD).value(f) == raw
    s = toy_system()
    forces = [feature_cv(ft, s, f"f{k}") for k, ft in enumerate(FEATS)]
    from .learned_toys import z_of
    fz = np.r_[f, z_of(X0)]
    r = ResidualComposite(forces, W, MU, SD, z_force=funnel_z_force(HOST, GUEST, AXIS), z_min=ZMIN, z_max=ZMAX,
                          tables=TABLES, clamp=(s_ref(X0) + 0.2, s_ref(X0) + 0.9))
    assert r.value(fz) == pytest.approx(s_ref(X0) + 0.2)


def test_fused_and_per_feature_clamp_agree():
    raw = s_plain(X1)
    for kind in ("below", "inside", "above"):
        X = X1
        lo, hi = (raw + o for o in EDGES[kind])
        s1, s2 = toy_system(), toy_system()
        a = composite_cv(FEATS, W, MU, SD, s1)
        a.clamp = (lo, hi)
        b = fused_sw_composite(FEATS, W, MU, SD, s2)
        b.clamp = (lo, hi)
        ea, fa = _ef(_ctx(a, s1), X)
        eb, fb = _ef(_ctx(b, s2), X)
        assert eb == pytest.approx(ea, abs=1e-10)
        assert np.allclose(fa, fb, atol=1e-9)


def test_learned_cv_spec_sets_the_clamp_for_rupture_only(tmp_path):
    s = toy_system()
    for resid in (False, True):
        path = write_npz_r2(str(tmp_path / f"r{resid}.npz"), resid)
        c = learned_cv_spec(path, s, 1, fused=not resid)[0].resolve(None, 5, None).cv
        d = np.load(path)
        assert c.clamp == (float(d["sw_lo"]), float(d["sw_hi"]))
    r1 = tmp_path / "r1.npz"
    np.savez(r1, kept=np.arange(3), weights=np.c_[W, -W], mu=MU, sd=SD, grid_min=[-3.0, -3.0],
             grid_max=[3.0, 3.0], hill_width=[0.2, 0.2],
             **__import__("autopath.metadynamics.learned.features", fromlist=["x"]).features_to_dict(FEATS))
    for c in (learned_cv_spec(str(r1), s, 2)[k].resolve(None, 5, None).cv for k in range(2)):
        assert c.clamp is None


def test_md_run_never_leaves_the_clamp(tmp_path):
    path = write_npz_r2(str(tmp_path / "r.npz"), False)
    d = dict(np.load(path))
    c = s_plain(np.round(X0 * 10, 3) / 10)
    d["sw_lo"], d["sw_hi"] = c + 0.5, c + 1.5                  # walker starts below sw_lo
    d["sw_grid_min"], d["sw_grid_max"] = c - 0.1, c + 2.1
    np.savez(path, **d)
    top, pdb = toy_pdb(tmp_path)
    s = toy_system()
    specs = learned_cv_spec(path, s, 1)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[1], platform="Reference",
                        out_dir=str(tmp_path / "out"))
    md.run(pdb_file=pdb, system=s, run_id="w", cv_specs=specs, mMD_time=0.02, bias_factor=10,
           hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"), rct_ustride=1)
    colvar = np.load(tmp_path / "out" / "COLVAR_w.npy")
    hills = np.load(tmp_path / "out" / "HILLS_w.npz")
    for v in (colvar[:, -1], hills["cv"][:, -1]):
        assert v.min() >= d["sw_lo"] - 1e-9 and v.max() <= d["sw_hi"] + 1e-9
    assert colvar[0, -1] == pytest.approx(float(d["sw_lo"]))


def _run(d, s, lo=-1.0, hi=1.0):
    d.mkdir()
    np.save(d / "COLVAR_w.npy", np.c_[np.zeros(len(s)), s])
    np.savez(d / "HILLS_w.npz", cv_names=np.array(["funnel_z", "learned_sw"]))
    np.savez(d / "cv.npz", sw_lo=lo, sw_hi=hi)
    return str(d), str(d / "cv.npz")


def test_check_sw_range_counts_clamp_edges_as_in_range(tmp_path, caplog):
    import logging
    from autopath.metadynamics.report import check_sw_range
    s = np.linspace(-0.9, 0.9, 1000)
    s[:100] = np.float32(-1.0)                                  # float32 on GPU
    s[100:130] = np.float32(1.0) * (1 + 5e-8)
    out = check_sw_range(*_run(tmp_path / "a", s))
    assert out["fraction"] == 0.0
    assert out["frac_at_lo_clamp"] == pytest.approx(0.1) and out["frac_at_hi_clamp"] == pytest.approx(0.03)
    s2 = np.linspace(-0.9, 0.9, 1000)
    s2[:100] = -1.0
    with caplog.at_level(logging.WARNING, logger="autopath"):
        check_sw_range(*_run(tmp_path / "b", s2))
    assert "clamp" in caplog.text
    s3 = s.copy()
    s3[500] = 1.2
    assert check_sw_range(*_run(tmp_path / "c", s3))["fraction"] == pytest.approx(0.001)
