"""learned_z: 2D (funnel z, s_w) metad, COLVAR-z reweighting and orchestrator wiring."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app.metadynamics import BiasVariable

from autopath.metadynamics import (BoundaryCorrectedMetadynamics, MetadynamicsMD, ResidualComposite,
                                   TransitionTemperedMetadynamics)
from autopath.metadynamics.BoundaryCorrection import MAX_CVS, boundary_factor, composite_bias_force
from autopath.metadynamics.CV import funnel_z_cv
from autopath.metadynamics.learned.features import Feature, features_to_dict
from autopath.metadynamics.learned.forces import (feature_cv, funnel_z_force, learned_cv_spec,
                                                  tt_basins_from_classes)

from .learned_toys import (AXIS, FEATS, GUEST, HOST, MU, SD, TABLES, W, X0, Z0, ZMAX, ZMIN, s_plain, s_ref,
                           toy_pdb, toy_system, write_npz_r2, z_of)

REF = mm.Platform.getPlatformByName("Reference")
SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "04_Full-Pipeline" / "run_single_walker_control.py"


def _specs(tmp_path, residualised=True):
    zspec = funnel_z_cv(HOST, GUEST, AXIS, grid_min=Z0 - 1.0, grid_max=Z0 + 1.0, hill_width=0.05,
                        grid_points=81)
    sw = learned_cv_spec(write_npz_r2(str(tmp_path / "r2.npz"), residualised), toy_system(), 1,
                         n_plain_cvs=1, fused=not residualised)
    return [zspec, *sw]


def _run(tmp_path, specs, monkeypatch=None, spy=None, name=None, **kw):
    top, pdb = toy_pdb(tmp_path)
    if spy is not None:
        monkeypatch.setattr(sys.modules[MetadynamicsMD.__module__], name, spy)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[1], platform="Reference",
                        out_dir=str(tmp_path / "out"))
    md.run(pdb_file=pdb, system=toy_system(), run_id="w", cv_specs=specs, mMD_time=0.02, bias_factor=10,
           hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"), rct_ustride=1, **kw)


@pytest.mark.parametrize("residualised", [True, False])
def test_learned_z_2d_metad_records_both_variables(tmp_path, residualised):
    _run(tmp_path, _specs(tmp_path, residualised))
    colvar = np.load(tmp_path / "out" / "COLVAR_w.npy")
    hills = np.load(tmp_path / "out" / "HILLS_w.npz")
    assert colvar.shape == (11, 2) and hills["cv"].shape == (10, 2)
    assert [str(n) for n in hills["cv_names"]] == ["funnel_z", "learned_sw"]
    assert np.array_equal(colvar[1:], hills["cv"])
    Xp = np.round(X0 * 10, 3) / 10                       # PDB precision
    sw0 = s_ref(Xp) if residualised else s_plain(Xp)
    assert colvar[0] == pytest.approx([z_of(Xp), sw0], rel=1e-4)


def test_learned_z_2d_boundary_factor_reaches_the_s_w_bias(tmp_path, monkeypatch):
    made = []

    class Spy(BoundaryCorrectedMetadynamics):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    _run(tmp_path, _specs(tmp_path), monkeypatch, Spy, "BoundaryCorrectedMetadynamics")
    meta, = made
    zv, sv = meta.variables
    c = s_ref(X0)
    assert meta.bounds[1] == pytest.approx((c - 2.0, c + 2.0))
    assert meta.boundaryFactor.shape == (sv.gridWidth, zv.gridWidth)
    assert np.array_equal(meta.boundaryFactor, boundary_factor([zv, sv], meta.bounds))
    # s_w factor alone, outer-multiplied with the z one
    sw_only = boundary_factor([BiasVariable(sv.force, sv.minValue, sv.maxValue, sv.biasWidth,
                                            gridWidth=sv.gridWidth)], [meta.bounds[1]])
    assert sw_only.max() > 1.5 and meta.boundaryFactor.max() > sw_only.max()
    zg, sg = (np.linspace(v.minValue, v.maxValue, v.gridWidth) for v in (zv, sv))
    g = sum(h * np.exp(-0.5 * (sg[:, None] - ps) ** 2 / sv.biasWidth ** 2
                       - 0.5 * (zg[None, :] - pz) ** 2 / zv.biasWidth ** 2) for _, pz, ps, h in meta.hills)
    assert len(meta.hills) == 10
    assert np.allclose(meta._selfBias, g * meta.boundaryFactor, atol=1e-3)


def test_tt_on_a_composite_variable_in_2d(tmp_path, monkeypatch):
    made = []

    class Spy(TransitionTemperedMetadynamics):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    c = s_ref(X0)
    _run(tmp_path, _specs(tmp_path), monkeypatch, Spy, "TransitionTemperedMetadynamics",
         tt_basins=[[Z0 - 0.3, c - 1.0], [Z0 + 0.3, c + 1.0]], tt_bias_factor=5.0)
    meta, = made
    assert str(np.load(tmp_path / "out" / "HILLS_w.npz")["tempering"]) == "wt+tt"
    assert len(meta.vstar_log) == 10 and np.isfinite(meta.vstar_log).all()
    # grid axes run (s_w, z); link the basins with an L-shaped ridge, then cut it
    (j1, i1), (j2, i2) = meta.basins
    bias = np.full(meta._totalBias.shape, -5.0)
    bias[j1, min(i1, i2):max(i1, i2) + 1] = 7.0
    bias[min(j1, j2):max(j1, j2) + 1, i2] = 7.0
    meta._totalBias = bias.copy()
    assert meta.vstar() == pytest.approx(7.0)
    bias[j1, (i1 + i2) // 2] = -5.0
    meta._totalBias = bias
    assert meta.vstar() == pytest.approx(-5.0)


def test_tt_basins_from_classes_follow_the_bias_variable_order(tmp_path):
    specs = _specs(tmp_path)
    c = s_ref(X0)
    got = tt_basins_from_classes(str(tmp_path / "r2.npz"), specs)
    assert got == [pytest.approx([Z0 - 0.3, c - 1.0]), pytest.approx([Z0 + 0.3, c + 1.0])]
    with pytest.raises(ValueError, match="not supported"):
        tt_basins_from_classes(str(tmp_path / "r2.npz"), specs[::-1])
    with pytest.raises(ValueError, match="not supported"):
        tt_basins_from_classes(str(tmp_path / "r2.npz"), specs[1:])


def _control_script_error(*args):
    r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stderr


def test_control_script_tt_basins_from_classes_flag_validation():
    rc, err = _control_script_error("--cv", "learned_z", "--learned-cv-file", "x.npz",
                                    "--tt-basins", "1,2", "3,4", "--tt-basins-from-classes")
    assert rc == 2 and "--tt-basins" in err and "not allowed" in err
    rc, err = _control_script_error("--cv", "path", "--tt-basins-from-classes")
    assert rc == 2 and "learned_z" in err


def _mixed(order):
    s = toy_system()
    resid = ResidualComposite([feature_cv(f, s, f"f{k}") for k, f in enumerate(FEATS)], W, MU, SD,
                              z_force=funnel_z_force(HOST, GUEST, AXIS), z_min=ZMIN, z_max=ZMAX, tables=TABLES)
    plain = funnel_z_force(HOST, GUEST, AXIS)
    vs = {"r": SimpleNamespace(force=resid, periodic=False), "p": SimpleNamespace(force=plain, periodic=False)}
    return s, [vs[k] for k in order]


def _mixed_value(X, order):
    val = {"r": s_ref(X), "p": z_of(X)}
    return sum((k + 1) * val[o] for k, o in enumerate(order))


@pytest.mark.parametrize("order", ["pr", "rp"])
def test_mixed_residual_and_plain_composite_force_matches_numpy(order):
    s, vs = _mixed(order)
    force, cv_map = composite_bias_force(vs)
    n_res = len(FEATS) + 1
    if order == "pr":
        assert cv_map[0] == ("cv", 0) and cv_map[1][2] == list(range(1, n_res + 1))
    else:
        assert cv_map[0][2] == list(range(n_res)) and cv_map[1] == ("cv", n_res)
    head, rest = force.getEnergyFunction().split(";", 1)
    args = head[len("table("):-1].split(", ")
    force.setEnergyFunction(" + ".join(f"{k + 1}*{a}" for k, a in enumerate(args)) + ";" + rest)
    s.addForce(force)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), REF)

    def energy(X):
        ctx.setPositions(X)
        st = ctx.getState(getEnergy=True, getForces=True)
        return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
                st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))

    rng = np.random.default_rng(3)
    X = X0 + rng.normal(0, 0.03, X0.shape)
    e, f = energy(X)
    tether = 50 * ((X - X0) ** 2).sum()
    assert e - tether == pytest.approx(_mixed_value(X, order), rel=1e-6)
    h = 1e-5
    num = np.zeros_like(X)
    for i in range(X.shape[0]):
        for d in range(3):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, d] += h
            Xm[i, d] -= h
            num[i, d] = -(_mixed_value(Xp, order) - _mixed_value(Xm, order)) / (2 * h)
    num += -100 * (X - X0)                                   # tether
    assert f == pytest.approx(num, rel=2e-3, abs=2e-3)


def _npz_with_n_features(path, n):
    feats = [Feature(f"p{i}", "pair", "F1", pairs=[(0, 1)], r0=0.35) for i in range(n)]
    k = len(feats)
    np.savez(path, kept=np.arange(k), weights=np.ones((k, 2)), mu=np.zeros(k), sd=np.ones(k), component=1,
             sw_weights=np.ones(k), residualised=True, z_min=ZMIN, z_max=ZMAX, z_tables=np.zeros((k, 10)),
             funnel_host=np.array(HOST), funnel_guest=np.array(GUEST), funnel_axis=AXIS / np.linalg.norm(AXIS),
             sw_lo=-1.0, sw_hi=1.0, sw_grid_min=-2.0, sw_grid_max=2.0, sw_hill_width=0.2,
             basin_B=[0.0, -1.0], basin_U=[1.0, 1.0], **features_to_dict(feats))
    return path


def test_cv_budget_is_checked_at_spec_build_time(tmp_path):
    ok = _npz_with_n_features(str(tmp_path / "ok.npz"), MAX_CVS - 2)           # + z of the residual + funnel z
    assert len(learned_cv_spec(ok, toy_system(), 1, n_plain_cvs=1, fused=False)) == 1
    big = _npz_with_n_features(str(tmp_path / "big.npz"), MAX_CVS - 1)
    with pytest.raises(ValueError, match="collective variables.*not supported"):
        learned_cv_spec(big, toy_system(), 1, n_plain_cvs=1, fused=False)


@pytest.fixture
def reweight_stubs(monkeypatch):
    """Stub the free-energy correction and final bias; returns the list of W(z) arrays passed in."""
    from autopath.metadynamics import report as rep
    seen = []
    monkeypatch.setattr(rep, "correct_fe_for_funnel",
                        lambda fes, fp, **kw: seen.append(np.asarray(fes)) or {"dG_bind_std_kj_mol": -1.0})
    monkeypatch.setattr(rep, "_final_bias_at_frames", lambda out_dir, rid, s: np.zeros(len(s)))
    return seen


def _colvar_z_run(d, z, clamped=0, zmax=2.6):
    d.mkdir()
    rng = np.random.default_rng(1)
    z = np.r_[z, np.full(clamped, zmax)]
    base = dict(time_ns=np.linspace(0, 10, len(z)), bias_kj=np.zeros(len(z)), rct_kj=np.zeros(len(z)),
                record_dt_ns=10 / len(z))
    np.savez(d / "REWEIGHT_w.npz", **base)
    np.save(d / "COLVAR_w.npy", np.c_[z, rng.normal(size=len(z))])
    np.savez(d / "HILLS_w.npz", cv_names=np.array(["funnel_z", "learned_sw"]))
    return str(d)


def test_z_reweighted_dG_from_colvar_drops_clamped_frames(tmp_path, reweight_stubs):
    from autopath.metadynamics import report as rep
    seen = reweight_stubs
    from tests.test_z_reweight import _fp
    rng = np.random.default_rng(0)
    z = np.r_[rng.normal(0.1, 0.02, 3000), rng.uniform(1.0, 2.0, 3000)]
    free = rep.z_reweighted_dG(_colvar_z_run(tmp_path / "free", z), fp=_fp(), temperature=300.0, burn=0.0)
    clamp = rep.z_reweighted_dG(_colvar_z_run(tmp_path / "clamp", z, clamped=500), fp=_fp(),
                                temperature=300.0, burn=0.0)
    assert free["n_dropped"] == 0 and clamp["n_dropped"] == 500
    assert free["n_frames"] == clamp["n_frames"] == len(z)
    np.testing.assert_allclose(seen[0], seen[2])             # the pile of clamped frames never enters W(z)
    np.testing.assert_allclose(seen[1], seen[3])


def test_z_reweighted_dG_drops_float32_clamped_colvar_frames(tmp_path, reweight_stubs):
    from autopath.metadynamics import report as rep
    from tests.test_z_reweight import _fp
    zmax = 2.6495475863484215
    fp = _fp(z_max=zmax * 10.0)
    z = np.r_[np.linspace(0.1, 2.0, 100), np.full(7, zmax - 1e-3)]
    clamped = float(np.float32(zmax))
    assert clamped != zmax
    out = rep.z_reweighted_dG(_colvar_z_run(tmp_path / "f32", z, clamped=5, zmax=clamped), fp=fp,
                              temperature=300.0, burn=0.0)
    assert out["n_dropped"] == 5 and out["n_frames"] == 107


def test_check_funnel_match(tmp_path):
    from autopath.metadynamics.learned.forces import check_funnel_match
    npz = str(tmp_path / "l.npz")
    np.savez(npz, funnel_host=np.array([1, 2, 3]), funnel_guest=np.array([7, 8]),
             funnel_axis=np.array([0.0, 0.0, 2.0]))
    fp = {"host_index": [1, 2, 3], "guest_index": [7, 8], "unbinding_axis": np.array([0.0, 0.0, 1.0])}
    check_funnel_match(npz, fp)
    for bad in ({**fp, "guest_index": [7, 9]}, {**fp, "host_index": [1, 2]},
                {**fp, "unbinding_axis": np.array([0.0, 1.0, 0.0])}):
        with pytest.raises(ValueError, match="differs from --funnel-from.*not supported"):
            check_funnel_match(npz, bad)


def test_z_reweighted_dG_monitor_keeps_frames_at_z_max(tmp_path, reweight_stubs):
    from autopath.metadynamics import report as rep
    seen = reweight_stubs
    from tests.test_z_reweight import _fp
    z = np.r_[np.linspace(0.1, 2.0, 100), np.full(10, 2.6)]
    np.savez(tmp_path / "REWEIGHT_w.npz", time_ns=np.arange(110.0), bias_kj=np.zeros(110), rct_kj=np.zeros(110),
             monitor_funnel_z=z, record_dt_ns=1.0)
    np.save(tmp_path / "COLVAR_w.npy", z[:, None])
    out = rep.z_reweighted_dG(str(tmp_path), fp=_fp(), temperature=300.0, burn=0.0)
    assert out["n_frames"] == 110 and out["n_dropped"] == 0


def test_z_reweighted_dG_reads_a_biased_funnel_z_column_like_a_monitor(tmp_path, reweight_stubs):
    from autopath.metadynamics import report as rep
    seen = reweight_stubs
    from tests.test_z_reweight import _fp
    rng = np.random.default_rng(0)
    z = np.r_[rng.normal(0.1, 0.02, 3000), rng.uniform(1.0, 2.0, 3000)]
    mon, col = tmp_path / "mon", tmp_path / "col"
    mon.mkdir()
    base = dict(time_ns=np.linspace(0, 10, len(z)), bias_kj=np.zeros(len(z)), rct_kj=np.zeros(len(z)),
                record_dt_ns=10 / len(z))
    np.savez(mon / "REWEIGHT_w.npz", monitor_funnel_z=z, **base)
    np.save(mon / "COLVAR_w.npy", rng.normal(size=(len(z), 1)))
    a = rep.z_reweighted_dG(str(mon), fp=_fp(), temperature=300.0, burn=0.0)
    b = rep.z_reweighted_dG(_colvar_z_run(col, z), fp=_fp(), temperature=300.0, burn=0.0)
    assert a["n_frames"] == b["n_frames"] == len(z) and b["reason"] == ""
    np.testing.assert_array_equal(seen[0], seen[2])
    np.testing.assert_array_equal(seen[1], seen[3])


def test_learned_z_option_accepted_and_needs_funnel():
    from autopath.autopath_core import AutoPath
    assert AutoPath(pdb_path="examples/data/3ptb.pdb", do_fix_pdb=False, mMD_cv="learned_z",
                    mMD_use_funnel_potential=True).mMD_cv == "learned_z"
    with pytest.raises(ValueError, match="learned_z"):
        AutoPath(pdb_path="examples/data/3ptb.pdb", do_fix_pdb=False, mMD_cv="learned_z")
    with pytest.raises(ValueError, match="preseed"):
        AutoPath(pdb_path="examples/data/3ptb.pdb", do_fix_pdb=False, mMD_cv="learned_z",
                 mMD_use_funnel_potential=True, mMD_preseed_bias=True)


def test_learned_z_has_no_z_monitor():
    from autopath.autopath_core import funnel_z_monitor
    fp = {"host_index": [0, 1], "guest_index": [2], "unbinding_axis": [0.0, 0.0, 1.0]}
    assert funnel_z_monitor(object(), fp, "learned_z") is None
    assert list(funnel_z_monitor(object(), fp, "learned")) == ["funnel_z"]


def test_sw_range_fraction_on_synthetic_colvar():
    from autopath.metadynamics.report import sw_range_fraction
    s = np.r_[np.linspace(-1.0, 1.0, 990), np.full(6, 1.5), np.full(4, -2.0)]
    assert sw_range_fraction(s, -1.0, 1.0) == pytest.approx(0.01)
    assert sw_range_fraction(np.linspace(-1.0, 1.0, 50), -1.0, 1.0) == 0.0     # the edges are inside


def _sw_run(d, n_out, n=2000, lo=-1.0, hi=1.0):
    d.mkdir()
    s = np.linspace(lo, hi, n)
    s[:n_out] = hi + 0.5
    np.save(d / "COLVAR_w.npy", np.c_[np.zeros(n), s])
    np.savez(d / "HILLS_w.npz", cv_names=np.array(["funnel_z", "learned_sw"]))
    np.savez(d / "cv.npz", sw_lo=lo, sw_hi=hi)
    return str(d), str(d / "cv.npz")


def test_check_sw_range_warns_and_strict_raises(tmp_path, caplog):
    import logging
    from autopath.metadynamics.report import check_sw_range
    ok = check_sw_range(*_sw_run(tmp_path / "ok", 1))                          # 0.05 %: silent
    assert ok["fraction"] == pytest.approx(0.0005) and ok["n"] == 2000
    with caplog.at_level(logging.WARNING, logger="autopath"):
        warn = check_sw_range(*_sw_run(tmp_path / "warn", 10))                 # 0.5 %: warning only
    assert warn["fraction"] == pytest.approx(0.005) and "training range" in caplog.text
    assert check_sw_range(*_sw_run(tmp_path / "warn2", 10), strict=True)["fraction"] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="s_w left its training range: boundary correction invalid"):
        check_sw_range(*_sw_run(tmp_path / "bad", 40), strict=True)            # 2 %
    (tmp_path / "none").mkdir()
    np.savez(tmp_path / "none" / "cv.npz", sw_lo=-1.0, sw_hi=1.0)
    assert "learned_sw" in check_sw_range(str(tmp_path / "none"), str(tmp_path / "none" / "cv.npz"))["reason"]


def test_build_learned_z_cv_learns_rupture_classes_with_system_masses(monkeypatch):
    import autopath.metadynamics.learned as learned
    from autopath.autopath_core import build_learned_z_cv
    seen = {}
    monkeypatch.setattr(learned, "learn_cv", lambda *a, **k: seen.update(k, args=a) or "x.npz")
    monkeypatch.setattr(learned, "learned_cv_spec", lambda npz, system, n, **k: [npz, n, k])
    s = mm.System()
    for m in (12.0, 1.5, 16.0):
        s.addParticle(m)
    fp = {"host_index": [0], "guest_index": [2], "unbinding_axis": [0.0, 0.0, 1.0]}
    assert build_learned_z_cv(s, "sys", "l.sdf", [2], [0, 1], "t", "o", "forward", fp) == \
        ["x.npz", 1, {"n_plain_cvs": 1}]
    assert seen["classes"] == "rupture" and seen["funnel"] is fp and seen["hidden"] is False
    assert seen["masses"] == pytest.approx([12.0, 1.5, 16.0]) and seen["direction"] == "forward"
    assert seen["system"] is s
    assert seen["args"][:6] == ("sys/system.pdb", "l.sdf", [2], [0, 1], "t", "o")


def test_learned_cv_out_dir_name():
    from autopath.autopath_core import LEARNED_Z_DIRNAME, learned_cv_out_dir
    assert learned_cv_out_dir("s/sMD", None) == "s/sMD/learned_cv"
    assert learned_cv_out_dir("s/sMD", None, name="learned_cv_rupture") == "s/sMD/learned_cv_rupture"
    assert learned_cv_out_dir("s/sMD", "p7", name=LEARNED_Z_DIRNAME) == "s/sMD/learned_cv_rupture/p7"
