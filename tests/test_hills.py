import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from openmm.app.metadynamics import BiasVariable

from autopath.metadynamics import (BoundaryCorrectedMetadynamics, CVSpec, MetadynamicsMD,
                                   TransitionTemperedMetadynamics, bias_from_hills, fes_history,
                                   hills_record, load_hills)
from autopath.metadynamics.Hills import fes_scale

T = 300.0
PI = np.pi


def _variables(kind):
    def cv(expr):
        f = mm.CustomExternalForce(expr)
        f.addParticle(0, [])
        return f
    if kind == "1d":
        return [BiasVariable(cv("x"), 0.2, 3.6, 0.03, gridWidth=341)], [(-np.inf, 3.0)]   # cap only
    if kind == "2d":
        return ([BiasVariable(cv("x"), 0.0, 1.0, 0.05, gridWidth=81),
                 BiasVariable(cv("y"), -1.0, 1.0, 0.08, gridWidth=61)], [(0.0, 1.0), (-1.0, 1.0)])
    if kind == "periodic":
        return ([BiasVariable(cv("x"), -PI, PI, 0.3, periodic=True, gridWidth=73),
                 BiasVariable(cv("y"), -PI, PI, 0.2, periodic=True, gridWidth=91)], [None, None])
    return [BiasVariable(cv("x"), 0.0, 1.0, 0.05, gridWidth=101)], [None]                  # off


def _meta(kind, cls=BoundaryCorrectedMetadynamics, bias_dir=None, **kw):
    s = mm.System()
    s.addParticle(12.0)
    variables, bounds = _variables(kind)
    if bias_dir:
        bias_dir.mkdir(exist_ok=True)
    extra = dict(saveFrequency=10, biasDir=str(bias_dir)) if bias_dir else {}
    meta = cls(s, variables, T * u.kelvin, 10, 1.0, 10, bounds=bounds, **extra, **kw)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    return meta, ctx


def _deposit(meta, ctx, n, rng, t0=0.0):
    for _ in range(n):
        p = [rng.uniform(v.minValue - 0.1, v.maxValue + 0.1) for v in meta.variables]
        meta._addGaussian(p, rng.uniform(0.1, 2.0) * u.kilojoule_per_mole, ctx)


def _specs(meta):
    return [CVSpec(cv=None, grid_min=v.minValue, grid_max=v.maxValue, hill_width=v.biasWidth,
                   grid_points=v.gridWidth, name=f"cv{i}") for i, v in enumerate(meta.variables)]


def _save(tmp_path, meta, run_id, tempering="wt", started_at=0.0):
    rec = hills_record(meta, run_id, started_at, 0.002, _specs(meta), tempering,
                       record_dt_ns=0.01, dcd_interval_ns=0.01)
    np.savez(tmp_path / f"HILLS_{run_id}.npz", **rec)
    return rec


@pytest.mark.parametrize("kind", ["1d", "2d", "periodic", "off"])
def test_rebuild_is_exactly_the_deposited_bias(tmp_path, kind):
    meta, ctx = _meta(kind)
    _deposit(meta, ctx, 300, np.random.default_rng(1))
    _save(tmp_path, meta, "w")
    (w,) = load_hills(tmp_path)
    assert w["hills_version"] == 2 and w["cv"].shape == (300, len(meta.variables))
    assert np.allclose(bias_from_hills([w]), meta._totalBias, rtol=0, atol=1e-8)


def test_shared_walkers_rebuild_to_the_synced_total(tmp_path):
    rng = np.random.default_rng(2)
    a, ca = _meta("2d", bias_dir=tmp_path / "b")
    b, cb = _meta("2d", bias_dir=tmp_path / "b")
    for _ in range(4):
        _deposit(a, ca, 20, rng)
        a._syncWithDisk()
        _deposit(b, cb, 20, rng)
        b._syncWithDisk()
    a._syncWithDisk()
    _save(tmp_path, a, "a")
    _save(tmp_path, b, "b")
    rebuilt = bias_from_hills(load_hills(tmp_path))
    assert np.allclose(rebuilt, a._totalBias, atol=1e-8)
    assert np.allclose(rebuilt, b._totalBias, atol=1e-8)


def _walker(tmp_path, run_id, wall, started_at, seed):
    meta, ctx = _meta("1d", bias_dir=tmp_path / "shared")
    _deposit(meta, ctx, len(wall), np.random.default_rng(seed))
    meta.hill_wall = list(wall)
    meta.hills = [(10000 * (i + 1), *row[1:]) for i, row in enumerate(meta.hills)]   # 0.02 ns per hill (2 fs steps)
    return _save(tmp_path, meta, run_id, started_at=started_at)


def test_history_ends_at_the_final_fes_and_follows_wall_clock(tmp_path):
    # "z" starts first but sorts last by name: deposition order must come from the clock
    _walker(tmp_path, "z_first", np.arange(100.0, 150.0), 100.0, 3)
    _walker(tmp_path, "a_second", np.arange(200.0, 250.0), 200.0, 4)
    walkers = load_hills(tmp_path)
    (h,) = fes_history(walkers, n=10)
    final = -fes_scale(walkers[0]) * bias_from_hills(walkers)
    assert np.allclose(h["fes"][-1], final, atol=1e-8)
    assert h["t_ns"][-1] == pytest.approx(2.0)                        # 1 ns + 1 ns, end to end
    mid = np.flatnonzero(h["wall_s"] < 200.0)[-1]                     # only "z_first" deposited
    only_z = [w for w in walkers if w["run_id"] == "z_first"]
    assert np.allclose(h["fes"][mid], -fes_scale(walkers[0]) * bias_from_hills(only_z), atol=1e-8)
    assert h["t_ns"][mid] == pytest.approx(1.0)
    assert np.all(np.diff(h["t_ns"]) >= 0)


def test_history_of_concurrent_walkers_cuts_at_wall_time(tmp_path):
    _walker(tmp_path, "w1", np.arange(0.0, 100.0, 2.0), 0.0, 5)       # interleaved deposition
    _walker(tmp_path, "w2", np.arange(1.0, 101.0, 2.0), 1.0, 6)
    walkers = load_hills(tmp_path)
    (h,) = fes_history(walkers, n=4)
    for c, t, f in zip(h["wall_s"], h["t_ns"], h["fes"]):
        assert np.allclose(f, -fes_scale(walkers[0]) * bias_from_hills(walkers, until=c), atol=1e-8)
        n = sum(int(np.sum(w["wall_s"] <= c)) for w in walkers)
        assert t == pytest.approx(0.02 * n)


def test_md_run_writes_v2_that_rebuilds_the_final_free_energy(tmp_path):
    from ._metad_toy import toy, x_spec
    top, s, pdb = toy(tmp_path)
    md = MetadynamicsMD(topology=top, ligand_atoms=[0], pocket_atoms=[0],
                        platform="Reference", out_dir=str(tmp_path))
    md.run(pdb_file=pdb, system=s, run_id="w", cv_specs=[x_spec()], mMD_time=0.04,
           bias_factor=20, hill_height=1.0, biasFrequency=2, bias_dir=str(tmp_path / "b"),
           tt_basins=[[0.7], [1.4]], tt_bias_factor=5)
    (w,) = load_hills(tmp_path)
    assert w["tempering"] == "wt+tt" and w["tt_bias_factor"] == 5 and len(w["wall_s"]) == 20
    assert w["bias_dir"].endswith("/b") and not np.isnan(w["bounds_lo"][0])     # edge-corrected
    fe = np.load(tmp_path / "FE_w.npy")
    assert np.allclose(-fes_scale(w) * bias_from_hills([w]), fe, atol=1e-8)


def test_old_hills_files_are_skipped(tmp_path):
    np.savez(tmp_path / "HILLS_old.npz", time_ns=np.zeros(1), cv=np.zeros((1, 1)), height_kj=np.ones(1))
    assert load_hills(tmp_path) == []


def test_in_progress_tmp_files_are_not_walkers(tmp_path):
    meta, ctx = _meta("1d")
    _deposit(meta, ctx, 20, np.random.default_rng(5))
    _save(tmp_path, meta, "w")
    (tmp_path / "HILLS_w.tmp.npz").write_bytes((tmp_path / "HILLS_w.npz").read_bytes())
    assert [w["run_id"] for w in load_hills(str(tmp_path))] == ["w"]
