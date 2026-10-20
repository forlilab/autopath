import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics import report as R
from autopath.metadynamics.Analysis import correct_fe_for_funnel
from autopath.metadynamics.Hills import load_hills

GAMMA = 15.0
S = np.linspace(0.0, 1.0, 125)
F = -50.0 * np.exp(-((S - 0.05) / 0.04) ** 2)           # bound well, flat unbound plateau
SIG = 0.04 / np.sqrt(2.0)                                # one hill at s = 0.05 is exactly this well
BIAS_DEPTH = 50.0 * (GAMMA - 1.0) / GAMMA                # total hill height giving F
T_NS = 5.0


def _fp():
    rng = np.random.default_rng(0)
    return {"z_cc": 14.0 * unit.angstrom, "z_max": 26.0 * unit.angstrom,
            "R_cylinder": 4.0 * unit.angstrom,
            "axial_distances": np.r_[rng.uniform(1, 3, 200), rng.uniform(3, 20, 800)]}


def _walker(d, run_id, bias_dir, walls, total, heights=None, finished=True, z=None):
    """COLVAR plus a HILLS v2 file: hills at s = 0.05 summing to ``total`` kJ/mol."""
    z = np.r_[np.full(50, 2.0), np.full(50, 24.0), np.full(50, 2.0)] if z is None else z
    np.save(d / f"COLVAR_{run_id}.npy", (z / 26.0)[:, None])
    n = len(walls)
    h = np.full(n, total / n) if heights is None else np.asarray(heights, float)
    np.savez(d / f"HILLS_{run_id}.npz", hills_version=2, run_id=run_id, finished=finished,
             bias_dir=str(bias_dir), started_at=float(walls[0]),
             time_ns=np.linspace(T_NS / n, T_NS, n), wall_s=np.asarray(walls, float),
             cv=np.full((n, 1), 0.05), height_kj=h,
             grid_min=np.array([0.0]), grid_max=np.array([1.0]), grid_points=np.array([125]),
             sigma=np.array([SIG]), periodic=np.array([False]),
             bounds_lo=np.array([np.nan]), bounds_hi=np.array([np.nan]),
             bias_factor=GAMMA, temperature=300.0, tempering="wt", cv_names=np.array(["s"]),
             cv_length_per_unit_ang=np.array([20.0]), origin=np.array([np.nan]),
             record_dt_ns=T_NS / len(z), dcd_interval_ns=0.05)


def _shared(tmp_path, concurrent=True, name="shared"):
    d = tmp_path / name
    d.mkdir()
    for k, rid in enumerate(("w1", "w2", "w3")):
        walls = 1000.0 + 60.0 * np.arange(10) + (k if concurrent else 700.0 * k)
        _walker(d, rid, d, walls, BIAS_DEPTH / 3.0)
    return d


def _isolated(tmp_path):
    d = tmp_path / "isolated"
    d.mkdir()
    for rid, depth, done in (("m1", 1.0, True), ("m2", 1.2, True), ("m3", 0.9, False)):
        _walker(d, rid, d / f"bias_{rid}", 1000.0 + 60.0 * np.arange(10), BIAS_DEPTH * depth,
                finished=done)
    return d


def test_mode_detection(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    _walker(single, "only", single, 1000.0 + np.arange(10), BIAS_DEPTH)
    assert R.detect_mode(load_hills(single)).layout == "single"
    assert R.detect_mode(load_hills(_isolated(tmp_path))).layout == "isolated"
    conc = R.detect_mode(load_hills(_shared(tmp_path, concurrent=True)))
    assert (conc.layout, conc.schedule) == ("shared", "concurrent")
    seq = R.detect_mode(load_hills(_shared(tmp_path, concurrent=False, name="seq")))
    assert (seq.layout, seq.schedule) == ("shared", "sequential")


def test_shared_report_uses_one_surface_and_one_correction(tmp_path):
    d = _shared(tmp_path)
    out = R.report(str(d), 300.0, 0.0, 1.0, cv_length_per_unit_ang=20.0,
                   funnel_params=_fp())
    expected = correct_fe_for_funnel(F, _fp(), temperature=300.0, grid_min=0.0, grid_max=1.0,
                                     cv_length_per_unit_ang=20.0)
    assert out["dG_bind_std_kj"] == pytest.approx(expected["dG_bind_std_kj_mol"])
    assert out["n_round_trips"] == 3                    # one per walker, on the CV
    assert out["pKd"] is not None and isinstance(out["convergence"]["converged"], bool)
    assert "files" not in out and not list(tmp_path.rglob("*.png"))


def test_diagnose_returns_the_report_numbers_and_writes_no_png(tmp_path):
    from autopath.metadynamics.Analysis import MetadynamicsAnalysis
    d = _shared(tmp_path)                                  # no funnel_params.npz: well depth
    out = MetadynamicsAnalysis(out_dir=str(d)).diagnose(
        temperature=300.0, grid_min=0.0, grid_max=1.0, cv_length_per_unit_ang=20.0)
    (label, depth), = out["well_depth_kj"]
    assert depth == pytest.approx(-50.0, abs=1.0)
    assert out["n_round_trips"] == 3 and out["convergence"]["verdict"]
    assert not list(tmp_path.rglob("*.png"))


def test_isolated_report_is_mean_and_sd_of_complete_walkers(tmp_path):
    d = _isolated(tmp_path)
    out = R.report(str(d), 300.0, 0.0, 1.0, cv_length_per_unit_ang=20.0,
                   funnel_params=_fp())
    per = {lab: dg for lab, dg, done in out["estimates"]}
    complete = [dg for lab, dg, done in out["estimates"] if done]
    assert set(per) == {"m1", "m2", "m3"} and len(complete) == 2      # m3 never finished
    assert out["dG_bind_std_kj"] == pytest.approx(np.mean(complete))
    assert out["dG_bind_std_sd_kj"] == pytest.approx(np.std(complete, ddof=1))


def test_concurrent_history_is_on_aggregate_time(tmp_path):
    walkers = load_hills(_shared(tmp_path, concurrent=True))
    est = R.build_estimates(walkers, _fp(), 300.0, (0.0, 1.0), 20.0)[0]
    assert est.t_ns[-1] == pytest.approx(3 * T_NS)
    assert np.all(np.diff(est.t_ns) >= 0)
    # at the first cut each walker has deposited one hill: 3 x 0.5 ns, not 0.5 end to end
    assert est.t_ns[0] == pytest.approx(1.5)
    assert np.allclose(est.snapshots[-1], F, atol=1e-9)


def test_report_without_a_funnel_gives_the_uncorrected_well_depth(tmp_path):
    d = _shared(tmp_path)
    out = R.report(str(d), 300.0, 0.0, 1.0, funnel_params=None)
    assert out["funnel"] is False and out["dG_bind_std_kj"] is None
    (label, depth), = out["well_depth_kj"]
    assert depth == pytest.approx(-50.0, abs=1.0)          # the synthetic well
    # no funnel axis: round trips use fractions of the CV grid (this trace reaches 92%)
    assert all(st["coordinate"] == "cv" for st in out["walker_stats"].values())
    assert out["n_round_trips"] == 3
    assert "files" not in out and not list(tmp_path.rglob("*.png"))


def test_sequential_walkers_are_replayed_in_start_order(tmp_path):
    """The history follows the wall clock (c, a, b), not the walker names."""
    d = tmp_path / "seq_order"
    d.mkdir()
    for k, (rid, start) in enumerate((("c", 1000.0), ("a", 2000.0), ("b", 3000.0))):
        _walker(d, rid, d, start + 60.0 * np.arange(10), BIAS_DEPTH * (k + 1) / 6.0)
    walkers = load_hills(d)
    mode = R.detect_mode(walkers)
    assert (mode.layout, mode.schedule) == ("shared", "sequential")
    est = R.build_estimates(walkers, _fp(), 300.0, (0.0, 1.0), 20.0)[0]
    after_c = np.flatnonzero(np.isclose(est.t_ns, T_NS))[0]       # c done, a not started
    assert est.snapshots[after_c].min() == pytest.approx(F.min() / 6.0)   # c alone, not a
    assert est.snapshots[-1].min() == pytest.approx(F.min())


def test_walker_without_hills_v2_is_not_reported(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    np.save(d / "COLVAR_x.npy", np.zeros((10, 1)))
    np.savez(d / "HILLS_x.npz", time_ns=np.zeros(1), cv=np.zeros((1, 1)), height_kj=np.ones(1))
    out = R.report(str(d), 300.0, 0.0, 1.0, funnel_params=None)
    assert out["estimates"] == [] and out["n_walkers"] == 0


def test_funnel_without_a_cv_length_tracks_the_well_depth(tmp_path):
    d = _shared(tmp_path)
    out = R.report(str(d), 300.0, 0.0, 1.0, funnel_params=_fp())   # no cv_length_per_unit_ang
    assert out["dG_bind_std_kj"] is None
    (label, depth), = out["well_depth_kj"]
    assert depth == pytest.approx(-50.0, abs=1.0)
    assert out["convergence"]["verdict"] and not list(tmp_path.rglob("*.png"))
