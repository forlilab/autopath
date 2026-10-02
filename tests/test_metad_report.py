import os

import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics import report as R
from autopath.metadynamics.Analysis import correct_fe_for_funnel

GAMMA = 15.0
S = np.linspace(0.0, 1.0, 125)
F = -50.0 * np.exp(-((S - 0.05) / 0.04) ** 2)           # bound well, flat unbound plateau


def _fp():
    rng = np.random.default_rng(0)
    return {"z_cc": 14.0 * unit.angstrom, "z_max": 26.0 * unit.angstrom,
            "R_cylinder": 4.0 * unit.angstrom,
            "axial_distances": np.r_[rng.uniform(1, 3, 200), rng.uniform(3, 20, 800)]}


def _bias(fes):
    return -((GAMMA - 1.0) / GAMMA) * fes


def _walker(d, run_id, frac=1.0, z=None, start=None, end=None, hist=True, n_snap=10, t_ns=5.0):
    """COLVAR, FUNNELZ and FEhist for one walker; start/end fix its wall-clock window."""
    z = np.r_[np.full(50, 2.0), np.full(50, 18.0), np.full(50, 2.0)] if z is None else z
    np.save(d / f"COLVAR_{run_id}.npy", (z / 26.0)[:, None])
    np.save(d / f"FUNNELZ_{run_id}.npy", z)
    if hist:
        snaps = np.array([F * frac * (k + 1) / n_snap for k in range(n_snap)])
        p = d / f"FEhist_{run_id}.npz"
        np.savez(p, fes_kj=snaps, time_ns=np.linspace(t_ns / n_snap, t_ns, n_snap),
                 started_at=np.array(start if start is not None else 1000.0))
        if end is not None:
            os.utime(p, (end, end))


def _shared(tmp_path, concurrent=True):
    d = tmp_path / "shared"
    d.mkdir()
    for k, rid in enumerate(("w1", "w2", "w3")):
        np.save(d / f"bias_{100 + k}_9.npy", _bias(F) / 3.0)          # partial self-biases
        if concurrent:
            _walker(d, rid, frac=1.0, start=1000.0, end=1600.0)
        else:
            _walker(d, rid, frac=1.0, start=1000.0 + 700 * k, end=1600.0 + 700 * k)
    return d


def _isolated(tmp_path):
    d = tmp_path / "isolated"
    d.mkdir()
    for k, (rid, depth, done) in enumerate((("m1", 1.0, True), ("m2", 1.2, True), ("m3", 0.9, False))):
        sub = d / f"bias_{rid}"
        sub.mkdir()
        np.save(sub / f"bias_{200 + k}_9.npy", _bias(F * depth))
        _walker(d, rid, frac=depth, hist=done)
    return d


def test_mode_detection(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    np.save(single / "bias_1_3.npy", _bias(F))
    _walker(single, "only")
    assert R.detect_mode(str(single)).layout == "single"
    assert R.detect_mode(str(_isolated(tmp_path))).layout == "isolated"
    conc = R.detect_mode(str(_shared(tmp_path, concurrent=True)))
    assert (conc.layout, conc.schedule) == ("shared", "concurrent")
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    for k, rid in enumerate(("a", "b")):
        np.save(seq_dir / f"bias_{k}_1.npy", _bias(F) / 2)
        _walker(seq_dir, rid, start=1000.0 + 700 * k, end=1600.0 + 700 * k)
    seq = R.detect_mode(str(seq_dir))
    assert (seq.layout, seq.schedule) == ("shared", "sequential")


def test_shared_report_uses_one_surface_and_one_correction(tmp_path):
    d = _shared(tmp_path)
    out = R.report(str(d), 300.0, GAMMA, 0.0, 1.0, "s", cv_length_per_unit_ang=20.0,
                   out_prefix=str(tmp_path / "rep"), funnel_params=_fp())
    expected = correct_fe_for_funnel(-(GAMMA / (GAMMA - 1.0)) * _bias(F), _fp(), temperature=300.0,
                                     grid_min=0.0, grid_max=1.0, cv_length_per_unit_ang=20.0)
    assert out["dG_bind_std_kj"] == pytest.approx(expected["dG_bind_std_kj_mol"])
    assert out["n_round_trips"] == 3                    # one per walker, on FUNNELZ
    for f in out["files"].values():
        assert os.path.exists(f)


def test_isolated_report_is_mean_and_sd_of_complete_walkers(tmp_path):
    d = _isolated(tmp_path)
    out = R.report(str(d), 300.0, GAMMA, 0.0, 1.0, "s", cv_length_per_unit_ang=20.0,
                   out_prefix=str(tmp_path / "iso"), funnel_params=_fp())
    per = {lab: dg for lab, dg, done in out["estimates"]}
    complete = [dg for lab, dg, done in out["estimates"] if done]
    assert set(per) == {"m1", "m2", "m3"} and len(complete) == 2      # m3 never finished
    assert out["dG_bind_std_kj"] == pytest.approx(np.mean(complete))
    assert out["dG_bind_std_sd_kj"] == pytest.approx(np.std(complete, ddof=1))


def test_concurrent_history_is_on_aggregate_time(tmp_path):
    d = _shared(tmp_path, concurrent=True)
    mode = R.detect_mode(str(d))
    est = R.build_estimates(str(d), mode, _fp(), 300.0, GAMMA, (0.0, 1.0), 20.0)[0]
    assert est.t_ns[-1] == pytest.approx(15.0)                       # 3 walkers x 5 ns
    assert np.all(np.diff(est.t_ns) >= 0)
    # walkers that ran together take their first snapshots at the same moment, when
    # the shared bias had seen 3 x 0.5 ns; laid end to end they would be 0.5, 1.0, 1.5
    np.testing.assert_allclose(est.t_ns[:3], 1.5)


def test_deposition_rate_falls_as_the_bias_converges():
    t = np.linspace(0.5, 20, 40)
    snaps = np.array([F * (1 - np.exp(-ti / 5.0)) for ti in t])     # bias saturates
    tr, rate = R._deposition_rate(t, snaps, (0.0, 1.0), GAMMA)
    assert rate.size and np.all(rate > 0)
    assert rate[-1] < 0.1 * rate[0]


def test_report_without_a_funnel_gives_the_uncorrected_well_depth(tmp_path):
    d = _shared(tmp_path)
    out = R.report(str(d), 300.0, GAMMA, 0.0, 1.0, "s", out_prefix=str(tmp_path / "nf"),
                   funnel_params=None)
    assert out["funnel"] is False and out["dG_bind_std_kj"] is None
    (label, depth), = out["well_depth_kj"]
    assert depth == pytest.approx(-50.0, abs=1.0)          # the synthetic well
    # no funnel axis: round trips use fractions of the CV grid (this trace stops at 69%)
    assert all(st["coordinate"] == "cv" for st in out["walker_stats"].values())
    assert out["n_round_trips"] == 0
    for f in out["files"].values():
        assert os.path.exists(f)


def test_sequential_walkers_are_replayed_in_start_order(tmp_path):
    """Sequential walkers are laid end to end by start time, not by name."""
    d = tmp_path / "seq_order"
    d.mkdir()
    np.save(d / "bias_0_1.npy", _bias(F))
    # wall-clock order c, a, b; the bias grows by one unit per walker
    for k, (rid, start) in enumerate((("c", 1000.0), ("a", 2000.0), ("b", 3000.0))):
        _walker(d, rid, frac=k + 1.0, start=start, end=start + 600.0)
    mode = R.detect_mode(str(d))
    assert (mode.layout, mode.schedule) == ("shared", "sequential")
    est = R.build_estimates(str(d), mode, _fp(), 300.0, GAMMA, (0.0, 1.0), 20.0)[0]
    depth = est.snapshots[9::10].min(axis=1)              # end of each walker
    assert np.all(np.diff(depth) < 0)                     # the well only deepens
