"""Rupture (B/TS/U) labels, rupture-class learn_cv and its report."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from autopath.metadynamics.learned.classes import RUPTURE_ORDER, rupture_labels
from autopath.metadynamics.learned.select import fisher
from autopath.metadynamics.learned.sw import z_spline_values

from .learned_toys import FUNNEL, RUPTURE_FEATS, pull_files, rupture_data, toy_system

DT = 0.05


def _trace(f_of_t, t_end=100.0, noise=0.0, seed=0):
    t = np.round(np.arange(0.0, t_end + DT / 2, DT), 6)
    f = f_of_t(t)
    if noise:
        f = f + np.random.default_rng(seed).normal(0.0, noise, len(t))
    return pd.DataFrame({"time": t, "force": f, "r_after": 0.8 + 0.02 * t})


def _single(t):
    # 0 until 30.5 ps, linear rise to 100 at 50.5 ps, drop to the drag plateau 10 at 56.5 ps
    return np.select([t < 30.5, t < 50.5, t < 56.5],
                     [0.0, 5.0 * (t - 30.5), 100.0 - 15.0 * (t - 50.5)], 10.0)


def _two(t):
    p1 = np.clip(100.0 - 100.0 * np.abs(t - 20.5) / 3.0, 0.0, None)            # 100 at 20.5 ps
    p2 = np.select([t < 44.5, t < 50.5, t < 56.5],
                   [0.0, 200.0 * (t - 44.5) / 6.0, 200.0 - 190.0 * (t - 50.5) / 6.0], 10.0)
    return p1 + p2


def _idx(lab, c):
    return np.flatnonzero(lab == c).tolist()


def test_single_peak_classes_and_windows_are_exact():
    lab, info = rupture_labels(_trace(_single), 101, window_ps=DT)      # one-row window: no smoothing
    assert info["reason"] == ""
    assert info["base_b"] == 0.0 and info["base_u"] == 10.0 and info["dF"] == pytest.approx(90.0)
    assert info["t_rise"] == pytest.approx(34.5, abs=0.1)
    assert info["t_peak"] == pytest.approx(50.5, abs=1e-9)
    assert info["t_slip"] == pytest.approx(53.5, abs=0.1)
    assert info["t_free"] == pytest.approx(55.9, abs=0.1)
    assert _idx(lab, "B") == list(range(35))
    assert _idx(lab, "TS") == [51, 52, 53]
    assert _idx(lab, "U") == list(range(56, 101))
    assert set(lab) == set(RUPTURE_ORDER) | {""}


def test_noisy_peak_is_smoothed():
    lab, info = rupture_labels(_trace(_single, noise=10.0, seed=1), 101)
    assert info["reason"] == ""
    assert 49.0 <= info["t_peak"] <= 52.0
    ts = _idx(lab, "TS")
    assert ts and min(ts) >= 50 and max(ts) <= 55
    assert max(_idx(lab, "B")) <= 36
    u = _idx(lab, "U")
    assert min(u) >= 54 and len(u) >= 40


def test_two_peaks_use_the_global_maximum():
    lab, info = rupture_labels(_trace(_two), 101, window_ps=DT)
    assert info["reason"] == ""
    assert info["t_peak"] == pytest.approx(50.5, abs=1e-9)
    assert _idx(lab, "TS") == [51, 52, 53]
    assert _idx(lab, "B") == list(range(19))          # B ends at the first rise, before the small peak


def test_late_peak_is_excluded():
    lab, info = rupture_labels(_trace(lambda t: 2.0 * t), 101, window_ps=DT)
    assert info["reason"] == "peak in the last 10% of the pull"
    assert (lab == "").all()


def test_flat_trace_is_excluded():
    lab, info = rupture_labels(_trace(lambda t: np.full_like(t, 5.0)), 50, window_ps=DT)
    assert info["reason"] == "no force rise"
    assert (lab == "").all() and len(lab) == 50


def test_no_settling_is_excluded():
    # plateau 10 for most of the tail, but the last rows rise above the free threshold
    def f(t):
        return np.where(t > 99.0, 40.0, _single(t))
    lab, info = rupture_labels(_trace(f), 101, window_ps=DT)
    assert info["base_u"] == 10.0                     # 20 of 201 tail rows at 40: the median stays 10
    assert info["reason"] == "force never settles to the unbound baseline"
    assert (lab == "").all()


def test_frame_mapping_uses_time_fraction_and_reads_dat_files(tmp_path):
    df = _trace(_single)
    p = tmp_path / "sMD_replica-1_v0.01_forward.dat"
    with open(p, "w") as fh:
        fh.write("# spring_constant_kJ_mol_nm2=3765.6\n# steps_per_move=12\n")
        df.assign(step=np.arange(len(df))).to_csv(fh, index=False)
    lab, info = rupture_labels(str(p), 11, window_ps=DT)                # 11 frames at t = 0, 10, ..., 100
    assert lab.tolist() == ["B"] * 4 + ["", ""] + ["U"] * 5
    assert info["t_peak"] == pytest.approx(50.5)


def test_start_of_pull_spike_is_not_the_rupture():
    def f(t):
        return _single(t) + 500.0 * (t < 0.12)
    lab, info = rupture_labels(_trace(f), 101, window_ps=DT)
    assert info["reason"] == ""
    assert info["t_peak"] == pytest.approx(50.5, abs=1e-9)
    assert info["base_b"] == 0.0
    assert _idx(lab, "TS") == [51, 52, 53]
    assert _idx(lab, "B") == list(range(35))
    assert _idx(lab, "U") == list(range(56, 101))


def test_wide_start_spike_with_default_window_matches_spike_free_trace():
    lab0, info0 = rupture_labels(_trace(_single), 101)
    lab, info = rupture_labels(_trace(lambda t: _single(t) + 500.0 * (t < 0.5)), 101)
    assert info["reason"] == "" and info0["reason"] == ""
    assert info["base_b"] == 0.0
    assert lab.tolist() == lab0.tolist()
    for k in ("t_rise", "t_peak", "t_slip", "t_free"):
        assert info[k] == pytest.approx(info0[k])


def test_too_short_trace_raises():
    df = pd.DataFrame({"time": [0.0], "force": [1.0]})
    with pytest.raises(ValueError, match="too few rows"):
        rupture_labels(df, 5)


def _run_learn(monkeypatch, tmp_path, data):
    from autopath.metadynamics.learned import pipeline
    V, frames, lab, z, diag = data
    pulls = [pull_files(tmp_path)]
    (tmp_path / "t.pdb").write_text("pdb")
    (tmp_path / "l.sdf").write_text("sdf")
    monkeypatch.setattr(pipeline, "find_pulls", lambda *a: pulls)
    monkeypatch.setattr(pipeline, "_extract", lambda *a, **k: (RUPTURE_FEATS, V, frames, lab, None, z, diag))
    out = str(tmp_path / "out")
    path = pipeline.learn_cv(str(tmp_path / "t.pdb"), str(tmp_path / "l.sdf"), [0, 1], [2, 3, 4],
                             str(tmp_path), out, hidden=False, classes="rupture", funnel=FUNNEL,
                             masses=[12.0] * 5, mosaic_resolution=0.5)
    return path, out


def test_learn_cv_rupture_writes_s_w_and_report(monkeypatch, tmp_path):
    V, frames, lab, z, diag = data = rupture_data()
    path, out = _run_learn(monkeypatch, tmp_path, data)
    d = np.load(path, allow_pickle=False)
    kept = d["kept"]
    X = (V[:, kept] - d["mu"]) / d["sd"]
    if bool(d["residualised"]):
        X = X - z_spline_values(d["z_tables"], float(d["z_min"]), float(d["z_max"]), z)
    s = X @ d["sw_weights"]
    m = lab != ""
    assert int(d["component"]) in (0, 1) and d["sw_weights"].shape == (len(kept),)
    assert s[lab == "B"].mean() < s[lab == "TS"].mean()
    assert fisher(s[m], lab[m] == "TS") > 1.0
    hw = float(d["sw_hill_width"])
    assert hw == pytest.approx(0.5 * s[lab == "B"].std())
    assert float(d["sw_lo"]) == pytest.approx(s[m].min()) and float(d["sw_hi"]) == pytest.approx(s[m].max())
    assert float(d["sw_grid_min"]) == pytest.approx(float(d["sw_lo"]) - 3 * hw)
    assert float(d["sw_grid_max"]) == pytest.approx(float(d["sw_hi"]) + 3 * hw)
    assert list(d["funnel_host"]) == [2, 3] and np.isfinite(float(d["corr_sz"]))
    assert list(d["class_order"]) == ["B", "TS", "U"]
    rep = json.load(open(os.path.join(out, "rupture_report.json")))
    assert rep["class_counts"] == {"B": 1350, "TS": 450, "U": 1080}
    assert rep["class_frames_per_pull_median"] == {"B": 150, "TS": 50, "U": 120}
    assert rep["n_pulls"] == 10 and rep["n_pulls_labelled"] == 9
    assert rep["excluded"] == {"no slip after the peak": 1}
    assert rep["B_r_median"] < rep["ts_r_median"] < rep["U_r_median"]
    assert rep["residualised"] == bool(d["residualised"])
    assert rep["v0_barrier"]["status"] == "skipped" and "analysis_traces" in rep["v0_barrier"]["reason"]
    assert os.path.exists(os.path.join(out, "report.png"))


def test_rupture_npz_stores_basins_in_z_nm_and_s_w(monkeypatch, tmp_path):
    V, frames, lab, z, diag = data = rupture_data()
    path, _ = _run_learn(monkeypatch, tmp_path, data)
    d = np.load(path, allow_pickle=False)
    bB, bU = d["basin_B"], d["basin_U"]
    assert bB.shape == bU.shape == (2,) and np.isfinite(bB).all() and np.isfinite(bU).all()
    assert bB[0] < bU[0]
    assert bB[0] == pytest.approx(np.median(z[lab == "B"])) and bU[0] == pytest.approx(np.median(z[lab == "U"]))
    X = (V[:, d["kept"]] - d["mu"]) / d["sd"]
    if bool(d["residualised"]):
        X = X - z_spline_values(d["z_tables"], float(d["z_min"]), float(d["z_max"]), z)
    s = X @ d["sw_weights"]
    assert bB[1] == pytest.approx(np.median(s[lab == "B"])) and bU[1] == pytest.approx(np.median(s[lab == "U"]))


def test_rupture_npz_builds_the_bias_variable(monkeypatch, tmp_path):
    from autopath.metadynamics import LinearComposite
    from autopath.metadynamics.learned.forces import is_rupture_cv, learned_cv_spec
    path, _ = _run_learn(monkeypatch, tmp_path, rupture_data())
    assert is_rupture_cv(path)
    spec = learned_cv_spec(path, toy_system(), 1)[0]
    assert spec.name == "learned_sw"
    assert isinstance(spec.resolve(None, 5, None).cv, LinearComposite)


def test_learn_cv_rupture_too_few_ts_frames_names_the_class(monkeypatch, tmp_path):
    V, frames, lab, z, diag = rupture_data()
    lab = lab.copy()
    lab[lab == "TS"] = ""
    lab[200] = "TS"
    with pytest.raises(ValueError, match=r"class 'TS' has 1 frame.*'B': 1350"):
        _run_learn(monkeypatch, tmp_path, (V, frames, lab, z, diag))


def test_learn_cv_rupture_needs_funnel_and_masses(tmp_path):
    from autopath.metadynamics.learned.pipeline import learn_cv
    with pytest.raises(ValueError, match="needs funnel"):
        learn_cv("t", "l", [0], [1], "d", str(tmp_path), classes="rupture", masses=[1.0])
    with pytest.raises(ValueError, match="needs funnel"):
        learn_cv("t", "l", [0], [1], "d", str(tmp_path), classes="rupture", funnel=FUNNEL)
    with pytest.raises(ValueError, match="not supported"):
        learn_cv("t", "l", [0], [1], "d", str(tmp_path), classes="r", funnel=FUNNEL)


def test_cache_key_covers_exclusions_and_funnel(tmp_path):
    from autopath.metadynamics.learned.pipeline import _cache_key, _funnel_geometry
    pull = pull_files(tmp_path)
    (tmp_path / "t.pdb").write_text("pdb")
    (tmp_path / "l.sdf").write_text("sdf")
    args = ([pull], 4, "forward", str(tmp_path / "t.pdb"), str(tmp_path / "l.sdf"), [1], [2])
    g = _funnel_geometry(FUNNEL)

    def key(**kw):
        base = dict(code_hash="a", classes="rupture", exclude_families=("F3", "F5"), funnel=g,
                    masses=[12.0] * 5)
        base.update(kw)
        return _cache_key(*args, **base)

    k = key()
    assert k != key(exclude_families=("F3",))
    assert k != key(funnel=_funnel_geometry(dict(FUNNEL, unbinding_axis=[0.0, 1.0, 0.0])))
    assert k != key(masses=[12.0, 12.0, 13.0, 12.0, 12.0])                 # a host mass changed
    assert k == key(masses=[12.0] * 4 + [99.0])                            # atom 4 is neither host nor guest
    assert k != _cache_key(*args, code_hash="a")                           # classes "r", no funnel


def test_rupture_report_single_speed_and_multimodality():
    from autopath.metadynamics.learned.pipeline import multimodal, rupture_report
    V, frames, lab, z, diag = rupture_data(speeds=(0.01,))
    rep = rupture_report(frames, lab, z, V, [f"f{k}" for k in range(5)], diag)
    assert np.isnan(rep["ts_speed_max_smd"]) and list(rep["ts_speed_means"]) == ["0.01"]
    rng = np.random.default_rng(0)
    assert multimodal(np.r_[rng.normal(0, 0.1, 500), rng.normal(2, 0.1, 500)])[0] is True
    assert multimodal(rng.normal(0, 1, 2000))[0] is False
    assert multimodal(np.arange(5.0))[0] is None


def test_rupture_report_speed_disagreement():
    from autopath.metadynamics.learned.pipeline import rupture_report
    V, frames, lab, z, diag = rupture_data()
    names = [f"f{k}" for k in range(5)]
    assert rupture_report(frames, lab, z, V, names, diag)["ts_speed_max_smd"] < 0.5
    V = V.copy()
    V[(lab == "TS") & (frames.speed.values == 0.02), 0] += 3.0
    assert rupture_report(frames, lab, z, V, names, diag)["ts_speed_max_smd"] > 1.5


def _write_pmfs(d, peaks):
    """mixture_pmfs.csv with a v->0 (speed 0) profile per estimator; peaks = {estimator: r of the barrier or None}."""
    r = np.arange(0.8, 2.6, 0.01)
    rows = []
    for est, rp in peaks.items():
        dG = 40.0 * np.exp(-0.5 * ((r - rp) / 0.08) ** 2) if rp is not None else 30.0 * (r - 0.8)
        rows.append(pd.DataFrame({"step": np.arange(len(r)), "speed": 0.0, "estimator": est,
                                  "r_coord": r, "dG": dG}))
        rows.append(pd.DataFrame({"step": np.arange(len(r)), "speed": 0.01, "estimator": est,
                                  "r_coord": r, "dG": dG + 100.0}))        # per-speed rows are ignored
    os.makedirs(d, exist_ok=True)
    pd.concat(rows).to_csv(os.path.join(d, "mixture_pmfs.csv"), index=False)


def test_rupture_report_v0_barrier_compares_r_ts_with_ts_frames(tmp_path):
    from autopath.metadynamics.learned.pipeline import rupture_report
    V, frames, lab, z, diag = rupture_data()
    ts_r = np.percentile(frames.r.values[lab == "TS"], [5, 25, 50, 75, 95])
    _write_pmfs(str(tmp_path), {"cumulant": float(ts_r[2]), "jarzynski": 2.2, "force": None})
    rep = rupture_report(frames, lab, z, V, [f"f{k}" for k in range(5)], diag, analysis_dir=str(tmp_path))
    b = rep["v0_barrier"]
    assert b["status"] == "ok"
    assert b["ts_r_quantiles"] == pytest.approx(ts_r.tolist())
    c = b["estimators"]["cumulant"]
    assert c["r_ts"] == pytest.approx(ts_r[2], abs=0.02)
    assert c["within_ts_iqr"] is True and 0.4 < c["ts_r_percentile"] < 0.6
    assert b["estimators"]["jarzynski"]["within_ts_iqr"] is False
    assert b["estimators"]["jarzynski"]["ts_r_percentile"] == 1.0
    assert b["estimators"]["force"]["r_ts"] is None
    json.dumps(rep)


def test_rupture_report_v0_barrier_skips_with_a_reason(tmp_path):
    from autopath.metadynamics.learned.pipeline import rupture_report
    V, frames, lab, z, diag = rupture_data()
    names = [f"f{k}" for k in range(5)]
    b = rupture_report(frames, lab, z, V, names, diag, analysis_dir=str(tmp_path / "nope"))["v0_barrier"]
    assert b["status"] == "skipped" and "mixture_pmfs.csv" in b["reason"]
    os.makedirs(tmp_path / "x")
    pd.DataFrame({"step": [0], "speed": [0.01], "estimator": ["cumulant"], "r_coord": [1.0], "dG": [0.0]}
                 ).to_csv(tmp_path / "x" / "mixture_pmfs.csv", index=False)
    b = rupture_report(frames, lab, z, V, names, diag, analysis_dir=str(tmp_path / "x"))["v0_barrier"]
    assert b["status"] == "skipped" and "v->0" in b["reason"]


TRAJ = "examples/04_Full-Pipeline/3ptb/sMD/trajectories"
PDB3 = "examples/04_Full-Pipeline/3ptb/system.pdb"
EQ3 = "examples/04_Full-Pipeline/3ptb/equilibration/3ptb_equilibrated.pdb"
XML3 = "examples/04_Full-Pipeline/3ptb/system.xml"
FUNNEL3 = "examples/04_Full-Pipeline/3ptb/funnel_params_narrow_k100.npz"
needs_3ptb = pytest.mark.skipif(not all(os.path.exists(p) for p in (TRAJ, PDB3, EQ3, XML3, FUNNEL3)),
                                reason="3ptb sMD data not on disk")


@pytest.fixture(scope="module")
def rupture_3ptb(tmp_path_factory):
    import MDAnalysis as mda
    import openmm as mm
    from openmm import unit
    from autopath.autopath_core import learned_cv_atom_sets
    from autopath.metadynamics.learned.pipeline import learn_cv
    lig, pocket = learned_cv_atom_sets(PDB3, "resname UNK", positions=mda.Universe(EQ3).atoms.positions)
    s = mm.XmlSerializer.deserialize(open(XML3).read())
    masses = [s.getParticleMass(i).value_in_unit(unit.dalton) for i in range(s.getNumParticles())]
    out = str(tmp_path_factory.mktemp("rupture3ptb"))
    path = learn_cv(PDB3, "examples/data/3ptb.sdf", lig, pocket, TRAJ, out, stride=4, hidden=False,
                    classes="rupture", funnel=FUNNEL3, masses=masses)
    return path, out


@pytest.mark.learned_slow
@needs_3ptb
def test_learn_cv_rupture_on_3ptb(rupture_3ptb, caplog):
    import openmm as mm
    from autopath.metadynamics.learned.forces import learned_cv_spec
    path, out = rupture_3ptb
    rep = json.load(open(os.path.join(out, "rupture_report.json")))
    assert rep["n_pulls_labelled"] >= 0.8 * rep["n_pulls"], rep["excluded"]
    assert rep["B_r_median"] < rep["ts_r_median"] < rep["U_r_median"]
    d = np.load(path, allow_pickle=False)
    assert not any(str(n).startswith(("F3", "F5")) for n in d["names"])
    assert np.isfinite(d["sw_weights"]).all() and np.isfinite(float(d["sw_lo"])) and np.isfinite(float(d["sw_hi"]))
    assert 0 <= int(d["component"]) < 2 and np.isfinite(d["basin_B"]).all() and np.isfinite(d["basin_U"]).all()
    spec = learned_cv_spec(path, mm.XmlSerializer.deserialize(open(XML3).read()), 1)[0]
    assert spec.resolve(None, 0, None).cv is not None and spec.grid_min < spec.grid_max
    print("SLOW_NUMBERS", json.dumps({k: rep[k] for k in (
        "n_pulls", "n_pulls_labelled", "excluded", "class_counts", "class_frames_per_pull_median", "kept",
        "component", "corr_sz", "corr_sz_resid", "residualised", "ts_r_median", "ts_r_iqr", "ts_z_median",
        "B_r_median", "U_r_median", "v0_barrier", "sw_lo", "sw_hi", "sw_hill_width", "cv_acc")}))
