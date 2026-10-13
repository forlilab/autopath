import numpy as np
import pytest
from autopath.metadynamics.summary.data import load_run
from ._summary_fixtures import make_run


def test_shared_two_walkers(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), layout="shared", n_walkers=2))
    assert run.layout == "shared" and [w.rid for w in run.walkers] == ["w0", "w1"]
    assert [c.name for c in run.cvs] == ["funnel_z"] and run.cvs[0].ang_per_unit == 10.0
    w = run.walkers[0]
    assert w.cvs["funnel_z"].shape == w.t.shape == w.bias.shape == w.ct.shape


def test_isolated_layout(tmp_path):
    assert load_run(make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3)).layout == "isolated"


def test_single_walker(tmp_path):
    assert load_run(make_run(str(tmp_path / "r"), n_walkers=1)).layout == "single"


def test_extension_segments_joined(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=1, extend=True, n_rows=400))
    w = run.walkers[0]
    assert len(w.t) == 2 * 399 and np.all(np.diff(w.t) > 0)       # last partial row of each segment dropped
    assert w.segments == [pytest.approx(3.99, abs=1e-9)]            # 399 rows x 0.01 ns
    assert np.all(np.diff(w.hill_t) > 0)


def test_monitored_cv_loaded(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("dist",), monitors=("funnel_z",)))
    assert run.monitored == ["funnel_z"] and "funnel_z" in run.walkers[0].cvs


def test_2d_cvs(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=()))
    assert [c.name for c in run.cvs] == ["funnel_z", "learned_sw"]
    assert set(run.walkers[0].cvs) == {"funnel_z", "learned_sw"}


def test_truncated_walker_skipped(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=2, truncate=("w1", "reweight-empty")))
    assert [w.rid for w in run.walkers] == ["w0"] and any("w1" in s for s in run.skipped)


def test_length_mismatch_truncates(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=2, truncate=("w1", "length")))
    w = [w for w in run.walkers if w.rid == "w1"][0]
    assert len(w.t) == len(w.cvs["funnel_z"]) == len(w.bias)


def test_no_reweight_files(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), reweight=False))
    assert run.has_reweight is False and run.walkers[0].bias is None


def test_tempering_and_temperature(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), tempering="wt+tt"))
    assert run.tempering == "wt+tt" and run.temperature == 300.0 and run.bias_factor == 20.0


def test_record_frame_mapping(tmp_path):
    w = load_run(make_run(str(tmp_path / "r"))).walkers[0]
    # record every 0.01 ns, DCD every 0.05 ns: rows 5, 10, ... map to frames 0, 1, ...
    assert w.record_frame[5] == 0 and w.record_frame[10] == 1 and w.record_frame[3] == -1


def test_empty_dir_raises(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(ValueError, match="no HILLS"):
        load_run(str(tmp_path / "e"))


def _funnel_npz(path):
    from openmm import unit
    from autopath.metadynamics.Funnel import save_funnel_params
    A = unit.angstrom
    p = {"host_index": [0], "guest_index": [1], "k_xy": 10.0 * unit.kilocalorie_per_mole / A**2,
         "z_cc": 14.0 * A, "z_max": 26.0 * A, "alpha": 45.0 * unit.degrees, "R_cylinder": 4.0 * A,
         "unbinding_axis": np.array([0.0, 0.0, 1.0]), "axial_distances": np.array([4.0, 4.0]),
         "com_trajectory": np.zeros((2, 3)), "host_com_trajectory": np.zeros((2, 3)),
         "guest_com_trajectory": np.zeros((2, 3)), "radial_distances": np.zeros(2),
         "trajectory_length": 2, "force_group": 10}
    save_funnel_params(p, str(path))
    return str(path) + ".npz"


def test_funnel_from_hills_metadata(tmp_path):
    f = _funnel_npz(tmp_path / "fp")
    run = load_run(make_run(str(tmp_path / "r"), extra_meta={"funnel_file": f}))
    assert run.funnel is not None and "z_cc" in run.funnel and not run.skipped


def test_missing_metadata_funnel_recorded(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), extra_meta={"funnel_file": str(tmp_path / "gone.npz")}))
    assert run.funnel is None and any("funnel_file" in s and "not found" in s for s in run.skipped)


def test_explicit_missing_funnel_raises(tmp_path):
    r = make_run(str(tmp_path / "r"))
    with pytest.raises(FileNotFoundError):
        load_run(r, funnel="missing.npz")


def test_explicit_funnel_accepts_path(tmp_path):
    from pathlib import Path
    f = _funnel_npz(tmp_path / "fp")
    assert load_run(make_run(str(tmp_path / "r")), funnel=Path(f)).funnel is not None


def test_learned_cv_from_metadata(tmp_path):
    f = tmp_path / "lcv.npz"
    np.savez(f, w=np.arange(3.0))
    run = load_run(make_run(str(tmp_path / "r"), extra_meta={"learned_cv_file": str(f)}))
    assert list(run.learned_cv["w"]) == [0.0, 1.0, 2.0]
    run = load_run(make_run(str(tmp_path / "r2"), extra_meta={"learned_cv_file": str(tmp_path / "gone.npz")}))
    assert run.learned_cv is None and any("learned_cv_file" in s for s in run.skipped)


def test_failed_segment_lists_later_segments(tmp_path):
    run = load_run(make_run(str(tmp_path / "r"), n_walkers=2, extend=True, truncate=("w1", "reweight-empty")))
    assert [w.rid for w in run.walkers] == ["w0"]
    assert any(s.startswith("w1:") for s in run.skipped)
    assert any(s.startswith("w1_x2:") and "earlier segment failed" in s for s in run.skipped)
