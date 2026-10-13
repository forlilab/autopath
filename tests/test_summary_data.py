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
