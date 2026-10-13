"""load_run / report on runs that are still being written, and walker consistency checks."""
import os

import numpy as np
import pytest

from autopath.metadynamics.summary import report
from autopath.metadynamics.summary.analysis import bands, binding
from autopath.metadynamics.summary.data import load_run
from ._summary_fixtures import add_fresh, make_run


@pytest.mark.parametrize("layout", ["shared", "isolated"])
@pytest.mark.parametrize("n_hills,n_rows", [(0, 1), (1, 1), (1, 2)])
def test_fresh_walker_is_skipped_and_report_runs(tmp_path, layout, n_hills, n_rows):
    d = make_run(str(tmp_path / "r"), layout=layout, n_walkers=2)
    bd = os.path.join(d, "bias_w2") if layout == "isolated" else None
    add_fresh(d, "w2", n_hills=n_hills, n_rows=n_rows, bias_dir=bd)
    run = load_run(d)
    assert [w.rid for w in run.walkers] == ["w0", "w1"]
    assert any(s.startswith("w2: no records yet") for s in run.skipped)
    s = report(d, out_dir=str(tmp_path / "o"), step_ns=0.5)
    assert set(s["walkers"]) == {"w0", "w1"} and any("w2: no records yet" in x for x in s["skipped_walkers"])


def test_fresh_extension_skipped_base_intact(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    ref = load_run(d).walkers[0]
    add_fresh(d, "w0_x2", n_hills=0, n_rows=1)
    run = load_run(d)
    w = run.walkers[0]
    assert w.rid == "w0" and w.segments == [] and len(w.t) == len(ref.t) == 399
    np.testing.assert_array_equal(w.cvs["funnel_z"], ref.cvs["funnel_z"])
    assert any(s.startswith("w0_x2: no records yet") for s in run.skipped)
    report(d, out_dir=str(tmp_path / "o"), step_ns=0.5)


def test_every_walker_empty_not_supported(tmp_path):
    d = str(tmp_path / "r")
    make_run(d, n_walkers=1)
    add_fresh(d, "w0", n_hills=0, n_rows=1, like="w0")
    with pytest.raises(ValueError, match="no usable records yet: not supported"):
        report(d, out_dir=str(tmp_path / "o"))


def test_lone_extension_without_base_skipped(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    os.rename(os.path.join(d, "HILLS_w1.npz"), os.path.join(d, "HILLS_w1_x2.npz"))
    run = load_run(d)
    assert [w.rid for w in run.walkers] == ["w0"]
    assert any(s.startswith("w1_x2:") and "base segment w1 missing" in s for s in run.skipped)


def test_cv_names_mismatch_not_supported(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    f = os.path.join(d, "HILLS_w1.npz")
    h = dict(np.load(f))
    h["cv_names"] = np.array(["other"])
    np.savez(f, **h)
    with pytest.raises(ValueError, match="cv_names differs.*not supported"):
        load_run(d)


def test_metadata_from_first_used_walker(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    add_fresh(d, "a0", n_hills=0, n_rows=1)               # sorts first, skipped
    f = os.path.join(d, "HILLS_a0.npz")
    h = dict(np.load(f))
    h["tempering"] = "tt"
    np.savez(f, **h)
    run = load_run(d)
    assert run.tempering == "wt" and [w.rid for w in run.walkers] == ["w0", "w1"]


def test_monitor_missing_in_one_segment_listed(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=1, extend=True, cvs=("dist",), monitors=("funnel_z",))
    f = os.path.join(d, "REWEIGHT_w0_x2.npz")
    r = {k: v for k, v in np.load(f).items() if k != "monitor_funnel_z"}
    np.savez(f, **r)
    run = load_run(d)
    assert "funnel_z" not in run.walkers[0].cvs and run.monitored == []
    assert any("w0: monitor funnel_z missing in 1 segment" in s for s in run.skipped)


def test_monitor_missing_in_one_walker_listed(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2, cvs=("dist",), monitors=("funnel_z",))
    f = os.path.join(d, "REWEIGHT_w1.npz")
    np.savez(f, **{k: v for k, v in np.load(f).items() if k != "monitor_funnel_z"})
    run = load_run(d)
    assert run.monitored == [] and any("monitor funnel_z: not logged for every walker" in s for s in run.skipped)
    bands(run)


def test_walker_without_reweight_excluded_others_proceed(tmp_path):
    d = make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3)
    os.remove(os.path.join(d, "REWEIGHT_w1.npz"))
    run = load_run(d)
    assert run.has_reweight and run.walkers[1].bias is None
    assert any(s.startswith("w1: no REWEIGHT data") for s in run.skipped)
    c = binding(run, bands(run), cv="funnel_z")
    assert set(c.per_walker) == {"w0", "w2"} and c.n_walkers == 3
    s = report(d, out_dir=str(tmp_path / "o"), step_ns=0.5)
    assert set(s["binding"]["per_walker"]) == {"w0", "w2"} and set(s["blocks"]) == {"w0", "w2"}


def test_shared_walker_without_reweight_left_out_of_pool(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=2)
    os.remove(os.path.join(d, "REWEIGHT_w1.npz"))
    run = load_run(d)
    v = binding(run, bands(run), cv="funnel_z", burn=0.0).pooled
    assert v["n_frames"] == len(run.walkers[0].t)


def test_mixed_layout_not_supported(tmp_path):
    d = make_run(str(tmp_path / "r"), n_walkers=3)
    f = os.path.join(d, "HILLS_w2.npz")
    h = dict(np.load(f))
    h["bias_dir"] = os.path.join(d, "bias_w2")
    np.savez(f, **h)
    with pytest.raises(ValueError, match="mixed shared/private bias dirs: not supported"):
        load_run(d)
