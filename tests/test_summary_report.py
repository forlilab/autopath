import json
import os
import subprocess
import sys

import numpy as np

from autopath.metadynamics.summary import check_convergence, report
from autopath.metadynamics.summary.convergence import (
    bootstrap_sd_below, clamp_fraction_below, last_blocks_within, min_recrossings, walker_spread_within)
from ._summary_fixtures import make_run

KJ = 4.184


def test_report_writes_figures_and_json(tmp_path):
    d = make_run(str(tmp_path / "r"))
    s = report(d, out_dir=str(tmp_path / "o"), step_ns=0.5)
    for f in ("metad_cvs.png", "metad_bias.png", "metad_fe.png", "metad_summary.json"):
        assert os.path.getsize(tmp_path / "o" / f) > 100
    j = json.load(open(tmp_path / "o" / "metad_summary.json"))
    assert s["layout"] == j["layout"] == "shared"
    for k in ("walkers", "bands", "skipped_panels", "skipped_walkers", "clamp", "binding", "blocks", "params",
              "temperature", "bias_factor", "tempering", "cvs"):
        assert k in j
    assert "time_ns" in j["walkers"]["w0"] and "verdict" not in j


def test_json_nan_is_null(tmp_path):
    d = make_run(str(tmp_path / "r"), n_rows=60)
    s = report(d, out_dir=str(tmp_path / "o"), step_ns=0.1, t_rej=0.2)
    txt = open(tmp_path / "o" / "metad_summary.json").read()
    assert "NaN" not in txt
    json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(c)))
    assert s["layout"] == "shared"


def test_isolated_clamp_per_walker(tmp_path):
    d = make_run(str(tmp_path / "r"), layout="isolated")
    s = report(d, out_dir=str(tmp_path / "o"), step_ns=0.5)
    assert set(s["clamp"]["funnel_z"]) == {"w0", "w1"}
    assert set(s["blocks"]) == {"w0", "w1"}


def test_not_write_into_run_dir_when_out_given(tmp_path):
    d = make_run(str(tmp_path / "r")); before = set(os.listdir(d))
    report(d, out_dir=str(tmp_path / "o"), step_ns=0.5)
    assert set(os.listdir(d)) == before


def test_bands_override(tmp_path):
    d = make_run(str(tmp_path / "r"))
    ov = {"funnel_z": {"bound": [-0.4, 0.5], "unbound": [1.5, 2.3], "bound_end": "low"}}
    s = report(d, bands=ov, out_dir=str(tmp_path / "o"), step_ns=0.5)
    assert s["bands"]["funnel_z"]["bound"] == (-0.4, 0.5)


def test_cli(tmp_path):
    d = make_run(str(tmp_path / "r"))
    bj = tmp_path / "b.json"
    bj.write_text(json.dumps({"funnel_z": {"bound": [-0.4, 0.5], "unbound": [1.5, 2.3], "bound_end": "low"}}))
    env = dict(os.environ, PYTHONPATH=os.getcwd() + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-m", "autopath.metadynamics.summary", d, "--out", str(tmp_path / "o"),
                        "--step-ns", "0.5", "--bands", str(bj)], capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(tmp_path / "o" / "metad_summary.json")


def _s(**kw):
    blk = lambda d, sd: dict(last_blocks_diff=d * KJ, boot_sd=sd * KJ)
    base = dict(layout="isolated", funnel=True, cvs=[dict(name="funnel_z")],
                walkers={"w0": dict(recrossings={"funnel_z": 4}), "w1": dict(recrossings={"funnel_z": 1})},
                clamp={"funnel_z": {"w0": 0.01, "w1": 0.2}},
                binding=dict(spread=8.0 * KJ), blocks={"w0": blk(10, 8), "w1": blk(10, 5)})
    base.update(kw); return base


def test_criteria_fail_and_empty():
    v = check_convergence(_s(), [min_recrossings(3), walker_spread_within(1.0), last_blocks_within(1.0),
                                 bootstrap_sd_below(1.0), clamp_fraction_below(0.1)])
    assert v.needs_more_sampling and not any(v.passed.values()) and len(v.reasons) == 5
    assert all("kcal" in r or "recross" in r or "clamp" in r for r in v.reasons)
    v2 = check_convergence(_s(), [])
    assert v2.needs_more_sampling is False and v2.passed == {}


def test_criteria_pass_and_nan():
    s = _s(walkers={"w0": dict(recrossings={"funnel_z": 4}), "w1": dict(recrossings={"funnel_z": 3})},
           clamp={"funnel_z": {"w0": 0.0}}, binding=dict(spread=0.5 * KJ),
           blocks={"w0": dict(last_blocks_diff=0.1 * KJ, boot_sd=0.1 * KJ), "w1": dict(last_blocks_diff=0.1, boot_sd=0.1)})
    cr = [min_recrossings(3), walker_spread_within(1.0), last_blocks_within(1.0), bootstrap_sd_below(1.0),
          clamp_fraction_below(0.1)]
    assert not check_convergence(s, cr).needs_more_sampling
    s["blocks"]["w1"]["boot_sd"] = float("nan")
    assert not check_convergence(s, [bootstrap_sd_below(1.0)]).passed["bootstrap_sd_below(1.0)"]
    s["blocks"]["w1"]["boot_sd"] = None
    assert check_convergence(s, [bootstrap_sd_below(1.0)]).needs_more_sampling


def test_shared_pools_recrossings_and_spread_na():
    s = _s(layout="shared", blocks=dict(last_blocks_diff=0.5, boot_sd=0.4), binding=dict(pooled={}, spread=None))
    assert min_recrossings(5)(s)[0] and not min_recrossings(6)(s)[0]
    ok, reason = walker_spread_within(1.0)(s)
    assert ok and "not applicable (shared)" in reason
    assert last_blocks_within(1.0)(s)[0] and bootstrap_sd_below(1.0)(s)[0]


def test_min_recrossings_default_cv_choice():
    w = {"w0": dict(recrossings={"funnel_z": 5, "dist": 0})}
    s = _s(layout="single", walkers=w, cvs=[dict(name="funnel_z"), dict(name="dist")])
    assert min_recrossings(3)(s)[0]
    s["funnel"] = False
    assert not min_recrossings(3)(s)[0]
    assert min_recrossings(3, cv="funnel_z")(s)[0]
