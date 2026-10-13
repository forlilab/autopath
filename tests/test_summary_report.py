import json
import os
import subprocess
import sys

import numpy as np
import pytest

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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
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


def _all():
    # fixture runs have no funnel: band ΔF numbers
    return [min_recrossings(3), last_blocks_within(1.0, allow_band=True), walker_spread_within(1.0, allow_band=True),
            bootstrap_sd_below(1.0, allow_band=True), clamp_fraction_below(0.05)]


def _both(d, out, **kw):
    s = report(d, out_dir=out, **kw)
    with open(os.path.join(out, "metad_summary.json")) as fh:
        return [s, json.load(fh)]


def test_names_include_cv():
    assert min_recrossings(3, "a").__name__ != min_recrossings(3, "b").__name__
    assert min_recrossings(3, "a").__name__ == "min_recrossings(3, a)"


def test_e2e_isolated_none_blocks(tmp_path):
    d = make_run(str(tmp_path / "r"), layout="isolated")
    for s in _both(d, str(tmp_path / "o"), step_ns=0.5):
        assert s["blocks"]["w0"] is None
        v = check_convergence(s, _all())
        assert v.needs_more_sampling and not v.passed["last_blocks_within(1.0)"]
        assert any("w0: blocks undefined (ΔG°(t) too short)" in r for r in v.reasons)


@pytest.mark.parametrize("layout", ["isolated", "shared"])
def test_e2e_crossing_run_finite_blocks(tmp_path, layout):
    d = make_run(str(tmp_path / "r"), layout=layout, cross=True)
    for s in _both(d, str(tmp_path / "o"), step_ns=0.2):
        assert all(w["recrossings"]["funnel_z"] > 0 for w in s["walkers"].values())
        v = check_convergence(s, _all() + [min_recrossings(1)])
        assert v.passed["min_recrossings(1)"]
        blk = list(s["blocks"].values()) if layout == "isolated" else [s["blocks"]]
        assert all(b is not None and np.isfinite(b["boot_sd"] if b["boot_sd"] is not None else np.nan) for b in blk)
        assert not any("too short" in r for r in v.reasons)
        assert len(v.passed) == 6


def test_e2e_shared_none_blocks(tmp_path):
    d = make_run(str(tmp_path / "r"))
    for s in _both(d, str(tmp_path / "o"), step_ns=0.5):
        v = check_convergence(s, _all())
        assert any("all: blocks undefined" in r for r in v.reasons)
        assert v.passed["walker_spread_within(1.0)"]


def test_clamp_undefined_and_empty():
    ok, r = clamp_fraction_below(0.1)(_s(clamp={}))
    assert ok and "not applicable" in r
    ok, r = clamp_fraction_below(0.1)(_s(clamp={"funnel_z": {"w0": None}}))
    assert not ok and "undefined" in r


def test_spread_fails_on_undefined_walkers():
    s = _s(binding=dict(spread=0.0, n_walkers=3, n_defined=1, per_walker={"w0": dict(kind="dG0")}))
    ok, r = walker_spread_within(0.5)(s)
    assert not ok and "2 of 3 walkers undefined" in r
    s = _s(binding=dict(spread=float("nan"), n_walkers=1, n_defined=1, per_walker={"w0": dict(kind="dG0")}))
    ok, r = walker_spread_within(0.5)(s)
    assert not ok and "spread undefined" in r


def test_spread_fails_end_to_end_when_walker_never_reaches_bulk(tmp_path):
    from openmm import unit
    A = unit.angstrom
    fp = {"z_cc": 12.0 * A, "z_max": 24.0 * A, "z_low": -1.5 * A, "R_cylinder": 1.0 * A, "alpha": 31.5 * unit.degrees}
    d = make_run(str(tmp_path / "r"), layout="isolated", n_walkers=3, cross=True)
    for rid in ("w1", "w2"):
        f = os.path.join(d, f"COLVAR_{rid}.npy")
        np.save(f, np.full_like(np.load(f), 0.1))
    s = report(d, funnel=fp, out_dir=str(tmp_path / "o"), step_ns=0.5)
    assert s["binding"]["n_walkers"] == 3 and s["binding"]["n_defined"] == 1
    v = check_convergence(s, [walker_spread_within(0.5)])
    assert v.needs_more_sampling and "2 of 3 walkers undefined" in v.reasons[0]


def test_band_dF_fails_dG_criteria_unless_allowed():
    s = _s(binding=dict(spread=0.1 * KJ, n_walkers=2, n_defined=2,
                        per_walker={"w0": dict(kind="band_dF"), "w1": dict(kind="band_dF")}),
           blocks={"w0": dict(last_blocks_diff=0.0, boot_sd=0.0), "w1": dict(last_blocks_diff=0.0, boot_sd=0.0)})
    for c in (last_blocks_within(1.0), bootstrap_sd_below(1.0), walker_spread_within(1.0)):
        ok, r = c(s)
        assert not ok and "kind is band ΔF, not ΔG°" in r
    for c in (last_blocks_within(1.0, allow_band=True), bootstrap_sd_below(1.0, allow_band=True),
              walker_spread_within(1.0, allow_band=True)):
        assert c(s)[0]


def test_min_recrossings_unknown_cv():
    ok, r = min_recrossings(1, cv="nope")(_s())
    assert not ok and "unknown CV nope" in r
    ok, r = min_recrossings(1, cv="nope")(_s(layout="shared"))
    assert not ok and "unknown CV nope" in r


def test_min_ess_unit():
    from autopath.metadynamics.summary.convergence import min_ess
    s = _s(layout="shared", binding=dict(pooled=dict(ess=120.0, kind="dG0")))
    assert min_ess(50)(s)[0] and not min_ess(200)(s)[0]
    s = _s(binding=dict(per_walker={"w0": dict(ess=80.0), "w1": dict(ess=float("nan"))}))
    ok, r = min_ess(50)(s)
    assert not ok and "w1: ESS undefined" in r and "w0" not in r
    s = _s(binding=dict(per_walker={"w0": dict(ess=80.0)}))
    ok, r = min_ess(50)(s)
    assert not ok and "w1: ESS undefined" in r
    assert not min_ess(1)(_s(layout="shared", binding=None))[0]


@pytest.mark.parametrize("layout", ["shared", "isolated"])
def test_min_ess_end_to_end(tmp_path, layout):
    from autopath.metadynamics.summary.convergence import min_ess
    d = make_run(str(tmp_path / "r"), layout=layout)
    for s in _both(d, str(tmp_path / "o"), step_ns=0.1):
        vals = [s["binding"]["pooled"]] if layout == "shared" else list(s["binding"]["per_walker"].values())
        e = min(v["ess"] for v in vals)
        assert e >= 1
        assert check_convergence(s, [min_ess(1)]).passed["min_ess(1)"]
        v = check_convergence(s, [min_ess(e + 1)])
        assert v.needs_more_sampling and "ESS below" in v.reasons[0]
        blk = [s["blocks"]] if layout == "shared" else list(s["blocks"].values())
        assert all(len(b["block_ess"]) == 10 for b in blk)
        ser = [s["series"]["pooled"]] if layout == "shared" else list(s["series"]["per_walker"].values())
        assert all(len(x["ess"]) == len(x["t"]) for x in ser)
        assert s["fes_ess"]["map"] is not None


def test_bands_carry_unit(tmp_path):
    d = make_run(str(tmp_path / "r"), cvs=("funnel_z", "learned_sw"), monitors=())
    for s in _both(d, str(tmp_path / "o"), step_ns=0.5):
        assert s["bands"]["funnel_z"]["unit"] == "nm" and s["bands"]["learned_sw"]["unit"] == "CV units"
