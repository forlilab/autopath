import json
import os

import pandas as pd

from autopath.pulling import SMDData
from autopath.pulling.AnalysisSMD import SMDAnalysis

from .hs_toys import flat, simulate_pulls, write_dat

HS_FILES = ("hs_g0.csv", "hs_slices.csv", "hs_row_weights.csv.gz", "hs_excluded.json")


def _smd(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=10, speed=0.01, length_nm=0.1, seed=3)
    files = write_dat(rows, tmp_path / "logs")
    smd = SMDData(files, temperature=300.0, completion_threshold_nm=0.0)
    if "path" not in smd.raw_data:
        smd.raw_data["path"] = 0
    return smd


def test_stage_writes_outputs_and_leaves_results(tmp_path):
    smd = _smd(tmp_path)
    before = smd.results.copy() if hasattr(smd.results, "copy") else smd.results
    out = tmp_path / "out"
    an = SMDAnalysis(do_plots=False, outdir=str(out), path_model="null",
                     hummer_szabo=dict(f_sources=("jarzynski",), n_boot=5))
    res = an._run_hummer_szabo(smd)
    assert res is an.hs_result and res is not None
    for f in HS_FILES:
        assert (out / f).exists()
    g0 = pd.read_csv(out / "hs_g0.csv")
    assert {"f_source", "scope", "r_bin", "G0", "G0_sd"} <= set(g0.columns)
    assert len(pd.read_csv(out / "hs_row_weights.csv.gz")) == len(res.row_weights)
    assert isinstance(json.loads((out / "hs_excluded.json").read_text()), dict)
    if isinstance(before, pd.DataFrame):
        pd.testing.assert_frame_equal(smd.results, before)


def test_stage_off_by_default(tmp_path):
    out = tmp_path / "out"
    an = SMDAnalysis(do_plots=False, outdir=str(out), path_model="null")
    assert an.hs_result is None
    assert an._run_hummer_szabo(_smd(tmp_path)) is None
    assert not any(f.startswith("hs_") for f in os.listdir(out))


class _Stop(Exception):
    pass


def test_run_fits_hs_on_rows_before_support_trim(tmp_path, monkeypatch):
    # Path 2 has fewer pulls than min_samples_per_step, so run()'s trim removes it entirely.
    rows = simulate_pulls(*flat(), n_pulls=13, speed=0.01, length_nm=0.1, seed=5)
    files = write_dat(rows, tmp_path / "logs")
    smd = SMDData(files, temperature=300.0, completion_threshold_nm=0.0)
    small = set(sorted(smd.raw_data.trajname.unique())[:3])
    mapping = {t: (2 if t in small else 1) for t in smd.raw_data.trajname.unique()}

    an = SMDAnalysis(do_plots=False, outdir=str(tmp_path / "out"), path_model="null",
                     estimators=["jarzynski", "cumulant"],
                     hummer_szabo=dict(f_sources=("jarzynski",), n_boot=0))
    monkeypatch.setattr(an, "_build_cluster_feature_df", lambda *a, **kw: (pd.DataFrame(), {}))
    monkeypatch.setattr(an.path_model, "fit_transform", lambda *a, **kw: mapping)
    # Flat-landscape dG fails the negative-dG path filter; that filter is not under test.
    monkeypatch.setattr(an, "_path_filtering", lambda d: d)
    seen, real = [], an._run_hummer_szabo

    def spy(data):
        seen.append(real(data))
        raise _Stop

    monkeypatch.setattr(an, "_run_hummer_szabo", spy)
    try:
        an.run(smd, cluster_to_boundary=False)
    except _Stop:
        pass
    assert len(seen) == 1, "run() did not reach the HS stage"
    rw = seen[0].row_weights
    assert set(rw.path) == {1, 2}
    assert set(rw.loc[rw.path == 2, "trajname"]) == small
    assert 2 not in set(smd.raw_data.path)


def test_json_default_rejects_unknown_types():
    import numpy as np
    import pytest

    from autopath.pulling.AnalysisSMD import _json_default

    out = json.loads(json.dumps({"a": np.int64(3), "b": np.arange(2), "c": np.float32(0.5)},
                                default=_json_default))
    assert out == {"a": 3, "b": [0, 1], "c": 0.5}
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=_json_default)
