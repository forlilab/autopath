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
