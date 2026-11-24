import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from autopath.pulling.outputs import PULL_COLUMNS, load_traces, pull_table

from .smd_toy_run import analysed_dir, toy_smd

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def legacy_dir(tmp_path_factory):
    return analysed_dir(tmp_path_factory.mktemp("legacy"), legacy_outputs=True)


def test_traces_roundtrip_equals_legacy(legacy_dir):
    a = load_traces(str(legacy_dir))
    b = pd.read_csv(legacy_dir / "sMD_processed_data.csv")
    pd.testing.assert_frame_equal(a[sorted(a)], b[sorted(b)])
    assert not set(PULL_COLUMNS) & set(pd.read_csv(legacy_dir / "traces.csv.gz").columns)


def test_legacy_off_by_default(tmp_path):
    out = analysed_dir(tmp_path)
    assert not (out / "sMD_processed_data.csv").exists()
    pulls = pd.read_csv(out / "pulls.csv")
    assert len(pulls) == len(list((tmp_path / "pulls").glob("*.dat")))
    assert (pulls["stop_reason"] == "unknown").all()


def test_load_traces_falls_back_to_legacy(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_traces(str(tmp_path))
    df = pd.DataFrame({"trajname": ["a"], "speed": [0.01], "step": [1]})
    df.to_csv(tmp_path / "sMD_processed_data.csv", index=False)
    pd.testing.assert_frame_equal(load_traces(str(tmp_path)), df)


def _stamp(files, stamps):
    for fn, st in zip(files, stamps):
        txt = Path(fn).read_text()
        Path(fn).write_text(f"# start_time={st}\n" + txt if st else txt)


def test_pull_index_from_start_time(tmp_path):
    smd = toy_smd(tmp_path, speeds=(0.01,), n_pulls=3)
    files = sorted(smd.log_files)
    _stamp(files, ["2026-01-03T00:00:00", "2026-01-02T00:00:00", "2026-01-01T00:00:00"])
    smd = type(smd)(files, temperature=300.0, completion_threshold_nm=0.0)
    p = pull_table(smd, smd.raw_data).set_index("trajname")
    assert p.loc[[os.path.basename(f)[:-4] for f in files], "pull_index"].tolist() == [3, 2, 1]


def test_pull_index_mixed_headers_falls_back(tmp_path, caplog):
    from autopath.pulling import SMDData
    smd = toy_smd(tmp_path, speeds=(0.01,), n_pulls=3)
    files = sorted(smd.log_files)
    _stamp(files, ["2026-01-03T00:00:00", "2026-01-02T00:00:00", None])
    smd = SMDData(files, temperature=300.0, completion_threshold_nm=0.0)
    with caplog.at_level(logging.INFO, logger="autopath.pulling.outputs"):
        p = pull_table(smd, smd.raw_data)
    want = sorted(files, key=lambda f: (SMDData._replica_start_datetime(f), f))
    got = p.set_index("trajname")["pull_index"]
    assert [got[os.path.basename(f)[:-4]] for f in want] == [1, 2, 3]
    assert sum("pull order from" in r.getMessage() for r in caplog.records) == 1


def test_collect_readouts_reads_new_layout(tmp_path):
    d = analysed_dir(tmp_path / "work")
    dest = tmp_path / "root" / "sysA" / "sMD" / "analysis_traces"
    dest.parent.mkdir(parents=True)
    d.rename(dest)
    csv = tmp_path / "readouts.csv"
    r = subprocess.run([sys.executable, str(ROOT / "examples/03_Pull-it-out/collect_readouts.py"),
                        "--root", str(tmp_path / "root"), "--out", str(csv)],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr
    assert len(pd.read_csv(csv)) > 0
