import pandas as pd
import pytest
from autopath.pulling.SMDData import SMDData


def _trace():
    return pd.DataFrame({
        "trajname": ["t1"] * 6,
        "speed": [0.001] * 6,
        "step": [0, 1, 2, 3, 4, 5],
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "work": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })


def _geom():
    return pd.DataFrame({
        "trajname": ["t1", "t1", "t1"],
        "speed": [0.001] * 3,
        "step": [0, 2, 4],
        "time": [0.0, 2.0, 4.0],
        "geom_rog": [0.5, 0.6, 0.7],
    })


def test_merge_geom_impute_no_nan_full_rows():
    out = SMDData.merge_geom_features(_trace(), _geom(), mode="impute")
    assert len(out) == 6
    assert "geom_rog" in out.columns
    assert not out["geom_rog"].isnull().any()


def test_merge_geom_aligned_exact_only():
    out = SMDData.merge_geom_features(_trace(), _geom(), mode="aligned")
    assert len(out) == 3
    assert sorted(out["time"].tolist()) == [0.0, 2.0, 4.0]
    assert not out["geom_rog"].isnull().any()


def test_merge_geom_bad_mode():
    with pytest.raises(ValueError):
        SMDData.merge_geom_features(_trace(), _geom(), mode="nope")


def test_load_geom_features_missing_returns_empty(tmp_path):
    log = tmp_path / "sMD_replica-1_v0.001_forward.dat"
    log.write_text("step,time,r_target,r_before,r_after,force,U_cvpack,dW_protocol,lag_nm\n"
                   "0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0\n")
    smd = SMDData.__new__(SMDData)
    smd.log_files = [str(log)]
    df = smd.load_geom_features()
    assert df.empty


def test_load_geom_features_reads_sidecar(tmp_path):
    log = tmp_path / "sMD_replica-1_v0.001_forward.dat"
    log.write_text("step,time,r_target,r_before,r_after,force,U_cvpack,dW_protocol,lag_nm\n"
                   "0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0\n")
    geom = tmp_path / "sMD_replica-1_v0.001_forward_geom.dat"
    geom.write_text("step,time,geom_rog,geom_npr1\n0,0.0,0.5,0.3\n2,2.0,0.6,0.4\n")
    smd = SMDData.__new__(SMDData)
    smd.log_files = [str(log)]
    df = smd.load_geom_features()
    assert list(df["geom_rog"]) == [0.5, 0.6]
    assert df["trajname"].iloc[0] == "sMD_replica-1_v0.001_forward"
    assert df["speed"].iloc[0] == 0.001
