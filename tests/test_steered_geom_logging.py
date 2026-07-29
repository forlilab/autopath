import numpy as np
import pytest
from autopath.pulling.steered_md import SteeredMD


def _make_stub(shape_feats=("rog", "npr1")):
    from autopath.pulling.geom_kernel import ShapeDescriptorCalculator
    smd = SteeredMD.__new__(SteeredMD)
    smd.groupA_atoms = [0, 1, 2, 3]
    smd.subset_protein_CA = np.array([4, 5, 6, 7, 8])
    smd._lig_heavy_idx = [0, 1, 2, 3]
    smd._shape_calc = ShapeDescriptorCalculator(["C", "C", "C", "O"], list(shape_feats))
    return smd


def test_resolve_geom_features_disabled():
    smd = _make_stub()
    smd.log_geom_features = False
    assert smd._resolve_geom_features() == []


def test_resolve_geom_features_default_true():
    smd = _make_stub()
    smd.log_geom_features = True
    feats = smd._resolve_geom_features()
    assert "nc" in feats and "rog" in feats and "npr1" in feats


def test_resolve_geom_features_accepts_asphericity():
    smd = _make_stub()
    smd.log_geom_features = ["rog", "asphericity"]
    assert smd._resolve_geom_features() == ["rog", "asphericity"]


def test_resolve_geom_features_rejects_unknown():
    smd = _make_stub()
    smd.log_geom_features = ["rog", "not_a_descriptor"]
    with pytest.raises(ValueError):
        smd._resolve_geom_features()


def test_compute_geom_row_keys_and_values():
    smd = _make_stub(shape_feats=("rog", "npr1"))
    smd.log_geom_features = ["nc", "mindist", "rog", "npr1"]
    rng = np.random.default_rng(0)
    pos_nm = rng.normal(size=(9, 3)) * 0.3
    row = smd._compute_geom_row(pos_nm)
    assert set(row) == {"nc", "mindist", "rog", "npr1"}
    assert row["mindist"] >= 0.0
    assert 0.0 <= row["npr1"] <= 1.0


def test_compute_geom_row_exit_direction_is_unit_vector():
    smd = _make_stub(shape_feats=("rog",))
    smd.log_geom_features = ["exit_1", "exit_2", "exit_3", "rog"]
    pos_nm = np.random.default_rng(2).normal(size=(9, 3)) * 0.3
    row = smd._compute_geom_row(pos_nm)
    assert {"exit_1", "exit_2", "exit_3", "rog"} <= set(row)
    e = np.array([row["exit_1"], row["exit_2"], row["exit_3"]])
    assert np.isclose(np.sum(e ** 2), 1.0, atol=1e-6)   # projected unit vector


def test_compute_geom_row_exit_skipped_without_pocket():
    smd = _make_stub(shape_feats=("rog",))
    smd.subset_protein_CA = None
    smd.log_geom_features = ["exit_1", "exit_2", "exit_3", "rog"]
    row = smd._compute_geom_row(np.random.default_rng(3).normal(size=(9, 3)))
    assert set(row) == {"rog"}


def test_compute_geom_row_shape_only_when_pocket_missing():
    smd = _make_stub(shape_feats=("rog", "npr1"))
    smd.subset_protein_CA = None          # no pocket -> nc/mindist skipped
    smd.log_geom_features = ["nc", "mindist", "rog", "npr1"]
    pos_nm = np.random.default_rng(1).normal(size=(9, 3)) * 0.3
    row = smd._compute_geom_row(pos_nm)
    assert set(row) == {"rog", "npr1"}


# Integration tests need a minimal OpenMM sMD fixture; skip cleanly if absent.
@pytest.mark.slow
def test_pull_writes_sidecar_when_enabled(tmp_path):
    pytest.importorskip("openmm")
    fixture = pytest.importorskip(
        "tests.smd_fixture", reason="no minimal sMD fixture available")
    import os
    import pandas as pd
    smd, kwargs = fixture.make_two_group_smd(out_dir=str(tmp_path),
                                             log_geom_features=True)
    run_id = smd.run(**kwargs)
    assert os.path.exists(f"{tmp_path}/sMD_{run_id}_geom.dat")
    df = pd.read_csv(f"{tmp_path}/sMD_{run_id}_geom.dat")
    assert "geom_rog" in df.columns and "geom_npr1" in df.columns


@pytest.mark.slow
def test_pull_no_sidecar_when_disabled(tmp_path):
    pytest.importorskip("openmm")
    fixture = pytest.importorskip(
        "tests.smd_fixture", reason="no minimal sMD fixture available")
    import glob
    smd, kwargs = fixture.make_two_group_smd(out_dir=str(tmp_path),
                                             log_geom_features=False)
    run_id = smd.run(**kwargs)
    assert glob.glob(f"{tmp_path}/*_geom.dat") == []
