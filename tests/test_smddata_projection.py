import numpy as np
import pandas as pd
import pytest

from autopath.pulling import HummerSzaboEstimator, SMDData

from .hs_toys import flat, simulate_pulls, write_dat


def _files(tmp_path, cv=None, shift=0.0, n_pulls=6, seed=0, id0=0):
    rows = simulate_pulls(*flat(), n_pulls=n_pulls, speed=0.01, r_start=0.1, length_nm=0.4, seed=seed)
    rows = rows.copy()
    rows["trajname"] = rows["trajname"].str.replace(
        r"replica-(\d+)", lambda m: f"replica-{int(m.group(1)) + id0}", regex=True)
    files = write_dat(rows, tmp_path)
    for fn in files:
        d = pd.read_csv(fn, comment="#")
        for c in ("r_target", "r_before", "r_after"):
            d[c] = d[c] + shift
        with open(fn, "w") as f:
            f.write("# spring_constant_kJ_mol_nm2=3765.6\n")
            if cv is not None:
                f.write(f"# cv={cv}\n")
            d.to_csv(f, index=False)
    return files


def test_projection_pull_with_negative_z_is_kept(tmp_path):
    files = _files(tmp_path, cv="projection", shift=-0.3)
    smd = SMDData(files, temperature=300.0)
    assert smd.cv_type == "projection"
    assert smd.raw_data["r_after"].min() < 0
    assert smd.raw_data["trajname"].nunique() == len(files)


def test_distance_logs_unchanged(tmp_path):
    ok = _files(tmp_path / "ok", shift=0.5)
    smd = SMDData(ok, temperature=300.0)
    assert smd.cv_type == "distance"
    assert smd.raw_data["trajname"].nunique() == len(ok)
    smd = SMDData(_files(tmp_path / "ex", cv="distance", shift=0.5), temperature=300.0)
    assert smd.cv_type == "distance"
    # Pulls reaching below the threshold are still dropped for distance logs.
    files = _files(tmp_path / "drop", shift=-0.3, n_pulls=3) + _files(tmp_path / "keep", shift=0.5, n_pulls=3, id0=100)
    smd = SMDData(files, temperature=300.0)
    assert smd.raw_data["trajname"].nunique() == 3


def test_mixed_cv_types_not_supported(tmp_path):
    files = _files(tmp_path / "a", cv="projection") + _files(tmp_path / "b", id0=100)
    with pytest.raises(ValueError, match="not supported"):
        SMDData(files, temperature=300.0)


def test_hs_fit_on_projection_logs(tmp_path):
    files = _files(tmp_path, cv="projection", shift=-0.3, n_pulls=10)
    res = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=0).fit(SMDData(files, temperature=300.0))
    assert np.isfinite(res.row_weights.logw).all()
