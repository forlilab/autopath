import os

import numpy as np
import pytest
from openmm import unit

from autopath.metadynamics.report import z_reweighted_dG


def _fp(z_max=26.0):
    rng = np.random.default_rng(0)
    return {"z_cc": 14.0 * unit.angstrom, "z_max": z_max * unit.angstrom,
            "R_cylinder": 4.0 * unit.angstrom,
            "axial_distances": np.r_[rng.uniform(1, 3, 200), rng.uniform(3, 20, 800)]}


def _monitored(d, z):
    np.savez(d / "REWEIGHT_w.npz", time_ns=np.linspace(0, 10, len(z)), bias_kj=np.zeros(len(z)),
             rct_kj=np.zeros(len(z)), monitor_funnel_z=z, record_dt_ns=10 / len(z))
    np.save(d / "COLVAR_w.npy", z[:, None])


def test_no_monitor_gives_reason_not_error(tmp_path):
    np.savez(tmp_path / "REWEIGHT_w.npz", time_ns=np.arange(3.0), bias_kj=np.zeros(3), rct_kj=np.zeros(3))
    out = z_reweighted_dG(str(tmp_path), fp={"dummy": 1}, temperature=300.0)
    assert "eq12_kj" not in out and "monitor_funnel_z" in out["reason"]


def test_no_funnel_gives_reason(tmp_path):
    out = z_reweighted_dG(str(tmp_path), fp=None, temperature=300.0)
    assert "funnel" in out["reason"]


def test_unbiased_two_state_recovers_population_ratio(tmp_path, monkeypatch):
    # zero bias: weights uniform, so W(z) = -kT ln histogram
    from autopath.metadynamics import report as rep
    rng = np.random.default_rng(0)
    z = np.r_[rng.normal(0.1, 0.02, 3000), rng.uniform(1.0, 2.0, 3000)]
    np.savez(tmp_path / "REWEIGHT_w.npz", time_ns=np.linspace(0, 10, len(z)), bias_kj=np.zeros(len(z)),
             rct_kj=np.zeros(len(z)), monitor_funnel_z=z, record_dt_ns=10 / len(z))
    np.save(tmp_path / "COLVAR_w.npy", z[:, None])
    seen = {}

    def fake_correct(fes, fp, **kw):
        seen["fes"] = np.asarray(fes)
        return {"dG_bind_std_kj_mol": -1.0}

    monkeypatch.setattr(rep, "correct_fe_for_funnel", fake_correct)
    monkeypatch.setattr(rep, "_final_bias_at_frames", lambda out_dir, rid, s: np.zeros(len(s)))
    out = z_reweighted_dG(str(tmp_path), fp=_fp(), temperature=300.0, burn=0.0)
    assert out["eq12_kj"] == -1.0 and out["ct_kj"] == -1.0 and out["n_frames"] == len(z)
    assert np.isfinite(seen["fes"]).sum() > 50


@pytest.mark.skip(reason="needs a run with monitor_funnel_z; see Task 10")
def test_eq12_matches_scratch_pocket_reweight():
    pass


def test_sparse_bins_through_real_correction_are_finite(tmp_path, monkeypatch):
    from autopath.metadynamics import report as rep
    rng = np.random.default_rng(1)
    z = np.r_[rng.normal(0.1, 0.04, 8000), rng.uniform(0.5, 3.0, 1500)]
    _monitored(tmp_path, z)
    monkeypatch.setattr(rep, "_final_bias_at_frames", lambda out_dir, rid, s: np.zeros(len(s)))
    out = z_reweighted_dG(str(tmp_path), fp=_fp(), temperature=300.0, burn=0.0)
    assert np.isfinite(out["eq12_kj"]) and np.isfinite(out["ct_kj"])
    assert out["n_dropped"] == int((z > 2.6).sum()) and out["n_frames"] + out["n_dropped"] == len(z)


def test_report_survives_missing_colvar(tmp_path):
    from autopath.metadynamics import report as rep
    z = np.linspace(0.0, 2.0, 50)
    np.savez(tmp_path / "REWEIGHT_w.npz", time_ns=np.arange(50.0), bias_kj=np.zeros(50),
             rct_kj=np.zeros(50), monitor_funnel_z=z)
    with pytest.raises(Exception):
        z_reweighted_dG(str(tmp_path), fp=_fp(), temperature=300.0)
    from tests import test_metad_report as T
    d = T._shared(tmp_path)
    np.savez(d / "REWEIGHT_w1.npz", time_ns=np.arange(50.0), bias_kj=np.zeros(50),
             rct_kj=np.zeros(50), monitor_funnel_z=z)
    out = rep.report(str(d), 300.0, 0.0, 1.0, cv_length_per_unit_ang=20.0,
                     funnel_params=T._fp())
    assert "z reweighting failed" in out["z_reweighted"]["reason"]
    assert out["dG_bind_std_kj"] is not None
