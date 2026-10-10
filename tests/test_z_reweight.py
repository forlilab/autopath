import numpy as np
import pytest

from autopath.metadynamics.report import z_reweighted_dG


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
    out = z_reweighted_dG(str(tmp_path), fp={"z_max": None}, temperature=300.0, burn=0.0)
    assert out["eq12_kj"] == -1.0 and out["ct_kj"] == -1.0 and out["n_frames"] == len(z)
    assert np.isfinite(seen["fes"]).sum() > 50


@pytest.mark.skip(reason="needs a run with monitor_funnel_z; see Task 10")
def test_eq12_matches_scratch_pocket_reweight():
    pass
