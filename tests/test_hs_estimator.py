import numpy as np
import pandas as pd
import pytest

from autopath.pulling import HummerSzaboEstimator, SMDData
from autopath.pulling.hummer_szabo import F_SOURCES

from .hs_toys import flat, harmonic, simulate_pulls, write_dat


def _smd(tmp_path, rows, header_k=True, direction="forward", **kw):
    files = write_dat(rows, tmp_path, header_k=header_k, direction=direction)
    kw.setdefault("completion_threshold_nm", 0.0)
    return SMDData(files, temperature=300.0, **kw)


def _relabel(*groups):
    # Sequential replica ids across groups so every log file gets a unique integer id.
    out, n = [], 0
    for g in groups:
        g = g.copy()
        names = list(dict.fromkeys(g["trajname"]))
        new = {t: f"replica-{n + i}_v{g.loc[g.trajname == t, 'speed'].iloc[0]}" for i, t in enumerate(names)}
        g["trajname"] = g["trajname"].map(new)
        n += len(names)
        out.append(g)
    rows = pd.concat(out, ignore_index=True)
    rows.attrs.update(groups[0].attrs)
    return rows


def _aligned_dev(g, G0, min_ess):
    m = (g.ess_pulls >= min_ess).to_numpy()
    assert m.sum() > 3
    d = g.G0.to_numpy()[m] - G0(g.r_bin.to_numpy()[m])
    return d - d.mean(), m


def test_autostop_pulls_are_trimmed_and_recover_g0(tmp_path):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    rows = simulate_pulls(G0, dG0, n_pulls=80, speed=0.005, r_start=0.9, length_nm=0.2,
                          stop_steps=[150] * 5 + [None] * 75, seed=9)
    res = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=0).fit(_smd(tmp_path, rows))
    assert res.excluded["n_rows_unsupported"] == 75 * (200 - 151)
    assert res.excluded["n_pulls_dropped"] == 0
    assert res.row_weights.step.max() <= 150
    d, _ = _aligned_dev(res.g0.query("scope == 'all'"), G0, 30)
    assert np.max(np.abs(d)) < 1.0


def test_missing_k_header_is_recovered(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=10, speed=0.01, length_nm=0.1, seed=10)
    est = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=0)
    a = est.fit(_smd(tmp_path / "hdr", rows)).row_weights
    smd = _smd(tmp_path / "nohdr", rows, header_k=False)
    assert smd.spring_constant is None
    b = est.fit(smd).row_weights
    assert np.isfinite(b.logw).all()
    np.testing.assert_allclose(b.logw.to_numpy(), a.logw.to_numpy(), rtol=1e-8)


# Own-lambda0 pulls start as a delta at r(0); start sd 0.01 nm set by the equilibration spring.
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_own_lambda0_realistic_spread(tmp_path, seed):
    kT, kappa = 0.008314462618 * 300.0, 500.0
    G0, dG0 = harmonic(r0=1.0, kappa=kappa)
    rows = simulate_pulls(G0, dG0, n_pulls=60, speed=0.005, r_start=0.9, length_nm=0.2,
                          lam0="own", k_eq=kT / 0.01 ** 2 - kappa, seed=seed)
    res = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=50).fit(_smd(tmp_path, rows))
    assert res.excluded["lam0_sd_nm"] == pytest.approx(0.01, abs=0.003)
    g = res.g0.query("scope == 'all'")
    d, m = _aligned_dev(g, G0, 5)
    assert np.all(np.abs(d) <= 3 * g.G0_sd.to_numpy()[m] + 0.3)


def test_two_paths_joint_matches_truth_and_reports_per_path(tmp_path):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    a = simulate_pulls(G0, dG0, n_pulls=40, speed=0.005, r_start=0.9, length_nm=0.2, seed=14)
    b = simulate_pulls(G0, dG0, n_pulls=40, speed=0.005, r_start=0.9, length_nm=0.2, seed=15)
    smd = _smd(tmp_path, _relabel(a, b))
    smd.raw_data["path"] = np.where(smd.raw_data["repid"] < 40, 1, 2)
    res = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=0).fit(smd)

    d, _ = _aligned_dev(res.g0.query("scope == 'all'"), G0, 30)
    assert np.max(np.abs(d)) < 1.0
    per = res.g0.query("scope == 'path'")
    assert set(per.path) == {1, 2} and per.speed.isna().all()
    assert res.g0.query("scope == 'all'")[["path", "speed"]].isna().all().all()

    rw = res.row_weights
    assert np.exp(rw.logw).sum() == pytest.approx(1.0)
    for _, g in rw.groupby("path"):
        assert np.exp(g.logw_path).sum() == pytest.approx(1.0)
        assert np.exp(g.logw).sum() == pytest.approx(0.5, abs=0.15)


def test_results_table_untouched(tmp_path):
    smd = _smd(tmp_path, simulate_pulls(*flat(), n_pulls=6, speed=0.01, length_nm=0.1))
    before = None if smd.results is None else smd.results.copy()
    raw = smd.raw_data.copy()
    HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=5).fit(smd)
    assert (smd.results is None and before is None) or smd.results.equals(before)
    pd.testing.assert_frame_equal(smd.raw_data, raw)


def test_mixed_step_size_within_speed_is_not_supported(tmp_path):
    a = simulate_pulls(*flat(), n_pulls=3, speed=0.01, dx=0.001, length_nm=0.1, seed=12)
    b = simulate_pulls(*flat(), n_pulls=3, speed=0.01, dx=0.0012, length_nm=0.1, seed=13)
    with pytest.raises(ValueError, match="not supported"):
        HummerSzaboEstimator(f_sources=("jarzynski",)).fit(_smd(tmp_path, _relabel(a, b)))


def test_backward_pulls_are_not_supported(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=3, speed=0.01, length_nm=0.1, seed=16)
    smd = _smd(tmp_path, rows, direction="backward", completion_threshold_nm=100.0)
    assert smd.pulling_direction == "backward"
    with pytest.raises(ValueError, match="not supported"):
        HummerSzaboEstimator(f_sources=("jarzynski",)).fit(smd)


def test_two_speeds_report_lam0_spread_and_force_gaps(tmp_path):
    G0, dG0 = harmonic(r0=1.0, kappa=500.0)
    slow = simulate_pulls(G0, dG0, n_pulls=20, speed=0.005, r_start=0.9, length_nm=0.2,
                          lam0="own", r0_jitter_nm=0.02, seed=17)
    fast = simulate_pulls(G0, dG0, n_pulls=20, speed=0.01, r_start=0.9, length_nm=0.1,
                          lam0="own", r0_jitter_nm=0.02, seed=18)
    smd = _smd(tmp_path, _relabel(slow, fast))
    res = HummerSzaboEstimator(n_boot=0).fit(smd)

    raw = smd.raw_data.sort_values(["trajname", "step"])
    first = raw.groupby("trajname")[["speed", "r_target"]].first()
    med = (first["r_target"] - 0.001).groupby(first["speed"]).median()
    assert res.excluded["lam0_spread_nm"] == pytest.approx(med.max() - med.min(), abs=1e-12)
    assert res.excluded["lam0_spread_nm"] > 0
    by_hand = (first["r_target"] - 0.001).groupby(first["speed"]).std().max()
    assert res.excluded["lam0_sd_nm"] == pytest.approx(by_hand, rel=1e-9)
    assert "skipped_sources" not in res.excluded or not res.excluded["skipped_sources"]
    assert set(res.g0.f_source) == set(F_SOURCES)

    no_f = res.excluded["n_rows_no_F"]
    assert no_f["force"] == 20 * 100 and no_f["jarzynski"] == 0
    f = res.row_weights.query("f_source == 'force'")
    assert np.isneginf(f.logw).sum() == no_f["force"]

    sp = res.g0.query("scope == 'speed'")
    assert set(sp.speed) == {0.005, 0.01} and sp.path.isna().all()
    assert "force" not in set(sp.f_source)
    assert set(res.slices.speed) == {0.005, 0.01}
    assert {f"F_{s}" for s in F_SOURCES} <= set(res.slices.columns)


def test_single_speed_skips_force(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=8, speed=0.01, length_nm=0.05, seed=19)
    rmin = rows.groupby("trajname")["r_after"].min()
    cut = float(rmin.sort_values().iloc[2:4].mean())
    res = HummerSzaboEstimator(n_boot=0).fit(_smd(tmp_path, rows, completion_threshold_nm=cut))
    assert res.excluded["n_pulls_dropped"] == 3
    assert res.row_weights.trajname.nunique() == 5
    assert res.excluded["skipped_sources"] == {"force": "single speed"}
    assert "force" not in set(res.g0.f_source)
    assert set(res.g0.f_source) == set(F_SOURCES) - {"force"}
    assert res.excluded["lam0_spread_nm"] == 0.0


def test_non_protocol_work_is_not_supported(tmp_path):
    rows = simulate_pulls(*flat(), n_pulls=4, speed=0.01, length_nm=0.05, seed=30)
    est = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=0)
    with pytest.raises(ValueError, match="not supported: .*protocol"):
        est.fit(_smd(tmp_path / "fdx", rows, work_mode="force_dx"))
    auto = _smd(tmp_path / "auto", rows, work_mode="auto")
    assert np.isfinite(est.fit(auto).row_weights.logw).all()
    auto.raw_data = auto.raw_data.drop(columns="dW_protocol")
    with pytest.raises(ValueError, match="not supported: .*protocol"):
        est.fit(auto)


def test_force_with_speed_dependent_dx_is_not_supported(tmp_path):
    a = simulate_pulls(*flat(), n_pulls=4, speed=0.01, dx=0.001, length_nm=0.05, seed=31)
    b = simulate_pulls(*flat(), n_pulls=4, speed=0.02, dx=0.002, length_nm=0.1, seed=32)
    smd = _smd(tmp_path, _relabel(a, b))
    with pytest.raises(ValueError, match="not supported: .*dx"):
        HummerSzaboEstimator(f_sources=("force",), n_boot=0).fit(smd)
    res = HummerSzaboEstimator(f_sources=("jarzynski",), n_boot=0).fit(smd)
    # Per-pull lambda0 = first target minus the speed's move; spread over per-speed medians.
    raw = smd.raw_data.sort_values(["trajname", "step"])
    first = raw.groupby("trajname")[["speed", "r_target"]].first()
    lam0 = first["r_target"] - first["speed"].map({0.01: 0.001, 0.02: 0.002})
    med = lam0.groupby(first["speed"]).median()
    assert res.excluded["lam0_spread_nm"] == pytest.approx(med.max() - med.min(), abs=1e-12)
    assert res.excluded["lam0_sd_nm"] == pytest.approx(lam0.groupby(first["speed"]).std().max(), abs=1e-12)
