import os

import numpy as np
import pytest

from autopath.metadynamics import permeation as P

HW, BULK = 3.0, (2.2, 2.8)


def _walker(path, rid, bias_dir, z, c, h, dt=0.01):
    n = len(h)
    rec = dict(hills_version=2, run_id=rid, finished=True, bias_dir=os.path.abspath(bias_dir), started_at=0.0,
               time_ns=np.arange(1, n + 1) * dt, wall_s=np.arange(n, dtype=float), cv=np.stack([z, c], 1),
               height_kj=np.asarray(h, float), grid_min=np.array([-HW, -1.0]), grid_max=np.array([HW, 1.0]),
               grid_points=np.array([121, 41]), sigma=np.array([0.2, 0.1]), periodic=np.array([False, False]),
               bounds_lo=np.array([np.nan, np.nan]), bounds_hi=np.array([np.nan, np.nan]),
               bias_factor=11.0, temperature=310.0, tempering="wt", cv_names=np.array(["membrane_z", "cos_theta"]),
               cv_length_per_unit_ang=np.array([10.0, np.nan]), origin=np.array([0.0, np.nan]),
               record_dt_ns=dt, dcd_interval_ns=0.1)
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, f"HILLS_{rid}.npz"), **rec)
    np.save(os.path.join(path, f"COLVAR_{rid}.npy"), np.stack([z, c], 1))


def _hills(rng, n, symmetric):
    z = rng.uniform(-1.5, 1.5, n)
    c = rng.uniform(-0.8, 0.8, n)
    if symmetric:                        # every hill mirrored through (z, cos) -> (-z, -cos)
        z, c = np.r_[z, -z], np.r_[c, -c]
    else:
        z = np.abs(z)
    return z, c, np.full(len(z), 0.5)


def test_mirrored_hills_give_a_symmetric_pmf_zeroed_in_bulk(tmp_path):
    z, c, h = _hills(np.random.default_rng(0), 200, symmetric=True)
    _walker(tmp_path, "w0", tmp_path, z, c, h)
    (e,) = P.pmf_history(str(tmp_path), BULK, n=4)
    assert e["pmf"].shape == (4, 121)
    assert np.abs(e["pmf"][-1, P.bulk_mask(e["z"], BULK)].mean()) < 1e-9
    vma, vaa = P.symmetry_error(e["z"], e["pmf"][-1], 2.8)
    assert vma < 1e-6
    assert e["pmf"][-1][60] < -5.0                         # bias filled the centre: F = -V is a well there


def test_one_sided_hills_are_asymmetric(tmp_path):
    z, c, h = _hills(np.random.default_rng(1), 200, symmetric=False)
    _walker(tmp_path, "w0", tmp_path, z, c, h)
    (e,) = P.pmf_history(str(tmp_path), BULK, n=2)
    assert P.symmetry_error(e["z"], e["pmf"][-1], 2.8)[0] > 5.0


def test_isolated_walkers_are_separate_estimates_and_shared_ones_one(tmp_path):
    rng = np.random.default_rng(2)
    iso, sh = tmp_path / "iso", tmp_path / "shared"
    for k in range(3):
        _walker(iso, f"w{k}", iso / f"bias_w{k}", *_hills(rng, 50, True))
        _walker(sh, f"w{k}", sh / "bias", *_hills(rng, 50, True))
    a = P.arm_estimates([str(iso)], BULK, n=5, n_t=5)
    b = P.arm_estimates([str(sh)], BULK, n=5, n_t=5)
    assert a["pmf"].shape[0] == 3 and b["pmf"].shape[0] == 1
    assert a["agg_ns"][-1] == pytest.approx(3 * a["t_ns"][-1])
    assert b["t_ns"][-1] == pytest.approx(3 * 1.0)         # 3 walkers x 100 hills x 0.01 ns, summed


def test_symmetrise_and_features():
    z = np.linspace(-3, 3, 61)
    f = 2.0 * np.exp(-z ** 2 / 0.1) - 1.5 * np.exp(-(np.abs(z) - 1.2) ** 2 / 0.1) + 0.3 * z
    s = P.symmetrise(f)
    assert np.allclose(s, s[::-1])
    feat = P.profile_features(z, s, inner=1.8)
    assert feat["centre"] == pytest.approx(2.0 - 1.5 * np.exp(-14.4), abs=1e-3)
    assert feat["z_well"] == pytest.approx(1.2, abs=0.05)


def test_symmetry_error_needs_a_symmetric_grid():
    with pytest.raises(ValueError, match="not supported"):
        P.symmetry_error(np.linspace(-3, 2, 11), np.zeros(11), 2.0)


def test_transits_count_full_crossings_and_ignore_the_periodic_image():
    through = [2.5, 1.5, 0.2, -1.5, -2.5]                 # bulk+ -> interior -> bulk-
    back = [-2.5, -0.5, 2.5]                               # and back
    bounce = [2.5, 1.5, 0.3, 1.5, 2.5]                     # in and out on the same side
    image = [2.9, -2.9]                                    # wraps through the water: not a crossing
    r = P.transits(np.array(through + back + bounce + image), bulk_lo=2.2, inner=1.0, half_width=3.0)
    assert r["full"] == 2 and r["first_full"] == 4
    assert r["half"] == 6


def test_walker_transits_reads_colvar_with_record_time(tmp_path):
    z = np.array([2.5, 0.0, -2.5, -2.5])
    _walker(tmp_path, "w0", tmp_path, z, np.zeros(4), np.full(4, 0.1))
    r = P.walker_transits(str(tmp_path), bulk_lo=2.2, inner=1.0)["w0"]
    assert r["full"] == 1 and r["first_full_ns"] == pytest.approx(0.02)


def test_fes2d_is_one_map_per_bias_with_z_on_columns(tmp_path):
    rng = np.random.default_rng(3)
    for k in range(2):
        _walker(tmp_path, f"w{k}", tmp_path / f"bias_w{k}", *_hills(rng, 50, True))
    maps = P.fes2d(str(tmp_path))
    assert len(maps) == 2
    m = maps[0]
    assert m["fes"].shape == (41, 121) and m["y_name"] == "cos_theta"
    assert m["fes"].min() == pytest.approx(0.0)
    # mirrored hills -> the map is symmetric under (z, cos) -> (-z, -cos)
    assert np.allclose(m["fes"], m["fes"][::-1, ::-1], atol=1e-6)
