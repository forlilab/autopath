# tests/test_learned_rupture.py
"""Rupture (B/TS/U) labels, rupture-class learn_cv and its report."""
import numpy as np
import pandas as pd
import pytest

from autopath.metadynamics.learned.classes import RUPTURE_ORDER, rupture_labels

DT = 0.05


def _trace(f_of_t, t_end=100.0, noise=0.0, seed=0):
    t = np.round(np.arange(0.0, t_end + DT / 2, DT), 6)
    f = f_of_t(t)
    if noise:
        f = f + np.random.default_rng(seed).normal(0.0, noise, len(t))
    return pd.DataFrame({"time": t, "force": f, "r_after": 0.8 + 0.02 * t})


def _single(t):
    # 0 until 30.5 ps, linear rise to 100 at 50.5 ps, drop to the drag plateau 10 at 56.5 ps
    return np.select([t < 30.5, t < 50.5, t < 56.5],
                     [0.0, 5.0 * (t - 30.5), 100.0 - 15.0 * (t - 50.5)], 10.0)


def _two(t):
    p1 = np.clip(100.0 - 100.0 * np.abs(t - 20.5) / 3.0, 0.0, None)            # 100 at 20.5 ps
    p2 = np.select([t < 44.5, t < 50.5, t < 56.5],
                   [0.0, 200.0 * (t - 44.5) / 6.0, 200.0 - 190.0 * (t - 50.5) / 6.0], 10.0)
    return p1 + p2


def _idx(lab, c):
    return np.flatnonzero(lab == c).tolist()


def test_single_peak_classes_and_windows_are_exact():
    lab, info = rupture_labels(_trace(_single), 101, window_ps=DT)      # one-row window: no smoothing
    assert info["reason"] == ""
    assert info["base_b"] == 0.0 and info["base_u"] == 10.0 and info["dF"] == pytest.approx(90.0)
    assert info["t_rise"] == pytest.approx(34.5, abs=0.1)
    assert info["t_peak"] == pytest.approx(50.5, abs=1e-9)
    assert info["t_slip"] == pytest.approx(53.5, abs=0.1)
    assert info["t_free"] == pytest.approx(55.9, abs=0.1)
    assert _idx(lab, "B") == list(range(35))
    assert _idx(lab, "TS") == [51, 52, 53]
    assert _idx(lab, "U") == list(range(56, 101))
    assert set(lab) == set(RUPTURE_ORDER) | {""}


def test_noisy_peak_is_smoothed():
    lab, info = rupture_labels(_trace(_single, noise=10.0, seed=1), 101)
    assert info["reason"] == ""
    assert 49.0 <= info["t_peak"] <= 52.0
    ts = _idx(lab, "TS")
    assert ts and min(ts) >= 50 and max(ts) <= 55
    assert max(_idx(lab, "B")) <= 36
    u = _idx(lab, "U")
    assert min(u) >= 54 and len(u) >= 40


def test_two_peaks_use_the_global_maximum():
    lab, info = rupture_labels(_trace(_two), 101, window_ps=DT)
    assert info["reason"] == ""
    assert info["t_peak"] == pytest.approx(50.5, abs=1e-9)
    assert _idx(lab, "TS") == [51, 52, 53]
    assert _idx(lab, "B") == list(range(19))          # B ends at the first rise, before the small peak


def test_late_peak_is_excluded():
    lab, info = rupture_labels(_trace(lambda t: 2.0 * t), 101, window_ps=DT)
    assert info["reason"] == "peak in the last 10% of the pull"
    assert (lab == "").all()


def test_flat_trace_is_excluded():
    lab, info = rupture_labels(_trace(lambda t: np.full_like(t, 5.0)), 50, window_ps=DT)
    assert info["reason"] == "no force rise"
    assert (lab == "").all() and len(lab) == 50


def test_no_settling_is_excluded():
    # plateau 10 for most of the tail, but the last rows rise above the free threshold
    def f(t):
        return np.where(t > 99.0, 40.0, _single(t))
    lab, info = rupture_labels(_trace(f), 101, window_ps=DT)
    assert info["base_u"] == 10.0                     # 20 of 201 tail rows at 40: the median stays 10
    assert info["reason"] == "force never settles to the unbound baseline"
    assert (lab == "").all()


def test_frame_mapping_uses_time_fraction_and_reads_dat_files(tmp_path):
    df = _trace(_single)
    p = tmp_path / "sMD_replica-1_v0.01_forward.dat"
    with open(p, "w") as fh:
        fh.write("# spring_constant_kJ_mol_nm2=3765.6\n# steps_per_move=12\n")
        df.assign(step=np.arange(len(df))).to_csv(fh, index=False)
    lab, info = rupture_labels(str(p), 11, window_ps=DT)                # 11 frames at t = 0, 10, ..., 100
    assert lab.tolist() == ["B"] * 4 + ["", ""] + ["U"] * 5
    assert info["t_peak"] == pytest.approx(50.5)


def test_start_of_pull_spike_is_not_the_rupture():
    def f(t):
        return _single(t) + 500.0 * (t < 0.12)
    lab, info = rupture_labels(_trace(f), 101, window_ps=DT)
    assert info["reason"] == ""
    assert info["t_peak"] == pytest.approx(50.5, abs=1e-9)
    assert info["base_b"] == 0.0
    assert _idx(lab, "TS") == [51, 52, 53]
    assert _idx(lab, "B") == list(range(35))
    assert _idx(lab, "U") == list(range(56, 101))
