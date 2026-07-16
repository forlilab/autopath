"""Tests for logged k + realized-speed handling and adaptive force sampling."""
import numpy as np, pandas as pd, pytest
from autopath.pulling.Estimators import recover_spring_constant, extrapolate_to_v0
from autopath.pulling.SMDData import SMDData


# ---------------------------------------------------------------- logged k
def test_recover_spring_constant_prefers_logged_k():
    # raw_data force column is deliberately inconsistent with the identity;
    # logged_k must win regardless.
    df = pd.DataFrame({'r_target': [1.0, 1.1], 'r_before': [1.02, 1.13],
                       'force': [999.0, -12.0]})
    assert recover_spring_constant(df, logged_k=3765.6) == pytest.approx(3765.6)


@pytest.mark.parametrize("bad", [None, 0.0, -5.0, np.nan])
def test_recover_spring_constant_ignores_invalid_logged_k(bad):
    k = 5000.0
    rt = np.linspace(1.0, 2.0, 30); rb = rt - 0.02
    df = pd.DataFrame({'r_target': rt, 'r_before': rb, 'force': -k * (rb - rt)})
    assert recover_spring_constant(df, logged_k=bad) == pytest.approx(k, rel=1e-9)


# ---------------------------------------------------- realized-speed regression
def _linear_results(a, b, nominal, realized):
    """param = a + b*realized_speed, labelled by nominal speed, 3 shared steps."""
    rows = []
    for step in range(3):
        for sp_nom, sp_real in zip(nominal, realized):
            rows.append(dict(step=step, r_coord=1.0 + 0.1 * step, speed=sp_nom,
                             estimator='force', param=a + b * sp_real))
    return pd.DataFrame(rows)


def test_extrapolate_to_v0_uses_realized_speed_map():
    a, b = 10.0, 500.0
    nominal = [0.01, 0.02]
    realized = [0.01, 0.0208]
    df = _linear_results(a, b, nominal, realized)
    rmap = dict(zip(nominal, realized))

    with_map = extrapolate_to_v0(df, param='param', realized_speed_map=rmap)
    # regressing on realized speed recovers the true intercept a and slope b
    np.testing.assert_allclose(with_map['param'].to_numpy(), a, atol=1e-6)
    np.testing.assert_allclose(with_map['param_slope'].to_numpy(), b, atol=1e-3)

    # without the map the (mislabelled) nominal x-axis gives a different intercept
    without = extrapolate_to_v0(df, param='param')
    assert not np.allclose(without['param'].to_numpy(), a, atol=1e-3)


def test_extrapolate_identity_when_map_is_identity():
    nominal = [0.01, 0.02]
    df = _linear_results(3.0, 700.0, nominal, nominal)
    a = extrapolate_to_v0(df, param='param')
    b = extrapolate_to_v0(df, param='param',
                          realized_speed_map={s: s for s in nominal})
    np.testing.assert_allclose(a['param'].to_numpy(), b['param'].to_numpy(), atol=1e-9)


# ------------------------------------------------------- header metadata parse
_HEADER = (
    "# spring_constant_kJ_mol_nm2=3765.6\n"
    "# requested_speed_nm_per_ps=0.02\n"
    "# realized_speed_nm_per_ps=0.0208333\n"
    "# steps_per_move=12\n"
    "# force_n_samples=3\n"
)
_COLS = "step,time,r_target,r_before,r_after,force,force_sem,U_cvpack,dW_protocol,lag_nm\n"


def _write_dat(path, speed_token, realized, n=40, k=3765.6):
    r_target = np.linspace(0.8, 1.4, n)
    r_before = r_target - 0.001
    r_after = r_target - 0.0005
    hdr = _HEADER.replace("0.0208333", f"{realized:.7g}")
    with open(path, "w") as f:
        f.write(hdr); f.write(_COLS)
        for i in range(n):
            force = -k * (r_before[i] - r_target[i])
            f.write(f"{i},{i*0.1},{r_target[i]},{r_before[i]},{r_after[i]},"
                    f"{force},0.1,0.0,0.0,0.0005\n")


def test_parse_log_metadata(tmp_path):
    p = tmp_path / "sMD_replica-1_v0.02_forward.dat"
    _write_dat(p, "0.02", 0.0208333)
    meta = SMDData.parse_log_metadata(str(p))
    assert meta['spring_constant_kJ_mol_nm2'] == pytest.approx(3765.6)
    assert meta['realized_speed_nm_per_ps'] == pytest.approx(0.0208333)
    assert meta['steps_per_move'] == 12
    # data still parses (comment lines skipped)
    df = pd.read_csv(p, comment='#')
    assert len(df) == 40 and {'force', 'force_sem'} <= set(df.columns)


def test_parse_log_metadata_absent_returns_empty(tmp_path):
    p = tmp_path / "sMD_replica-9_v0.02_forward.dat"
    with open(p, "w") as f:
        f.write(_COLS); f.write("0,0.0,1.0,0.999,0.9995,3.7656,0,0,0,0.0005\n")
    assert SMDData.parse_log_metadata(str(p)) == {}


def test_smddata_exposes_logged_k_and_realized_map(tmp_path):
    files = []
    for rid, (token, realized) in enumerate([("0.02", 0.0208333), ("0.01", 0.01)], start=1):
        p = tmp_path / f"sMD_replica-{rid}_v{token}_forward.dat"
        _write_dat(str(p), token, realized)
        files.append(str(p))
    d = SMDData(log_files=files, completion_threshold_nm=0.0)
    assert d.spring_constant == pytest.approx(3765.6)
    assert d.realized_speed_map[0.02] == pytest.approx(0.0208333)
    assert d.realized_speed_map[0.01] == pytest.approx(0.01)


def test_smddata_backward_compat_without_header(tmp_path):
    # logs without header -> realized speed falls back to nominal, k is None
    files = []
    for rid, token in enumerate(["0.02", "0.01"], start=1):
        p = tmp_path / f"sMD_replica-{rid}_v{token}_forward.dat"
        with open(p, "w") as f:
            f.write(_COLS)
            for i in range(40):
                rt = 0.8 + 0.6 * i / 39
                f.write(f"{i},{i*0.1},{rt},{rt-0.001},{rt-0.0005},3.7656,0.1,0,0,0.0005\n")
        files.append(str(p))
    d = SMDData(log_files=files, completion_threshold_nm=0.0)
    assert d.spring_constant is None
    assert d.realized_speed_map[0.02] == pytest.approx(0.02)
