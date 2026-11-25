import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_filter

from autopath.pulling import Diagnostics as D

from .smd_toy_run import analysed_dir


def _metrics_csv(path, cols):
    rows = []
    for sp in (0.01, 0.02):
        for n in (2, 3, 4):
            rows.append({"speed": sp, "path": "mixture", "n_replicas": n,
                         **{c: float(n) * (1 + sp) for c in cols}})
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    out = analysed_dir(tmp_path_factory.mktemp("toy"))
    _metrics_csv(out / "sMD_conv_vALL_metrics.csv", list(D._CONV_METRICS))
    return out


@pytest.fixture
def captured(monkeypatch):
    figs = []
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: figs.append(plt.gcf()))
    return figs


def _visible(fig):
    return [a for a in fig.axes if a.get_visible()]


def test_plot_analysis_writes_figures_1_to_4(toy):
    written = D.plot_analysis(str(toy))
    names = {p.rsplit("/", 1)[1] for p in written}
    assert names == {"sMD_convergence_metrics.svg", "free_energy.svg",
                     "Gamma_local.svg", "Gamma_cumulative.svg"}
    assert all((toy / n).stat().st_size > 0 for n in names)
    assert not (toy / "dG.svg").exists() and not (toy / "Wdiss.svg").exists()
    assert not list(toy.glob("work_profiles_*.svg"))


def test_free_energy_facets(toy, captured):
    D.plot_free_energy(pd.read_csv(toy / "mixture_pmfs.csv"), str(toy / "x.svg"))
    fig = captured[0]
    assert [a.get_title() for a in _visible(fig)] == ["0.02 nm/ps", "0.01 nm/ps", "v → 0"]
    assert any(ln.get_linestyle() == "--" for ln in fig.axes[0].lines)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "force (finite v)" in labels and "force" in labels


def test_gamma_local_grid(toy, captured):
    D.plot_gamma_local(pd.read_csv(toy / "friction.csv"), str(toy / "x.svg"))
    fig = captured[0]
    axes = np.array(fig.axes).reshape(3, 3)
    assert [a.get_title() for a in axes[0]] == ["0.02 nm/ps", "0.01 nm/ps", "v → 0"]
    assert all(a.get_title() == "" for a in axes[1:].ravel())
    for a in axes.ravel():
        al = [ln.get_alpha() for ln in a.lines]
        assert len(al) % 2 == 0
        assert al[0::2] == [0.3] * (len(al) // 2) or a in axes[2, :2]
        assert all(x == 1.0 for x in al[1::2]) or a in axes[2, :2]
    fin0, fin1, v0 = axes[2]
    assert fin0.get_shared_y_axes().joined(fin0, fin1)
    assert not fin0.get_shared_y_axes().joined(fin0, v0)


def test_display_smooth_short_and_long():
    y = np.array([1.0, 3.0, 2.0, 5.0])
    assert np.array_equal(D._display_smooth(np.arange(4) * 0.001, y), y)
    r = np.arange(400) * 0.001
    y = np.sin(r * 20) + np.random.default_rng(0).normal(size=400) * 0.1
    assert np.allclose(D._display_smooth(r, y), savgol_filter(y, 51, 3, mode="interp"))


def test_convergence_metrics_grid(tmp_path, captured):
    f6 = _metrics_csv(tmp_path / "a.csv", list(D._CONV_METRICS))
    D.plot_convergence_metrics([f6], str(tmp_path))
    assert len(_visible(captured[0])) == 6
    f3 = _metrics_csv(tmp_path / "b.csv", ["barrier_delta", "r_ts", "n_paths"])
    D.plot_convergence_metrics([f3], str(tmp_path), tolerances={"barrier_delta": 3.0})
    fig = captured[1]
    assert len(_visible(fig)) == 3 and len(fig.axes) - len(_visible(fig)) == 3
    ax = next(a for a in fig.axes if a.get_title() == "Barrier change")
    hl = [ln for ln in ax.lines if len(set(ln.get_ydata())) == 1 and list(ln.get_ydata())[0] == 3.0]
    assert len(hl) == 1
    assert "tolerance" in [t.get_text() for t in fig.legends[0].get_texts()]


def test_figures_single_speed(tmp_path, captured, caplog):
    out = analysed_dir(tmp_path, speeds=(0.01,), estimators=("cumulant", "jarzynski"))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        D.plot_free_energy(pd.read_csv(out / "mixture_pmfs.csv"), str(out / "a.svg"))
        D.plot_gamma_local(pd.read_csv(out / "friction.csv"), str(out / "b.svg"))
    assert [a.get_title() for a in _visible(captured[0])] == ["0.01 nm/ps"]
    assert len(captured[1].axes) == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_plot_failure_is_warning(toy, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(D, "plot_gamma_local", boom)
    (toy / "free_energy.svg").unlink(missing_ok=True)
    with caplog.at_level(logging.WARNING):
        written = D.plot_analysis(str(toy))
    assert (toy / "free_energy.svg").exists()
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and "Gamma_local" in warns[0].getMessage()
    assert str(toy / "free_energy.svg") in written
