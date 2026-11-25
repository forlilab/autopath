import logging
import shutil

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


def test_plot_analysis_writes_all_figures(toy):
    written = D.plot_analysis(str(toy))
    names = {p.rsplit("/", 1)[1] for p in written}
    assert names == {"sMD_convergence_metrics.svg", "free_energy.svg",
                     "Gamma_local.svg", "Gamma_cumulative.svg",
                     "paths.png", "path_decomposition.png", "jarzynski_convergence.svg"}
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


def test_paths_heatmap_order_and_topn(toy, captured):
    pdir = toy / "path_analysis"
    dist = np.load(pdir / "dtw_distance_vGlobal.npy")
    labels = pd.read_csv(pdir / "dtw_labels_vGlobal.csv")["path"].to_numpy()
    o = np.argsort(labels.astype(str), kind="stable")
    D.plot_paths(str(pdir), "Global", str(toy / "x.png"))
    fig = captured[0]
    img = fig.axes[0].collections[0].get_array()
    assert np.allclose(np.ma.filled(img, np.nan).reshape(dist.shape), dist[np.ix_(o, o)])
    n_feat = len(pd.read_csv(pdir / "pca_loadings_vGlobal.csv"))
    assert len(fig.axes[3].get_yticklabels()) == min(25, n_feat)
    D.plot_paths(str(pdir), "Global", str(toy / "x.png"), n_top_loadings=1)
    assert len(captured[1].axes[3].get_yticklabels()) == 1
    assert [ln.get_xdata()[0] for ln in captured[1].axes[1].lines] == [2]


def test_paths_final_labels_override(toy, captured):
    pdir = toy / "path_analysis"
    lab = pd.read_csv(pdir / "dtw_labels_vGlobal.csv")
    D.plot_paths(str(pdir), "Global", str(toy / "x.png"),
                 final_labels={t: "path-4" for t in lab["trajname"]})
    colors = {tuple(c.get_facecolor()[0][:3]) for c in captured[0].axes[2].collections[1:]}
    from matplotlib.colors import to_rgba
    assert colors == {to_rgba(D._path_color("path-4"))[:3]}


def test_null_model_skips_path_figures(tmp_path, caplog):
    out = analysed_dir(tmp_path, path_model="null")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        written = D.plot_analysis(str(out))
    names = {p.rsplit("/", 1)[1] for p in written}
    assert "paths.png" not in names and "path_decomposition.png" not in names
    assert not (out / "paths.png").exists() and not (out / "path_decomposition.png").exists()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def _relabel(src, dst, mapper, drop):
    shutil.copytree(src, dst)
    for name in ("path_pmfs.csv", "path_friction.csv", "path_quality.csv", "pulls.csv"):
        df = pd.read_csv(dst / name)
        df["path"] = [mapper(p, s) for p, s in zip(df["path"], df["speed"])]
        if name != "pulls.csv":
            df = df[~df["path"].isin(drop(df))]
        df.to_csv(dst / name, index=False)


@pytest.mark.parametrize("suffix", [False, True])
def test_path_decomposition_sparse_labels(toy, tmp_path, captured, caplog, suffix):
    ren = {"path-0": "path-1", "path-1": "path-3"}
    mapper = (lambda p, s: f"{ren[p]}_v{s:g}") if suffix else (lambda p, s: ren[p])
    dst = tmp_path / "d"
    _relabel(toy, dst, mapper, lambda df: [])
    for name in ("path_pmfs.csv", "path_friction.csv", "path_quality.csv"):
        df = pd.read_csv(dst / name)
        miss = df["path"].map(lambda p: p.startswith("path-3")) & (df["speed"] == 0.02)
        df[~miss].to_csv(dst / name, index=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        D.plot_path_decomposition(str(dst), str(dst / "x.png"))
    fig = captured[0]
    c1, c3 = D._cycle_colors()[1], D._cycle_colors()[3]
    fast, slow = fig.axes[0], fig.axes[1]
    assert c3 not in {ln.get_color() for ln in fast.lines} and c1 in {ln.get_color() for ln in fast.lines}
    assert {c1, c3} <= {ln.get_color() for ln in slow.lines}
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_jarzynski_figure_axes(toy, captured):
    from autopath.pulling.outputs import jarzynski_convergence, load_traces
    diag = jarzynski_convergence(load_traces(str(toy)), pd.read_csv(toy / "pulls.csv"), 300.0)
    D.plot_jarzynski_convergence(diag, str(toy / "x.svg"))
    fig = captured[0]
    assert len(fig.axes) == 4
    for ax in fig.axes:
        assert len(ax.lines) == 2 and ax.get_xlabel() == "Pulls"
    assert fig.axes[1].get_legend() is not None
    assert all(a.get_legend() is None for i, a in enumerate(fig.axes) if i != 1)


def test_plot_analysis_old_dir(toy, tmp_path, caplog):
    from autopath.pulling.outputs import load_traces
    old = tmp_path / "old"
    old.mkdir()
    for n in ("mixture_pmfs.csv", "friction.csv", "sMD_conv_vALL_metrics.csv"):
        shutil.copy(toy / n, old / n)
    load_traces(str(toy)).to_csv(old / "sMD_processed_data.csv", index=False)
    with caplog.at_level(logging.INFO):
        written = D.plot_analysis(str(old))
    assert {p.rsplit("/", 1)[1] for p in written} == {
        "sMD_convergence_metrics.svg", "free_energy.svg", "Gamma_local.svg", "Gamma_cumulative.svg"}
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    skips = [r for r in caplog.records if r.levelno == logging.INFO and "skipped" in r.getMessage()]
    assert len(skips) == 3


def test_plot_analysis_written_excludes_failed(toy, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(D, "plot_paths", boom)
    names = {p.rsplit("/", 1)[1] for p in D.plot_analysis(str(toy))}
    assert "paths.png" not in names and "path_decomposition.png" in names
