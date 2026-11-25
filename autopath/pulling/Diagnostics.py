from autopath.pulling.SMDData import SMDData
import pandas as pd
import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
# plt.rcParams["axes.edgecolor"] = 'black'

_STYLE = ["fivethirtyeight"]
_RC_OVERRIDES = {
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.facecolor": "white",
}

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import MDAnalysis as mda
from MDAnalysis.analysis import align, density
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

import re
import functools
import logging
logger = logging.getLogger(__name__)

def plot_convergence_traces(smd_conv_traces: list[str], outdir: str):
    """Plot PMF convergence traces as replica count increases sequentially.

    "Convergence" here means that the estimated PMF (or Wdiss, dG, …) is
    recomputed after adding each successive replica.  Each curve in the
    output represents the full profile at a given replica count, so the
    spread of curves shows how quickly the estimate stabilises.

    One figure is produced per quantity (e.g. ``dG``, ``Wdiss``).  Within
    each figure, panels are faceted by pulling speed.  Line style encodes
    the path/cluster; colour encodes the number of replicas used, running
    from cool (few replicas) to warm (many replicas) via the ``coolwarm_r``
    colormap normalised per speed panel.

    Parameters
    ----------
    smd_conv_traces : list of str
        Paths to CSV files produced by the sMD convergence analysis.
        Each file must contain columns: ``quantity``, ``speed``, ``path``,
        ``n_replicas``, ``r_coord``, ``value``.
    outdir : str
        Directory where figures are written.  One SVG per quantity:
        ``{outdir}/sMD_convergence_{quantity}.svg``.
    """
    if len(smd_conv_traces) == 0:
        logger.warning("No data files provided for convergence traces plotting.")
        return

    # load all data
    all_data = []
    for f in smd_conv_traces:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty traces file: {f}")
            continue
        all_data.append(df)

    if not all_data:
        logger.warning("No convergence traces data available for plotting.")
        return

    traces_df = pd.concat(all_data, ignore_index=True)

    # define line styles for clusters
    LINESTYLES = [
        "-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))
    ]

    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        for quantity, group in traces_df.groupby("quantity"):

            speeds = sorted(group["speed"].unique())
            fig, axes = plt.subplots(
                1, len(speeds),
                figsize=(5 * len(speeds), 4),
                sharey=True
            )

            if len(speeds) == 1:
                axes = [axes]

            for ax, speed in zip(axes, speeds):

                g_speed = group[group["speed"] == speed]

                # per-speed color normalization
                norm = mcolors.Normalize(
                    vmin=g_speed["n_replicas"].min(),
                    vmax=g_speed["n_replicas"].max()
                )
                cmap = cm.get_cmap("coolwarm_r")

                # iterate over clusters / paths
                paths = sorted(g_speed["path"].unique())

                for i, path in enumerate(paths):

                    g_path = g_speed[g_speed["path"] == path]
                    linestyle = LINESTYLES[i % len(LINESTYLES)]

                    for n_rep, gg in g_path.groupby("n_replicas"):
                        color = cmap(norm(n_rep))
                        # Plot in ascending r order.
                        gg = gg.sort_values("r_coord")

                        ax.plot(
                            gg["r_coord"],
                            gg["value"],
                            color=color,
                            linestyle=linestyle,
                            linewidth=2.0,
                            alpha=0.9,
                        )

                ax.set_title(f"speed = {speed} nm/ps")
                ax.set_xlabel("r_coord (nm)")
                ax.grid(True)

            axes[0].set_ylabel(f"{quantity} (kJ/mol)")
            plt.tight_layout()
            plt.savefig(f"{outdir}/sMD_convergence_{quantity}.svg", dpi=300)
            plt.close()

        return

_EST_ORDER = ["cumulant", "jarzynski", "force"]
_PATH_FALLBACK = "#8b8b8b"
_CONV_METRICS = {
    "dG_weighted-rmsd": ("ΔG RMSD", "ΔG RMSD vs previous (kJ/mol)"),
    "barrier_delta": ("Barrier change", "Barrier change (kJ/mol)"),
    "r_ts_delta": ("TS shift", "TS shift (nm)"),
    "barrier_height": ("Barrier height", "Barrier height (kJ/mol)"),
    "r_ts": ("TS position", "TS position (nm)"),
    "n_paths": ("Number of paths", "Number of paths"),
}
_CONV_SKIP = {"speed", "path", "n_replicas", "converged", "decision_quantity",
              "n_common_points", "reason", "estimator"}


def _cycle_colors() -> list[str]:
    with plt.style.context(_STYLE):
        return [c["color"] for c in plt.rcParams["axes.prop_cycle"]]


def _speed_order(speeds) -> list[float]:
    """Finite speeds fastest first, then v -> 0 (speed 0)."""
    s = sorted({float(x) for x in speeds})
    return [x for x in s if x > 0][::-1] + [x for x in s if x == 0]


def _speed_title(v: float) -> str:
    return "v → 0" if v == 0 else f"{v:g} nm/ps"


def _estimator_colors(names) -> dict:
    """Fixed cycle colours for the standard estimators, next free ones otherwise."""
    cyc = _cycle_colors()
    out = {e: cyc[i] for i, e in enumerate(_EST_ORDER)}
    free = len(_EST_ORDER)
    for n in names:
        if n not in out:
            out[n] = cyc[free % len(cyc)]
            free += 1
    return {n: out[n] for n in names}


def _path_color(label: str) -> str:
    """Cycle colour k for a label starting ``path-k``; grey otherwise."""
    m = re.match(r"path-(\d+)", str(label))
    if not m:
        return _PATH_FALLBACK
    cyc = _cycle_colors()
    return cyc[int(m.group(1)) % len(cyc)]


def _display_smooth(r: np.ndarray, y: np.ndarray, width_nm: float = 0.05,
                    polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smooth over ``width_nm`` of r; display only."""
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    dr = np.diff(r)
    if len(y) < 2 or dr.size == 0 or not np.median(dr) > 0:
        return y
    n = int(round(width_nm / np.median(dr)))
    n += n % 2 == 0
    n = min(n, len(y) - (len(y) % 2 == 0))
    if n <= polyorder:
        return y
    return savgol_filter(y, n, polyorder, mode="interp")


def _ordered_estimators(df: pd.DataFrame) -> list[str]:
    present = set(df["estimator"])
    return [e for e in _EST_ORDER if e in present] + sorted(present - set(_EST_ORDER))


def _save(outfname: str) -> None:
    plt.tight_layout()
    plt.savefig(outfname, dpi=150, bbox_inches="tight")


def _closes_figures(fn):
    """Close the figures ``fn`` opened, also when it raises."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        before = set(plt.get_fignums())
        try:
            return fn(*a, **k)
        finally:
            for n in set(plt.get_fignums()) - before:
                plt.close(n)
    return wrapper


@_closes_figures
def plot_convergence_metrics(smd_conv_metrics: list[str], outdir: str, tolerances: dict = None):
    """Plot convergence metrics vs. replica count, one panel per metric.

    Values are averaged across paths per (speed, n_replicas); one line per
    speed and one shared legend.

    Parameters
    ----------
    smd_conv_metrics : list of str
        CSV files with columns ``speed``, ``n_replicas`` and metric columns.
    outdir : str
        Writes ``{outdir}/sMD_convergence_metrics.svg``.
    tolerances : dict or None
        ``{metric_name: threshold}``; drawn as a dashed grey line on the
        matching panel.
    """
    if len(smd_conv_metrics) == 0:
        logger.warning("No data files provided for convergence metrics plotting.")
        return

    all_data = []
    for f in smd_conv_metrics:
        try:
            all_data.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty metrics file: {f}")
    if not all_data:
        logger.warning("No convergence metrics data available for plotting.")
        return

    df = pd.concat(all_data, ignore_index=True)
    if df.empty:
        logger.warning("No data available for convergence metrics plotting.")
        return

    metrics = [m for m in _CONV_METRICS if m in df.columns]
    metrics += [c for c in df.columns
                if c not in _CONV_SKIP and c not in _CONV_METRICS
                and pd.api.types.is_numeric_dtype(df[c])]
    if not metrics:
        logger.warning("No numeric convergence metrics found to plot.")
        return

    os.makedirs(outdir, exist_ok=True)
    speeds = sorted(df["speed"].unique())
    ncols = 3
    nrows = max(2, int(np.ceil(len(metrics) / ncols)))

    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        cyc = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
        sc = {s: cyc[i % len(cyc)] for i, s in enumerate(speeds)}
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, metric in zip(axes, metrics):
            agg = (df[["speed", "n_replicas", metric]].dropna()
                   .groupby(["speed", "n_replicas"], as_index=False)[metric].mean()
                   .sort_values(["speed", "n_replicas"]))
            for s in speeds:
                sub = agg[agg["speed"] == s]
                if len(sub):
                    ax.plot(sub["n_replicas"], sub[metric], color=sc[s], marker="o",
                            linewidth=2.0, alpha=0.95)
            if tolerances and metric in tolerances:
                ax.axhline(tolerances[metric], color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
            title, ylabel = _CONV_METRICS.get(metric, (metric, metric))
            if metric == "n_paths":
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, min_n_ticks=1))
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Replicas")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True)
        for ax in axes[len(metrics):]:
            ax.set_visible(False)

        handles = [Line2D([0], [0], color=sc[s], marker="o", lw=2, label=f"{s:g} nm/ps")
                   for s in speeds]
        if tolerances and any(m in tolerances for m in metrics):
            handles.append(Line2D([0], [0], color="gray", ls="--", lw=1.5, label="tolerance"))
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        _save(os.path.join(outdir, "sMD_convergence_metrics.svg"))


@_closes_figures
def plot_free_energy(mixture: pd.DataFrame, outfname: str) -> None:
    """One facet per speed (fastest first, v -> 0 last), one line per estimator."""
    so = _speed_order(mixture["speed"])
    ests = _ordered_estimators(mixture)
    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        col = _estimator_colors(ests)
        fig, axes = plt.subplots(1, len(so), figsize=(4 * len(so), 4), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, s in zip(axes, so):
            for e in ests:
                d = mixture[(mixture["speed"] == s) & (mixture["estimator"] == e)].sort_values("r_coord")
                if len(d):
                    ax.plot(d["r_coord"], d["dG"], color=col[e],
                            ls="--" if (e == "force" and s > 0) else "-")
            ax.set_title(_speed_title(s))
            ax.set_xlabel("r (nm)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        axes[0].set_ylabel("ΔG (kJ/mol)")
        hs = [Line2D([0], [0], color=col[e], label=e) for e in ests]
        if ((mixture["estimator"] == "force") & (mixture["speed"] > 0)).any():
            hs.append(Line2D([0], [0], color=col["force"], ls="--", label="force (finite v)"))
        fig.legend(handles=hs, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        _save(outfname)


@_closes_figures
def plot_gamma_local(friction: pd.DataFrame, outfname: str) -> None:
    """Γ(r) grid, one row per estimator and one column per speed.

    Raw curve faint, display-smoothed curve solid; y is shared along a row.
    In the force row finite speeds share one y and v -> 0 keeps its own.
    """
    so = _speed_order(friction["speed"])
    ests = _ordered_estimators(friction)
    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        col = _estimator_colors(ests)
        fig, axes = plt.subplots(len(ests), len(so), figsize=(4 * len(so), 3 * len(ests)),
                                 sharex=True, squeeze=False)
        for i, e in enumerate(ests):
            for j, s in enumerate(so):
                ax = axes[i, j]
                d = friction[(friction["speed"] == s) & (friction["estimator"] == e)]
                d = d.sort_values(["method", "r_coord"])
                fin_f = e == "force" and s > 0
                for _, g in d.groupby("method"):
                    r, y = g["r_coord"].to_numpy(), g["Gamma"].to_numpy()
                    ax.plot(r, y, color=col[e], alpha=0.15 if fin_f else 0.3)
                    ax.plot(r, _display_smooth(r, y), color=col[e], alpha=0.5 if fin_f else 1.0)
                if i == 0:
                    ax.set_title(_speed_title(s))
                if i == len(ests) - 1:
                    ax.set_xlabel("r (nm)")
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            grp = [axes[i, j] for j, s in enumerate(so) if s > 0] if e == "force" else list(axes[i])
            if grp:
                lo = min(a.get_ylim()[0] for a in grp)
                hi = max(a.get_ylim()[1] for a in grp)
                for a in grp[1:]:
                    a.sharey(grp[0])
                    a.tick_params(labelleft=False)
                grp[0].set_ylim(lo, hi)
            axes[i, 0].set_ylabel(f"{e}\nΓ (kJ·ps/mol/nm²)")
            if e == "force" and 0 in so:
                axes[i, -1].set_ylabel("")
        _save(outfname)


@_closes_figures
def plot_gamma_cumulative(friction: pd.DataFrame, outfname: str) -> None:
    """Cumulative friction per speed; force only at v -> 0."""
    df = friction[~((friction["estimator"] == "force") & (friction["speed"] > 0))]
    so = _speed_order(df["speed"])
    ests = _ordered_estimators(df)
    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        col = _estimator_colors(ests)
        fig, axes = plt.subplots(1, len(so), figsize=(4 * len(so), 4), squeeze=False)
        axes = axes[0]
        for ax, s in zip(axes, so):
            for e in ests:
                d = df[(df["speed"] == s) & (df["estimator"] == e)].sort_values(["method", "r_coord"])
                for _, g in d.groupby("method"):
                    ax.plot(g["r_coord"], g["Gamma_integrated"], color=col[e])
            ax.set_title(_speed_title(s))
            ax.set_xlabel("r (nm)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        axes[0].set_ylabel("∫Γ dr (kJ·ps/mol/nm)")
        fig.legend(handles=[Line2D([0], [0], color=col[e], label=e) for e in ests],
                   loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        _save(outfname)


def _path_key(label) -> str:
    """Label without its per-speed suffix (``path-0_v0.01`` -> ``path-0``)."""
    return str(label).split("_")[0]


def _path_sort(labels) -> list:
    def k(x):
        m = re.match(r"path-(\d+)", str(x))
        return (0, int(m.group(1)), str(x)) if m else (1, 0, str(x))
    return sorted(set(labels), key=k)


@_closes_figures
def plot_paths(path_dir: str, group: str, outfname: str, final_labels: dict | None = None,
               n_top_loadings: int = 25) -> None:
    """DTW matrix, silhouette, PCA space and top feature loadings of one path model.

    Parameters
    ----------
    path_dir : str
        The ``path_analysis`` directory.
    group : str
        Speed group suffix of the ``*_v{group}.*`` files (``"Global"`` or a speed).
    outfname : str
        Output image.
    final_labels : dict or None
        ``{trajname: path}`` overriding the as-clustered labels (post-merge).
    n_top_loadings : int
        Features shown, ranked by sqrt(PC1^2 + PC2^2).
    """
    import json

    f = lambda stem, ext: os.path.join(path_dir, f"{stem}_v{group}.{ext}")
    dist = np.load(f("dtw_distance", "npy"))
    lab = pd.read_csv(f("dtw_labels", "csv"))
    scores = pd.read_csv(f("pca_scores", "csv"))
    load = pd.read_csv(f("pca_loadings", "csv"))
    with open(f("path_model", "json")) as fh:
        model = json.load(fh)
    fl = final_labels or {}
    labels = np.array([fl.get(t, p) for t, p in zip(lab["trajname"], lab["path"])], dtype=str)
    evr = model.get("explained_variance_ratio", [np.nan, np.nan])
    sil = {int(k): v for k, v in model.get("silhouette", {}).items()}

    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        ax = axes.ravel()
        order = np.argsort(labels, kind="stable")
        sns.heatmap(dist[np.ix_(order, order)], cmap="viridis", ax=ax[0],
                    xticklabels=False, yticklabels=False)
        sl = labels[order]
        for b in np.where(sl[1:] != sl[:-1])[0] + 1:
            ax[0].axhline(b, color="white", lw=0.8)
            ax[0].axvline(b, color="white", lw=0.8)
        ax[0].set(title="DTW distance", xlabel="Trajectories", ylabel="Trajectories")
        ax[0].grid(False)

        if sil:
            ax[1].plot(list(sil), list(sil.values()), marker="o")
            ax[1].set_xticks(list(sil))
        ax[1].axvline(model["K"], color="red", linestyle="--", label=f"K = {model['K']}")
        ax[1].set(title="Silhouette score", xlabel="Number of paths", ylabel="Silhouette score")

        ax[2].scatter(scores["PC1"], scores["PC2"], alpha=0.75, color="lightgray", s=50, linewidth=0)
        pl = scores["trajname"].map(lambda t: fl.get(t)).fillna(scores["path"])
        med = scores[scores["is_medoid"].astype(bool)].assign(_p=pl)
        for p in _path_sort(med["_p"]):
            d = med[med["_p"] == p]
            ax[2].scatter(d["PC1"], d["PC2"], s=20, alpha=0.9, color=_path_color(p),
                          label=_path_key(p))
        if len(med):
            handles, names = ax[2].get_legend_handles_labels()
            seen = dict(zip(names, handles))
            ax[2].legend(seen.values(), seen.keys(), frameon=False)
        ax[2].set(title="PCA space", xlabel=f"PC1 ({evr[0] * 100:.1f}%)",
                  ylabel=f"PC2 ({evr[1] * 100:.1f}%)")

        contrib = np.hypot(load["PC1"], load["PC2"]).to_numpy()
        idx = np.argsort(contrib)[::-1][:n_top_loadings][::-1]
        y = np.arange(len(idx))
        w = 0.35
        ax[3].barh(y + w / 2, load["PC1"].to_numpy()[idx], w, color="#6d904f",
                   label=f"PC1 ({evr[0] * 100:.1f}%)")
        ax[3].barh(y - w / 2, load["PC2"].to_numpy()[idx], w, color="#810f7c",
                   label=f"PC2 ({evr[1] * 100:.1f}%)")
        ax[3].set_yticks(y)
        ax[3].set_yticklabels(load["feature"].to_numpy()[idx])
        ax[3].axvline(0, color="black", lw=0.8)
        ax[3].set(xlabel="Loadings", title="Feature contributions")
        ax[3].legend(frameon=False)
        _save(outfname)


def _sigma_ratio(traces: pd.DataFrame, speed: float):
    """r and sigma_W within-path / pooled for one speed."""
    g = traces[traces["speed"] == speed]
    W = g.pivot_table(index="step", columns="trajname", values="work")
    r = g.groupby("step")["r_target"].mean().reindex(W.index).to_numpy()
    path = g.drop_duplicates("trajname").set_index("trajname")["path"].reindex(W.columns)
    w = W.to_numpy()
    ss = np.zeros(len(w))
    dof = np.zeros(len(w))
    # steps past a path's last pull are all-NaN; their ratio is NaN by design
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for p in path.dropna().unique():
            x = w[:, (path == p).to_numpy()]
            ss += np.nansum((x - np.nanmean(x, axis=1, keepdims=True)) ** 2, axis=1)
            dof += np.sum(~np.isnan(x), axis=1) - 1
        return r, np.sqrt(ss / dof) / np.nanstd(w, axis=1, ddof=1)


@_closes_figures
def plot_path_decomposition(outdir: str, outfname: str) -> bool | None:
    """Per-path dG, friction and weights against the mixture, per speed.

    Parameters
    ----------
    outdir : str
        Analysis directory holding ``path_pmfs.csv``, ``path_friction.csv``,
        ``path_quality.csv``, ``mixture_pmfs.csv``, ``friction.csv`` and the traces.
    outfname : str
        Output image.
    """
    from autopath.pulling.outputs import load_traces

    rd = lambda n: pd.read_csv(os.path.join(outdir, n))
    pp = rd("path_pmfs.csv")
    pp = pp[pp["estimator"] == "cumulant"]
    if pp.empty:
        logger.info("path_decomposition skipped: no cumulant")
        return False
    pf = rd("path_friction.csv") if os.path.exists(os.path.join(outdir, "path_friction.csv")) else None
    mix, fr, pq = rd("mixture_pmfs.csv"), rd("friction.csv"), rd("path_quality.csv")
    speeds = [s for s in _speed_order(pp["speed"]) if s > 0]
    ncols = max(3, len(speeds))
    paths = _path_sort(pp["path"])
    keys = _path_sort(_path_key(p) for p in paths)
    key_col = {k: _path_color(k) for k in keys}
    pq = pq[pq["estimator"] == pq["estimator"].iloc[0]]

    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 9.5), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)

        def on(ax):
            ax.set_visible(True)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            return ax

        mx = mix[(mix["estimator"] == "cumulant")]
        fx = fr[(fr["estimator"] == "cumulant") & (fr["method"] == "derivative")]
        pfc = pf[(pf["estimator"] == "cumulant")] if pf is not None else None
        for j, s in enumerate(speeds):
            ax = on(axes[0, j])
            for p, g in pp[pp["speed"] == s].groupby("path"):
                g = g.sort_values("r_coord")
                ax.plot(g["r_coord"], g["dG"], color=_path_color(p))
            g = mx[mx["speed"] == s].sort_values("r_coord")
            ax.plot(g["r_coord"], g["dG"], color="k")
            ax.set_title(_speed_title(s))
            ax = on(axes[1, j])
            if pfc is not None:
                for p, g in pfc[(pfc["speed"] == s) & (pfc["method"] == "derivative")].groupby("path"):
                    g = g.sort_values("r_coord")
                    ax.plot(g["r_coord"], g["Gamma_integrated"], color=_path_color(p))
            g = fx[fx["speed"] == s].sort_values("r_coord")
            ax.plot(g["r_coord"], g["Gamma_integrated"], color="k")
            ax.set_xlabel("r (nm)")
            if j:
                for i in (0, 1):
                    axes[i, j].sharey(axes[i, 0])
                    axes[i, j].tick_params(labelleft=False)
        axes[0, 0].set_ylabel("cumulant\nΔG (kJ/mol)")
        axes[1, 0].set_ylabel("cumulant\n∫Γ dr (kJ·ps/mol/nm)")
        axes[1, 0].legend(handles=[Line2D([0], [0], color=key_col[k], label=k) for k in keys]
                          + [Line2D([0], [0], color="k", label="mixture")],
                          loc="upper left", frameon=False)

        fv = pf[(pf["estimator"] == "force") & (pf["speed"] == 0)] if pf is not None else None
        fa = fr[(fr["estimator"] == "force") & (fr["method"] == "regression") & (fr["speed"] == 0)]
        if fv is not None and len(fv) and len(fa):
            ax = on(axes[2, 0])
            for p, g in fv.groupby("path"):
                g = g.sort_values("r_coord")
                ax.plot(g["r_coord"], g["Gamma"], color=_path_color(p))
            g = fa.sort_values("r_coord")
            ax.plot(g["r_coord"], g["Gamma"], color="k")
            ax.set(xlabel="r (nm)", ylabel="force, v → 0\nΓ (kJ·ps/mol/nm²)")

        ax = on(axes[2, 1])
        bot = np.zeros(len(speeds))
        for k in keys:
            wv = np.array([pq[(pq["speed"] == s) & (pq["path"].map(_path_key) == k)]["weight"].sum()
                           for s in speeds])
            ax.bar(range(len(speeds)), wv, bottom=bot, color=key_col[k])
            bot += wv
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f"{s:g}" for s in speeds])
        ax.set(xlabel="Speed (nm/ps)", ylabel="Path weight")

        ax = on(axes[2, 2])
        cl = ["#6d904f", "#810f7c", "#8b8b8b"]
        tr = load_traces(outdir, ["trajname", "speed", "step", "work", "r_target", "path"])
        for k, s in enumerate(speeds):
            r, rat = _sigma_ratio(tr, s)
            ax.plot(r, rat, color=cl[k % len(cl)], label=f"{s:g} nm/ps")
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set(xlabel="r (nm)", ylabel="σ$_W$ within-path / pooled")
        ax.set_ylim(top=max(ax.get_ylim()[1], 1.15))
        ax.legend(frameon=False, loc="upper left")
        _save(outfname)


@_closes_figures
def plot_jarzynski_convergence(diag: pd.DataFrame, outfname: str) -> None:
    """Jarzynski estimator health against pull count, one line per speed.

    Parameters
    ----------
    diag : pandas.DataFrame
        Output of :func:`autopath.pulling.outputs.jarzynski_convergence`.
    outfname : str
        Output image.
    """
    cols = ["top_pull_weight", "ess_over_n", "tail_position", "beta_sigma_W"]
    lab = ["Top-pull weight", "Kish ESS / n", "(ΔG$_J$ − W$_{min}$) / (kT ln n)", "βσ$_W$"]
    with plt.style.context(_STYLE), plt.rc_context(_RC_OVERRIDES):
        cyc = _cycle_colors()
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.ravel()
        for k, (s, g) in enumerate(diag.groupby("speed")):
            g = g.sort_values("n")
            for ax, c in zip(axes, cols):
                ax.plot(g["n"], g[c], color=cyc[k % len(cyc)], label=f"{s:g} nm/ps")
        for ax, y in zip(axes, lab):
            ax.set(xlabel="Pulls", ylabel=y)
        axes[1].legend(frameon=False)
        _save(outfname)


def plot_analysis(outdir: str, temperature: float | None = None, tolerances: dict | None = None,
                  n_top_loadings: int = 25) -> list[str]:
    """Render the sMD analysis figures from the files in ``outdir``.

    Missing inputs skip the figure (INFO); a failing figure logs a WARNING
    and the rest continue.

    Returns
    -------
    list of str
        Paths of the figures written.
    """
    import json

    if temperature is None:
        try:
            with open(os.path.join(outdir, "summary.json")) as fh:
                temperature = json.load(fh).get("temperature_K")
        except (OSError, ValueError):
            temperature = None
        if temperature is None:
            temperature = 300.0
            logger.info("temperature not found, using 300 K")

    written: list[str] = []

    def _try(name, fn, *args):
        try:
            return fn(*args) is not False
        except Exception as exc:
            logger.warning(f"{name} not written: {exc}")
            return False

    def _need(figure, fname):
        path = os.path.join(outdir, fname)
        if os.path.exists(path):
            return path
        logger.info(f"{figure} skipped: no {fname}")
        return None

    def _done(ok, out):
        if ok:
            written.append(out)

    def _figure(name, fname, csv, fn):
        src = _need(name, csv)
        if src is None:
            return
        out = os.path.join(outdir, fname)
        _done(_try(name, lambda: fn(pd.read_csv(src), out)), out)

    src = _need("sMD_convergence_metrics", "sMD_conv_vALL_metrics.csv")
    if src is not None:
        out = os.path.join(outdir, "sMD_convergence_metrics.svg")
        _done(_try("sMD_convergence_metrics", plot_convergence_metrics, [src], outdir, tolerances), out)
    _figure("free_energy", "free_energy.svg", "mixture_pmfs.csv", plot_free_energy)
    _figure("Gamma_local", "Gamma_local.svg", "friction.csv", plot_gamma_local)
    _figure("Gamma_cumulative", "Gamma_cumulative.svg", "friction.csv", plot_gamma_cumulative)

    import glob
    pulls_csv = os.path.join(outdir, "pulls.csv")
    try:
        pulls = pd.read_csv(pulls_csv) if os.path.exists(pulls_csv) else None
    except Exception as exc:
        logger.warning(f"pulls.csv unreadable: {exc}")
        pulls = None
    pdir = os.path.join(outdir, "path_analysis")
    models = sorted(glob.glob(os.path.join(pdir, "path_model_v*.json")))
    if not models:
        logger.info("paths skipped: no path_analysis/path_model_v*.json")
    final = dict(zip(pulls["trajname"], pulls["path"])) if pulls is not None and "path" in pulls else None
    for m in models:
        g = os.path.basename(m)[len("path_model_v"):-len(".json")]
        try:
            with open(m) as fh:
                K = json.load(fh).get("K")
        except (OSError, ValueError):
            K = None
        if K is None:
            logger.info(f"paths_v{g} skipped: no K in {os.path.basename(m)}")
            continue
        if K <= 1:
            continue
        out = os.path.join(outdir, "paths.png" if g == "Global" else f"paths_v{g}.png")
        _done(_try(f"paths_v{g}", plot_paths, pdir, g, out, final, n_top_loadings), out)

    src = _need("path_decomposition", "path_pmfs.csv")
    if src is not None:
        out = os.path.join(outdir, "path_decomposition.png")
        try:
            multi = pd.read_csv(src)["path"].map(_path_key).nunique() > 1
        except Exception:
            multi = False
        if multi:
            _done(_try("path_decomposition", plot_path_decomposition, outdir, out), out)
        else:
            logger.info("path_decomposition skipped: one path")

    if _need("jarzynski_convergence", "pulls.csv") is not None:
        from autopath.pulling.outputs import jarzynski_convergence, load_traces
        out = os.path.join(outdir, "jarzynski_convergence.svg")

        def _jc():
            diag = jarzynski_convergence(load_traces(outdir), pulls, temperature)
            if diag.empty:
                logger.info("jarzynski_convergence skipped: fewer than 2 pulls per speed")
                return False
            plot_jarzynski_convergence(diag, out)

        _done(_try("jarzynski_convergence", _jc), out)
    return written

def _parse_speed_from_path(path_name: str) -> float | None:
    """Extract pulling speed (nm/ps) from a path name such as ``'path-0_v0.001'``.

    Returns ``None`` when the pattern ``_v{number}`` is absent (e.g. when
    clustering was performed across speeds without a per-speed suffix).
    """
    m = re.search(r'[_\-]v([\d.]+)', path_name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _get_friction_profile_for_path(
    path_name: str,
    friction_profiles: dict,
    friction_speed_override: float | None,
) -> "pd.Series | None":
    """Return the per-speed friction profile for *path_name*.

    Speed is parsed from the path name using the ``_v{number}`` suffix
    convention (e.g. ``path-0_v0.001`` → 0.001 nm/ps).  Speed=0 entries
    are always excluded from *friction_profiles* before this function is
    called because Gamma is undefined (the derivative-based estimator
    requires v > 0).

    Lookup order:
    1. Speed parsed from the path name (e.g. ``path-0_v0.001`` → 0.001 nm/ps).
    2. *friction_speed_override* when the name carries no speed suffix.
    3. Fastest available speed as a last resort.

    Returns ``None`` when *friction_profiles* is empty.
    """
    if not friction_profiles:
        return None
    path_speed = _parse_speed_from_path(path_name)
    if path_speed is None:
        path_speed = (friction_speed_override if friction_speed_override is not None
                      else max(friction_profiles))
        how = "override" if friction_speed_override is not None else "fastest available"
        logger.warning(
            f"Path '{path_name}' has no speed suffix; colouring by friction at "
            f"speed={path_speed} nm/ps ({how})."
        )
    if path_speed in friction_profiles:
        return friction_profiles[path_speed]
    # nearest speed fallback
    nearest = min(friction_profiles, key=lambda s: abs(s - path_speed))
    logger.info(
        f"No friction profile for speed={path_speed} nm/ps (path='{path_name}'); "
        f"using nearest available speed={nearest} nm/ps."
    )
    return friction_profiles[nearest]


def _build_friction_grid(dens, friction_profile: pd.Series, center_pos_nm: np.ndarray):
    """Build a volumetric friction map on the same grid as a density object.

    For each voxel, computes Euclidean distance from *center_pos_nm* (the
    membrane anchor, e.g. DUM atom) and looks up Gamma(r) from the friction
    profile via linear interpolation.  Values are clipped at 0 (negative
    Gamma arises from noise, not physics).

    Parameters
    ----------
    dens : gridData.Grid
        Density object whose grid geometry (origin, delta, shape) is reused.
    friction_profile : pd.Series
        Index = r_coord (nm), values = Gamma (kJ·ps/mol/nm²).
    center_pos_nm : (3,) array
        3-D position of the membrane center anchor in **nm**.

    Returns
    -------
    gridData.Grid
        Friction map on the same voxel grid as *dens*.
    """
    from scipy.interpolate import interp1d
    import gridData

    r_vals = friction_profile.index.to_numpy(dtype=float)
    g_vals = friction_profile.to_numpy(dtype=float)
    # Replace any residual NaN (e.g. from clipped/noisy profiles) with 0 before
    # interpolation — NaN propagates through interp1d and makes the entire grid NaN.
    n_nan = int(np.sum(~np.isfinite(g_vals)))
    if n_nan:
        logger.warning(
            f"_build_friction_grid: {n_nan}/{len(g_vals)} non-finite values "
            f"in friction profile replaced with 0 before interpolation."
        )
    g_vals = np.where(np.isfinite(g_vals), g_vals, 0.0)
    g_vals = np.clip(g_vals, 0, None)
    order = np.argsort(r_vals)
    r_vals, g_vals = r_vals[order], g_vals[order]

    gamma_interp = interp1d(
        r_vals, g_vals, kind="linear",
        bounds_error=False,
        fill_value=(g_vals[0], g_vals[-1]),
    )

    shape = dens.grid.shape
    origin_A = np.asarray(dens.origin)   # Å
    delta_A  = np.asarray(dens.delta)    # Å / voxel

    ix, iy, iz = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    x = origin_A[0] + ix * delta_A[0]
    y = origin_A[1] + iy * delta_A[1]
    z = origin_A[2] + iz * delta_A[2]

    center_A = center_pos_nm * 10.0      # nm → Å
    r_nm = np.sqrt((x - center_A[0])**2 + (y - center_A[1])**2 + (z - center_A[2])**2) / 10.0

    gamma_grid = gamma_interp(r_nm)
    return gridData.Grid(gamma_grid, origin=dens.origin, delta=dens.delta)


def _transparency_ladder(first: float, last: float, n: int) -> List[float]:
    """Linear stick-transparency ramp across *n* conformations.

    Returns *n* values running from *first* (earliest snapshot) to *last*
    (final snapshot) inclusive. A single conformation gets *last*, so the
    fully-rendered end state is never dropped.
    """
    if n <= 0:
        return []
    if n == 1:
        return [last]
    return [first + (last - first) * i / (n - 1) for i in range(n)]


def make_unbinding_paths_visualization(
    paths: Dict[str, List[Tuple[str, str]]],
    reference_pdb: str,
    ligand_select: str,
    outdir: str = "path_analysis",
    align_sel: str = "protein and backbone",
    grid_spacing: float = 0.375,
    cartoon_color: str = "grey90",
    sample_stride: int = 2,
    pocket_select: str = None,
    n_lig_conformations: int = 5,
    stick_transparency_ladder: Optional[Tuple[float, float]] = None,
    output_format: str = "pml",
    color_by_friction: bool = False,
    friction_csv: str = None,
    friction_estimator: str = "cumulant",
    friction_center_select: str | None = None,
    friction_speed: float | None = None,
) -> str:
    """Generate ligand-path density maps as a PyMOL session or script.

    Args:
        n_lig_conformations: Number of ligand snapshots sampled evenly along each
            path's medoid trajectory.
        stick_transparency_ladder: ``None`` (default) keeps the legacy behaviour —
            all conformations go into one multi-state object at a single
            ``stick_transparency``, so PyMOL shows only state 1 unless
            ``all_states`` is enabled.  Pass a ``(first, last)`` pair such as
            ``(0.9, 0.0)`` to instead emit **one object per conformation**, all
            displayed simultaneously, with stick transparency interpolated
            linearly from *first* (earliest snapshot, most transparent) to *last*
            (final snapshot, most opaque).  This renders the unbinding
            progression as a fading trail.  Values are PyMOL transparencies in
            ``[0, 1]``; the per-path objects are collected into a PyMOL group.
        output_format: ``"pse"`` saves a portable, self-contained session file;
            ``"pml"`` writes a script + auxiliary .dx files next to it.
            PSE is self-contained (all data embedded); PML references
            auxiliary ``.dx`` files that must stay alongside the script.
            ``"none"`` writes nothing and returns None.
        color_by_friction: When ``True`` (and *friction_csv* is provided), the
            isosurfaces are coloured by local friction Γ(r) as a blue (low) →
            white → red (high) gradient.  Default ``False``: each path is
            instead coloured with a distinct colour from the ``fivethirtyeight``
            palette (the same cycle used by :func:`plot_path_decomposition`), applied
            to both the density isosurface and the ligand sticks.
        friction_csv: Path to ``friction.csv`` produced by
            :class:`~autopath.pulling.Estimators.FrictionEstimator`.
            Only used when *color_by_friction* is ``True``.
        friction_estimator: Which estimator row to use from *friction_csv*
            (``"cumulant"`` or ``"jarzynski"``).  Default ``"cumulant"``.
        friction_center_select: MDAnalysis selection for the CV reference
            anchor used to map voxel distances to r_coord values.
            Default ``None``: the pocket COM (*pocket_select*) is used directly
            when provided (pocket-unbinding convention).  Pass an explicit
            selection string (e.g. ``"resname DUM"`` for membrane-pull systems)
            to override.  If the explicit selection matches 0 atoms, the
            pocket COM fallback is attempted; if that also fails, friction
            colouring is disabled.
        friction_speed: Pulling speed (nm/ps) whose derivative-friction profile
            is used for colouring.  ``None`` (default) picks the fastest
            non-zero speed present in *friction_csv*, which typically gives the
            widest spatial coverage and the smoothest Gamma profile.  The
            derivative at speed=0 is always excluded (its Gamma is undefined).

    Notes
    -----
    **Friction colour ramp strategy**: the ramp limits are computed from the
    5th–95th percentile of Gamma values across the per-path profiles that were
    actually used to build the volumetric grids.  A *shared* ramp (same
    colour scale for all paths) keeps colours comparable across paths even
    when trajectories were pulled at different speeds; per-path friction maps
    ensure the spatial Γ(r) pattern reflects each path's own simulation
    conditions.
    """
    if output_format not in ("pse", "pml", "none"):
        raise ValueError(f"output_format must be 'pse', 'pml' or 'none', got '{output_format}'")
    if output_format == "none":
        logger.info("Unbinding paths visualization disabled (output_format='none').")
        return None

    if stick_transparency_ladder is not None:
        if len(stick_transparency_ladder) != 2:
            raise ValueError(
                "stick_transparency_ladder must be a (first, last) pair, got "
                f"{stick_transparency_ladder!r}"
            )
        if not all(0.0 <= float(v) <= 1.0 for v in stick_transparency_ladder):
            raise ValueError(
                "stick_transparency_ladder values must lie in [0, 1] (PyMOL "
                f"transparency), got {stick_transparency_ladder!r}"
            )
        stick_transparency_ladder = (
            float(stick_transparency_ladder[0]), float(stick_transparency_ladder[1])
        )

    level = 0.000002
    surface_transparency = 0.2
    cartoon_transparency = 0.1
    stick_transparency = 0.15

    os.makedirs(outdir, exist_ok=True)
    outdir = Path(outdir)
    protein_abs = str(Path(reference_pdb).resolve())
    u_ref = mda.Universe(protein_abs)

    # ---------------------------------------------------------------------- #
    # Load per-speed friction profiles (optional)                              #
    # ---------------------------------------------------------------------- #
    # friction_profiles: speed (nm/ps) → Gamma(r) Series.
    # Each path is later coloured with the profile that matches its own speed
    # (parsed from the path name, e.g. "path-0_v0.001" → 0.001 nm/ps).
    # speed=0 rows are always excluded because their Gamma is NaN (derivative
    # is undefined at v=0). friction_speed acts as a fallback for paths whose
    # name carries no speed suffix.
    friction_profiles: dict = {}
    friction_center_nm: Optional[np.ndarray] = None

    if color_by_friction and friction_csv is not None:
        try:
            fdf_all = pd.read_csv(friction_csv)
            fdf_deriv = fdf_all[
                (fdf_all["method"] == "derivative") & (fdf_all["estimator"] == friction_estimator)
            ]
            valid_speeds = sorted(s for s in fdf_deriv["speed"].unique() if s > 0)
            if not valid_speeds:
                logger.warning(
                    f"No non-zero derivative/{friction_estimator} speeds in {friction_csv}. "
                    "Friction colouring disabled."
                )
            else:
                for spd in valid_speeds:
                    rows = (fdf_deriv[fdf_deriv["speed"] == spd]
                            .sort_values("r_coord")
                            .dropna(subset=["Gamma"]))
                    if not rows.empty:
                        friction_profiles[spd] = pd.Series(
                            rows["Gamma"].to_numpy(), index=rows["r_coord"].to_numpy()
                        )
                logger.info(
                    f"Loaded friction profiles for {len(friction_profiles)} speed(s): "
                    f"{sorted(friction_profiles)} nm/ps."
                )
        except Exception as exc:
            logger.warning(f"Could not load friction data from {friction_csv}: {exc}. Skipping.")

    if friction_profiles:
        # Determine the spatial anchor for Γ(r) lookup (same for all paths).
        # Anchor = the fixed point from which per-voxel distance is computed
        # to index into the friction profile.
        _center_selected = False
        try:
            if friction_center_select is not None:
                center_ag = u_ref.select_atoms(friction_center_select)
                if center_ag.n_atoms > 0:
                    friction_center_nm = center_ag.center_of_geometry() / 10.0  # Å → nm
                    _center_selected = True
                else:
                    logger.warning(
                        f"Friction centre selection '{friction_center_select}' found 0 atoms; "
                        "trying pocket_select fallback."
                    )

            if not _center_selected:
                if pocket_select is not None:
                    pocket_ag = u_ref.select_atoms(pocket_select)
                    if pocket_ag.n_atoms > 0:
                        friction_center_nm = pocket_ag.center_of_geometry() / 10.0  # Å → nm
                        logger.info("Using pocket_select COM as friction anchor.")
                    else:
                        logger.warning("pocket_select found 0 atoms. Friction colouring disabled.")
                else:
                    logger.info("No friction anchor provided. Friction colouring disabled.")
        except Exception as exc:
            logger.warning(
                f"Friction anchor determination failed: {exc}. Friction colouring disabled."
            )

        if friction_center_nm is not None:
            logger.info(
                f"Friction colouring enabled ({friction_estimator}). "
                f"Centre: {friction_center_nm} nm — "
                f"each path coloured by its own pulling speed."
            )

    # Colour each path with the fivethirtyeight palette — the same colour cycle
    # used by plot_path_decomposition — so path identity is consistent across figures.
    # Colours are stored as RGB (0–1) and registered as custom PyMOL colours
    # (indexed names sidestep path-name characters PyMOL dislikes).
    with plt.style.context(_STYLE):
        _palette_hex = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    path_colors = {name: f"ap_path_{i}" for i, name in enumerate(paths)}
    path_color_rgb = {
        f"ap_path_{i}": list(mcolors.to_rgb(_palette_hex[i % len(_palette_hex)]))
        for i, name in enumerate(paths)
    }

    # ------------------------------------------------------------------ #
    # Shared: compute density maps and extract ligand conformations        #
    # ------------------------------------------------------------------ #
    import tempfile
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)

        dx_files: Dict[str, str] = {}
        friction_dx_files: Dict[str, str] = {}
        lig_pdb_files: Dict[str, str] = {}
        # path -> [per-conformation pdb, ...]; only populated for the ladder mode
        lig_conf_files: Dict[str, List[str]] = {}

        for path_name, traj_list in paths.items():
            if not traj_list:
                logger.warning(f"No trajectories found for path {path_name}")
                continue

            top, traj = traj_list[0]
            u = mda.Universe(top, traj)
            if u_ref.select_atoms(align_sel).n_atoms > 0:
                align.AlignTraj(u, u_ref, select=align_sel, in_memory=True).run()
            else:
                logger.warning(
                    f"Alignment selection '{align_sel}' matched 0 atoms in reference PDB "
                    f"(membrane/no-protein system?). Skipping re-alignment — trajectories "
                    f"are assumed to be pre-aligned."
                )

            lig = u.select_atoms(ligand_select)
            if lig.n_atoms == 0:
                raise ValueError(f"No atoms found for '{ligand_select}' in {traj}.")

            da = density.DensityAnalysis(lig, delta=grid_spacing, padding=25.0)
            da.run(step=sample_stride)
            dens = da.results.density
            dens.grid = gaussian_filter(dens.grid, sigma=2.0)

            # For PML the .dx files are placed in outdir (referenced by the script).
            # For PSE they go to the temp dir and are cleaned up after saving.
            dx_dest = outdir if output_format == "pml" else tmpdir
            dx_path = str(dx_dest / f"{path_name}_density.dx")
            dens.export(dx_path)
            dx_files[path_name] = dx_path

            # Build friction volumetric map on the same grid as the density.
            # Each path uses the friction profile at its own pulling speed so that
            # the spatial Γ(r) pattern reflects the actual simulation conditions.
            if friction_center_nm is not None:
                path_profile = _get_friction_profile_for_path(
                    path_name, friction_profiles, friction_speed
                )
                if path_profile is not None:
                    try:
                        fric_grid = _build_friction_grid(dens, path_profile, friction_center_nm)
                        fric_dx_path = str(dx_dest / f"{path_name}_friction.dx")
                        fric_grid.export(fric_dx_path)
                        friction_dx_files[path_name] = fric_dx_path
                        logger.info(
                            f"Built friction grid for '{path_name}' "
                            f"(speed={_parse_speed_from_path(path_name)} nm/ps)."
                        )
                    except Exception as exc:
                        logger.warning(f"Could not build friction grid for {path_name}: {exc}")

            n_frames = len(u.trajectory)
            if n_lig_conformations >= n_frames:
                frame_indices = list(range(n_frames))
            else:
                frame_indices = [int(i * n_frames / n_lig_conformations) for i in range(n_lig_conformations)]

            # Legacy layout: one multi-state PDB per path.
            lig_pdb = str(tmpdir / f"{path_name}_lig.pdb")
            with mda.Writer(lig_pdb, multiframe=True, n_atoms=lig.n_atoms) as W:
                for idx in frame_indices:
                    u.trajectory[idx]
                    W.write(lig)
            lig_pdb_files[path_name] = lig_pdb

            # Ladder layout: one single-state PDB per conformation, so each can
            # carry its own stick transparency and all are shown at once (a
            # multi-state object would display only state 1).
            if stick_transparency_ladder is not None:
                conf_paths = []
                for i, idx in enumerate(frame_indices):
                    conf_pdb = str(tmpdir / f"{path_name}_lig_c{i:02d}.pdb")
                    u.trajectory[idx]
                    with mda.Writer(conf_pdb, n_atoms=lig.n_atoms) as W:
                        W.write(lig)
                    conf_paths.append(conf_pdb)
                lig_conf_files[path_name] = conf_paths

        # Determine friction colour range (shared ramp, 5th–95th percentile).
        friction_ramp_vals: Optional[List[float]] = None
        if friction_dx_files:
            all_gamma: List[float] = []
            for pname in friction_dx_files:
                pprofile = _get_friction_profile_for_path(pname, friction_profiles, friction_speed)
                if pprofile is not None:
                    all_gamma.extend(np.clip(pprofile.to_numpy(dtype=float), 0, None).tolist())
            if all_gamma:
                arr = np.asarray(all_gamma, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    g_lo  = float(np.percentile(arr, 5))
                    g_mid = float(np.percentile(arr, 60))
                    g_hi  = float(np.percentile(arr, 95))
                    friction_ramp_vals = [g_lo, g_mid, g_hi]
                    logger.info(
                        f"Friction ramp: low={g_lo:.0f}, mid={g_mid:.0f}, "
                        f"high={g_hi:.0f} kJ·ps/mol/nm²"
                    )
                else:
                    logger.warning(
                        "All Gamma values are NaN/inf; friction ramp could not be set. "
                        "Friction colouring disabled."
                    )
                    friction_dx_files.clear()

        pocket_resids: List[int] = []
        if pocket_select is not None:
            pocket = u_ref.select_atoms(pocket_select)
            if pocket.n_atoms == 0:
                raise ValueError(f"No atoms found for pocket selection '{pocket_select}' in reference PDB.")
            pocket_resids = sorted(set(pocket.resids))
            logger.info(f"Found pocket residues: {pocket_resids}")

        # ------------------------------------------------------------------ #
        # PSE branch: build session via PyMOL Python API                      #
        # ------------------------------------------------------------------ #
        if output_format == "pse":
            try:
                import pymol2
            except ImportError:
                raise ImportError("pymol2 is required. Install open-source PyMOL into your environment.")

            out_path = str(outdir / "unbinding_paths.pse")

            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd
                cmd.bg_color("white")
                cmd.set("antialias", 2)
                cmd.set("specular", 0.3)
                cmd.set("ray_shadow", 0)
                cmd.set("ray_opaque_background", 0)
                cmd.set("cartoon_transparency", cartoon_transparency)

                # Register per-path palette colours (fivethirtyeight, RGB 0–1)
                for col_name, rgb in path_color_rgb.items():
                    cmd.set_color(col_name, rgb)

                cmd.load(protein_abs, "prot")
                cmd.hide("everything", "prot")
                cmd.show("cartoon", "prot")
                cmd.color(cartoon_color, "prot")

                if pocket_resids:
                    resi_str = "+".join(map(str, pocket_resids))
                    cmd.select("pocket", f"resi {resi_str} and prot")
                    cmd.show("sticks", "pocket")
                    cmd.color("orange", "pocket")

                # Build shared friction ramp (one ramp covers all paths)
                use_friction_coloring = bool(friction_dx_files and friction_ramp_vals)
                if use_friction_coloring:
                    g_lo, g_mid, g_hi = friction_ramp_vals
                    # Load one friction map (they all share the same Gamma(r) function;
                    # we use path-0's map for the ramp calibration, then reload per path)
                    first_path = next(iter(friction_dx_files))
                    cmd.load(friction_dx_files[first_path], "friction_ramp_ref")
                    cmd.do("map_double friction_ramp_ref")
                    cmd.ramp_new(
                        "friction_ramp", "friction_ramp_ref",
                        [g_lo, g_mid, g_hi],
                        ["blue", "white", "red"],
                    )
                    cmd.hide("everything", "friction_ramp_ref")

                for path_name in dx_files:
                    col = path_colors[path_name]
                    map_obj  = f"map_{path_name}"
                    surf_obj = f"surf_{path_name}"
                    lig_obj  = f"lig_{path_name}"

                    cmd.load(dx_files[path_name], map_obj)
                    cmd.do(f"map_double {map_obj}")
                    cmd.isosurface(surf_obj, map_obj, level)
                    cmd.set("transparency", surface_transparency, surf_obj)
                    cmd.set("two_sided_lighting", 1, surf_obj)
                    cmd.hide("everything", map_obj)

                    if use_friction_coloring and path_name in friction_dx_files:
                        # Load this path's friction map and redefine the ramp against it,
                        # then colour the surface.  Using per-path maps ensures the ramp
                        # samples the correct spatial region for this isosurface.
                        fric_obj = f"fric_{path_name}"
                        cmd.load(friction_dx_files[path_name], fric_obj)
                        cmd.do(f"map_double {fric_obj}")
                        cmd.ramp_new(
                            f"ramp_{path_name}", fric_obj,
                            [g_lo, g_mid, g_hi],
                            ["blue", "white", "red"],
                        )
                        cmd.color(f"ramp_{path_name}", surf_obj)
                        cmd.hide("everything", fric_obj)
                    else:
                        cmd.color(col, surf_obj)

                    if stick_transparency_ladder is not None:
                        conf_paths = lig_conf_files.get(path_name, [])
                        ladder = _transparency_ladder(
                            *stick_transparency_ladder, len(conf_paths)
                        )
                        member_objs = []
                        for i, (conf_pdb, alpha) in enumerate(zip(conf_paths, ladder)):
                            conf_obj = f"{lig_obj}_c{i:02d}"
                            cmd.load(conf_pdb, conf_obj)
                            cmd.hide("everything", conf_obj)
                            cmd.show("sticks", conf_obj)
                            cmd.color(col, conf_obj)
                            cmd.set("stick_transparency", alpha, conf_obj)
                            member_objs.append(conf_obj)
                        if member_objs:
                            cmd.group(lig_obj, " ".join(member_objs))
                    else:
                        cmd.load(lig_pdb_files[path_name], lig_obj)
                        cmd.hide("everything", lig_obj)
                        cmd.show("sticks", lig_obj)
                        cmd.color(col, lig_obj)
                        cmd.set("stick_transparency", stick_transparency, lig_obj)

                if cmd.count_atoms(f"({ligand_select}) and prot") > 0:
                    cmd.create("lig_ref", f"({ligand_select}) and prot")
                    cmd.hide("everything", "lig_ref")
                    cmd.show("sticks", "lig_ref")
                    cmd.color("yellow", "lig_ref")

                cmd.zoom("prot", 10.0)
                cmd.save(out_path)

            return out_path

        # ------------------------------------------------------------------ #
        # PML branch: write a script; .dx files already in outdir             #
        # ------------------------------------------------------------------ #
        # Copy the ligand PDB conformations into outdir so the .pml can reference them
        for path_name, lig_pdb in lig_pdb_files.items():
            dest = str(outdir / f"{path_name}_lig.pdb")
            shutil.copy2(lig_pdb, dest)
            lig_pdb_files[path_name] = dest

        for path_name, conf_paths in lig_conf_files.items():
            dests = []
            for i, conf_pdb in enumerate(conf_paths):
                dest = str(outdir / f"{path_name}_lig_c{i:02d}.pdb")
                shutil.copy2(conf_pdb, dest)
                dests.append(dest)
            lig_conf_files[path_name] = dests

        # Copy the reference PDB into outdir for portability
        prot_copy = outdir / os.path.basename(reference_pdb)
        shutil.copy2(reference_pdb, prot_copy)


        pml_path = str(outdir / "unbinding_paths.pml")
        with open(pml_path, "w") as pml:
            pml.write("reinitialize\n")
            pml.write("bg_color white\n")
            pml.write("set ray_opaque_background, 0\n")
            pml.write("set antialias, 2\n")
            pml.write("set specular, 0.2\n")
            pml.write("set ray_shadow, off\n")
            pml.write(f"set cartoon_transparency, {cartoon_transparency:.2f}\n")
            # Register per-path palette colours (fivethirtyeight, RGB 0–1)
            for col_name, rgb in path_color_rgb.items():
                pml.write(f"set_color {col_name}, [{rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f}]\n")

            pml.write(f"load {os.path.basename(str(prot_copy))}, prot\n")
            pml.write("hide everything, prot\n")
            pml.write("show cartoon, prot\n")
            pml.write(f"color {cartoon_color}, prot\n")

            if pocket_resids:
                pml.write(f"select pocket, resi {'+'.join(map(str, pocket_resids))} and prot\n")
                pml.write("show sticks, pocket\n")
                pml.write("color orange, pocket\n")

            use_friction_coloring = bool(friction_dx_files and friction_ramp_vals)

            for path_name, dx_path in dx_files.items():
                col = path_colors[path_name]
                map_obj  = f"map_{path_name}"
                surf_obj = f"surf_{path_name}"
                lig_obj  = f"lig_{path_name}"
                pml.write(f"load {os.path.basename(dx_path)}, {map_obj}\n")
                pml.write(f"map_double {map_obj}\n")
                pml.write(f"isosurface {surf_obj}, {map_obj}, {level}\n")
                pml.write(f"set transparency, {surface_transparency:.2f}, {surf_obj}\n")
                pml.write(f"set two_sided_lighting, on, {surf_obj}\n")

                if use_friction_coloring and path_name in friction_dx_files:
                    g_lo, g_mid, g_hi = friction_ramp_vals
                    fric_obj  = f"fric_{path_name}"
                    ramp_name = f"ramp_{path_name}"
                    fric_base = os.path.basename(friction_dx_files[path_name])
                    pml.write(f"load {fric_base}, {fric_obj}\n")
                    pml.write(f"map_double {fric_obj}\n")
                    pml.write(
                        f"ramp_new {ramp_name}, {fric_obj}, "
                        f"[{g_lo:.2f}, {g_mid:.2f}, {g_hi:.2f}], "
                        f"[blue, white, red]\n"
                    )
                    pml.write(f"color {ramp_name}, {surf_obj}\n")
                    pml.write(f"hide everything, {fric_obj}\n")
                else:
                    pml.write(f"color {col}, {surf_obj}\n")

                if stick_transparency_ladder is not None:
                    conf_paths = lig_conf_files.get(path_name, [])
                    ladder = _transparency_ladder(
                        *stick_transparency_ladder, len(conf_paths)
                    )
                    member_objs = []
                    for i, (conf_pdb, alpha) in enumerate(zip(conf_paths, ladder)):
                        conf_obj = f"{lig_obj}_c{i:02d}"
                        pml.write(f"load {os.path.basename(conf_pdb)}, {conf_obj}\n")
                        pml.write(f"hide everything, {conf_obj}\n")
                        pml.write(f"show sticks, {conf_obj}\n")
                        pml.write(f"color {col}, {conf_obj}\n")
                        pml.write(f"set stick_transparency, {alpha:.3f}, {conf_obj}\n")
                        member_objs.append(conf_obj)
                    if member_objs:
                        pml.write(f"group {lig_obj}, {' '.join(member_objs)}\n")
                else:
                    pml.write(f"load {os.path.basename(lig_pdb_files[path_name])}, {lig_obj}\n")
                    pml.write(f"hide everything, {lig_obj}\n")
                    pml.write(f"show sticks, {lig_obj}\n")
                    pml.write(f"color {col}, {lig_obj}\n")
                    pml.write(f"set stick_transparency, {stick_transparency:.2f}, {lig_obj}\n")

            # `@script.pml` runs line-by-line, so a multi-line Python `if` block
            # raises IndentationError and its body executes unconditionally
            # anyway. Resolve the condition here instead and only emit the
            # commands when the reference PDB actually carries the ligand.
            if u_ref.select_atoms(ligand_select).n_atoms > 0:
                pml.write(f"create lig_ref, ({ligand_select}) and prot\n")
                pml.write("hide everything, lig_ref\n")
                pml.write("show sticks, lig_ref\n")
                pml.write("color yellow, lig_ref\n")
            pml.write("zoom prot, 10.0\n")

        return pml_path