"""End-of-run metadynamics report: FES, COLVAR and convergence figures plus ΔG°_b.

Run modes:

* single walker    its own bias and FES history;
* shared bias      summed self-biases form one surface; history laid end to end
                   (sequential) or on aggregate simulation time (concurrent);
* isolated biases  each walker is an independent estimate, reported alone and as
                   mean ± sd (biases are never averaged).

Every ΔG comes from correct_fe_for_funnel, so figures and log agree.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from autopath.metadynamics.Analysis import MetadynamicsAnalysis, bound_site_bins, correct_fe_for_funnel
from autopath.metadynamics.Funnel import load_funnel_params, funnel_basins

logger = logging.getLogger("autopath")

KJ_TO_KCAL = 1.0 / 4.184
COLVAR_DT_NS = 0.01          # MetadynamicsMD records COLVAR and FUNNELZ every 10 ps
_BIAS_RE = re.compile(r"bias_(\d+)_(\d+)\.npy")


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------
@dataclass
class RunMode:
    layout: str                      # "single" | "shared" | "isolated"
    schedule: Optional[str]          # "concurrent" | "sequential" (shared only)
    run_ids: list

    def describe(self) -> str:
        n = len(self.run_ids)
        if self.layout == "single":
            return "single walker"
        if self.layout == "isolated":
            return f"{n} walkers, isolated biases"
        return f"{n} walkers, shared bias ({self.schedule})"


def _hist_files(out_dir: str, run_id: str) -> list:
    """FEhist files of one walker in run order (a retry continues its walker)."""
    files = [f for f in glob(os.path.join(out_dir, f"FEhist_{run_id}*.npz"))
             if os.path.basename(f)[len("FEhist_"):-len(".npz")] in (run_id, f"{run_id}_retry")]
    return sorted(files, key=os.path.getmtime)


def _wall_window(hist_path: str) -> tuple:
    """(start, end) wall-clock seconds of the walker that wrote ``hist_path``."""
    end = os.path.getmtime(hist_path)
    data = np.load(hist_path)
    start = float(data["started_at"]) if "started_at" in data.files else np.nan
    if not np.isfinite(start) or start <= 0 or start > end:
        start = end
    return start, end


def detect_mode(out_dir: str) -> RunMode:
    run_ids = sorted(os.path.basename(f)[len("COLVAR_"):-len(".npy")]
                     for f in glob(os.path.join(out_dir, "COLVAR_*.npy")))
    run_ids = [r for r in run_ids if not r.endswith("_retry")]
    if len(run_ids) <= 1:
        return RunMode("single", None, run_ids)
    if MetadynamicsAnalysis.bias_layout(out_dir) == "isolated":
        return RunMode("isolated", None, run_ids)
    windows = sorted(_wall_window(h[0]) for h in
                     (_hist_files(out_dir, r) for r in run_ids) if h)
    concurrent = any(nxt[0] < cur[1] for cur, nxt in zip(windows, windows[1:]))
    return RunMode("shared", "concurrent" if concurrent or len(windows) < 2 else "sequential",
                   run_ids)


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------
@dataclass
class Estimate:
    label: str
    fes_kj: np.ndarray
    corr: Optional[dict]
    error: Optional[str]
    t_ns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dG_kj: np.ndarray = field(default_factory=lambda: np.zeros(0))
    resolved: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    snapshots: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    complete: bool = True

    @property
    def dG(self) -> Optional[float]:
        return None if self.corr is None else float(self.corr["dG_bind_std_kj_mol"])


def _latest_bias(directory: str) -> list:
    latest = {}
    for f in os.listdir(directory):
        m = _BIAS_RE.match(f)
        if m:
            wid, idx = int(m.group(1)), int(m.group(2))
            if wid not in latest or idx > latest[wid][0]:
                latest[wid] = (idx, os.path.join(directory, f))
    return [np.load(p) for _, p in latest.values()]


def _well_depth(fes_kj, grid=(0.0, 1.0), origin=None, n_tail=10):
    """FES minimum in the bound site relative to the unbound end of the grid (kJ/mol)."""
    fe = np.asarray(fes_kj, float)
    return float((fe - fe[-n_tail:].mean())[:bound_site_bins(*grid, fe.size, origin)].min())


def reference_range_from_funnel(fp, grid, cv_is_funnel_z, sigma_ang, out_dir=None,
                                cv_len=None, n_sigma=3.0, hill_width_cv=None):
    """Unbound reference window in CV units: the cylinder, clear of the wall.

    z in [z_cc, z_max - n_sigma·σ], avoiding the hill-deficit artifact near the wall.
    Non-z CVs are mapped via the median recorded funnel z per CV bin, else linearly
    from the bound site (grid_min) to the cap (grid_max). The window also stays
    n_sigma hill widths inside the grid edges. Returns None if no window fits.
    """
    z_cc = float(fp["z_cc"].value_in_unit(unit_ang()))
    z_max = float(fp["z_max"].value_in_unit(unit_ang()))
    z_lo, z_hi = z_cc, z_max - n_sigma * float(sigma_ang)
    if z_hi <= z_lo:
        logger.warning("Cylinder (%.1f-%.1f A) is shorter than %g sigma: no reference window.",
                       z_cc, z_max, n_sigma)
        return None
    if cv_is_funnel_z:
        lo, hi = z_lo / 10.0, z_hi / 10.0
    else:
        pairs = []
        for c in glob(os.path.join(out_dir or "", "COLVAR_*.npy")):
            zf = c.replace("COLVAR_", "FUNNELZ_")
            if os.path.exists(zf):
                cv = np.load(c)
                cv = cv[:, 0] if cv.ndim > 1 else cv
                z = np.load(zf)
                n = min(len(cv), len(z))
                pairs.append(np.c_[cv[:n], z[:n]])
        if pairs:
            P = np.concatenate(pairs)
            edges = np.linspace(grid[0], grid[1], 126)
            idx = np.clip(np.digitize(P[:, 0], edges) - 1, 0, 124)
            centres, med = [], []
            for b in range(125):
                zb = P[idx == b, 1]
                if zb.size >= 5:
                    centres.append(0.5 * (edges[b] + edges[b + 1]))
                    med.append(np.median(zb))
            inside = [c for c, m in zip(centres, med) if z_lo <= m <= z_hi]
            if not inside:
                logger.warning("No CV bin has its recorded funnel z inside the cylinder window.")
                return None
            lo, hi = min(inside), max(inside)
        else:
            from autopath.metadynamics.Funnel import funnel_basins as _fb
            z0 = 2.0 * _fb(fp)[0] - z_cc                     # = the sMD bound-site z
            span = (grid[1] - grid[0]) / max(z_max - z0, 1e-9)
            lo, hi = grid[0] + (z_lo - z0) * span, grid[0] + (z_hi - z0) * span
            logger.info("Reference window mapped linearly bound site -> cap (no FUNNELZ): "
                        "approximate.")
    margin = n_sigma * float(hill_width_cv) if hill_width_cv else 0.0
    lo, hi = max(lo, grid[0] + margin), min(hi, grid[1] - margin)
    return (lo, hi) if hi > lo else None


def unit_ang():
    from openmm import unit
    return unit.angstrom


def _correct(fes_kj, fp, temperature, grid, cv_len, reference_range=None, origin=None):
    if fp is None:
        return None, "no funnel: standard-state correction needs the funnel cylinder"
    try:
        return correct_fe_for_funnel(fes_kj, fp, temperature=temperature,
                                     grid_min=grid[0], grid_max=grid[1],
                                     cv_length_per_unit_ang=cv_len, verbose=False,
                                     reference_range=reference_range, cv_origin=origin), None
    except Exception as exc:                       # surface not interpretable yet
        return None, str(exc)


def _history(snapshots, fp, temperature, grid, cv_len, reference_range=None, origin=None):
    dG = np.full(len(snapshots), np.nan)
    ok = np.zeros(len(snapshots), bool)
    if fp is None:                             # no correction possible: track the well depth
        for i, fe in enumerate(snapshots):
            dG[i] = _well_depth(fe, grid, origin)
            ok[i] = dG[i] <= -10.0
        return dG, ok
    for i, fe in enumerate(snapshots):
        try:
            r = correct_fe_for_funnel(fe, fp, temperature=temperature, grid_min=grid[0],
                                      grid_max=grid[1], cv_length_per_unit_ang=cv_len,
                                      strict=False, verbose=False,
                                      reference_range=reference_range, cv_origin=origin)
            dG[i], ok[i] = r["dG_bind_std_kj_mol"], r["bound_state_sampled"]
        except Exception:
            pass
    return dG, ok


def _end_to_end(hist_lists):
    t_all, fes_all, offset = [], [], 0.0
    for files in hist_lists:
        for h in files:
            d = np.load(h)
            t = np.asarray(d["time_ns"], float)
            t_all.append(t + offset)
            fes_all.append(np.asarray(d["fes_kj"], float))
            offset = float(t_all[-1][-1]) if t.size else offset
    if not t_all:
        return np.zeros(0), np.zeros((0, 0))
    return np.concatenate(t_all), np.concatenate(fes_all)


def _aggregate_concurrent(hist_lists):
    """Merge concurrent walkers' snapshots on aggregate simulation time.

    A snapshot's aggregate time is the summed progress of all walkers at its
    wall-clock time (progress assumed linear in wall time).
    """
    walkers = []
    for files in hist_lists:
        if not files:
            continue
        d = np.load(files[-1])
        t = np.asarray(d["time_ns"], float)
        if not t.size:
            continue
        start, end = _wall_window(files[-1])
        walkers.append((t, np.asarray(d["fes_kj"], float), start, end))
    if not walkers:
        return np.zeros(0), np.zeros((0, 0))

    def progress(w, t, start, end):
        span = max(end - start, 1e-9)
        return float(t[-1]) * float(np.clip((w - start) / span, 0.0, 1.0))

    pts = []
    for t, fes, start, end in walkers:
        span = max(end - start, 1e-9)
        for ti, fe in zip(t, fes):
            w = start + span * ti / max(float(t[-1]), 1e-12)
            agg = sum(progress(w, tj, sj, ej) for tj, _, sj, ej in walkers)
            pts.append((agg, fe))
    pts.sort(key=lambda p: p[0])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def build_estimates(out_dir, mode, fp, temperature, bias_factor, grid, cv_len,
                    reference_range=None, origin=None) -> list:
    gamma = float(bias_factor)
    estimates = []
    if mode.layout == "isolated":
        for sub in sorted(glob(os.path.join(out_dir, "bias_*"))):
            if not os.path.isdir(sub):
                continue
            rid = os.path.basename(sub)[len("bias_"):]
            biases = _latest_bias(sub)
            if not biases:
                continue
            fes = -(gamma / (gamma - 1.0)) * np.sum(biases, axis=0)
            corr, err = _correct(fes, fp, temperature, grid, cv_len, reference_range, origin)
            hists = _hist_files(out_dir, rid)
            t, snaps = _end_to_end([hists])
            dG, ok = _history(snaps, fp, temperature, grid, cv_len, reference_range, origin)
            estimates.append(Estimate(rid, fes, corr, err, t, dG, ok, snaps, complete=bool(hists)))
        return estimates

    biases = _latest_bias(out_dir)
    for sub in glob(os.path.join(out_dir, "bias_*")):         # single walker in a subdir
        if os.path.isdir(sub):
            biases += _latest_bias(sub)
    if not biases:
        return estimates
    fes = -(gamma / (gamma - 1.0)) * np.sum(biases, axis=0)
    corr, err = _correct(fes, fp, temperature, grid, cv_len, reference_range, origin)
    hist_lists = [_hist_files(out_dir, r) for r in mode.run_ids]
    if mode.layout == "shared" and mode.schedule == "concurrent":
        t, snaps = _aggregate_concurrent(hist_lists)
    else:
        # sequential retries run after other walkers, so order files, not walkers
        order = sorted(([f] for h in hist_lists for f in h), key=lambda h: _wall_window(h[0])[0])
        t, snaps = _end_to_end(order)
    dG, ok = _history(snaps, fp, temperature, grid, cv_len, reference_range, origin)
    label = mode.run_ids[0] if mode.layout == "single" and mode.run_ids else "shared bias"
    estimates.append(Estimate(label, fes, corr, err, t, dG, ok, snaps,
                              complete=all(bool(h) for h in hist_lists)))
    return estimates


# ---------------------------------------------------------------------------
# Walker traces
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    run_id: str
    t_ns: np.ndarray
    cv: np.ndarray
    z_ang: Optional[np.ndarray]
    stats: dict


def walker_traces(out_dir, run_ids, basins, grid, cv_is_funnel_z=False) -> list:
    ma = MetadynamicsAnalysis.__new__(MetadynamicsAnalysis)
    traces = []
    for rid in run_ids:
        for r in (rid, f"{rid}_retry"):
            cpath = os.path.join(out_dir, f"COLVAR_{r}.npy")
            if not os.path.exists(cpath):
                continue
            col = np.load(cpath)
            cv = col[:, 0] if col.ndim > 1 else col
            zpath = os.path.join(out_dir, f"FUNNELZ_{r}.npy")
            z = np.load(zpath) if os.path.exists(zpath) else None
            if z is None and cv_is_funnel_z and basins is not None:
                z = cv * 10.0                      # the biased CV is z itself (nm)
                stats = ma.analyze_colvar_transitions(z[:, None], bound_threshold=basins[0],
                                                      unbound_threshold=basins[1])
                stats["coordinate"] = "funnel_z"
            else:
                stats = ma.walker_transitions(cpath, basins, grid_min=grid[0], grid_max=grid[1])
            traces.append(Trace(r, np.arange(len(cv)) * COLVAR_DT_NS, cv, z, stats))
    return traces


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
_STYLE = {"figure.facecolor": "white", "axes.facecolor": "white",
          "savefig.facecolor": "white", "savefig.edgecolor": "white",
          "axes.titlesize": 11, "legend.fontsize": 9}


def _styled(fn):
    """fivethirtyeight on white, the look of the rest of the metadynamics plots."""
    def wrapper(*args, **kwargs):
        with plt.style.context("fivethirtyeight"), plt.rc_context(_STYLE):
            return fn(*args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


def _cv_to_ang(cv_len, origin):
    if not cv_len:
        return None
    return (lambda x: (np.asarray(x) - origin) * cv_len,
            lambda a: np.asarray(a) / cv_len + origin)


@_styled
def plot_fes(estimates, mode, grid, cv_name, cv_len, verdict, out_path, corrected=True,
             origin=None):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    axis = None
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(estimates), 1)))
    shaded = False
    for k, e in enumerate(estimates):
        axis = np.linspace(grid[0], grid[1], len(e.fes_kj))
        many = len(estimates) > 1
        if e.corr is not None:
            ax.plot(axis, e.corr["fe_corrected"], color=colors[k] if many else "k",
                    lw=1.4 if many else 2.2,
                    label=f"{e.label}: ΔG°_b = {e.dG:.1f} kJ/mol" + ("" if e.complete else " (incomplete)"))
            if not shaded:
                ax.axvspan(*e.corr["site_range"], color="royalblue", alpha=0.08,
                           label="bound site (integrated)")
                ax.axvspan(*e.corr["plateau_range"], color="0.5", alpha=0.15,
                           label="unbound reference")
                shaded = True
        else:
            raw = e.fes_kj - e.fes_kj[-10:].mean()
            lab = (f"{e.label}: correction refused (raw FES)" if corrected
                   else f"{e.label}: well depth = {_well_depth(e.fes_kj, grid, origin):.1f} kJ/mol")
            ax.plot(axis, raw, color=colors[k] if many else "k", ls="--" if corrected else "-",
                    lw=1.0 if corrected else 2.0, label=lab)
    good = [e.dG for e in estimates if e.dG is not None and e.complete]
    if mode.layout == "isolated" and len(good) > 1:
        m, sd = float(np.mean(good)), float(np.std(good, ddof=1))
        ax.axhspan(m - sd, m + sd, color="darkgreen", alpha=0.10)
        ax.axhline(m, color="darkgreen", lw=1.2, ls="--",
                   label=f"mean ΔG°_b = {m:.1f} ± {sd:.1f} kJ/mol (n={len(good)})")
    elif good:
        ax.axhline(good[0], color="darkgreen", lw=1.0, ls="--")
    ax.axhline(0, color="0.4", lw=0.7, ls=":")
    ax.set_xlabel(cv_name)
    ax.set_ylabel("ΔG° (kJ/mol)" if corrected else "FES (kJ/mol), uncorrected")
    conv = _cv_to_ang(cv_len, grid[0] if origin is None else origin)
    if conv is not None:
        sec = ax.secondary_xaxis("top", functions=conv)
        sec.set_xlabel("distance along the CV (Å)")
    title = f"Free energy — {mode.describe()}"
    if good:
        dg = float(np.mean(good))
        title += f"\nΔG°_b = {dg:.1f} kJ/mol ({dg * KJ_TO_KCAL:.1f} kcal/mol)"
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    import textwrap
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.01, textwrap.fill(verdict, 130), fontsize=7.5, va="bottom",
             color="darkgreen" if verdict == "converged" else "firebrick")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@_styled
def plot_colvar(traces, mode, basins, grid, cv_name, out_path, z_max_ang=None):
    n = max(len(traces), 1)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.9 * n + 0.8), sharex=True, squeeze=False)
    total_rt = sum(t.stats["n_round_trips"] for t in traces)
    for ax, tr in zip(axes[:, 0], traces):
        use_z = tr.z_ang is not None and tr.stats.get("coordinate") == "funnel_z"
        y = tr.z_ang if use_z else tr.cv
        ax.plot(tr.t_ns[:len(y)], y, lw=0.8, color="k")
        lo = basins[0] if use_z else tr.stats["bound_threshold"]
        hi = basins[1] if use_z else tr.stats["unbound_threshold"]
        top = (z_max_ang if (use_z and z_max_ang) else float(np.nanmax(y)))
        ax.axhspan(min(0.0, float(np.nanmin(y))), lo, color="royalblue", alpha=0.10)
        ax.axhspan(hi, max(top, hi), color="tomato", alpha=0.10)
        for f in tr.stats["exit_frames"]:
            ax.axvline(f * COLVAR_DT_NS, color="tomato", lw=0.8, ls=":")
        for f in tr.stats["return_frames"]:
            ax.axvline(f * COLVAR_DT_NS, color="seagreen", lw=1.2)
        if use_z and not np.allclose(tr.cv * 10.0, tr.z_ang[:len(tr.cv)], atol=0.05):
            z = tr.z_ang[:len(tr.cv)]
            free = (tr.cv > grid[0] + 1e-4) & (tr.cv < grid[1] - 1e-4)   # CV is float32
            if free.any() and np.allclose(tr.cv[free] * 10.0, z[free], atol=0.05):
                # funnel z CV: differs from z only where clamped, so share the axis
                ax.plot(tr.t_ns[:len(tr.cv)], tr.cv * 10.0, lw=0.6, color="0.6",
                        label="biased CV (clamped)")
                ax.legend(fontsize=6, loc="upper right")
            else:
                ax2 = ax.twinx()
                ax2.plot(tr.t_ns, tr.cv, lw=0.6, color="0.6")
                ax2.set_ylabel(cv_name, fontsize=7, color="0.5")
        ax.set_ylabel("funnel z (Å)" if use_z else cv_name, fontsize=8)
        ax.set_title(f"{tr.run_id}: {tr.stats['n_round_trips']} round trip(s), "
                     f"{tr.stats['n_transitions']} exit(s); bound {100 * tr.stats['frac_bound']:.0f}%, "
                     f"unbound {100 * tr.stats['frac_unbound']:.0f}%", fontsize=8, loc="left")
    axes[-1, 0].set_xlabel("walker simulation time (ns)")
    fig.suptitle(f"COLVAR — {mode.describe()}  ·  {total_rt} round trip(s) in total "
                 f"(blue = bound, red = unbound; green = return, dotted = exit)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _deposition_rate(t_ns, snapshots, grid, bias_factor, window=6):
    """Rate at which bias is added (kJ/mol · CV-unit / ns), from the FES snapshots.

    V = -(γ-1)/γ · F. Its integral is made monotone (running max, removes sync dips)
    and the rate is a linear-fit slope over ``window`` snapshots.
    """
    V = -((bias_factor - 1.0) / bias_factor) * np.asarray(snapshots, float)
    dx = (grid[1] - grid[0]) / max(V.shape[1] - 1, 1)
    vol = np.maximum.accumulate(V.sum(axis=1) * dx)
    t = np.asarray(t_ns, float)
    out_t, out_r = [], []
    for i in range(0, len(t) - window + 1):
        tt, vv = t[i:i + window], vol[i:i + window]
        if np.ptp(tt) <= 0:
            continue
        slope = np.polyfit(tt, vv, 1)[0]
        if slope > 0:
            out_t.append(tt.mean())
            out_r.append(slope)
    return np.array(out_t), np.array(out_r)


@_styled
def plot_convergence(estimates, mode, grid, bias_factor, plateau_results, out_path,
                     corrected=True):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(estimates), 1)))
    xlab = ("aggregate simulation time (ns)" if mode.layout == "shared" and mode.schedule == "concurrent"
            else "accumulated simulation time (ns)" if mode.layout == "shared"
            else "walker simulation time (ns)")
    for k, e in enumerate(estimates):
        if not e.t_ns.size:
            continue
        ok = e.resolved & np.isfinite(e.dG_kj)
        a1.plot(e.t_ns, e.dG_kj, color=colors[k], lw=0.8, alpha=0.6)
        a1.plot(e.t_ns[ok], e.dG_kj[ok], "o", color=colors[k], ms=3, label=e.label)
        a1.plot(e.t_ns[~ok], e.dG_kj[~ok], "o", mfc="none", color=colors[k], ms=3)
        p = plateau_results.get(e.label)
        if p is not None and np.isfinite(p.get("drift_kj", np.nan)) and ok.any():
            t0 = 0.6 * e.t_ns[-1]
            last = e.dG_kj[ok][-1]
            a1.fill_between([t0, e.t_ns[-1]], last - 4.0, last + 4.0, color=colors[k], alpha=0.12)
        # bias deposition rate: d/dt of the integrated bias, V = -(γ-1)/γ F
        if e.snapshots.size and len(e.t_ns) > 3:
            t_r, rate = _deposition_rate(e.t_ns, e.snapshots, grid, bias_factor)
            if t_r.size:
                a3.plot(t_r, rate, color=colors[k], lw=1.2, label=e.label)
    a1.set_xlabel(xlab)
    a1.set_ylabel("ΔG°_b (kJ/mol)" if corrected else "well depth (kJ/mol)")
    a1.set_title("ΔG°_b(t)  (filled = bound basin resolved; band = ±4 kJ/mol over the last 40%)",
                 fontsize=9)
    if any(e.t_ns.size for e in estimates):
        a1.legend(fontsize=7)

    main = next((e for e in estimates if e.snapshots.size), None)
    if main is not None:
        axis = np.linspace(grid[0], grid[1], main.snapshots.shape[1])
        cmap = matplotlib.colormaps["viridis"]
        sel = np.unique(np.linspace(0, len(main.snapshots) - 1, min(12, len(main.snapshots))).astype(int))
        span = max(main.t_ns[-1] - main.t_ns[0], 1e-12)
        for i in sel:
            fe = main.snapshots[i]
            a2.plot(axis, fe - fe[-10:].mean(), color=cmap((main.t_ns[i] - main.t_ns[0]) / span),
                    lw=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(main.t_ns[0], main.t_ns[-1]))
        fig.colorbar(sm, ax=a2, label=xlab)
        a2.set_title(f"FES snapshots ({main.label}), unbound = 0", fontsize=9)
    a2.set_ylabel("FES (kJ/mol)")
    a3.set_yscale("log")
    a3.set_xlabel(xlab)
    a3.set_ylabel("bias deposition rate\n(kJ/mol · CV / ns)")
    a3.set_title("Bias deposition rate (falls as the well-tempered bias converges)", fontsize=9)
    fig.suptitle(f"Convergence — {mode.describe()}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def report(out_dir, temperature, bias_factor, grid_min=0.0, grid_max=1.0,
           cv_name="CV", cv_length_per_unit_ang=None, out_prefix=None,
           funnel_params=None, cv_is_funnel_z=False, hill_width=None) -> dict:
    grid = (float(grid_min), float(grid_max))
    origin = 0.0 if cv_is_funnel_z else grid[0]        # funnel z may extend below the site
    out_prefix = out_prefix or os.path.join(out_dir, "metad")
    fp_path = os.path.join(out_dir, "funnel_params.npz")
    fp = funnel_params if funnel_params is not None else (
        load_funnel_params(fp_path) if os.path.exists(fp_path) else None)
    # No funnel: no standard-state correction, round trips on CV grid fractions.
    basins = funnel_basins(fp) if fp is not None else None
    z_max = float(fp["z_max"].value_in_unit(fp["z_cc"].unit)) if fp is not None else None
    if fp is None:
        logger.info("No funnel: reporting the uncorrected FES and its well depth.")

    mode = detect_mode(out_dir)
    logger.info("Metadynamics report: %s", mode.describe())
    # Unbound zero from the funnel geometry, 3 hill widths clear of the wall.
    ref_range = None
    if fp is not None:
        if hill_width is not None and cv_length_per_unit_ang:
            sigma_ang = float(hill_width) * float(cv_length_per_unit_ang)
        else:
            sigma_ang = 1.0
            logger.info("Hill width unknown: wall margin assumes sigma = 1 A.")
        ref_range = reference_range_from_funnel(fp, grid, cv_is_funnel_z, sigma_ang,
                                                out_dir=out_dir, cv_len=cv_length_per_unit_ang,
                                                hill_width_cv=hill_width)
        if ref_range is None:
            logger.warning("No geometric reference window: falling back to the last grid points.")
        else:
            logger.info("Unbound reference: CV %.3f-%.3f (cylinder from z_cc to z_max - 3 sigma).",
                        *ref_range)
    traces = walker_traces(out_dir, mode.run_ids, basins, grid, cv_is_funnel_z)
    estimates = build_estimates(out_dir, mode, fp, temperature, bias_factor, grid,
                                cv_length_per_unit_ang, reference_range=ref_range,
                                origin=origin)

    ma = MetadynamicsAnalysis.__new__(MetadynamicsAnalysis)
    walker_stats = {t.run_id: t.stats for t in traces}

    def visited_both(ids):
        return any(walker_stats[i]["frac_bound"] > 0.01 and walker_stats[i]["frac_unbound"] > 0.01
                   for i in ids if i in walker_stats)

    plateau_results, verdicts = {}, {}
    for e in estimates:
        ids = [e.label, f"{e.label}_retry"] if mode.layout == "isolated" else list(walker_stats)
        if e.t_ns.size:
            plateau_results[e.label] = ma.dG_plateau(e.t_ns, e.dG_kj, sampled=e.resolved,
                                                     both_basins=visited_both(ids))
        trans = [walker_stats[i] for i in ids if i in walker_stats]
        verdicts[e.label] = ma.assess_convergence(trans, e.fes_kj, grid_min=grid[0],
                                                  grid_max=grid[1],
                                                  dG_plateau_result=plateau_results.get(e.label),
                                                  reference_range=ref_range)

    good = [e.dG for e in estimates if e.dG is not None and e.complete]
    if mode.layout == "isolated":
        dG_main = float(np.mean(good)) if good else None
        dG_sd = float(np.std(good, ddof=1)) if len(good) > 1 else None
        all_conv = bool(verdicts) and all(v["converged"] for v in verdicts.values())
    else:
        dG_main = good[0] if good else None
        dG_sd = None
        all_conv = bool(verdicts) and next(iter(verdicts.values()))["converged"]
    if len(verdicts) > 1:
        reasons = [f"{lab}: {r}" for lab, v in verdicts.items() for r in v.get("reasons", [])]
    else:
        reasons = [r for v in verdicts.values() for r in v.get("reasons", [])]
    verdict = "converged" if all_conv else "NOT converged — " + "; ".join(reasons)
    kT_log10 = 8.314e-3 * float(temperature) * np.log(10)
    pKd = -dG_main / kT_log10 if dG_main is not None else None

    files = {
        "fes": plot_fes(estimates, mode, grid, cv_name, cv_length_per_unit_ang, verdict,
                        f"{out_prefix}_FES.png", corrected=fp is not None, origin=origin),
        "colvar": plot_colvar(traces, mode, basins, grid, cv_name, f"{out_prefix}_COLVAR.png",
                              z_max_ang=z_max),
        "convergence": plot_convergence(estimates, mode, grid, float(bias_factor),
                                        plateau_results, f"{out_prefix}_convergence.png",
                                        corrected=fp is not None),
    }
    for e in estimates:
        logger.info("  %-28s ΔG°_b = %s kJ/mol%s", e.label,
                    f"{e.dG:.1f}" if e.dG is not None else f"n/a ({e.error})",
                    "" if e.complete else "  [incomplete walker]")
    if dG_main is not None:
        logger.info("  ΔG°_b = %.1f%s kJ/mol (%.1f kcal/mol), pKd = %.2f — %s",
                    dG_main, f" ± {dG_sd:.1f}" if dG_sd is not None else "",
                    dG_main * KJ_TO_KCAL, pKd, verdict)

    return {
        "mode": mode,
        "estimates": [(e.label, e.dG, e.complete) for e in estimates],
        "dG_bind_std_kj": dG_main,
        "funnel": fp is not None,
        "reference_range": ref_range,
        "well_depth_kj": [(e.label, _well_depth(e.fes_kj, grid, origin)) for e in estimates],
        "dG_bind_std_sd_kj": dG_sd,
        "dG_bind_std_kcal": dG_main * KJ_TO_KCAL if dG_main is not None else None,
        "pKd": pKd,
        "walker_stats": walker_stats,
        "n_walkers": len(traces),
        "n_converged_cv": sum(1 for s in walker_stats.values() if s["reached_unbound"]),
        "n_round_trips": sum(s["n_round_trips"] for s in walker_stats.values()),
        "convergence": {"converged": all_conv, "verdict": verdict, "per_estimate": verdicts},
        "files": files,
    }
