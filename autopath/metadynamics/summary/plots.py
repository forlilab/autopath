"""Summary figures: CVs (fig_cvs), bias deposition (fig_bias), free energy (fig_fe), estimators
(fig_estimators), block analysis (fig_blocks) and 2D maps (fig_fes2d).

fig_cvs, fig_bias, fig_estimators and fig_blocks return (fig, axes, skipped); fig_fe and fig_fes2d return
(fig, axes, skipped, numbers).
"""
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from openmm import unit

from ..Hills import bias_from_hills
from .analysis import (ESTIMATORS, FUNNEL_Z, KT_PER_K, LABELS, PRIMARY, RW_METHODS, _ang, aggregate_time,
                       ang_per_unit, bias_fes_scale, fe_numbers, fes_bias, fes_marginal, fes_reweighted, final_bias,
                       funnel_z_low, hill_mask, length_unit, recrossings, tt_basins)

KJ_TO_KCAL = 1 / 4.184

MAX_ROWS = 6
_BOUND, _UNBOUND = "#9ecae1", "#fdd0a2"
# Reweighting estimators validated all-pairs for colour-vision deficiency; bias is the neutral reference.
ESTIMATOR_COLORS = {"rw_ct": "#2a78d6", "rw_final": "#eb6834", "rw_balanced": "#1baf7a", "bias": "#52514e"}
ESTIMATOR_MARKERS = {"rw_ct": "o", "rw_final": "s", "rw_balanced": "D", "bias": None}
_INK, _MUTED, _REJECT, _HATCH = "#0b0b0b", "#52514e", "#ececea", "#b5b4ae"
ESTIMATOR_LABELS = dict(LABELS)


def _label(run, name):
    """Axis label ``name (unit)``; a CV without a length scale keeps the bare name."""
    u = length_unit(run, name)
    return f"{name} ({u})" if u else name


def _legend(ax, **kw):
    """Legend only when something is labelled."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)


def _w_ylim(ax, band, pad=0.15):
    """y-range from the curves between the bound and unbound bands; values outside are clipped."""
    if band is None:
        return
    edges = list(band.bound) + list(band.unbound)
    lo, hi = min(edges), max(edges)
    top = []
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        m = (x >= lo) & (x <= hi) & np.isfinite(y)
        if m.any():
            top.append(y[m].max())
    if top:
        t = max(top)
        ax.set_ylim(-0.05 * t - 0.2, (1 + pad) * t + 0.2)


def _names(run):
    return [c.name for c in run.cvs] + list(run.monitored)


def fig_cvs(run, bands_, out_path=None, axes=None):
    """CVs vs walker time with bound/unbound bands.

    One row per walker (up to ``MAX_ROWS``), otherwise one row with walkers overlaid; a path panel is added
    for two biased CVs.

    Returns
    -------
    fig, axes, skipped
    """
    skipped = []
    names = [n for n in _names(run) if all(n in w.cvs for w in run.walkers)]
    skipped += [f"CV {n}: not logged for every walker" for n in _names(run) if n not in names]
    two_d = len(run.cvs) == 2
    overlay = len(run.walkers) > MAX_ROWS
    nrow = 1 if overlay else len(run.walkers)
    ncol = len(names) + int(two_d)
    if axes is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.6 * nrow + 0.6), squeeze=False,
                                 constrained_layout=True)
    else:
        fig = axes.flat[0].figure
    lines = {}
    for i, w in enumerate(run.walkers):
        r = 0 if overlay else i
        for j, nm in enumerate(names):
            a, b = axes[r, j], bands_[nm]
            if not overlay or i == 0:
                a.axhspan(*b.bound, color=_BOUND, alpha=0.6, lw=0)
                a.axhspan(*b.unbound, color=_UNBOUND, alpha=0.6, lw=0)
            a.plot(w.t, w.cvs[nm], lw=0.5, label=w.rid if overlay else None)
            for s in w.segments:
                a.axvline(s, color="k", ls="--", lw=0.6)
            rc = recrossings(w, nm, b)
            if overlay:
                lines.setdefault(j, []).append(f"{w.rid}: {rc['n_cycles']} cyc, b {rc['frac_bound']:.2f}, "
                                               f"u {rc['frac_unbound']:.2f}")
            else:
                a.set_title(f"{w.rid} {nm}\n{rc['n_cycles']} cycles, bound {rc['frac_bound']:.2f}, "
                            f"unbound {rc['frac_unbound']:.2f}", fontsize=7, loc="left")
            a.set(xlabel="walker time (ns)", ylabel=_label(run, nm))
        if two_d:
            x, y = run.cvs[0].name, run.cvs[1].name
            sc = axes[r, -1].scatter(w.cvs[x], w.cvs[y], c=w.t, s=1, cmap="viridis")
            axes[r, -1].set_title("paths" if overlay else f"{w.rid} path", fontsize=7, loc="left")
            axes[r, -1].set(xlabel=_label(run, x), ylabel=_label(run, y))
            if not overlay or i == 0:
                fig.colorbar(sc, ax=axes[r, -1], label="time (ns)")
    if overlay:
        _legend(axes[0, 0], fontsize=6, ncol=2, frameon=False)
        for j, ls in lines.items():
            axes[0, j].text(0.01, 0.99, "\n".join(ls), transform=axes[0, j].transAxes, va="top", fontsize=5.5,
                            bbox=dict(fc="w", ec="0.8", alpha=0.8))
    fig.suptitle(f"CVs, {run.layout} bias, {len(run.walkers)} walkers; blue = bound band, orange = unbound band", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped


def _until(group, tt):
    """``bias_from_hills`` cut ``{segment run_id: local time}`` for joined walker time ``tt``."""
    out = {}
    for w in group:
        for h, off in zip(w.hills_raw, [0.0] + list(w.segments)):
            out[h["run_id"]] = tt - off
    return out


def _masked_bias(raw, nsig, until=None):
    """Bias on the grid, NaN away from deposited hills."""
    return np.where(hill_mask(raw, nsig, until), np.asarray(bias_from_hills(raw, until=until)), np.nan)


def _grid(ax):
    """Grid drawn behind the artists."""
    ax.set_axisbelow(True)
    ax.grid(True, color="0.35", alpha=0.45, lw=0.5)


def _range(maps):
    """Common (min, max) of the finite values of several maps."""
    fin = np.concatenate([np.asarray(m, float)[np.isfinite(m)].ravel() for m in maps] + [np.zeros(0)])
    return (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)


def _fill(fig, ax, x, y, Z, label, vrange=None):
    """Translucent filled map ``Z[iy, ix]`` with thin contours and a colorbar.

    ``vrange`` fixes the colour scale and contour levels (shared across panels).
    """
    Z = np.asarray(Z, float)
    lo, hi = _range([Z]) if vrange is None else vrange
    lv = np.linspace(lo, hi if hi > lo else lo + 1.0, 40)
    cf = ax.contourf(x, y, Z, levels=lv, cmap="Spectral_r", alpha=0.78)
    cf.set_edgecolor("face")
    ax.contour(x, y, Z, levels=lv[::max(1, len(lv) // 14)], colors="0.25", linewidths=0.3, alpha=0.5)
    fig.colorbar(cf, ax=ax, label=label)
    _grid(ax)
    return cf


def _cap(maps, fmax=None):
    """Colour cap (kcal/mol): ``fmax`` if given, else the 99th percentile of the finite values of ``maps``."""
    if fmax is not None:
        return float(fmax)
    fin = np.concatenate([np.asarray(m, float)[np.isfinite(m)].ravel() for m in maps] + [np.zeros(0)])
    return float(np.percentile(fin, 99)) if fin.size else 1.0


def _fill_capped(fig, ax, x, y, Z, label, fmax):
    """Filled map with 1-unit levels from 0 to ``fmax``; values above saturate at the top colour.

    ``label=None`` draws no colorbar (the caller adds a shared one).
    """
    lv = np.arange(0.0, fmax + 1e-9, 1.0)
    if len(lv) < 2:
        lv = np.array([0.0, max(fmax, 1e-6)])
    cf = ax.contourf(x, y, np.asarray(Z, float), levels=lv, cmap="Spectral_r", alpha=0.78, extend="max")
    cf.set_edgecolor("face")
    ax.contour(x, y, np.asarray(Z, float), levels=lv, colors="0.25", linewidths=0.3, alpha=0.5)
    if label is not None:
        fig.colorbar(cf, ax=ax, label=label)
    _grid(ax)
    return cf


def _stars(ax, xs, ys):
    ax.plot(xs, ys, ls="none", marker="*", ms=14, mfc="w", mec="k", zorder=5)


def _basin_coords(run, names, skipped):
    """Recorded TT basin coordinates for the biased CVs ``names``, as an (n_basins, len(names)) array.

    Returns None, with the reason in ``skipped``, when the run is not transition-tempered (no entry), the
    basins are not recorded, or a CV is not biased.
    """
    if "tt" not in run.tempering:
        return None
    b = tt_basins(run)
    if b is None:
        _note(skipped, "TT basins: not recorded")
        return None
    cv_names = [c.name for c in run.cvs]
    if b.shape[1] != len(cv_names):
        _note(skipped, "TT basins: shape does not match the biased CVs: not supported")
        return None
    miss = [n for n in names if n not in cv_names]
    if miss:
        _note(skipped, f"TT basins: {', '.join(miss)} is not a biased CV")
        return None
    return b[:, [cv_names.index(n) for n in names]]


def _note(skipped, msg):
    if msg not in skipped:
        skipped.append(msg)


def fig_bias(run, out_path=None, axes=None, n_snap=5, hill_mask_sigmas=2.0):
    """Hill heights, bias growth and, for transition tempering, V*(t), one row per bias group.

    Shared and single biases are one row (all walkers' hills, coloured per walker); isolated biases get one
    row per walker (up to ``MAX_ROWS``), never summed. Bias growth is shown for one CV as ``n_snap``
    snapshots. For two CVs the final bias map and the Boltzmann marginals of the bias-derived FES at
    ``n_snap`` times are drawn. Recorded transition-tempering
    basins are marked with stars on the map and the marginals. The bias map is drawn unmasked with a dashed
    outline around grid points within ``hill_mask_sigmas`` hill widths of a hill; the marginals leave out
    points farther than that from every hill deposited by then (NaN).

    Returns
    -------
    fig, axes, skipped
    """
    skipped = []
    nrow = len(run.cvs)
    isolated = run.layout == "isolated"
    groups = [[w] for w in run.walkers] if isolated else [run.walkers]
    if len(groups) > MAX_ROWS:
        skipped.append(f"bias rows: {len(groups) - MAX_ROWS} of {len(groups)} isolated walkers not drawn")
        groups = groups[:MAX_ROWS]
    nr = len(groups)
    has_tt = "tt" in run.tempering
    nc = 2 + (2 if nrow == 2 else 0) + int(has_tt)
    vcol = nc - 1
    if axes is None:
        fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.3 * nr + 0.6), squeeze=False, constrained_layout=True)
    else:
        fig = axes.flat[0].figure
    cols = {w.rid: f"C{i % 10}" for i, w in enumerate(run.walkers)}
    kt = KT_PER_K * run.temperature
    scale = bias_fes_scale(run)
    basins = _basin_coords(run, [c.name for c in run.cvs], skipped) if nrow == 2 else None
    if nrow != 2 and has_tt and tt_basins(run) is None:
        skipped.append("TT basins: not recorded")
    vrange = None
    if nrow == 2:
        vrange = _range([final_bias([h for w in g for h in w.hills_raw]) for g in groups])
    if nrow > 2:
        skipped.append(f"bias growth: {nrow} biased CVs not supported")
    vstar_rows = set()
    if has_tt:
        if not any("vstar_kj" in h for w in run.walkers for h in w.hills_raw):
            skipped.append("V*(t): not logged in HILLS")
    else:
        skipped.append("V*(t): not a transition-tempered run")
    for gi, g in enumerate(groups):
        raw = [h for w in g for h in w.hills_raw]
        h0 = raw[0]
        gmin, gmax = np.atleast_1d(h0["grid_min"]), np.atleast_1d(h0["grid_max"])
        tmax = max(float(w.hill_t[-1]) for w in g)
        tag = g[0].rid if isolated else ""
        times = np.linspace(tmax / n_snap, tmax, n_snap)
        ah = axes[gi, 0]
        for w in g:
            ah.plot(w.hill_t, w.hill_h, lw=0.6, alpha=0.6, color=cols[w.rid], label=w.rid)
        ah.set(yscale="log", xlabel="walker time (ns)", ylabel="hill height (kJ/mol)")
        ah.set_title(f"hill heights {tag}".strip(), fontsize=8, loc="left")
        if not isolated:
            _legend(ah, fontsize=7, ncol=2, frameon=False, loc="upper right")
        if nrow == 1:
            x = np.linspace(float(gmin[0]), float(gmax[0]), int(np.atleast_1d(h0["grid_points"])[0]))
            for tt in times:
                V = np.asarray(bias_from_hills(raw, until=_until(g, tt))).ravel()
                axes[gi, 1].plot(x, V, lw=1, label=f"{tag} t<={tt:.1f}".strip())
            axes[gi, 1].set(xlabel=_label(run, run.cvs[0].name), ylabel="V (kJ/mol)")
            axes[gi, 1].set_title(f"bias growth {tag}".strip(), fontsize=8, loc="left")
            _legend(axes[gi, 1], fontsize=7, ncol=2, frameon=False)
        elif nrow == 2:
            a = axes[gi, 1]
            xs = [np.linspace(float(gmin[k]), float(gmax[k]), int(np.atleast_1d(h0["grid_points"])[k]))
                  for k in range(2)]
            _fill(fig, a, xs[0], xs[1], final_bias(raw), "V (kJ/mol)", vrange)
            sampled = hill_mask(raw, hill_mask_sigmas)
            if sampled.any() and not sampled.all():
                a.contour(xs[0], xs[1], sampled.astype(float), levels=[0.5], colors="k", linestyles="--",
                          linewidths=0.8)
            a.set(xlabel=_label(run, run.cvs[0].name), ylabel=_label(run, run.cvs[1].name))
            a.set_title(f"final bias {tag}".strip(), fontsize=8, loc="left")
            if basins is not None:
                _stars(a, basins[:, 0], basins[:, 1])
            for tt in times:
                V = _masked_bias(raw, hill_mask_sigmas, _until(g, tt))
                for k in range(2):
                    m = fes_marginal(-scale * V, k, kt) * KJ_TO_KCAL
                    axes[gi, 2 + k].plot(xs[k], m, lw=1, label=f"t<={tt:.1f}")
            for k in range(2):
                b = axes[gi, 2 + k]
                b.set(xlabel=_label(run, run.cvs[k].name), ylabel="F (kcal/mol)")
                b.set_title(f"F marginal on {run.cvs[k].name} (from bias) {tag}".strip(), fontsize=8, loc="left")
                _grid(b)
                if basins is not None:
                    ln = b.lines[-1]
                    _stars(b, basins[:, k], np.interp(basins[:, k], ln.get_xdata(), ln.get_ydata()))
            _legend(axes[gi, 2], fontsize=7, frameon=False)
        if has_tt:
            have = [w for w in g if any("vstar_kj" in h for h in w.hills_raw)]
            vs = [np.concatenate([np.asarray(h["vstar_kj"]) for h in w.hills_raw if "vstar_kj" in h]) for w in have]
            if any(len(v) != len(w.hill_t) for v, w in zip(vs, have)):
                skipped.append("V* length != hill count: not supported")
                continue
            for w, v in zip(have, vs):
                axes[gi, vcol].plot(w.hill_t, v, lw=0.8, color=cols[w.rid], label=w.rid)
            if have:
                vstar_rows.add(gi)
                axes[gi, vcol].set(xlabel="walker time (ns)", ylabel="V* (kJ/mol)")
                axes[gi, vcol].set_title(f"TT tempering level {tag}".strip(), fontsize=8, loc="left")
                if not isolated:
                    _legend(axes[gi, vcol], fontsize=7, ncol=2, frameon=False)
    used = {(g, k) for g in range(nr) for k in range(2)}
    if nrow == 2:
        used |= {(g, k) for g in range(nr) for k in (2, 3)}
    used |= {(g, vcol) for g in vstar_rows}
    for r in range(nr):
        for k in range(nc):
            if (r, k) not in used:
                axes[r, k].axis("off")
            else:
                _grid(axes[r, k])
    fig.suptitle(f"bias deposition, {run.tempering}, gamma={run.bias_factor:g}, {run.layout}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped


_RXY = ("r_xy", "funnel_r_xy", "funnel_rxy")


def _rxy(run):
    """Name of the logged funnel radial coordinate, or None."""
    return next((n for n in _RXY if n in _names(run) and all(n in w.cvs for w in run.walkers)), None)


def _default_pair(run):
    """(funnel z, r_xy) when a funnel exists and r_xy is logged, else the first two CVs; plus a skip note."""
    names = _names(run)
    if run.funnel is not None and FUNNEL_Z in names:
        r = _rxy(run)
        if r is not None:
            return (FUNNEL_Z, r), None
        note = "default pair: r_xy not logged"
    else:
        note = None
    if FUNNEL_Z in names and len(names) > 1:
        return (FUNNEL_Z, next(n for n in names if n != FUNNEL_Z)), note
    return ((names[0], names[1]) if len(names) > 1 else (names[0], None)), note


def _funnel_outline(run, a):
    """Radial limit vs z (CV units, ``a`` Å per unit): cone below z_cc, cylinder beyond."""
    f = run.funnel
    z_cc, z_max, r0 = _ang(f, "z_cc"), _ang(f, "z_max"), _ang(f, "R_cylinder")
    tan = np.tan(float(f["alpha"].value_in_unit(unit.radian)))
    z = np.linspace(funnel_z_low(run, a), z_max, 100)
    return z / a, (r0 + np.maximum(z_cc - z, 0.0) * tan) / a


def _num(v):
    return f"{v * KJ_TO_KCAL:.2f}" if np.isfinite(v) else "n/a"


def _pick(c, key):
    return c.pooled if c.pooled is not None else c.per_walker[key]


def _pick_ess(c, key):
    return c.ess if not isinstance(c.ess, dict) else c.ess.get(key)


def _ess(v):
    return f"ESS {v:.0f}" if v is not None and np.isfinite(v) else "ESS n/a"


def _fe_axes(axes, n, iso, skipped, want_text=False):
    """Validate a user ``axes`` dict for fig_fe: ``ct`` and ``w`` (``n`` axes each); optional ``text``.

    Other keys are ignored.
    """
    miss = {"ct", "w"} - set(axes)
    if miss:
        raise ValueError(f"fig_fe axes missing {sorted(miss)}: not supported")
    out = dict(axes)
    v = out["ct"]
    v = [v] if not isinstance(v, (list, tuple, np.ndarray)) else list(np.ravel(v))
    if len(v) != n:
        raise ValueError(f"fig_fe axes['ct'] has {len(v)} axes for {n} rows: not supported")
    out["ct"] = v
    w = out["w"]
    w = [w] if not isinstance(w, (list, tuple, np.ndarray)) else list(np.ravel(w))
    if len(w) != n:
        raise ValueError(f"fig_fe axes['w'] has {len(w)} axes for {n} rows: not supported")
    out["w"] = w
    out.setdefault("text", None)
    if want_text and out["text"] is None:
        skipped.append("ΔG° numbers: no text axes given")
    return out


def _band_edges(ax, band, kind):
    """Bound edge dashed, unbound edge dotted: the band edges that face each other."""
    lo = band.bound_end == "low"
    be, ue = (band.bound[1], band.unbound[0]) if lo else (band.bound[0], band.unbound[1])
    line = ax.axvline if kind == "v" else ax.axhline
    line(be, color="k", ls="--", lw=1.2)
    line(ue, color="k", ls=":", lw=1.2)


_LS = ("-", "--", ":", "-.")


def _ct_panel(a, ws, lab, iso, run):
    """c(t) bold and the bias at the walker faint (kJ/mol), dotted lines at segment boundaries."""
    have = [w for w in ws if w.ct is not None]
    if not have:
        a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
        a.set_axis_off()
        return
    for w in have:
        i = run.walkers.index(w)
        col = f"C{i % 10}"
        a.plot(w.t, w.bias, lw=0.4, color=col, alpha=0.35, zorder=1)
        a.plot(w.t, w.ct, lw=2.2, color=col, ls=_LS[i % len(_LS)], label=f"{w.rid} c(t)", zorder=3)
        for s_ in w.segments:
            a.axvline(s_, color="k", lw=0.9, ls=":", alpha=0.7)
    a.set(xlabel="walker time (ns)", ylabel="kJ/mol")
    a.set_title("c(t) bold, V(s,t) faint \u2014 flat c(t) = bias quasi-static" + (f" ({lab})" if iso else ""),
                fontsize=8, loc="left")
    _legend(a, fontsize=6.5, ncol=1, frameon=False, loc="lower right")
    _grid(a)


def fig_fe(run, bands_, burn=0.2, t_rej=0.2, step_ns=1.0, out_path=None, axes=None, numbers=None,
           min_points=12, min_count=0, min_bin_ess=0.0, smooth_sigma=2.0,
           bins=None, bins_per_hill=4, bins_fallback=60, min_density=0.05, show_numbers=False):
    """c(t) with the bias at the walker, and W along the main CV for every estimator (kcal/mol).

    One row per bias group: shared and single biases pooled; isolated walkers get their own row and curve
    (up to ``MAX_ROWS``), never pooled. Left: c(t). Right: W of each estimator in
    ``numbers["estimators"]`` (colour = ``ESTIMATOR_COLORS``), site and bulk bands, and a legend of the
    estimator final values. ΔG°(t) and the blocks are drawn by ``fig_estimators`` and ``fig_blocks``; the
    2D maps by ``fig_fes2d``.

    Parameters
    ----------
    axes : dict, optional
        ``ct`` and ``w`` (one axis per drawn row each); optional ``text``.
    numbers : dict, optional
        ``fe_numbers`` output; computed with ``burn``, ``t_rej``, ``step_ns``, ``min_points`` if None.
    bins, bins_per_hill, bins_fallback, min_density, min_count, min_bin_ess, smooth_sigma
        Binning, smoothing and display masks of the reweighted W; see ``fes_reweighted``.
    show_numbers : bool
        Draw the ΔG° numbers as a text block (they are always in ``numbers``).

    Returns
    -------
    fig, axes, skipped, numbers
        ``numbers``: ``binding`` and ``series`` (Combined as dict or None), ``blocks`` and ``fes_ess``,
        ``fes_masked``, ``fes_kept``, ``fes_bins`` (key ``w`` only; the map entries come from ``fig_fes2d``);
        free energies in kJ/mol.
    """
    if numbers is None:
        numbers = fe_numbers(run, bands_, burn=burn, t_rej=t_rej, step_ns=step_ns, min_points=min_points)
    skipped = list(numbers["skipped"])
    nums = {"binding": None if numbers["binding"] is None else asdict(numbers["binding"]),
            "series": None if numbers["series"] is None else asdict(numbers["series"]),
            "blocks": numbers["blocks"],
            "binding_bias": None if numbers.get("binding_bias") is None else asdict(numbers["binding_bias"]),
            "dg_bias_series": None if numbers.get("dg_bias_series") is None else asdict(numbers["dg_bias_series"]),
            "estimator_gap": numbers.get("estimator_gap"), "fes_ess": None, "fes_masked": None, "fes_kept": None,
            "fes_bins": None}
    iso = run.layout == "isolated"
    groups = [(w.rid, [w]) for w in run.walkers] if iso else [("all", run.walkers)]
    shown = groups[:MAX_ROWS]
    if len(groups) > len(shown):
        skipped.append(f"map rows: {len(groups) - len(shown)} of {len(groups)} isolated walkers not drawn")
    n = len(shown)
    if axes is None:
        fig = plt.figure(figsize=(15 if show_numbers else 12, 3.4 * n + 0.8), constrained_layout=True)
        gs = fig.add_gridspec(n, 3 if show_numbers else 2, width_ratios=[1, 1, 0.7] if show_numbers else [1, 1])
        axes = dict(ct=[fig.add_subplot(gs[r, 0]) for r in range(n)],
                    w=[fig.add_subplot(gs[r, 1]) for r in range(n)],
                    text=fig.add_subplot(gs[:, 2]) if show_numbers else None)
    else:
        axes = _fe_axes(axes, n, iso, skipped, show_numbers)
        fig = axes["w"][0].figure
    ax_a, ax_w, ax_t = axes["ct"], axes["w"], axes["text"]
    main = numbers["cv"]
    for a, (lab, ws) in zip(ax_a, shown):
        _ct_panel(a, ws, lab, iso, run)
    if numbers["binding"] is None:
        for a in ax_w:
            a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
        fig.suptitle(f"free energy, {run.layout}", fontsize=9)
        if out_path:
            fig.savefig(out_path, dpi=110)
        return fig, axes, skipped, nums
    rw = {w.rid for w in run.reweighted()}
    kw = dict(bins=bins, burn=burn, min_count=min_count, min_bin_ess=min_bin_ess, smooth_sigma=smooth_sigma,
              bins_per_hill=bins_per_hill, bins_fallback=bins_fallback, min_density=min_density)
    f1 = fes_reweighted(run, main, **kw)
    nums["fes_bins"] = {"w": f1.bin_info}
    nums["fes_ess"] = {"w": f1.ess}
    nums["fes_masked"] = {"w": f1.n_masked if isinstance(f1.n_masked, dict) else {"pooled": f1.n_masked}}
    nums["fes_kept"] = {"w": f1.kept_frac if isinstance(f1.kept_frac, dict) else {"pooled": f1.kept_frac}}
    est = numbers.get("estimators") or {}
    prim = numbers.get("primary", PRIMARY)
    pe = est.get(prim) or dict(binding=numbers["binding"], series=numbers["series"], blocks=numbers["blocks"])
    fx = {m: fes_reweighted(run, main, method=m, **kw) for m in RW_METHODS if m != "rw_ct" and m in est}
    fx["rw_ct"] = f1
    bd, pb = numbers["binding"], pe["binding"]
    b = bands_[main]
    kind = _pick(bd, next(lab for lab, _ in shown if not iso or lab in rw))["kind"]
    sym = "ΔG°" if kind == "dG0" else "band ΔF"
    bb = numbers.get("binding_bias")
    gap = numbers.get("estimator_gap")
    wpts = _basin_coords(run, [main], skipped)
    lines = []
    for r, (lab, ws) in enumerate(shown):
        a = ax_w[r]
        if iso and lab not in rw:
            a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
            continue
        a.axvspan(*b.bound, color=_BOUND, alpha=0.6, lw=0)
        a.axvspan(*b.unbound, color=_UNBOUND, alpha=0.6, lw=0)
        _grid(a)
        curves = {}
        for m, fm in fx.items():
            if fm.pooled is not None or lab in fm.per_walker:
                cm = _pick(fm, lab)
                curves[m] = (cm[0], cm[1] * KJ_TO_KCAL)
        vb = None if bb is None else _pick(bb, lab)
        if vb is not None:
            curves["bias"] = (vb["z"], np.asarray(vb["W"]) * KJ_TO_KCAL)
        for m in ESTIMATORS:
            if m not in curves:
                continue
            v = _get(est[m]["binding"], lab) if m in est else None
            val = "n/a" if v is None else (f"{v['value'] * KJ_TO_KCAL:.2f}" if np.isfinite(v["value"])
                                           else f"n/a ({v['reason']})")
            ln, = a.plot(*curves[m], color=ESTIMATOR_COLORS[m], ls="-", lw=2.0,
                         label=f"{ESTIMATOR_LABELS[m]}: {sym} {val}")
            if m == "rw_ct" and wpts is not None:
                _stars(a, wpts[:, 0], np.interp(wpts[:, 0], ln.get_xdata(), ln.get_ydata()))
        ttl = f"{lab}: W along {main}; {sym} (kcal/mol)" if iso else f"W along {main}; {sym} (kcal/mol)"
        if iso and r == 0 and bd.mean is not None:
            ttl += (f"\nmean {_num(bd.mean)} ± {_num(bd.sd)}, spread {_num(bd.spread)}; "
                    f"{bd.n_defined}/{bd.n_walkers} walkers defined")
        a.set(xlabel=_label(run, main), ylabel="W (kcal/mol)")
        _w_ylim(a, bands_.get(main))
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.4 * (hi - lo))                       # headroom for the legend
        a.set_title(ttl, fontsize=8, loc="left")
        _legend(a, fontsize=7.5, frameon=False, loc="upper left")
        blk = pe["blocks"].get(lab) if iso else pe["blocks"]
        v = _pick(pb, lab)
        pre = [f"{lab}:"] if iso else []
        if blk is None:
            lines.append("\n".join(pre + [f"final {sym} {_num(v['value'])}", "blocks n/a", _ess(v["ess"])]))
            continue
        nb = len(blk["block_dg"])
        be = blk.get("block_ess")
        extra = []
        if vb is not None:
            extra.append(f"bias-derived {sym} {_num(vb['value'])}")
            g = gap["per_walker"].get(lab) if gap and gap["pooled"] is None else (gap or {}).get("pooled")
            extra.append(f"estimator gap (rw - bias) {_num(g) if g is not None else 'n/a'}")
        lines.append("\n".join(pre + [
            f"final {sym} {_num(v['value'])}",
            f"mean ± SD after t_rej {_num(blk['time_mean'])} ± {_num(blk['time_sd'])}",
            f"block bootstrap SD {_num(blk['boot_sd'])}",
            f"last two blocks differ by {_num(blk['last_blocks_diff'])}",
            f"valid blocks {blk['n_valid_blocks']}/{nb}",
            _ess(v["ess"]) + (f" (min block {_ess(min(be))})" if be else "")] + extra))
    if show_numbers and lines and ax_t is not None:
        ax_t.axis("off")
        ax_t.text(0.0, 1.0, "\n\n".join(lines), transform=ax_t.transAxes, fontsize=7, va="top")
    fig.suptitle(f"free energy, {run.layout}; burn {burn:g}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped, nums


def _get(c, lab):
    """Pooled value of a Combined, or walker ``lab``'s (None if absent)."""
    if c is None:
        return None
    return c.pooled if c.pooled is not None else c.per_walker.get(lab)


def fig_estimators(run, numbers, out_path=None, axes=None, t_rej=0.2):
    """ΔG°(t) of every estimator and their final values (kcal/mol), pooled or per isolated walker.

    One row per bias: shared and single pooled; isolated per walker (up to ``MAX_ROWS``), never pooled.
    Left: ΔG°(t) of each estimator in ``numbers["estimators"]`` (colour = estimator, solid); the
    y-range covers the points after ``t_rej`` of the run. Right: lollipop of the final values with the estimator spread in the title.

    Parameters
    ----------
    numbers : dict
        ``fe_numbers`` output.
    axes : ndarray of Axes, optional
        Shape (rows, 2).

    Returns
    -------
    fig, axes, skipped
    """
    skipped = []
    est = numbers.get("estimators") or {}
    iso = run.layout == "isolated"
    groups = [w.rid for w in run.walkers] if iso else ["all"]
    if len(groups) > MAX_ROWS:
        skipped.append(f"estimator rows: {len(groups) - MAX_ROWS} of {len(groups)} isolated walkers not drawn")
        groups = groups[:MAX_ROWS]
    n = len(groups)
    if axes is None:
        fig, axes = plt.subplots(n, 2, figsize=(13, 3.3 * n + 0.6), squeeze=False, constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[3, 1.2]))
    else:
        axes = np.asarray(axes)
        if axes.shape != (n, 2):
            raise ValueError(f"fig_estimators axes shape {axes.shape} != {(n, 2)}: not supported")
        fig = axes.flat[0].figure
    sp = numbers.get("estimator_spread") or {}
    kinds = {(_get(e["binding"], g) or {}).get("kind") for e in est.values() for g in groups} - {None}
    sym = "ΔG°" if kinds <= {"dG0"} else "band ΔF"
    allv = [v["value"] * KJ_TO_KCAL for e in est.values() for g in groups
            for v in [_get(e["binding"], g)] if v is not None and np.isfinite(v["value"])]
    lo, hi = min(allv + [0.0]), max(allv + [0.0])
    pad = 0.3 * (hi - lo) if hi > lo else 1.0
    xlim = (lo - (pad if lo < 0 else 0.05 * pad), hi + (pad if hi > 0 else 0.05 * pad))   # shared by all rows
    for r, lab in enumerate(groups):
        a, b = axes[r]
        finals, ys, x_all = [], [], []
        for m in ESTIMATORS:
            e = est.get(m)
            s, v = (_get(e["series"], lab), _get(e["binding"], lab)) if e is not None else (None, None)
            if s is None or v is None:
                continue
            x, y = np.asarray(s.get("t_aggregate", s["t"]), float), np.asarray(s["dg"], float) * KJ_TO_KCAL
            a.plot(x, y, color=ESTIMATOR_COLORS[m], ls="-", lw=2.2, label=ESTIMATOR_LABELS[m])
            ys.append((x, y))
            x_all.append(x)
            finals.append((m, v["value"]))
        if not finals:
            a.text(0.5, 0.5, "no estimator", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off(); b.set_axis_off()
            skipped.append(f"{lab}: estimators: none defined")
            continue
        x0 = t_rej * max(float(x.max()) for x in x_all if len(x))
        late = np.concatenate([y[(x >= x0) & np.isfinite(y)] for x, y in ys])
        if late.size:
            lo, hi = late.min(), late.max()
            pad = 0.15 * (hi - lo) if hi > lo else 1.0
            a.set_ylim(lo - pad, hi + pad)
        a.axvline(x0, color="0.5", ls=":", lw=0.8, zorder=0)
        a.set(xlabel="walker time (ns)" if iso else "aggregate walker time (ns, summed over walkers)",
              ylabel=f"{sym} (kcal/mol)")
        a.set_title(f"{lab}: {sym}(t) per estimator (dotted: t_rej)", fontsize=8, loc="left")
        _grid(a)
        _legend(a, fontsize=7, frameon=False, loc="upper right")
        b.set_xlim(*xlim)
        for i, (m, v) in enumerate(finals):
            if np.isfinite(v):
                y = v * KJ_TO_KCAL
                b.hlines(i, 0.0, y, color=ESTIMATOR_COLORS[m], lw=2.2)
                b.plot([y], [i], marker="o", ms=8, color=ESTIMATOR_COLORS[m], ls="none")
                b.annotate(f"{y:.2f}", (y, i), xytext=(8 if y >= 0 else -8, 0), textcoords="offset points",
                           ha="left" if y >= 0 else "right", va="center", fontsize=8)
            else:
                b.text(0.5, i, "undefined", transform=b.get_yaxis_transform(), ha="center", va="center", fontsize=8)
        b.axvline(0.0, color="0.4", lw=0.8)
        b.set_yticks(range(len(finals)))
        b.set_yticklabels([ESTIMATOR_LABELS[m] for m, _ in finals], fontsize=8)
        b.set_ylim(-0.6, len(finals) - 0.4)
        b.invert_yaxis()
        spr = sp.get("pooled") if not iso else (sp.get("per_walker") or {}).get(lab)
        b.set_title(f"final {sym} (kcal/mol); spread {_num(spr) if spr is not None else 'n/a'}", fontsize=8,
                    loc="left")
        b.set_xlabel(f"{sym} (kcal/mol)")
        _grid(b)
    fig.suptitle(f"{sym} estimators, {run.layout}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped


def _est_blocks(est, m, lab, iso):
    """Block dict of estimator ``m`` for group ``lab`` (None if absent)."""
    e = est.get(m)
    if e is None or e["blocks"] is None:
        return None
    return e["blocks"].get(lab) if iso else e["blocks"]


def _agg_axis(ax, rws, t_end):
    """Top axis with the walker time summed over walkers; a monotone relabelling of walker time."""
    tt = np.linspace(0.0, t_end, 2001)
    ag = aggregate_time(rws, tt)
    sec = ax.secondary_xaxis("top", functions=(lambda t: np.interp(t, tt, ag), lambda s: np.interp(s, ag, tt)))
    sec.set_xlabel("aggregate walker time (ns, summed over walkers)", fontsize=8, color=_MUTED)
    sec.tick_params(labelsize=7, colors=_MUTED)
    return sec


def _right_labels(ax, items, lo, hi):
    """Direct labels ``(text, y)`` at the right edge, nudged apart by 7% of the y-range."""
    gap = 0.07 * (hi - lo)
    ys = []
    for text, y in sorted(items, key=lambda it: it[1]):
        y = max(y, ys[-1] + gap) if ys else y
        ys.append(y)
        ax.text(0.995, y, text, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5,
                color=_INK, zorder=5)


def fig_blocks(run, numbers, out_path=None, axes=None, t_rej=0.2):
    """Block analysis, one row per bias group, three stacked panels on walker time.

    Rows are shared and single biases pooled and isolated walkers (up to ``MAX_ROWS``), never pooled. Blocks
    are equal windows of walker time after ``t_rej``; a pooled block holds every walker's frames in its window,
    and a top axis gives the walker time summed over walkers. (a) ΔG° of each block per reweighting estimator
    (colour and marker = estimator), the final ΔG° of every estimator dashed and labelled, the primary
    estimator's block-bootstrap ±SD as a band, undefined blocks hatched and each block's round-trip count on
    top. (b) site and bulk ESS per block of the primary estimator (log y). (c) ESS/N(t) of the primary
    estimator.

    Parameters
    ----------
    numbers : dict
        ``fe_numbers`` output.
    axes : ndarray of Axes, optional
        Shape (rows, 3).

    Returns
    -------
    fig, axes, skipped
    """
    skipped = []
    est = numbers.get("estimators") or {}
    prim = numbers.get("primary", PRIMARY)
    iso = run.layout == "isolated"
    groups = [w.rid for w in run.walkers] if iso else ["all"]
    if len(groups) > MAX_ROWS:
        skipped.append(f"block rows: {len(groups) - MAX_ROWS} of {len(groups)} isolated walkers not drawn")
        groups = groups[:MAX_ROWS]
    n = len(groups)
    leg = [None] * n
    if axes is None:
        fig = plt.figure(figsize=(11, 5.4 * n + 0.4), constrained_layout=True)
        gs = fig.add_gridspec(3 * n, 2, height_ratios=[3, 1.3, 1.1] * n, width_ratios=[1, 0.22])
        axes = np.empty((n, 3), object)
        for r in range(n):
            axes[r, 0] = fig.add_subplot(gs[3 * r, 0])
            for k in (1, 2):
                axes[r, k] = fig.add_subplot(gs[3 * r + k, 0], sharex=axes[r, 0])
            leg[r] = fig.add_subplot(gs[3 * r:3 * r + 3, 1])
            leg[r].axis("off")
    else:
        axes = np.asarray(axes)
        if axes.shape != (n, 3):
            raise ValueError(f"fig_blocks axes shape {axes.shape} != {(n, 3)}: not supported")
        fig = axes.flat[0].figure
    rws = run.reweighted()
    sp = est.get(prim)
    kinds = {(_get(e["binding"], g) or {}).get("kind") for e in est.values() for g in groups} - {None}
    sym = "ΔG°" if kinds <= {"dG0"} else "band ΔF"
    rwm = [m for m in RW_METHODS if m in est]
    pc = ESTIMATOR_COLORS[prim]
    for r, lab in enumerate(groups):
        a, b, c = axes[r]
        for x in (a, b):
            x.tick_params(labelbottom=False)
        pb = _est_blocks(est, prim, lab, iso)
        ser = None if sp is None else _get(sp["series"], lab)
        if pb is None or ser is None:
            a.text(0.5, 0.5, "blocks n/a", ha="center", va="center", transform=a.transAxes)
            for x in (a, b, c):
                x.set_axis_off()
            a.set_title(f"{lab}: block analysis (t_rej {t_rej:g}, blocks n/a)", fontsize=9, loc="left")
            continue
        nb = len(pb["block_dg"])
        t_end = float(ser["t"][-1])
        ed = np.linspace(t_rej * t_end, t_end, nb + 1)
        xm = 0.5 * (ed[:-1] + ed[1:])
        bad = [k for k, v in enumerate(pb["block_dg"]) if not np.isfinite(v)]
        for x in (a, b, c):
            x.axvspan(0.0, ed[0], color=_REJECT, lw=0, zorder=0)
            for k in bad:
                x.axvspan(ed[k], ed[k + 1], facecolor="none", edgecolor=_HATCH, hatch="///", lw=0, zorder=0)
            x.set_xlim(0.0, t_end)
            x.tick_params(labelsize=8)
            _grid(x)

        final = {}
        for m in ESTIMATORS:
            e = est.get(m)
            v = None if e is None else _get(e["binding"], lab)
            if v is not None and np.isfinite(v["value"]):
                final[m] = v["value"] * KJ_TO_KCAL
                a.axhline(final[m], color=ESTIMATOR_COLORS[m], ls="--", lw=1.0, zorder=1)
        ys = list(final.values())
        finals = [(f"{ESTIMATOR_LABELS[m]} {y:+.2f}", y) for m, y in final.items()]
        pf = final.get(prim)
        sd = pb.get("boot_sd")
        if pf is not None and sd is not None and np.isfinite(sd):
            s = sd * KJ_TO_KCAL
            a.axhspan(pf - s, pf + s, color=pc, alpha=0.12, lw=0, zorder=0)
            ys += [pf - s, pf + s]
        for m in rwm:
            blk = _est_blocks(est, m, lab, iso)
            if blk is None or len(blk["block_dg"]) != nb:
                continue
            y = np.asarray(blk["block_dg"], float) * KJ_TO_KCAL
            a.plot(xm, y, color=ESTIMATOR_COLORS[m], lw=1.2, marker=ESTIMATOR_MARKERS[m], ms=6.5, mec="white",
                   mew=0.8, zorder=3)
            ys += list(y[np.isfinite(y)])
        lo, hi = (min(ys), max(ys)) if ys else (-1.0, 1.0)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad
        a.set_ylim(lo, hi + 0.12 * (hi - lo))                         # headroom for the round-trip row
        _right_labels(a, finals, lo, hi)
        rc = pb.get("block_recross", [])
        for k, v in enumerate(rc):
            a.text(xm[k], 0.975, str(v), transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                   color=_INK)
        if rc:
            a.text(ed[0] - 0.01 * t_end, 0.975, "round trips", transform=a.get_xaxis_transform(), ha="right",
                   va="top", fontsize=7.5, color=_MUTED)
        a.text(0.5 * ed[0], 0.04, f"rejected\n(t_rej {t_rej:g})", transform=a.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=7, color=_MUTED)
        a.set_ylabel(f"block {sym} (kcal/mol)", fontsize=9)
        valid, ntrip = nb - len(bad), int(np.sum(rc)) if rc else None
        stats = [f"{ESTIMATOR_LABELS[prim]}: bootstrap SD {_num(sd * KJ_TO_KCAL) if sd is not None else 'n/a'} "
                 f"kcal/mol", f"{valid}/{nb} blocks defined"] + ([f"{ntrip} round trips"] if ntrip is not None else [])
        a.set_title(f"{lab}: block analysis (t_rej {t_rej:g}, n blocks {nb})  ·  " + "  ·  ".join(stats),
                    fontsize=9, loc="left")
        if not iso and len(rws) > 1:
            _agg_axis(a, rws, t_end)
            for i, w in enumerate(rws):                                  # later blocks lose this walker
                te = float(w.t[-1])
                if te < t_end - 1e-9:
                    for x in (a, b, c):
                        x.axvline(te, color=_MUTED, ls=":", lw=1.0, zorder=1)
                    a.text(te, 0.04, f"walker {i} ends ", transform=a.get_xaxis_transform(), rotation=90,
                           ha="right", va="bottom", fontsize=7, color=_MUTED)

        if "block_ess_site" in pb:
            si = np.asarray(pb["block_ess_site"], float)
            bu = np.asarray(pb["block_ess_bulk"], float)
            b.plot(xm, np.where(si > 0, si, np.nan), color=_INK, lw=1.0, marker="^", ms=6, mfc="white", mew=1.0)
            b.plot(xm, np.where(bu > 0, bu, np.nan), color=_MUTED, lw=1.0, ls="--", marker="v", ms=6)
            if ((si > 0) | (bu > 0)).any():
                b.set_yscale("log")
                b.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
                b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                b.yaxis.set_minor_formatter(NullFormatter())
        else:
            b.text(0.5, 0.5, "no site/bulk ESS", ha="center", va="center", transform=b.transAxes, fontsize=8)
        b.set_ylabel("block ESS", fontsize=9)
        if "ess_frac" in ser:
            c.plot(np.asarray(ser["t"], float), 100.0 * np.asarray(ser["ess_frac"], float), color=pc, lw=1.6)
            c.set_ylim(bottom=0.0)
        c.set_ylabel("ESS/N (%)", fontsize=9)
        c.set_xlabel("walker time (ns)" if iso else "walker time (ns); a block pools every walker's frames in it",
                     fontsize=9)

        handles = [Line2D([], [], color=ESTIMATOR_COLORS[m], lw=1.2, marker=ESTIMATOR_MARKERS[m], ms=6.5,
                          mec="white") for m in rwm]
        labels = [f"{ESTIMATOR_LABELS[m]} (block)" for m in rwm]
        handles += [Line2D([], [], color=_MUTED, ls="--", lw=1.0), Patch(color=pc, alpha=0.12),
                    Patch(facecolor="none", edgecolor=_HATCH, hatch="///"),
                    Line2D([], [], color=_INK, lw=1.0, marker="^", mfc="white", ms=6),
                    Line2D([], [], color=_MUTED, lw=1.0, ls="--", marker="v", ms=6),
                    Line2D([], [], color=pc, lw=1.6)]
        labels += [f"final {sym} (labelled)", f"bootstrap ±SD ({ESTIMATOR_LABELS[prim]})", "block undefined",
                   f"site ESS ({ESTIMATOR_LABELS[prim]})", f"bulk ESS ({ESTIMATOR_LABELS[prim]})",
                   f"ESS/N(t) ({ESTIMATOR_LABELS[prim]})"]
        if leg[r] is not None:
            leg[r].legend(handles, labels, fontsize=7.5, frameon=False, loc="upper left", borderaxespad=0)
        else:
            a.legend(handles, labels, fontsize=6.5, frameon=False, loc="lower left", ncol=3)
    fig.suptitle(f"{sym} block analysis, {run.layout}", fontsize=10)
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig, axes, skipped


def _bias_map(run, pair, hill_mask_sigmas):
    """Bias-derived F as ``{key: (x, y, F[iy, ix])}`` in kcal/mol for ``pair``; ``(None, reason)`` if not drawable."""
    names = [c.name for c in run.cvs]
    if set(pair) != set(names):
        return None, "bias-derived map: pair is not the biased CVs"
    fb = fes_bias(run, hill_mask_sigmas)
    out = {}
    for key, (grids, F) in ([("pooled", fb.pooled)] if fb.pooled is not None else list(fb.per_walker.items())):
        gx, gy = (grids[names.index(pair[0])], grids[names.index(pair[1])])
        F = np.asarray(F, float) * KJ_TO_KCAL                       # stored as (last CV, ..., first CV)
        if F.shape != (len(grids[1]), len(grids[0])):
            raise ValueError(f"bias grid shape {F.shape} does not match the CV grids: not supported")
        out[key] = (gx, gy, F if tuple(pair) == tuple(names) else F.T)
    return out, None


MAP_COLUMNS = ("rw_ct", "rw_final", "rw_balanced", "bias")


def fig_fes2d(run, bands_, pair=None, burn=0.2, out_path=None, axes=None, min_count=0, min_bin_ess=0.0,
              smooth_sigma=2.0, fmax=None, bins=None, bins_per_hill=4, bins_fallback=60, min_density=0.05,
              hill_mask_sigmas=2.0, primary=PRIMARY):
    """F maps (kcal/mol) of every estimator: reweighted with c(t), with the final bias, balanced, and the
    bias-derived map from the final bias, in that column order.

    One row for shared and single biases (pooled); one row per walker, up to ``MAX_ROWS``, for isolated
    biases (never pooled). All panels share limits, contour levels and one colourbar. The bias-derived map
    needs ``pair`` to be the two biased CVs (either order) and is skipped otherwise; a reweighting method
    that is not supported leaves its column blank (noted in ``skipped``); with other than two biased CVs the
    whole figure is skipped (``fig`` is None).

    Parameters
    ----------
    pair : tuple of str, optional
        (x, y) CV names; default the funnel (z, r_xy) pair, else the first two CVs.
    axes : ndarray of Axes, optional
        Shape (rows, 4).
    fmax : float, optional
        Colour cap (kcal/mol); default the 99th percentile of the unmasked values of the ``primary``
        estimator's maps (of the bias-derived maps without REWEIGHT data). Other maps saturate above it.
    hill_mask_sigmas : float
        Bias-derived map is blank farther than this many hill widths from every hill.
    burn, min_count, min_bin_ess, smooth_sigma, bins, bins_per_hill, bins_fallback, min_density
        Reweighted map binning and masks; see ``fes_reweighted``.

    Returns
    -------
    fig, axes, skipped, numbers
        ``numbers``: ``fmax``, ``fes_bins``, ``fes_ess``, ``fes_masked``, ``fes_kept`` (key ``map``, the
        ``rw_ct`` map) in the layout of ``fig_fe``; None when the figure is skipped.
    """
    skipped = []
    nums = {"fes_ess": None, "fes_masked": None, "fes_kept": None, "fes_bins": None, "fmax": None}
    if len(run.cvs) != 2:
        skipped.append(f"2D maps: {len(run.cvs)} biased CVs, needs 2: not supported")
        return None, None, skipped, nums
    if pair is None:
        pair, note = _default_pair(run)
        if note:
            skipped.append(note)
    x, y = pair
    if y is None:
        skipped.append("2D maps: pair is 1D: not supported")
        return None, None, skipped, nums
    iso = run.layout == "isolated"
    groups = [w.rid for w in run.walkers] if iso else ["all"]
    if len(groups) > MAX_ROWS:
        skipped.append(f"map rows: {len(groups) - MAX_ROWS} of {len(groups)} isolated walkers not drawn")
        groups = groups[:MAX_ROWS]
    n = len(groups)
    bm, why = _bias_map(run, pair, hill_mask_sigmas)
    if why:
        skipped.append(why)
    frs = {}
    if run.has_reweight:
        for m in RW_METHODS:
            try:
                frs[m] = fes_reweighted(run, x, y, bins=bins, burn=burn, min_count=min_count,
                                        min_bin_ess=min_bin_ess, smooth_sigma=smooth_sigma,
                                        bins_per_hill=bins_per_hill, bins_fallback=bins_fallback,
                                        min_density=min_density, method=m)
            except ValueError as e:
                if m == "rw_ct":
                    raise
                skipped.append(f"{LABELS[m]} map: {e}")
        fr = frs["rw_ct"]
        nums["fes_bins"] = {"map": fr.bin_info}
        nums["fes_ess"] = {"map": fr.ess}
        nums["fes_masked"] = {"map": fr.n_masked if isinstance(fr.n_masked, dict) else {"pooled": fr.n_masked}}
        nums["fes_kept"] = {"map": fr.kept_frac if isinstance(fr.kept_frac, dict) else {"pooled": fr.kept_frac}}
    else:
        skipped.append("reweighted map: no REWEIGHT data")
    rw = {w.rid for w in run.reweighted()}
    key = lambda lab: lab if iso else "pooled"
    res = {m: {} for m in MAP_COLUMNS}
    for m, fr_ in frs.items():
        for lab in groups:
            if not iso or lab in rw:
                xc, yc, F = _pick(fr_, key(lab))
                res[m][lab] = (xc, yc, (np.asarray(F, float) * KJ_TO_KCAL).T)
    if bm:
        res["bias"] = {lab: bm[key(lab)] for lab in groups}
    base = primary if res.get(primary) else "bias"
    cap = _cap([r[2] for r in res[base].values()], fmax)
    nums["fmax"] = cap
    ext = [(r[0], r[1]) for r in res["rw_ct"].values()] or [(b[0], b[1]) for b in res["bias"].values()]
    lims = ((min(e[0][0] for e in ext), max(e[0][-1] for e in ext)),
            (min(e[1][0] for e in ext), max(e[1][-1] for e in ext)))
    nc = len(MAP_COLUMNS)
    if axes is None:
        fig, axes = plt.subplots(n, nc, figsize=(5.0 * nc + 1.0, 4.2 * n + 0.6), squeeze=False,
                                 constrained_layout=True)
    else:
        axes = np.asarray(axes)
        if axes.shape != (n, nc):
            raise ValueError(f"fig_fes2d axes shape {axes.shape} != {(n, nc)}: not supported")
        fig = axes.flat[0].figure
    smooth = f", smoothed σ={smooth_sigma:g} bins" if smooth_sigma > 0 else ""
    scale = bias_fes_scale(run)
    pts = _basin_coords(run, [x, y], skipped)
    outline = run.funnel is not None and y == _rxy(run) and x == FUNNEL_Z
    if run.funnel is None:
        skipped.append("funnel outline: no funnel")
    elif not outline:
        skipped.append("funnel outline: needs the (funnel_z, r_xy) pair")
    drawn, used = None, []
    for r, lab in enumerate(groups):
        for c, m in enumerate(MAP_COLUMNS):
            a = axes[r, c]
            got = res[m].get(lab)
            if got is None:
                if m == "rw_ct":
                    a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
                a.set_axis_off()
                continue
            xc, yc, Z = got
            if np.isfinite(Z).any():
                drawn = _fill_capped(fig, a, xc, yc, Z, None, cap)
                used.append(a)
            elif m != "bias":
                skipped.append(f"{lab}: {LABELS[m]} map: no weight")
            a.set(xlabel=_label(run, x), ylabel=_label(run, y), xlim=lims[0], ylim=lims[1])
            for v, kind in ((x, "v"), (y, "h")):
                if v in bands_:
                    _band_edges(a, bands_[v], kind)
            _grid(a)
            if pts is not None:
                _stars(a, pts[:, 0], pts[:, 1])
            if outline:
                a.plot(*_funnel_outline(run, ang_per_unit(run, x)), "k--", lw=1)
            if m == "bias":
                a.set_title(f"{lab}: bias-derived F = −{scale:.3g}·V (final bias)", fontsize=8, loc="left")
            else:
                name = "reweighted F" if m == "rw_ct" else f"{LABELS[m]} F"
                a.set_title(f"{lab}: {name} ({_ess(_pick_ess(frs[m], key(lab)))}{smooth})", fontsize=8,
                            loc="left")
    if drawn is not None:
        fig.colorbar(drawn, ax=used, label="F (kcal/mol)")
    fig.suptitle(f"free energy maps, {run.layout}; burn {burn:g}; colour cap from {LABELS[base]}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped, nums
