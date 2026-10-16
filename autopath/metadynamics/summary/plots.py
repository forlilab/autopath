"""Summary figures: CVs (fig_cvs), bias deposition (fig_bias), free energy (fig_fe) and 2D maps (fig_fes2d).

fig_cvs and fig_bias return (fig, axes, skipped); fig_fe and fig_fes2d return (fig, axes, skipped, numbers).
"""
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from openmm import unit

from ..Hills import bias_from_hills
from .analysis import (FUNNEL_Z, KT_PER_K, _ang, aggregate_time, ang_per_unit, bias_fes_scale, fe_numbers,
                       fes_bias, fes_marginal, fes_reweighted, funnel_z_low, hill_mask, length_unit, recrossings, tt_basins)

KJ_TO_KCAL = 1 / 4.184

MAX_ROWS = 6
_BOUND, _UNBOUND = "#9ecae1", "#fdd0a2"


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
    """Hill heights, bias growth and, for transition tempering, V*(t).

    Isolated biases are drawn per walker, never summed. Bias growth is shown for one CV as ``n_snap``
    snapshots. For two CVs the final bias map and the Boltzmann marginals of the bias-derived FES at
    ``n_snap`` times are drawn, per walker if isolated (up to ``MAX_ROWS``). Recorded transition-tempering
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
    if nrow > 1 and len(groups) > MAX_ROWS:
        skipped.append(f"bias maps: {len(groups) - MAX_ROWS} of {len(groups)} isolated walkers not drawn")
        groups = groups[:MAX_ROWS]
    nr = len(groups) if nrow == 2 else 1
    has_tt = "tt" in run.tempering
    nc = 2 + (2 if nrow == 2 else 0) + int(has_tt)
    vcol = nc - 1
    if axes is None:
        fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.6 * nr), squeeze=False, constrained_layout=True)
    else:
        fig = axes.flat[0].figure
    cols = {w.rid: f"C{i % 10}" for i, w in enumerate(run.walkers)}
    for w in run.walkers:
        axes[0, 0].plot(w.hill_t, w.hill_h, lw=0.6, alpha=0.6, color=cols[w.rid], label=w.rid)
    axes[0, 0].set(yscale="log", xlabel="walker time (ns)", ylabel="hill height (kJ/mol)")
    axes[0, 0].set_title("hill heights", fontsize=8, loc="left")
    _legend(axes[0, 0], fontsize=7, ncol=2, frameon=False, loc="upper right")
    kt = KT_PER_K * run.temperature
    scale = bias_fes_scale(run)
    basins = _basin_coords(run, [c.name for c in run.cvs], skipped) if nrow == 2 else None
    if nrow != 2 and has_tt and tt_basins(run) is None:
        skipped.append("TT basins: not recorded")
    vrange = None
    if nrow == 2:
        vrange = _range([np.asarray(bias_from_hills([h for w in g for h in w.hills_raw])) for g in groups])
    for gi, g in enumerate(groups):
        raw = [h for w in g for h in w.hills_raw]
        h0 = raw[0]
        gmin, gmax = np.atleast_1d(h0["grid_min"]), np.atleast_1d(h0["grid_max"])
        tmax = max(float(w.hill_t[-1]) for w in g)
        tag = g[0].rid if isolated else ""
        times = np.linspace(tmax / n_snap, tmax, n_snap)
        if nrow == 1:
            x = np.linspace(float(gmin[0]), float(gmax[0]), int(np.atleast_1d(h0["grid_points"])[0]))
            for tt in times:
                V = np.asarray(bias_from_hills(raw, until=_until(g, tt))).ravel()
                axes[0, 1].plot(x, V, lw=1, label=f"{tag} t<={tt:.1f}".strip())
            axes[0, 1].set(xlabel=_label(run, run.cvs[0].name), ylabel="V (kJ/mol)")
            axes[0, 1].set_title("bias growth", fontsize=8, loc="left")
            _grid(axes[0, 1])
        elif nrow == 2:
            a = axes[gi, 1]
            xs = [np.linspace(float(gmin[k]), float(gmax[k]), int(np.atleast_1d(h0["grid_points"])[k]))
                  for k in range(2)]
            _fill(fig, a, xs[0], xs[1], np.asarray(bias_from_hills(raw)), "V (kJ/mol)", vrange)
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
        else:
            skipped.append(f"bias growth: {nrow} biased CVs not supported")
            break
    if nrow == 1:
        _legend(axes[0, 1], fontsize=7, ncol=2, frameon=False)
    vstar = False
    if has_tt:
        have = [w for w in run.walkers if any("vstar_kj" in h for h in w.hills_raw)]
        for w in have:
            v = np.concatenate([np.asarray(h["vstar_kj"]) for h in w.hills_raw if "vstar_kj" in h])
            if len(v) != len(w.hill_t):
                skipped.append("V* length != hill count: not supported")
                vstar = False
                axes[0, vcol].cla()
                break
            axes[0, vcol].plot(w.hill_t, v, lw=0.8, color=cols[w.rid], label=w.rid)
            vstar = True
        if vstar:
            axes[0, vcol].set(xlabel="walker time (ns)", ylabel="V* (kJ/mol)")
            axes[0, vcol].set_title("TT tempering level", fontsize=8, loc="left")
            _legend(axes[0, vcol], fontsize=7, ncol=2, frameon=False)
        elif not have:
            skipped.append("V*(t): not logged in HILLS")
    else:
        skipped.append("V*(t): not a transition-tempered run")
    used = {(0, 0)} | ({(0, vcol)} if vstar else set())
    n_g = len(groups) if nrow == 2 else 1
    used |= {(g, 1) for g in range(n_g)}
    if nrow == 2:
        used |= {(g, k) for g in range(n_g) for k in (2, 3)}
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


def _dg_pair(fig, spec):
    """ΔG°(t) axes (top 3/4 of ``spec``) and an ESS/N strip below sharing its x axis."""
    sub = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
    d = fig.add_subplot(sub[0])
    e = fig.add_subplot(sub[1], sharex=d)
    d.tick_params(labelbottom=False)
    return d, e


def _fe_axes(axes, n, iso, skipped, want_text=False):
    """Validate a user ``axes`` dict for fig_fe: ``ct`` (``n`` axes), ``w``, ``dg``; optional ``ess``, ``text``.

    Other keys (e.g. the former ``map`` and ``bars``) are ignored.
    """
    miss = {"ct", "w", "dg"} - set(axes)
    if miss:
        raise ValueError(f"fig_fe axes missing {sorted(miss)}: not supported")
    out = dict(axes)
    v = out["ct"]
    v = [v] if not isinstance(v, (list, tuple, np.ndarray)) else list(np.ravel(v))
    if len(v) != n:
        raise ValueError(f"fig_fe axes['ct'] has {len(v)} axes for {n} rows: not supported")
    out["ct"] = v
    out.setdefault("text", None)
    out.setdefault("ess", None)
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
        a.plot(w.t, w.ct, lw=1.3, color=col, ls=_LS[i % len(_LS)], label=f"{w.rid} c(t)", zorder=3)
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
    """c(t) and bias at the walker, W along the main CV and ΔG°(t) with block numbers (kcal/mol).

    Shared and single biases are pooled; isolated walkers get their own c(t) rows and curves and are never
    pooled. The 2D maps are drawn by ``fig_fes2d``.

    Parameters
    ----------
    axes : dict, optional
        ``ct`` (one axis per drawn row), ``w``, ``dg``; optional ``ess`` and ``text``.
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
        multi = iso and n >= 2
        fig = plt.figure(figsize=(17, max(3.2 * n, 8.0) + 0.8) if multi else (17, 4.8), constrained_layout=True)
        gs = fig.add_gridspec(n, 3)
        ax_a = [fig.add_subplot(gs[r, 0]) for r in range(n)]  # c(t)
        ax_t = None
        if multi:
            hr = [1, 1, 0.5] if show_numbers else [1, 1]
            sub = gs[:, 1:3].subgridspec(len(hr), 1, height_ratios=hr)
            ax_c = fig.add_subplot(sub[0])
            ax_d, ax_r = _dg_pair(fig, sub[1])
            ax_t = fig.add_subplot(sub[2]) if show_numbers else None
        else:
            ax_c = fig.add_subplot(gs[:, 1])
            if show_numbers:
                sub = gs[:, 2].subgridspec(2, 1, height_ratios=[3, 1.4])
                ax_d, ax_r = _dg_pair(fig, sub[0])
                ax_t = fig.add_subplot(sub[1])
            else:
                ax_d, ax_r = _dg_pair(fig, gs[:, 2])
        axes = dict(ct=ax_a, w=ax_c, dg=ax_d, ess=ax_r, text=ax_t)
    else:
        axes = _fe_axes(axes, n, iso, skipped, show_numbers)
        ax_a, ax_c, ax_d, ax_r = (axes[k] for k in ("ct", "w", "dg", "ess"))
        fig = ax_c.figure
    ax_t = axes["text"]
    main = numbers["cv"]
    for a, (lab, ws) in zip(ax_a, shown):
        _ct_panel(a, ws, lab, iso, run)
    if numbers["binding"] is None:
        for a in [ax_c, ax_d]:
            a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
        fig.suptitle(f"free energy, {run.layout}", fontsize=9)
        if out_path:
            fig.savefig(out_path, dpi=110)
        return fig, axes, skipped, nums
    rw = {w.rid for w in run.reweighted()}
    rgroups = [(lab, ws) for lab, ws in groups if not iso or lab in rw]
    kw = dict(bins=bins, burn=burn, min_count=min_count, min_bin_ess=min_bin_ess, smooth_sigma=smooth_sigma,
              bins_per_hill=bins_per_hill, bins_fallback=bins_fallback, min_density=min_density)
    f1 = fes_reweighted(run, main, **kw)
    nums["fes_bins"] = {"w": f1.bin_info}
    nums["fes_ess"] = {"w": f1.ess}
    nums["fes_masked"] = {"w": f1.n_masked if isinstance(f1.n_masked, dict) else {"pooled": f1.n_masked}}
    nums["fes_kept"] = {"w": f1.kept_frac if isinstance(f1.kept_frac, dict) else {"pooled": f1.kept_frac}}
    bd, ser = numbers["binding"], numbers["series"]
    b = bands_[main]
    ax_c.axvspan(*b.bound, color=_BOUND, alpha=0.6, lw=0)
    ax_c.axvspan(*b.unbound, color=_UNBOUND, alpha=0.6, lw=0)
    for a_ in (ax_c, ax_d):
        _grid(a_)
    kind = _pick(bd, rgroups[0][0])["kind"]
    sym = "ΔG°" if kind == "dG0" else "band ΔF"
    lines = []
    bias_curves, segs, ycur, xrej = [], [], [], []
    rws = run.reweighted()

    def _agg(t):
        return t if iso else aggregate_time(rws, t)
    bb, bser, gap = numbers.get("binding_bias"), numbers.get("dg_bias_series"), numbers.get("estimator_gap")
    for r, (lab, ws) in enumerate(rgroups):
        col = f"C{r % 10}" if iso else "k"
        c1 = _pick(f1, lab)
        v = _pick(bd, lab)
        val = f"{v['value'] * KJ_TO_KCAL:.2f}" if np.isfinite(v["value"]) else f"n/a ({v['reason']})"
        vb = None if bb is None else _pick(bb, lab)
        bval = "n/a" if vb is None else _num(vb["value"])
        ax_c.plot(c1[0], c1[1] * KJ_TO_KCAL, color=col, lw=1,
                  label=f"{lab}: {sym} {val} rw / {bval} bias (kcal/mol), {_ess(v['ess'])}")
        if vb is not None:
            bias_curves.append((vb, col))
        if bser is not None:
            sb = _pick(bser, lab)
            ax_d.plot(sb.get("t_aggregate", sb["t"]), sb["dg"] * KJ_TO_KCAL, color=col, lw=1, ls="--")
            ycur.append((lab, sb.get("t_aggregate", sb["t"]), sb["dg"] * KJ_TO_KCAL))
        extra = []
        if vb is not None:
            extra.append(f"bias-derived {sym} {_num(vb['value'])}")
            g = gap["per_walker"].get(lab) if gap and gap["pooled"] is None else (gap or {}).get("pooled")
            extra.append(f"estimator gap (rw - bias) {_num(g) if g is not None else 'n/a'}")
        s = _pick(ser, lab)
        tx = s.get("t_aggregate", s["t"])
        ax_d.plot(tx, s["dg"] * KJ_TO_KCAL, color=col, lw=1)
        ycur.append((lab, tx, s["dg"] * KJ_TO_KCAL))
        blk = numbers["blocks"].get(lab) if iso else numbers["blocks"]
        if len(s["t"]):
            ax_d.axvline(_agg(t_rej * s["t"][-1]), color=col, ls=":", lw=0.8)
            xrej.append(_agg(t_rej * s["t"][-1]))
        pre = [f"{lab}:"] if iso else []
        if blk is None:
            lines.append("\n".join(pre + [f"final {sym} {_num(v['value'])}", "blocks n/a", _ess(v["ess"])] + extra))
            continue
        nb = len(blk["block_dg"])
        be = blk.get("block_ess")
        lines.append("\n".join(pre + [
            f"final {sym} {_num(v['value'])}",
            f"mean ± SD after t_rej {_num(blk['time_mean'])} ± {_num(blk['time_sd'])}",
            f"block bootstrap SD {_num(blk['boot_sd'])}",
            f"last two blocks differ by {_num(blk['last_blocks_diff'])}",
            f"valid blocks {blk['n_valid_blocks']}/{nb}",
            _ess(v["ess"]) + (f" (min block {_ess(min(be))})" if be else "")] + extra))
        ed = _agg(np.linspace(t_rej * s["t"][-1], s["t"][-1], nb + 1))
        segs.append((lab, col, blk, ed))
    mins = [np.minimum(b["block_ess_site"], b["block_ess_bulk"]) for _, _, b, _ in segs if "block_ess_site" in b]
    pos = np.concatenate(mins)[np.concatenate(mins) > 0] if mins else np.zeros(0)
    if pos.size:
        norm = LogNorm(vmin=pos.min(), vmax=max(pos.max(), pos.min() * 1.01))
        cmap = plt.get_cmap("viridis")
        for lab, col, blk, ed in segs:
            if "block_ess_site" not in blk:
                continue
            mn = np.minimum(blk["block_ess_site"], blk["block_ess_bulk"])
            first = True
            for k, bv in enumerate(blk["block_dg"]):
                xm = 0.5 * (ed[k] + ed[k + 1])
                if np.isfinite(bv):
                    c = cmap(norm(max(mn[k], norm.vmin)))
                    y = bv * KJ_TO_KCAL
                    ax_d.hlines(y, ed[k], ed[k + 1], color=c, lw=1.2 if iso else 2.5)
                    first = False
                    ax_d.annotate(str(blk["block_recross"][k]), (xm, y), xytext=(0, 1.5), textcoords="offset points",
                                  ha="center", va="bottom", fontsize=6, color="0.25" if not iso else col)
                else:
                    ax_d.plot([xm], [0], marker="o", mfc="none", mec=col if iso else "k", ms=5, ls="none",
                              transform=ax_d.get_xaxis_transform(), clip_on=False)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax_d, pad=0.01, fraction=0.04)
        cb.set_label("block min(site, bulk) ESS", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        axes["cbar"] = cb.ax
    if ax_r is not None:
        for lab, col, blk, ed in segs:
            s_ = _pick(ser, lab)
            if "ess_frac" in s_:
                ax_r.plot(s_.get("t_aggregate", s_["t"]), s_["ess_frac"], color=col if iso else "0.4", lw=0.8)
        ax_r.set_yscale("log")
        ax_r.set_ylabel("ESS/N", fontsize=7)
        ax_r.tick_params(labelsize=6)
        _grid(ax_r)
    axes["ess"] = ax_r
    ttl = f"W along {main}; {sym} (kcal/mol)"
    if iso and bd.mean is not None:
        ttl += (f"\nmean {_num(bd.mean)} ± {_num(bd.sd)}, spread {_num(bd.spread)}; "
                f"{bd.n_defined}/{bd.n_walkers} walkers defined")
    wpts = _basin_coords(run, [main], skipped)
    if wpts is not None:
        for line in [ln for ln in ax_c.get_lines() if ln.get_marker() in (None, "None")][:len(rgroups)]:
            _stars(ax_c, wpts[:, 0], np.interp(wpts[:, 0], line.get_xdata(), line.get_ydata()))
    for vb, col in bias_curves:
        ax_c.plot(vb["z"], np.asarray(vb["W"]) * KJ_TO_KCAL, color=col, lw=1, ls="--")
    ax_c.set(xlabel=_label(run, main), ylabel="W (kcal/mol)")
    _w_ylim(ax_c, bands_.get(main))
    ax_c.set_title(ttl, fontsize=8, loc="left")
    _legend(ax_c, fontsize=7, frameon=False)
    ax_x = ax_r if ax_r is not None else ax_d
    ax_x.set_xlabel("walker time (ns)" if iso else "aggregate walker time (ns, summed over walkers)")
    ax_d.set_ylabel(f"{sym} (kcal/mol)")
    if not iso:
        ax_x.xaxis.label.set_fontsize(8)
    ax_d.set_title(f"{sym} vs simulation time", fontsize=8, loc="left")
    if xrej:
        x0 = min(xrej)
        ys = [y[(x >= x0) & np.isfinite(y)] for _, x, y in ycur]
        ys += [np.array([b * KJ_TO_KCAL for b in blk["block_dg"] if np.isfinite(b)]) for _, _, blk, _ in segs]
        ys = np.concatenate(ys) if ys else np.zeros(0)
        if ys.size:
            lo, hi = ys.min(), ys.max()
            pad = 0.15 * (hi - lo) if hi > lo else 1.0
            ax_d.set_ylim(lo - pad, hi + pad + 0.5 * (hi - lo + 2 * pad))   # headroom for the legend
    keys = [(Line2D([], [], color="0.2", lw=1), f"reweighted {sym}(t) from all frames up to t"),
            (Line2D([], [], color="0.2", lw=1, ls="--"), f"bias-derived {sym}(t)"),
            (Line2D([], [], color="0.2", lw=0.8, ls=":"), "start of statistics (t_rej)"),
            (Line2D([], [], color="0.2", lw=2.5), f"block {sym} (colour = min(site, bulk) ESS)"),
            (Line2D([], [], color="0.2", marker="o", mfc="none", ls="none"), "block undefined")]
    ax_d.legend(*zip(*keys), fontsize=6, frameon=False, loc="upper right")
    if show_numbers and lines and ax_t is not None:
        ax_t.axis("off")
        ax_t.text(0.0, 1.0, "\n".join(lines), transform=ax_t.transAxes, fontsize=6.5, va="top")
    fig.suptitle(f"free energy, {run.layout}; burn {burn:g}, rejection {t_rej:g}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped, nums


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


def fig_fes2d(run, bands_, pair=None, burn=0.2, out_path=None, axes=None, min_count=0, min_bin_ess=0.0,
              smooth_sigma=2.0, fmax=None, bins=None, bins_per_hill=4, bins_fallback=60, min_density=0.05,
              hill_mask_sigmas=2.0):
    """Reweighted F map (left) and bias-derived F map from the final bias (right), kcal/mol.

    One row for shared and single biases (pooled); one row per walker, up to ``MAX_ROWS``, for isolated
    biases (never pooled). Both panels of every row share limits, contour levels and one colourbar. The
    bias-derived map needs ``pair`` to be the two biased CVs (either order) and is skipped otherwise; with
    other than two biased CVs the whole figure is skipped (``fig`` is None).

    Parameters
    ----------
    pair : tuple of str, optional
        (x, y) CV names; default the funnel (z, r_xy) pair, else the first two CVs.
    axes : ndarray of Axes, optional
        Shape (rows, 2).
    fmax : float, optional
        Colour cap (kcal/mol); default the 99th percentile of the unmasked values of both estimators.
    hill_mask_sigmas : float
        Bias-derived map is blank farther than this many hill widths from every hill.
    burn, min_count, min_bin_ess, smooth_sigma, bins, bins_per_hill, bins_fallback, min_density
        Reweighted map binning and masks; see ``fes_reweighted``.

    Returns
    -------
    fig, axes, skipped, numbers
        ``numbers``: ``fmax``, ``fes_bins``, ``fes_ess``, ``fes_masked``, ``fes_kept`` (key ``map``) in the
        layout of ``fig_fe``; None when the figure is skipped.
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
    fr = None
    if run.has_reweight:
        fr = fes_reweighted(run, x, y, bins=bins, burn=burn, min_count=min_count, min_bin_ess=min_bin_ess,
                            smooth_sigma=smooth_sigma, bins_per_hill=bins_per_hill, bins_fallback=bins_fallback,
                            min_density=min_density)
        nums["fes_bins"] = {"map": fr.bin_info}
        nums["fes_ess"] = {"map": fr.ess}
        nums["fes_masked"] = {"map": fr.n_masked if isinstance(fr.n_masked, dict) else {"pooled": fr.n_masked}}
        nums["fes_kept"] = {"map": fr.kept_frac if isinstance(fr.kept_frac, dict) else {"pooled": fr.kept_frac}}
    else:
        skipped.append("reweighted map: no REWEIGHT data")
    rw = {w.rid for w in run.reweighted()}
    key = lambda lab: lab if iso else "pooled"
    rres = {lab: _pick(fr, key(lab)) for lab in groups if fr is not None and (not iso or lab in rw)}
    bres = {lab: bm[key(lab)] for lab in groups} if bm else {}
    vals = [r[2] * KJ_TO_KCAL for r in rres.values()] + [b[2] for b in bres.values()]
    cap = _cap(vals, fmax)
    nums["fmax"] = cap
    ext = [(r[0], r[1]) for r in rres.values()] or [(b[0], b[1]) for b in bres.values()]
    lims = ((min(e[0][0] for e in ext), max(e[0][-1] for e in ext)),
            (min(e[1][0] for e in ext), max(e[1][-1] for e in ext)))
    if axes is None:
        fig, axes = plt.subplots(n, 2, figsize=(11.5, 4.2 * n + 0.6), squeeze=False, constrained_layout=True)
    else:
        axes = np.asarray(axes)
        if axes.shape != (n, 2):
            raise ValueError(f"fig_fes2d axes shape {axes.shape} != {(n, 2)}: not supported")
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
        for c in range(2):
            a = axes[r, c]
            res = rres.get(lab) if c == 0 else bres.get(lab)
            if res is None:
                if c == 0:
                    a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
                a.set_axis_off()
                continue
            xc, yc, F = res
            Z = (F * KJ_TO_KCAL).T if c == 0 else F
            if np.isfinite(Z).any():
                drawn = _fill_capped(fig, a, xc, yc, Z, None, cap)
                used.append(a)
            elif c == 0:
                skipped.append(f"{lab}: reweighted map: no weight")
            a.set(xlabel=_label(run, x), ylabel=_label(run, y), xlim=lims[0], ylim=lims[1])
            for v, kind in ((x, "v"), (y, "h")):
                if v in bands_:
                    _band_edges(a, bands_[v], kind)
            _grid(a)
            if pts is not None:
                _stars(a, pts[:, 0], pts[:, 1])
            if outline:
                a.plot(*_funnel_outline(run, ang_per_unit(run, x)), "k--", lw=1)
            if c == 0:
                a.set_title(f"{lab}: reweighted F ({_ess(_pick_ess(fr, key(lab)))}{smooth})", fontsize=8, loc="left")
            else:
                a.set_title(f"{lab}: bias-derived F = −{scale:.3g}·V (final bias)", fontsize=8, loc="left")
    if drawn is not None:
        fig.colorbar(drawn, ax=used, label="F (kcal/mol)")
    fig.suptitle(f"free energy maps, {run.layout}; burn {burn:g}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped, nums
