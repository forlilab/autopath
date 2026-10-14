"""Summary figures: CVs (fig_cvs), bias deposition (fig_bias) and free energy (fig_fe).

fig_cvs and fig_bias return (fig, axes, skipped); fig_fe returns (fig, axes, skipped, numbers).
"""
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from openmm import unit

from ..Hills import bias_from_hills
from .analysis import (FUNNEL_Z, KT_PER_K, _ang, ang_per_unit, bias_fes_scale, fe_numbers, fes_marginal,
                       fes_reweighted, funnel_z_low, hill_mask, length_unit, recrossings, tt_basins)

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


def _fe_axes(axes, n, iso, skipped):
    """Validate a user ``axes`` dict for fig_fe; rows ``ct``/``map`` as lists of ``n`` axes."""
    miss = {"ct", "map", "w", "dg"} - set(axes)
    if miss:
        raise ValueError(f"fig_fe axes missing {sorted(miss)}: not supported")
    out = dict(axes)
    for k in ("ct", "map"):
        v = out[k]
        v = [v] if not isinstance(v, (list, tuple, np.ndarray)) else list(np.ravel(v))
        if len(v) != n:
            raise ValueError(f"fig_fe axes[{k!r}] has {len(v)} axes for {n} rows: not supported")
        out[k] = v
    out.setdefault("bars", None)
    out.setdefault("text", None)
    if out["text"] is None:
        skipped.append("ΔG° numbers: no text axes given")
    if iso and out["bars"] is None:
        skipped.append("per-walker bars: no axes given")
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


def fig_fe(run, bands_, pair=None, burn=0.2, t_rej=0.2, step_ns=1.0, out_path=None, axes=None, numbers=None,
           min_points=12, min_count=5, min_bin_ess=3.0):
    """c(t) and bias at the walker, reweighted map, W along the main CV and ΔG°(t) with block numbers (kcal/mol).

    Shared and single biases are pooled; isolated walkers get their own rows and curves and are never pooled.

    Parameters
    ----------
    axes : dict, optional
        ``ct`` and ``map`` (one axis per drawn row), ``w``, ``dg`` and, for isolated runs, ``bars``.
    numbers : dict, optional
        ``fe_numbers`` output; computed with ``burn``, ``t_rej``, ``step_ns``, ``min_points`` if None.

    Returns
    -------
    fig, axes, skipped, numbers
        ``numbers``: ``binding`` and ``series`` (Combined as dict or None), ``blocks`` and ``fes_ess``
        (Kish ESS of the reweighted map and of W, float or ``{rid: float}``); free energies in kJ/mol.
    """
    if numbers is None:
        numbers = fe_numbers(run, bands_, burn=burn, t_rej=t_rej, step_ns=step_ns, min_points=min_points)
    skipped = list(numbers["skipped"])
    nums = {"binding": None if numbers["binding"] is None else asdict(numbers["binding"]),
            "series": None if numbers["series"] is None else asdict(numbers["series"]),
            "blocks": numbers["blocks"], "fes_ess": None, "fes_masked": None}
    iso = run.layout == "isolated"
    groups = [(w.rid, [w]) for w in run.walkers] if iso else [("all", run.walkers)]
    shown = groups[:MAX_ROWS]
    if len(groups) > len(shown):
        skipped.append(f"map rows: {len(groups) - len(shown)} of {len(groups)} isolated walkers not drawn")
    n = len(shown)
    if axes is None:
        fig = plt.figure(figsize=(21, 3.8 * n + 0.8), constrained_layout=True)
        gs = fig.add_gridspec(n, 4)
        ax_a = [fig.add_subplot(gs[r, 0]) for r in range(n)]  # c(t)
        ax_b = [fig.add_subplot(gs[r, 1]) for r in range(n)]
        ax_c = fig.add_subplot(gs[:, 2])
        if iso:
            sub = gs[:, 3].subgridspec(3, 1, height_ratios=[3, 1.3, 2.6])
            ax_d, ax_e, ax_t = (fig.add_subplot(sub[k]) for k in range(3))
        else:
            sub = gs[:, 3].subgridspec(2, 1, height_ratios=[3, 1.4])
            ax_d, ax_t, ax_e = fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), None
        axes = dict(ct=ax_a, map=ax_b, w=ax_c, dg=ax_d, bars=ax_e, text=ax_t)
    else:
        axes = _fe_axes(axes, n, iso, skipped)
        ax_a, ax_b, ax_c, ax_d, ax_e = (axes[k] for k in ("ct", "map", "w", "dg", "bars"))
        fig = ax_c.figure
    ax_t = axes["text"]
    main = numbers["cv"]
    if pair is None:
        pair, note = _default_pair(run)
        if note:
            skipped.append(note)
    x, y = pair
    for a, (lab, ws) in zip(ax_a, shown):
        _ct_panel(a, ws, lab, iso, run)
    if numbers["binding"] is None:
        for a in ax_b + [ax_c, ax_d]:
            a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
        fig.suptitle(f"free energy, {run.layout}", fontsize=9)
        if out_path:
            fig.savefig(out_path, dpi=110)
        return fig, axes, skipped, nums
    rw = {w.rid for w in run.reweighted()}
    rgroups = [(lab, ws) for lab, ws in groups if not iso or lab in rw]
    fr = fes_reweighted(run, x, y, burn=burn, min_count=min_count, min_bin_ess=min_bin_ess)
    f1 = fes_reweighted(run, main, burn=burn, min_count=min_count, min_bin_ess=min_bin_ess)
    nums["fes_ess"] = {"map": fr.ess, "w": f1.ess}
    nums["fes_masked"] = {k: (v.n_masked if isinstance(v.n_masked, dict) else {"pooled": v.n_masked})
                          for k, v in (("map", fr), ("w", f1))}
    bd, ser = numbers["binding"], numbers["series"]
    b = bands_[main]
    mapped = [lab for lab, _ in shown if not iso or lab in rw]
    vrange = lims = None
    if y is not None and mapped:
        rs = [_pick(fr, lab) for lab in mapped]
        vrange = _range([r[2] * KJ_TO_KCAL for r in rs])
        lims = ((min(r[0][0] for r in rs), max(r[0][-1] for r in rs)),
                (min(r[1][0] for r in rs), max(r[1][-1] for r in rs)))
    for a, (lab, ws) in zip(ax_b, shown):
        if iso and lab not in rw:
            a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
            continue
        res = _pick(fr, lab)
        if y is None:
            a.plot(res[0], res[1] * KJ_TO_KCAL)
            a.set(xlabel=_label(run, x), ylabel="F (kcal/mol)")
        else:
            xc, yc, F2 = res
            if np.isfinite(F2).any():
                _fill(fig, a, xc, yc, (F2 * KJ_TO_KCAL).T, "F (kcal/mol)", vrange)
            else:
                skipped.append(f"{lab}: reweighted map: no weight")
            a.set(xlabel=_label(run, x), ylabel=_label(run, y), xlim=lims[0], ylim=lims[1])
            if y in bands_:
                _band_edges(a, bands_[y], "h")
        if x in bands_:
            _band_edges(a, bands_[x], "v")
        _grid(a)
        pts = _basin_coords(run, [x] if y is None else [x, y], skipped)
        if pts is not None:
            if y is None:
                _stars(a, pts[:, 0], np.interp(pts[:, 0], res[0], res[1] * KJ_TO_KCAL))
            else:
                _stars(a, pts[:, 0], pts[:, 1])
        if run.funnel is not None and y == _rxy(run) and x == FUNNEL_Z:
            a.plot(*_funnel_outline(run, ang_per_unit(run, x)), "k--", lw=1)
        a.set_title(f"{lab}: reweighted F({x}{', ' + y if y else ''}), {_ess(_pick_ess(fr, lab))}",
                    fontsize=8, loc="left")
    ax_c.axvspan(*b.bound, color=_BOUND, alpha=0.6, lw=0)
    ax_c.axvspan(*b.unbound, color=_UNBOUND, alpha=0.6, lw=0)
    for a_ in (ax_c, ax_d):
        _grid(a_)
    if ax_e is not None:
        _grid(ax_e)
    kind = _pick(bd, rgroups[0][0])["kind"]
    sym = "ΔG°" if kind == "dG0" else "band ΔF"
    lines = []
    if run.funnel is None:
        skipped.append("funnel outline: no funnel")
    elif not (x == FUNNEL_Z and y is not None and y == _rxy(run)):
        skipped.append("funnel outline: needs the (funnel_z, r_xy) pair")
    bars = []
    for r, (lab, ws) in enumerate(rgroups):
        col = f"C{r % 10}" if iso else "k"
        c1 = _pick(f1, lab)
        v = _pick(bd, lab)
        val = f"{v['value'] * KJ_TO_KCAL:.2f}" if np.isfinite(v["value"]) else f"n/a ({v['reason']})"
        ax_c.plot(c1[0], c1[1] * KJ_TO_KCAL, color=col, lw=1, label=f"{lab}: {sym} {val}, {_ess(v['ess'])}")
        s = _pick(ser, lab)
        ax_d.plot(s["t"], s["dg"] * KJ_TO_KCAL, color=col, lw=1,
                  label=f"{lab}: {sym}(t): from all frames up to t" if iso else f"{sym}(t): from all frames up to t")
        blk = numbers["blocks"].get(lab) if iso else numbers["blocks"]
        if len(s["t"]):
            ax_d.axvline(t_rej * s["t"][-1], color=col, ls=":", lw=0.8,
                         label="start of statistics (t_rej)" if r == 0 else None)
        pre = [f"{lab}:"] if iso else []
        if blk is None:
            lines.append("\n".join(pre + [f"final {sym} {_num(v['value'])}", "blocks n/a", _ess(v["ess"])]))
            continue
        nb = len(blk["block_dg"])
        be = blk.get("block_ess")
        lines.append("\n".join(pre + [
            f"final {sym} {_num(v['value'])}",
            f"mean ± SD after t_rej {_num(blk['time_mean'])} ± {_num(blk['time_sd'])}",
            f"block bootstrap SD {_num(blk['boot_sd'])}",
            f"last two blocks differ by {_num(blk['last_blocks_diff'])}",
            f"valid blocks {blk['n_valid_blocks']}/{nb}",
            _ess(v["ess"]) + (f" (min block {_ess(min(be))})" if be else "")]))
        if iso:
            bars.append((lab, col, blk["time_mean"], blk["time_sd"]))
        else:
            ed = np.linspace(t_rej * s["t"][-1], s["t"][-1], nb + 1)
            first = True
            for k, bv in enumerate(blk["block_dg"]):
                if np.isfinite(bv):
                    ax_d.hlines(bv * KJ_TO_KCAL, ed[k], ed[k + 1], color="C3", lw=2.5,
                                label=f"block {sym} (each block alone)" if first else None)
                    first = False
    ttl = f"W along {main}; {sym} (kcal/mol)"
    if iso and bd.mean is not None:
        ttl += (f"\nmean {_num(bd.mean)} ± {_num(bd.sd)}, spread {_num(bd.spread)}; "
                f"{bd.n_defined}/{bd.n_walkers} walkers defined")
    wpts = _basin_coords(run, [main], skipped)
    if wpts is not None:
        for line in [ln for ln in ax_c.get_lines() if ln.get_marker() in (None, "None")][:len(rgroups)]:
            _stars(ax_c, wpts[:, 0], np.interp(wpts[:, 0], line.get_xdata(), line.get_ydata()))
    ax_c.set(xlabel=_label(run, main), ylabel="W (kcal/mol)")
    ax_c.set_title(ttl, fontsize=8, loc="left")
    _legend(ax_c, fontsize=7, frameon=False)
    ax_d.set(xlabel="walker time (ns)", ylabel=f"{sym} (kcal/mol)")
    ax_d.set_title(f"{sym} vs simulation time", fontsize=8, loc="left")
    lo, hi = ax_d.get_ylim()
    ax_d.set_ylim(lo, hi + 0.4 * (hi - lo))                 # headroom keeps the legend off the curves
    _legend(ax_d, fontsize=7, frameon=False, loc="upper right")
    if iso and ax_e is not None:
        for i, (lab, col, m, sd) in enumerate(bars):
            if np.isfinite(m):
                ax_e.bar(i, m * KJ_TO_KCAL, yerr=sd * KJ_TO_KCAL, color=col, capsize=3)
        ax_e.set_xticks(range(len(bars)), [b[0] for b in bars], rotation=30, ha="right", fontsize=6)
        ax_e.set(ylabel=f"time mean {sym} (kcal/mol)")
        ax_e.set_title("per walker, mean ± SD of ΔG(t) after rejection", fontsize=8, loc="left")
    if lines and ax_t is not None:
        ax_t.axis("off")
        ax_t.text(0.0, 1.0, "\n".join(lines), transform=ax_t.transAxes, fontsize=6.5, va="top")
    fig.suptitle(f"free energy, {run.layout}; burn {burn:g}, rejection {t_rej:g}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped, nums
