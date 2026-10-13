"""Summary figures: CVs and bias deposition. Each returns (fig, axes, skipped)."""
import numpy as np
import matplotlib.pyplot as plt
from openmm import unit

from dataclasses import asdict

from ..Hills import bias_from_hills
from .analysis import (FUNNEL_Z, _frames_in_window, _value, binding, block_analysis, clamp_fractions, dg_series,
                       fes_bias, fes_reweighted, recrossings)

KJ_TO_KCAL = 1 / 4.184

MAX_ROWS = 6
_BOUND, _UNBOUND = "#9ecae1", "#fdd0a2"


def _label(run, name):
    """Axis label ``name (unit)``; a CV without a length scale keeps the bare name."""
    spec = next((c for c in run.cvs if c.name == name), None)
    a = spec.ang_per_unit if spec is not None else (10.0 if name == "funnel_z" else None)
    unit = {10.0: "nm", 1.0: "A"}.get(a) if a is not None else None
    return f"{name} ({unit})" if unit else name


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
            if not overlay:
                rc = recrossings(w, nm, b)
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
        axes[0, 0].legend(fontsize=6, ncol=2, frameon=False)
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


def fig_bias(run, out_path=None, axes=None, n_snap=5):
    """Hill heights, deposition positions, bias growth and, for transition tempering, V*(t).

    Isolated biases are drawn per walker, never summed. Bias growth is shown for one CV as ``n_snap``
    snapshots. For two CVs the final bias map and the bias marginals (mean over the other CV) at ``n_snap``
    times are drawn, per walker if isolated (up to ``MAX_ROWS``).

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
    nr = max(nrow, len(groups) if nrow > 1 else 1)
    has_tt = "tt" in run.tempering
    nc = 3 + (2 if nrow == 2 else 0) + int(has_tt)
    vcol = nc - 1
    if axes is None:
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False, constrained_layout=True)
    else:
        fig = axes.flat[0].figure
    cols = {w.rid: f"C{i % 10}" for i, w in enumerate(run.walkers)}
    for w in run.walkers:
        axes[0, 0].plot(w.hill_t, w.hill_h, lw=0.6, alpha=0.6, color=cols[w.rid], label=w.rid)
    axes[0, 0].set(yscale="log", xlabel="walker time (ns)", ylabel="hill height (kJ/mol)")
    axes[0, 0].set_title("hill heights", fontsize=8, loc="left")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False, loc="lower left")
    clamp = clamp_fractions(run)
    for j, c in enumerate(run.cvs):
        a = axes[j, 1]
        sets = [(w.rid, w.hill_cv[:, j], cols[w.rid]) for w in run.walkers] if isolated else \
            [("pooled", np.concatenate([w.hill_cv[:, j] for w in run.walkers]), "0.4")]
        bins = np.linspace(c.grid_min, c.grid_max, 61)
        for lab, x, col in sets:
            a.hist(x, bins=bins, histtype="step", color=col, lw=1, label=lab if isolated else None)
        if c.bounds is not None:
            for edge in c.bounds:
                if np.isfinite(edge):
                    a.axvline(edge, color="r", lw=1)
            a.text(0.02, 0.97, "at clamp:\n" + "\n".join(f"{k} {v:.3f}" for k, v in clamp[c.name].items()),
                   transform=a.transAxes, va="top", fontsize=7)
        a.set(xlabel=_label(run, c.name), ylabel="hills")
        a.set_title(f"deposition on {c.name}", fontsize=8, loc="left")
    for gi, g in enumerate(groups):
        raw = [h for w in g for h in w.hills_raw]
        h0 = raw[0]
        gmin, gmax = np.atleast_1d(h0["grid_min"]), np.atleast_1d(h0["grid_max"])
        tmax = max(float(w.hill_t[-1]) for w in g)
        tag = g[0].rid if isolated else "bias"
        times = np.linspace(tmax / n_snap, tmax, n_snap)
        if nrow == 1:
            x = np.linspace(float(gmin[0]), float(gmax[0]), int(np.atleast_1d(h0["grid_points"])[0]))
            for tt in times:
                V = np.asarray(bias_from_hills(raw, until=_until(g, tt))).ravel()
                axes[0, 2].plot(x, V, lw=1, label=f"{tag} t<={tt:.1f}")
            axes[0, 2].set(xlabel=_label(run, run.cvs[0].name), ylabel="V (kJ/mol)")
            axes[0, 2].set_title("bias growth", fontsize=8, loc="left")
        elif nrow == 2:
            a = axes[gi, 2]
            im = a.imshow(np.asarray(bias_from_hills(raw)), origin="lower", aspect="auto",
                          extent=[gmin[0], gmax[0], gmin[1], gmax[1]])
            fig.colorbar(im, ax=a, label="V (kJ/mol)")
            a.set(xlabel=_label(run, run.cvs[0].name), ylabel=_label(run, run.cvs[1].name))
            a.set_title(f"final bias {tag}", fontsize=8, loc="left")
            xs = [np.linspace(float(gmin[k]), float(gmax[k]), int(np.atleast_1d(h0["grid_points"])[k]))
                  for k in range(2)]
            for tt in times:
                V = np.asarray(bias_from_hills(raw, until=_until(g, tt)))
                for k in range(2):
                    axes[gi, 3 + k].plot(xs[k], V.mean(axis=k), lw=1, label=f"t<={tt:.1f}")
            for k in range(2):
                axes[gi, 3 + k].set(xlabel=_label(run, run.cvs[k].name), ylabel="mean V (kJ/mol)")
                axes[gi, 3 + k].set_title(f"bias marginal on {run.cvs[k].name} {tag}", fontsize=8, loc="left")
            axes[gi, 3].legend(fontsize=7, frameon=False)
        else:
            skipped.append(f"bias growth: {nrow} biased CVs not supported")
            break
    if nrow == 1:
        axes[0, 2].legend(fontsize=7, ncol=2, frameon=False)
    vstar = False
    if "tt" in run.tempering:
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
            axes[0, vcol].legend(fontsize=7, ncol=2, frameon=False)
        elif not have:
            skipped.append("V*(t): not logged in HILLS")
    else:
        skipped.append("V*(t): not a transition-tempered run")
    used = {(0, 0)} | {(j, 1) for j in range(nrow)} | ({(0, vcol)} if vstar else set())
    n_g = len(groups) if nrow == 2 else 1
    used |= {(g, 2) for g in range(n_g)}
    if nrow == 2:
        used |= {(g, k) for g in range(n_g) for k in (3, 4)}
    for r in range(nr):
        for k in range(nc):
            if (r, k) not in used:
                axes[r, k].axis("off")
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


def _funnel_outline(run, a, x, y):
    """Radial limit vs z (CV units): cone below z_cc, cylinder beyond."""
    f = run.funnel
    ang = lambda k: float(f[k].value_in_unit(unit.angstrom))
    z_cc, z_max, r0 = ang("z_cc"), ang("z_max"), ang("R_cylinder")
    tan = np.tan(float(f["alpha"].value_in_unit(unit.radian)))
    z = np.linspace(ang("z_low") if "z_low" in f else 0.0, z_max, 100)
    return z / a, (r0 + np.maximum(z_cc - z, 0.0) * tan) / a


def _num(v):
    return f"{v * KJ_TO_KCAL:.2f}" if np.isfinite(v) else "n/a"


def _pick(c, key):
    return c.pooled if c.pooled is not None else c.per_walker[key]


def fig_fe(run, bands_, pair=None, burn=0.2, t_rej=0.2, step_ns=1.0, out_path=None):
    """Bias FES, reweighted map, W along the main CV and ΔG°(t) with block numbers (kcal/mol).

    Shared and single biases are pooled; isolated walkers get their own rows and curves and are never pooled.

    Returns
    -------
    fig, axes, skipped, numbers
        ``numbers = {"binding": Combined as dict or None, "blocks": dict | {rid: dict} | None}``; free
        energies in kJ/mol.
    """
    skipped, nums = [], {"binding": None, "blocks": None}
    iso = run.layout == "isolated"
    groups = [(w.rid, [w]) for w in run.walkers] if iso else [("all", run.walkers)]
    shown = groups[:MAX_ROWS]
    if len(groups) > len(shown):
        skipped.append(f"map rows: {len(groups) - len(shown)} of {len(groups)} isolated walkers not drawn")
    n = len(shown)
    fig = plt.figure(figsize=(21, 3.8 * n + 0.8), constrained_layout=True)
    gs = fig.add_gridspec(n, 4)
    ax_a = [fig.add_subplot(gs[r, 0]) for r in range(n)]
    ax_b = [fig.add_subplot(gs[r, 1]) for r in range(n)]
    ax_c = fig.add_subplot(gs[:, 2])
    if iso:
        sub = gs[:, 3].subgridspec(2, 1, height_ratios=[3, 1.3])
        ax_d, ax_e = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    else:
        ax_d, ax_e = fig.add_subplot(gs[:, 3]), None
    axes = dict(bias=ax_a, map=ax_b, w=ax_c, dg=ax_d, bars=ax_e)
    fb = fes_bias(run)
    main = FUNNEL_Z if FUNNEL_Z in bands_ else run.cvs[0].name
    if pair is None:
        pair, note = _default_pair(run)
        if note:
            skipped.append(note)
    x, y = pair
    for a, (lab, ws) in zip(ax_a, shown):
        grids, F = _pick(fb, lab)
        if F.ndim == 1:
            a.plot(grids[0], F * KJ_TO_KCAL, "k")
            a.set(xlabel=_label(run, run.cvs[0].name), ylabel="F (kcal/mol)")
        else:
            im = a.imshow(F * KJ_TO_KCAL, origin="lower", aspect="auto",
                          extent=[grids[0][0], grids[0][-1], grids[1][0], grids[1][-1]])
            fig.colorbar(im, ax=a, label="F (kcal/mol)")
            a.set(xlabel=_label(run, run.cvs[0].name), ylabel=_label(run, run.cvs[1].name))
        a.set_title(f"{lab}: bias-derived FES (not reweighted)", fontsize=8, loc="left")
    if not run.has_reweight:
        skipped.append("reweighted FES, ΔG° and blocks: no REWEIGHT data")
        for a in ax_b + [ax_c, ax_d]:
            a.text(0.5, 0.5, "no REWEIGHT data", ha="center", va="center", transform=a.transAxes)
            a.set_axis_off()
        fig.suptitle(f"free energy, {run.layout}", fontsize=9)
        if out_path:
            fig.savefig(out_path, dpi=110)
        return fig, axes, skipped, nums
    fr = fes_reweighted(run, x, y, burn=burn)
    bd = binding(run, bands_, cv=main, burn=burn)
    ser = dg_series(run, bands_, cv=main, burn=burn, step_ns=step_ns)
    f1 = fes_reweighted(run, main, burn=burn)
    nums["binding"] = asdict(bd)
    nums["blocks"] = {} if iso else None
    b = bands_[main]
    for a, (lab, ws) in zip(ax_b, shown):
        res = _pick(fr, lab)
        if y is None:
            a.plot(res[0], res[1] * KJ_TO_KCAL)
            a.set(xlabel=_label(run, x), ylabel="F (kcal/mol)")
        else:
            xc, yc, F2 = res
            cf = a.contourf(xc, yc, (F2 * KJ_TO_KCAL).T, levels=20, cmap="rainbow")
            fig.colorbar(cf, ax=a, label="F (kcal/mol)")
            a.set(xlabel=_label(run, x), ylabel=_label(run, y))
            if y in bands_:
                a.axhspan(*bands_[y].bound, color=_BOUND, alpha=0.3, lw=0)
                a.axhspan(*bands_[y].unbound, color=_UNBOUND, alpha=0.3, lw=0)
        if x in bands_:
            a.axvspan(*bands_[x].bound, color=_BOUND, alpha=0.3, lw=0)
            a.axvspan(*bands_[x].unbound, color=_UNBOUND, alpha=0.3, lw=0)
        if run.funnel is not None and y == _rxy(run) and x == FUNNEL_Z:
            zs = next((c.ang_per_unit for c in run.cvs if c.name == x), None) or 10.0
            a.plot(*_funnel_outline(run, zs, x, y), "k--", lw=1)
        a.set_title(f"{lab}: reweighted F({x}{', ' + y if y else ''})", fontsize=8, loc="left")
    ax_c.axvspan(*b.bound, color=_BOUND, alpha=0.6, lw=0)
    ax_c.axvspan(*b.unbound, color=_UNBOUND, alpha=0.6, lw=0)
    kind = _pick(bd, groups[0][0])["kind"]
    sym = "ΔG°" if kind == "dG0" else "band ΔF"
    lines = []
    if run.funnel is None:
        skipped.append("funnel outline: no funnel")
    elif not (x == FUNNEL_Z and y is not None and y == _rxy(run)):
        skipped.append("funnel outline: needs the (funnel_z, r_xy) pair")
    bars = []
    for r, (lab, ws) in enumerate(groups):
        col = f"C{r % 10}" if iso else "k"
        c1 = _pick(f1, lab)
        v = _pick(bd, lab)
        val = f"{v['value'] * KJ_TO_KCAL:.2f}" if np.isfinite(v["value"]) else f"n/a ({v['reason']})"
        ax_c.plot(c1[0], c1[1] * KJ_TO_KCAL, color=col, lw=1, label=f"{lab}: {sym} {val}")
        s = _pick(ser, lab)
        ax_d.plot(s["t"], s["dg"] * KJ_TO_KCAL, color=col, lw=1, label=lab if iso else None)
        main_ws = ws

        def block_value(lo, hi, ws_=main_ws):
            return _value(run, bands_, main, *_frames_in_window(ws_, run, main, lo, hi))["value"]
        if len(s["t"]) < 12:
            skipped.append(f"{lab}: blocks: ΔG°(t) has {len(s['t'])} < 12 points")
            blk = None
        else:
            blk = block_analysis(s["t"], s["dg"], block_value, t_rej=t_rej)
        if iso:
            nums["blocks"][lab] = blk
        else:
            nums["blocks"] = blk
        if len(s["t"]):
            ax_d.axvline(t_rej * s["t"][-1], color=col, ls=":", lw=0.8)
        if blk is not None:
            nb = len(blk["block_dg"])
            lines.append(f"{lab + ': ' if iso else ''}time mean {_num(blk['time_mean'])} ± {_num(blk['time_sd'])}\n"
                         f"{'  ' if iso else ''}boot SD {_num(blk['boot_sd'])}, last-blocks Δ "
                         f"{_num(blk['last_blocks_diff'])}, {blk['n_valid_blocks']}/{nb} blocks")
            if iso:
                bars.append((lab, col, blk["time_mean"], blk["time_sd"]))
            else:
                ed = np.linspace(t_rej * s["t"][-1], s["t"][-1], nb + 1)
                for k, v in enumerate(blk["block_dg"]):
                    if np.isfinite(v):
                        ax_d.hlines(v * KJ_TO_KCAL, ed[k], ed[k + 1], color="C3", lw=2.5,
                                    label=f"block {sym}" if k == 0 else None)
                ax_d.legend(fontsize=7, frameon=False, loc="upper right")
    ttl = f"W along {main}; {sym} (kcal/mol)"
    if iso and bd.mean is not None:
        ttl += f"\nmean {bd.mean * KJ_TO_KCAL:.2f} ± {bd.sd * KJ_TO_KCAL:.2f}, spread {bd.spread * KJ_TO_KCAL:.2f}"
    ax_c.set(xlabel=_label(run, main), ylabel="W (kcal/mol)")
    ax_c.set_title(ttl, fontsize=8, loc="left")
    ax_c.legend(fontsize=7, frameon=False)
    ax_d.set(xlabel="per-walker time (ns)", ylabel=f"{sym} (kcal/mol)")
    ax_d.set_title(f"{sym}(t); dotted = rejection time", fontsize=8, loc="left")
    if iso:
        ax_d.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    if iso:
        for i, (lab, col, m, sd) in enumerate(bars):
            if np.isfinite(m):
                ax_e.bar(i, m * KJ_TO_KCAL, yerr=sd * KJ_TO_KCAL, color=col, capsize=3)
        ax_e.set_xticks(range(len(bars)), [b[0] for b in bars], rotation=30, ha="right", fontsize=6)
        ax_e.set(ylabel=f"time mean {sym} (kcal/mol)")
        ax_e.set_title("per walker, mean ± SD of ΔG(t) after rejection", fontsize=8, loc="left")
    if lines:
        ax_d.text(0.02, 0.02, "\n".join(lines), transform=ax_d.transAxes, fontsize=6.5, va="bottom",
                  bbox=dict(fc="w", ec="0.8", alpha=0.8))
    fig.suptitle(f"free energy, {run.layout}; burn {burn:g}, rejection {t_rej:g}", fontsize=9)
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped, nums
