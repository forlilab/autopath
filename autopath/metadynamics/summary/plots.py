"""Summary figures: CVs and bias deposition. Each returns (fig, axes, skipped)."""
import numpy as np
import matplotlib.pyplot as plt

from ..Hills import bias_from_hills
from .analysis import clamp_fractions, recrossings

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
