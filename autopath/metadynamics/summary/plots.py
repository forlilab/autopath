"""Summary figures: CVs and bias deposition. Each returns (fig, axes, skipped)."""
import numpy as np
import matplotlib.pyplot as plt

from ..Hills import bias_from_hills
from .analysis import recrossings

MAX_ROWS = 6
_BOUND, _UNBOUND = "#9ecae1", "#fdd0a2"


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
                a.set_title(f"{w.rid} {nm}: {rc['n_cycles']} cycles, bound {rc['frac_bound']:.2f}, "
                            f"unbound {rc['frac_unbound']:.2f}", fontsize=8, loc="left")
            a.set(xlabel="walker time (ns)", ylabel=nm)
        if two_d:
            x, y = run.cvs[0].name, run.cvs[1].name
            axes[r, -1].scatter(w.cvs[x], w.cvs[y], c=w.t, s=1, cmap="viridis")
            axes[r, -1].set(xlabel=x, ylabel=y, title="paths" if overlay else f"{w.rid}: path (colour = time)")
    if overlay:
        axes[0, 0].legend(fontsize=6, ncol=2, frameon=False)
    fig.suptitle(f"CVs, {run.layout} bias, {len(run.walkers)} walkers; blue = bound band, orange = unbound band")
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
    snapshots and for two CVs as the final bias map (per walker if isolated, up to ``MAX_ROWS``).

    Returns
    -------
    fig, axes, skipped
    """
    skipped = []
    nrow = len(run.cvs)
    groups = [run.walkers] if run.layout != "isolated" else [[w] for w in run.walkers]
    if nrow > 1 and len(groups) > MAX_ROWS:
        skipped.append(f"bias maps: {len(groups) - MAX_ROWS} of {len(groups)} isolated walkers not drawn")
        groups = groups[:MAX_ROWS]
    nr = max(nrow, len(groups) if nrow > 1 else 1)
    if axes is None:
        fig, axes = plt.subplots(nr, 4, figsize=(19, 3.6 * nr), squeeze=False, constrained_layout=True)
    else:
        fig = axes.flat[0].figure
    for w in run.walkers:
        axes[0, 0].plot(w.hill_t, w.hill_h, lw=0.5, label=w.rid)
    axes[0, 0].set(yscale="log", xlabel="walker time (ns)", ylabel="hill height (kJ/mol)", title="hill heights")
    axes[0, 0].legend(fontsize=6, ncol=2, frameon=False)
    for j, c in enumerate(run.cvs):
        allc = np.concatenate([w.hill_cv[:, j] for w in run.walkers])
        a = axes[j, 1]
        a.hist(allc, bins=60, color="0.5")
        clamp = 0.0
        if c.bounds is not None:
            for edge in c.bounds:
                if np.isfinite(edge):
                    a.axvline(edge, color="r", lw=1)
            lo, hi = c.bounds
            tol = 1e-6 * (c.grid_max - c.grid_min)
            clamp = float(np.mean((allc <= lo + tol) | (allc >= hi - tol)))
        a.set(xlabel=c.name, ylabel="hills", title=f"deposition on {c.name}; at clamp {clamp:.3f}")
    for gi, g in enumerate(groups):
        raw = [h for w in g for h in w.hills_raw]
        h0 = raw[0]
        gmin, gmax = np.atleast_1d(h0["grid_min"]), np.atleast_1d(h0["grid_max"])
        tmax = max(float(w.hill_t[-1]) for w in g)
        tag = g[0].rid if run.layout == "isolated" else "bias"
        if nrow == 1:
            x = np.linspace(float(gmin[0]), float(gmax[0]), int(np.atleast_1d(h0["grid_points"])[0]))
            for tt in np.linspace(tmax / n_snap, tmax, n_snap):
                V = np.asarray(bias_from_hills(raw, until=_until(g, tt))).ravel()
                axes[0, 2].plot(x, V, lw=1, label=f"{tag} t<={tt:.1f}")
            axes[0, 2].set(xlabel=run.cvs[0].name, ylabel="V (kJ/mol)", title="bias growth")
        elif nrow == 2:
            a = axes[gi, 2]
            im = a.imshow(np.asarray(bias_from_hills(raw)), origin="lower", aspect="auto",
                          extent=[gmin[0], gmax[0], gmin[1], gmax[1]])
            fig.colorbar(im, ax=a, label="V (kJ/mol)")
            a.set(xlabel=run.cvs[0].name, ylabel=run.cvs[1].name, title=f"final bias {tag}")
        else:
            skipped.append(f"bias growth: {nrow} biased CVs not supported")
            break
    if nrow == 1:
        axes[0, 2].legend(fontsize=6, ncol=2, frameon=False)
    if "tt" in run.tempering:
        have = [w for w in run.walkers if any("vstar_kj" in h for h in w.hills_raw)]
        for w in have:
            v = np.concatenate([np.asarray(h["vstar_kj"]) for h in w.hills_raw if "vstar_kj" in h])
            axes[0, 3].plot(w.hill_t[:len(v)] if len(v) <= len(w.hill_t) else np.linspace(0, w.hill_t[-1], len(v)),
                            v, lw=0.8, label=w.rid)
        axes[0, 3].set(xlabel="walker time (ns)", ylabel="V* (kJ/mol)", title="TT global tempering level")
        if have:
            axes[0, 3].legend(fontsize=6, ncol=2, frameon=False)
        else:
            skipped.append("V*(t): not logged in HILLS")
    else:
        skipped.append("V*(t): not a transition-tempered run")
    if "tt" not in run.tempering or not any("vstar_kj" in h for w in run.walkers for h in w.hills_raw):
        axes[0, 3].axis("off")
    used = {(0, 0), (0, 3)} | {(j, 1) for j in range(nrow)} | {(g, 2) for g in range(len(groups) if nrow == 2 else 1)}
    for r in range(nr):
        for k in range(4):
            if (r, k) not in used:
                axes[r, k].axis("off")
    fig.suptitle(f"bias deposition, {run.tempering}, gamma={run.bias_factor:g}, {run.layout}")
    if out_path:
        fig.savefig(out_path, dpi=110)
    return fig, axes, skipped
