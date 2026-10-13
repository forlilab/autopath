"""Bands, recrossings, reweighting, the shared/isolated combination rule, free energies and ΔG°."""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from openmm import unit

from ..Hills import bias_from_hills, fes_scale

FUNNEL_Z = "funnel_z"
SITE_FRAC = 0.2
BULK_MARGIN_ANG = 1.5
EDGE_FRAC = 0.1


@dataclass
class Band:
    bound: tuple
    unbound: tuple
    bound_end: str
    source: str


def _ang(fp, key):
    return float(fp[key].value_in_unit(unit.angstrom))


def _default_band(run, name):
    spec = next((c for c in run.cvs if c.name == name), None)
    if name == FUNNEL_Z and run.funnel is not None:
        a = 10.0 if spec is None or spec.ang_per_unit is None else spec.ang_per_unit
        if "z_low" in run.funnel:
            z_low = _ang(run.funnel, "z_low")
        elif spec is not None:
            z_low = spec.grid_min * a
        else:  # monitor-only funnel z: no grid, use the observed minimum
            z_low = float(min(np.min(w.cvs[name]) for w in run.walkers)) * a
        zmax, zcc = _ang(run.funnel, "z_max"), _ang(run.funnel, "z_cc")
        return Band((z_low / a, SITE_FRAC * zmax / a), (zcc / a, (zmax - BULK_MARGIN_ANG) / a), "low", "funnel")
    lcv = run.learned_cv
    if name == "learned_sw" and lcv is not None and "basin_B" in lcv and "basin_U" in lcv:
        # basins are class medians; hill width is half the bound-class SD, so ±2 widths = ±1 SD (σ_U not stored)
        w = 2.0 * float(lcv["sw_hill_width"])
        b, u = float(np.atleast_1d(lcv["basin_B"])[-1]), float(np.atleast_1d(lcv["basin_U"])[-1])
        return Band((b - w, b + w), (u - w, u + w), "low" if b < u else "high", "learned_cv")
    if spec is None:
        allv = np.concatenate([w.cvs[name] for w in run.walkers])
        lo, hi = float(np.min(allv)), float(np.max(allv))
    else:
        lo, hi = spec.grid_min, spec.grid_max
    e = EDGE_FRAC * (hi - lo)
    return Band((lo, lo + e), (hi - e, hi), "low", "grid")


def bands(run, overrides=None) -> dict:
    """Bound / unbound interval per CV (biased and monitored).

    Defaults: funnel z from the funnel geometry, a learned CV from its B/U basins ± its hill width, otherwise the
    lowest/highest ``EDGE_FRAC`` of the grid with the bound end low. ``overrides`` maps a CV name to any of
    ``bound``, ``unbound``, ``bound_end``.
    """
    out = {}
    for name in [c.name for c in run.cvs] + list(run.monitored):
        b = _default_band(run, name)
        ov = (overrides or {}).get(name, {})
        if ov.get("bound_end") and ov["bound_end"] != b.bound_end:
            b = Band(b.unbound, b.bound, ov["bound_end"], b.source)
        if "bound" in ov or "unbound" in ov:
            b = Band(tuple(ov.get("bound", b.bound)), tuple(ov.get("unbound", b.unbound)), b.bound_end, "override")
        out[name] = b
    return out


def recrossings(walker, cv, band) -> dict:
    """Full bound→unbound→bound cycles and time fractions in each band."""
    x = np.asarray(walker.cvs[cv])
    inb = (x >= band.bound[0]) & (x <= band.bound[1])
    inu = (x >= band.unbound[0]) & (x <= band.unbound[1])
    state, n, seen_u = 0, 0, False
    for b, u in zip(inb, inu):
        if b:
            if state == 2 and seen_u:
                n += 1
            state, seen_u = 1, False
        elif u and state == 1:
            state, seen_u = 2, True
    return dict(n_cycles=int(n), frac_bound=float(inb.mean()), frac_unbound=float(inu.mean()))


KT_PER_K = 0.0083144626


@dataclass
class Combined:
    layout: str
    pooled: object = None
    per_walker: dict = field(default_factory=dict)
    mean: Optional[float] = None
    sd: Optional[float] = None
    spread: Optional[float] = None


def _weights_arrays(bias, ct, temperature, burn=0.2, normalise=True):
    """Kept-frame mask and c(t) weights exp[(V - c)/kT]: normalised, or log-weights if not ``normalise``."""
    if bias is None:
        raise ValueError("no REWEIGHT data: reweighting not supported")
    n = len(bias)
    m = np.arange(n) >= int(burn * n)
    lw = (bias[m] - ct[m]) / (KT_PER_K * temperature)
    if not normalise:
        return m, lw
    w = np.exp(lw - lw.max())
    return m, w / w.sum()


def weights(walker, temperature, burn=0.2):
    """c(t) reweighting weights over frames after the first ``burn`` fraction of the walker.

    Returns
    -------
    mask : ndarray of bool
    w : ndarray, normalised over the kept frames
    """
    return _weights_arrays(walker.bias, walker.ct, temperature, burn)


def _free_energy(p, kt):
    """-kT ln p with empty bins as NaN, shifted to min 0."""
    with np.errstate(divide="ignore"):
        F = -kt * np.log(p)
    F[~np.isfinite(F)] = np.nan
    return F - np.nanmin(F)


def apply(run, fn: Callable, scalar: Optional[Callable] = None) -> Combined:
    """Shared/single bias: ``fn`` on all walkers pooled. Isolated: ``fn`` per walker, never pooled."""
    if run.layout in ("shared", "single"):
        return Combined(run.layout, pooled=fn(run.walkers))
    per = {w.rid: fn([w]) for w in run.walkers}
    c = Combined(run.layout, per_walker=per)
    if scalar is not None:
        v = np.array([scalar(r) for r in per.values()], float)
        v = v[np.isfinite(v)]
        if len(v):
            c.mean = float(v.mean())
            c.sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            c.spread = float(v.max() - v.min())
    return c


def fes_bias(run) -> Combined:
    """Bias-derived FES (kJ/mol, min 0): -γ/(γ-1)·V for well tempering, -V for transition tempering alone."""
    def f(ws):
        raw = [h for w in ws for h in w.hills_raw]
        V = np.asarray(bias_from_hills(raw))
        h0 = raw[0]
        grids = [np.linspace(float(a), float(b), int(n)) for a, b, n in
                 zip(np.atleast_1d(h0["grid_min"]), np.atleast_1d(h0["grid_max"]), np.atleast_1d(h0["grid_points"]))]
        F = -fes_scale(h0) * V
        return grids, F - np.nanmin(F)
    return apply(run, f)


def fes_reweighted(run, x, y=None, bins=60, burn=0.2) -> Combined:
    """Reweighted FES (kJ/mol, min 0) on one or two logged CVs (biased or monitored)."""
    if not run.has_reweight:
        raise ValueError("no REWEIGHT data: reweighted FES not supported")
    kt = KT_PER_K * run.temperature

    def f(ws):
        xs, ys, lws = [], [], []
        for w in ws:
            m, lw = _weights_arrays(w.bias, w.ct, run.temperature, burn, normalise=False)
            xs.append(w.cvs[x][m]); lws.append(lw)
            if y is not None:
                ys.append(w.cvs[y][m])
        lw = np.concatenate(lws)                  # shared bias: raw weights pooled, one common shift
        Wt = np.exp(lw - lw.max()); Wt /= Wt.sum()
        X = np.concatenate(xs)
        if y is None:
            p, e = np.histogram(X, bins=bins, weights=Wt)
            return 0.5 * (e[1:] + e[:-1]), _free_energy(p, kt)
        p, ex, ey = np.histogram2d(X, np.concatenate(ys), bins=bins, weights=Wt)
        return 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1]), _free_energy(p, kt)
    return apply(run, f)
