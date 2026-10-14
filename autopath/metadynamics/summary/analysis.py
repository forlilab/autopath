"""Bands, recrossings, reweighting, the shared/isolated combination rule, free energies and ΔG°."""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from openmm import unit

from ..Hills import bias_from_hills, fes_scale

FUNNEL_Z = "funnel_z"
SITE_FRAC = 0.2
BULK_MARGIN_ANG = 1.5
EDGE_FRAC = 0.1
HIGH_BOUND_PREFIX = "contacts"


@dataclass
class Band:
    bound: tuple
    unbound: tuple
    bound_end: str
    source: str


def _ang(fp, key):
    """Funnel quantity ``fp[key]`` in Å."""
    return float(fp[key].value_in_unit(unit.angstrom))


def _spec(run, name):
    return next((c for c in run.cvs if c.name == name), None)


def ang_per_unit(run, name):
    """Å per CV unit: the HILLS ``cv_length_per_unit_ang``, else 10 (nm) for funnel z, else None."""
    spec = _spec(run, name)
    if spec is not None and spec.ang_per_unit:
        return spec.ang_per_unit
    return 10.0 if name == FUNNEL_Z else None


def length_unit(run, name):
    """``"nm"``, ``"Å"`` or None (no or another length scale)."""
    return {10.0: "nm", 1.0: "Å"}.get(ang_per_unit(run, name))


def funnel_z_low(run, a):
    """Lowest funnel z (Å) for the bound band and the funnel outline.

    In order: the funnel ``z_low``; the biased funnel z grid minimum; the observed minimum (monitored z).
    ``a`` is Å per CV unit.
    """
    if "z_low" in run.funnel:
        return _ang(run.funnel, "z_low")
    spec = _spec(run, FUNNEL_Z)
    if spec is not None:
        return spec.grid_min * a
    return float(min(np.min(w.cvs[FUNNEL_Z]) for w in run.walkers)) * a


def _bound_high(spec, name):
    """Contact counts are bound at their high end: from the recorded CV type, else the name prefix."""
    kind = spec.kind if spec is not None else None
    return (kind if kind is not None else name).startswith(HIGH_BOUND_PREFIX)


def _default_band(run, name):
    spec = _spec(run, name)
    if name == FUNNEL_Z and run.funnel is not None:
        a = ang_per_unit(run, name)
        z_low = funnel_z_low(run, a)
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
        lo, hi, src = float(np.min(allv)), float(np.max(allv)), "observed"
    else:
        lo, hi, src = spec.grid_min, spec.grid_max, "grid"
        if spec.bounds is not None:
            blo, bhi = spec.bounds
            lo, hi = (blo if np.isfinite(blo) else lo), (bhi if np.isfinite(bhi) else hi)
            src = "clamp" if np.isfinite(blo) or np.isfinite(bhi) else src
    e = EDGE_FRAC * (hi - lo)
    low, high = (lo, lo + e), (hi - e, hi)
    if _bound_high(spec, name):
        return Band(high, low, "high", src)
    return Band(low, high, "low", src)


def bands(run, overrides=None) -> dict:
    """Bound / unbound interval per CV (biased and monitored), in CV units.

    Defaults: funnel z from the funnel geometry; a learned CV from its B/U basins ± 2 hill widths; otherwise
    the lowest/highest ``EDGE_FRAC`` of the clamp bounds (finite edges), else of the grid (observed range for
    a monitored CV). The bound end is high for contact counts (CV type ``contacts`` recorded in the HILLS
    metadata, else a name starting with ``contacts``) and low otherwise. ``overrides`` maps a CV name to any
    of ``bound``, ``unbound``, ``bound_end``.
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
    state, n = 0, 0
    for b, u in zip(inb, inu):
        if b:
            if state == 2:
                n += 1
            state = 1
        elif u and state == 1:
            state = 2
    return dict(n_cycles=int(n), frac_bound=float(inb.mean()), frac_unbound=float(inu.mean()))


KT_PER_K = 0.0083144626


def clamp_fractions(run) -> dict:
    """Fraction of hills deposited at a CV bound: ``{cv: {rid or "pooled": fraction}}``.

    Isolated walkers are reported one by one; shared and single runs have one bias and are pooled.
    CVs without bounds are omitted.
    """
    out = {}
    pooled = run.layout != "isolated"
    for j, c in enumerate(run.cvs):
        if c.bounds is None:
            continue
        lo, hi = c.bounds
        tol = 1e-6 * (c.grid_max - c.grid_min)

        def frac(x):
            return float(np.mean((x <= lo + tol) | (x >= hi - tol)))
        if pooled:
            out[c.name] = {"pooled": frac(np.concatenate([w.hill_cv[:, j] for w in run.walkers]))}
        else:
            out[c.name] = {w.rid: frac(w.hill_cv[:, j]) for w in run.walkers}
    return out


@dataclass
class Combined:
    layout: str
    pooled: object = None
    per_walker: dict = field(default_factory=dict)
    mean: Optional[float] = None
    sd: Optional[float] = None
    spread: Optional[float] = None
    n_walkers: Optional[int] = None
    n_defined: Optional[int] = None
    ess: object = None
    n_masked: object = None
    kept_frac: object = None


def kish_ess(w) -> float:
    """Kish effective sample size (Σw)²/Σw²; 0 for no frames or no weight."""
    w = np.asarray(w, float)
    s2 = float(np.sum(w ** 2))
    return float(np.sum(w)) ** 2 / s2 if s2 > 0 else 0.0


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
    return F - np.nanmin(F) if np.isfinite(F).any() else F


def apply(run, fn: Callable, scalar: Optional[Callable] = None, reweighted=False) -> Combined:
    """Shared/single bias: ``fn`` on all walkers pooled. Isolated: ``fn`` per walker, never pooled.

    ``reweighted`` uses only walkers with REWEIGHT data. For isolated runs with ``scalar``, ``n_walkers`` is
    the number of loaded walkers, ``n_defined`` the number with a finite scalar; mean, sd and spread are over
    the defined walkers (sd and spread NaN with fewer than two).
    """
    ws = run.reweighted() if reweighted else run.walkers
    if not ws:
        raise ValueError("no REWEIGHT data: reweighting not supported")
    if run.layout in ("shared", "single"):
        return Combined(run.layout, pooled=fn(ws))
    per = {w.rid: fn([w]) for w in ws}
    c = Combined(run.layout, per_walker=per)
    if scalar is not None:
        v = np.array([scalar(r) for r in per.values()], float)
        v = v[np.isfinite(v)]
        c.n_walkers, c.n_defined = len(run.walkers), len(v)
        if len(v):
            c.mean = float(v.mean())
            c.sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            c.spread = float(v.max() - v.min()) if len(v) > 1 else float("nan")
    return c


def fes_bias(run, hill_mask_sigmas=2.0) -> Combined:
    """Bias-derived FES (kJ/mol, min 0): -γ/(γ-1)·V for well tempering, -V for transition tempering alone.

    Grid points farther than ``hill_mask_sigmas`` hill widths from every deposited hill are NaN.
    """
    def f(ws):
        raw = [h for w in ws for h in w.hills_raw]
        V = np.asarray(bias_from_hills(raw))
        h0 = raw[0]
        grids = [np.linspace(float(a), float(b), int(n)) for a, b, n in
                 zip(np.atleast_1d(h0["grid_min"]), np.atleast_1d(h0["grid_max"]), np.atleast_1d(h0["grid_points"]))]
        F = np.where(hill_mask(raw, hill_mask_sigmas), -fes_scale(h0) * V, np.nan)
        return grids, F - np.nanmin(F)
    return apply(run, f)


def fes_marginal(F, axis, kt):
    """Boltzmann marginal ``-kt ln sum exp(-F/kt)`` over ``axis`` of ``F``, shifted to min 0 (units of ``F``).

    NaN entries of ``F`` are left out of the integral; positions with no finite entry are NaN.
    """
    F = np.asarray(F, float)
    ok = np.isfinite(F)
    a = np.where(ok, -F / kt, -np.inf)
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        g = -kt * (np.log(np.exp(a - m).sum(axis=axis)) + np.squeeze(m, axis=axis))
    g[~ok.any(axis=axis)] = np.nan
    return g - np.nanmin(g) if np.isfinite(g).any() else g


def hill_mask(raw, nsig=2.0, until=None):
    """Grid points within ``nsig`` hill widths (σ units) of a deposited hill centre.

    Parameters
    ----------
    raw : list of dict
        HILLS records of one bias (same grid).
    nsig : float
        Radius in σ units; ``inf`` marks every point sampled.
    until : dict, optional
        ``{run_id: t_ns}``: only hills deposited up to each time count.

    Returns
    -------
    ndarray of bool
        Shape of ``bias_from_hills`` output (axes in reverse variable order).
    """
    h0 = raw[0]
    gmin, gmax = np.atleast_1d(h0["grid_min"]).astype(float), np.atleast_1d(h0["grid_max"]).astype(float)
    npts = np.atleast_1d(h0["grid_points"]).astype(int)
    sig, per = np.atleast_1d(h0["sigma"]).astype(float), np.atleast_1d(h0["periodic"]).astype(bool)
    shape = tuple(reversed(npts))
    if not np.isfinite(nsig):
        return np.ones(shape, bool)
    cs = []
    for h in raw:
        c = np.asarray(h["cv"], float).reshape(len(h["height_kj"]), len(npts))
        if until is not None:
            c = c[np.asarray(h["time_ns"]) <= until.get(h["run_id"], np.inf)]
        cs.append(c)
    c = np.concatenate(cs)
    if not len(c):
        return np.zeros(shape, bool)
    grids = [np.linspace(a, b, n) for a, b, n in zip(gmin, gmax, npts)]
    pts = np.stack([g.ravel() for g in np.meshgrid(*grids, indexing="ij")], 1)
    box = np.where(per, (gmax - gmin) / sig, 0.0)

    def scale(x):
        u = (x - gmin) / sig
        return np.where(per, u % np.where(per, box, 1.0), u)
    tree = cKDTree(scale(c), boxsize=box if per.any() else None)
    d, _ = tree.query(scale(pts), distance_upper_bound=nsig)
    return np.isfinite(d).reshape(tuple(npts)).T


def bias_fes_scale(run):
    """Bias-to-FES prefactor ``-F/V`` of the run (see ``fes_scale``)."""
    return fes_scale(run.walkers[0].hills_raw[0])


def tt_basins(run):
    """Recorded transition-tempering basins as an (n_basins, n_cvs) array in CV units, or None."""
    for w in run.walkers:
        for h in w.hills_raw:
            if "tt_basins" in h:
                return np.atleast_2d(np.asarray(h["tt_basins"], float))
    return None


def _masked_hist(sample, Wt, bins, kt, min_count, min_bin_ess, smooth_sigma=0.0):
    """Weighted histogram -> (edges, F, n_masked, kept_frac).

    The weight, squared-weight and count histograms are smoothed with a Gaussian of ``smooth_sigma`` bins
    (0: none). Bins whose smoothed count is below ``min_count`` or whose neighbourhood ESS
    (smoothed weight)² / (smoothed squared weight) is below ``min_bin_ess`` are NaN; F is shifted to min 0
    after masking. ``n_masked`` counts bins with any smoothed weight that were masked; ``kept_frac`` is the
    unmasked fraction of the bins holding at least one frame.
    """
    p, edges = np.histogramdd(sample, bins=bins, weights=Wt)
    n = np.histogramdd(sample, bins=edges)[0]
    s2 = np.histogramdd(sample, bins=edges, weights=Wt ** 2)[0]
    if smooth_sigma > 0:
        tot = p.sum()
        p, n, s2 = (gaussian_filter(h, smooth_sigma, mode="nearest") for h in (p, n, s2))
        p = p * (tot / p.sum()) if p.sum() > 0 else p
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(s2 > 0, p ** 2 / s2, 0.0)
    bad = (n < min_count) | (ess < min_bin_ess)
    nm = int(np.sum(bad & (p > 0)))
    occ = np.histogramdd(sample, bins=edges)[0] > 0
    kept = float(np.sum(occ & ~bad & (p > 0)) / occ.sum()) if occ.any() else float("nan")
    p = np.where(bad, 0.0, p)
    return edges, _free_energy(p, kt), nm, kept


def fes_reweighted(run, x, y=None, bins=60, burn=0.2, min_count=5, min_bin_ess=3.0, smooth_sigma=2.0) -> Combined:
    """Reweighted FES (kJ/mol, min 0) on one or two logged CVs (biased or monitored).

    The weighted histogram is Gaussian-smoothed (``smooth_sigma`` bins, 0 = off) before F = -kT ln p; bins
    whose smoothed neighbourhood has fewer than ``min_count`` frames or an ESS below ``min_bin_ess`` are NaN,
    and the shift to min 0 follows. ``Combined.ess`` holds the Kish ESS of the frames used,
    ``n_masked`` the bins masked and ``kept_frac`` the unmasked fraction of the occupied bins (each a number,
    or ``{rid: value}`` if isolated).
    """
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
        X, Wt = _pool(xs, lws)
        if y is None:
            e, F, nm, kf = _masked_hist(X[:, None], Wt, [bins], kt, min_count, min_bin_ess, smooth_sigma)
            return (0.5 * (e[0][1:] + e[0][:-1]), F), kish_ess(Wt), nm, kf
        e, F, nm, kf = _masked_hist(np.stack([X, np.concatenate(ys)], 1), Wt, [bins, bins], kt, min_count,
                                    min_bin_ess, smooth_sigma)
        return (0.5 * (e[0][1:] + e[0][:-1]), 0.5 * (e[1][1:] + e[1][:-1]), F), kish_ess(Wt), nm, kf
    c = apply(run, f, reweighted=True)
    if c.pooled is not None:
        c.pooled, c.ess, c.n_masked, c.kept_frac = c.pooled
    else:
        c.ess = {k: v[1] for k, v in c.per_walker.items()}
        c.n_masked = {k: v[2] for k, v in c.per_walker.items()}
        c.kept_frac = {k: v[3] for k, v in c.per_walker.items()}
        c.per_walker = {k: v[0] for k, v in c.per_walker.items()}
    return c


C0_PER_A3 = 1.0 / 1661.0


def dg_from_frames(z_ang, w, bound, bulk, r_cyl_ang, temperature) -> float:
    """Standard-state ΔG° (kJ/mol) from weighted frames.

    ΔG° = -kT ln(C0 π R² ∫_bound e^{-β(W - W_bulk)} dz); NaN if the bound or bulk band has no weight.
    The bulk density uses the nominal bulk-band width.
    """
    z_ang, w = np.asarray(z_ang, float), np.asarray(w, float)
    site = (z_ang >= bound[0]) & (z_ang <= bound[1])
    blk = (z_ang >= bulk[0]) & (z_ang <= bulk[1])
    if w[site].sum() <= 0 or w[blk].sum() <= 0:
        return float("nan")
    p_bulk = w[blk].sum() / (bulk[1] - bulk[0])
    kt = KT_PER_K * temperature
    return float(-kt * np.log(C0_PER_A3 * np.pi * r_cyl_ang ** 2 * w[site].sum() / p_bulk))


def band_dF(x, w, band, temperature) -> float:
    """F(bound) - F(unbound) (kJ/mol) from weighted frames; NaN if either band is empty."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    pb = w[(x >= band.bound[0]) & (x <= band.bound[1])].sum()
    pu = w[(x >= band.unbound[0]) & (x <= band.unbound[1])].sum()
    if pb <= 0 or pu <= 0:
        return float("nan")
    return float(-KT_PER_K * temperature * np.log(pb / pu))


def _pool(vals, lws):
    """Concatenate values and raw log-weights; one global shift and one normalisation."""
    if not vals:
        return np.zeros(0), np.zeros(0)
    v, lw = np.concatenate(vals), np.concatenate(lws)
    if lw.size == 0:
        return v, lw
    w = np.exp(lw - lw.max())
    return v, w / w.sum()


def _frames(ws, run, cv, burn, t_max=None):
    """Pooled (values, weights) after truncating at ``t_max`` and dropping the first ``burn`` fraction."""
    vs, lws = [], []
    for w in ws:
        keep = np.ones(len(w.t), bool) if t_max is None else (w.t <= t_max)
        m, lw = _weights_arrays(w.bias[keep], w.ct[keep], run.temperature, burn, normalise=False)
        vs.append(w.cvs[cv][keep][m]); lws.append(lw)
    return _pool(vs, lws)


def _window_raw(ws, run, cv, lo, hi):
    """Per-walker values and raw log-weights of frames with lo <= t < hi."""
    vs, lws = [], []
    for w in ws:
        k = (w.t >= lo) & (w.t < hi)
        _, lw = _weights_arrays(w.bias[k], w.ct[k], run.temperature, 0.0, normalise=False)
        vs.append(w.cvs[cv][k]); lws.append(lw)
    return vs, lws


def _frames_in_window(ws, run, cv, lo, hi):
    """Pooled (values, weights) of frames with lo <= t < hi, no burn-in."""
    return _pool(*_window_raw(ws, run, cv, lo, hi))


def _frames_in_windows(ws, run, cv, windows, cache=None):
    """Pooled (values, weights) of the frames of every (lo, hi) window, repeated windows counted repeatedly.

    ``cache`` is an optional dict reused across calls to avoid recomputing per-window raw weights.
    """
    vs, lws = [], []
    for lo, hi in windows:
        if cache is None:
            raw = _window_raw(ws, run, cv, lo, hi)
        else:
            raw = cache.get((lo, hi))
            if raw is None:
                raw = cache[(lo, hi)] = _window_raw(ws, run, cv, lo, hi)
        vs += raw[0]; lws += raw[1]
    return _pool(vs, lws)


def _value(run, bands_, cv, values, weights):
    """ΔG° or band ΔF of weighted frames, with the Kish ESS and frame count."""
    b = bands_[cv]
    a = ang_per_unit(run, cv)
    n = dict(ess=kish_ess(weights), n_frames=int(len(values)))
    if cv == FUNNEL_Z and run.funnel is not None:
        if "R_cylinder" not in run.funnel:
            raise ValueError("funnel without R_cylinder: ΔG° not supported")
        v = dg_from_frames(values * a, weights, (b.bound[0] * a, b.bound[1] * a),
                           (b.unbound[0] * a, b.unbound[1] * a), _ang(run.funnel, "R_cylinder"), run.temperature)
        return dict(value=v, unit="kJ/mol", kind="dG0",
                    reason=None if np.isfinite(v) else "no weight in bound or bulk band", **n)
    v = band_dF(values, weights, b, run.temperature)
    return dict(value=v, unit="kJ/mol", kind="band_dF",
                reason=None if np.isfinite(v) else "no weight in bound or unbound band", **n)


def binding(run, bands_, cv=FUNNEL_Z, burn=0.2) -> Combined:
    """ΔG° from the funnel (kind "dG0") or the bound-unbound band difference (kind "band_dF").

    Each value dict carries ``ess`` (Kish) and ``n_frames`` of the frames used.
    """
    if not run.has_reweight:
        raise ValueError("no REWEIGHT data: ΔG° not supported")
    return apply(run, lambda ws: _value(run, bands_, cv, *_frames(ws, run, cv, burn)),
                 scalar=lambda r: r["value"], reweighted=True)


def dg_series(run, bands_, cv=FUNNEL_Z, burn=0.2, step_ns=1.0) -> Combined:
    """ΔG°(t): each point from every walker's frames with walker time <= t, first ``burn`` of each walker's
    truncated span dropped; ``ess`` per point.

    The grid ends exactly at T, the longest pooled walker's end time; shorter walkers contribute all their
    frames, so the last point equals ``binding``.
    """
    if not run.has_reweight:
        raise ValueError("no REWEIGHT data: ΔG°(t) not supported")

    def f(ws):
        T = max(w.t[-1] for w in ws)
        t = np.arange(step_ns, T + 1e-12, step_ns)
        if len(t) == 0 or abs(t[-1] - T) > 1e-9:
            t = np.append(t, T)
        v = [_value(run, bands_, cv, *_frames(ws, run, cv, burn, t_max=ti)) for ti in t]
        return dict(t=t, dg=np.array([x["value"] for x in v]), ess=np.array([x["ess"] for x in v]), unit="kJ/mol")
    return apply(run, f, reweighted=True)


def block_analysis(t, dg, block_value, t_rej=0.2, n_blocks=10, n_boot=2000, seed=0, windows_value=None,
                   block_ess=None) -> dict:
    """Convergence numbers after the rejection time ``t_rej`` (fraction of the run).

    Blocks are contiguous half-open windows [lo, hi) of walker time between ``t_rej * t[-1]`` and ``t[-1]``;
    the last window is closed at ``t[-1]``. The bootstrap resamples whole blocks with replacement and recomputes
    ΔG° from the pooled frames of the drawn blocks.

    Parameters
    ----------
    t, dg : array_like
        ΔG°(t) series.
    block_value : callable
        ``block_value(lo, hi)`` returns ΔG° from frames with lo <= t < hi (NaN if undefined).
    windows_value : callable
        ``windows_value([(lo, hi), ...])`` returns ΔG° from the frames of all windows, concatenated with
        multiplicity (NaN if undefined). Required.
    block_ess : callable, optional
        ``block_ess(lo, hi)`` returns the Kish ESS of the block's frames; adds ``block_ess``.

    Returns
    -------
    dict
        time_mean, time_sd, n_finite (finite ΔG°(t) points used), block_dg, n_valid_blocks, nan_blocks
        (indices), boot_sd (SD over finite replicates; NaN if fewer than 2), n_boot_nan, last_blocks_diff, unit.
    """
    if windows_value is None:
        raise ValueError("frame-block bootstrap needs windows_value: not supported")
    t, dg = np.asarray(t, float), np.asarray(dg, float)
    t0 = t_rej * t[-1]
    after = t >= t0
    edges = np.linspace(t0, t[-1], n_blocks + 1)
    his = edges[1:].copy()
    his[-1] = np.nextafter(his[-1], np.inf)
    wins = list(zip(edges[:-1], his))
    blocks = np.array([block_value(lo, hi) for lo, hi in wins], float)
    valid = np.isfinite(blocks)
    rng = np.random.default_rng(seed)
    boots = np.array([windows_value([wins[i] for i in rng.integers(0, n_blocks, n_blocks)])
                      for _ in range(n_boot)], float)
    fin = np.isfinite(boots)
    boot_sd = float(np.std(boots[fin])) if fin.sum() >= 2 else float("nan")
    ok = np.isfinite(dg[after])
    out = dict(time_mean=float(np.mean(dg[after][ok])) if ok.any() else float("nan"),
                time_sd=float(np.std(dg[after][ok])) if ok.any() else float("nan"),
                n_finite=int(ok.sum()), block_dg=[float(b) for b in blocks], n_valid_blocks=int(valid.sum()),
                nan_blocks=[int(k) for k in np.flatnonzero(~valid)], boot_sd=boot_sd,
                n_boot_nan=int((~fin).sum()),
                last_blocks_diff=float(abs(blocks[-1] - blocks[-2])) if valid[-2:].all() else float("nan"),
                unit="kJ/mol")
    if block_ess is not None:
        out["block_ess"] = [float(block_ess(lo, hi)) for lo, hi in wins]
    return out


def fe_numbers(run, bands_, cv=None, burn=0.2, t_rej=0.2, step_ns=1.0, min_points=12, n_blocks=10, n_boot=2000,
               seed=0) -> dict:
    """Binding free energy, ΔG°(t) and block analysis along one CV (per walker if isolated).

    Parameters
    ----------
    cv : str, optional
        Default funnel z when it has a band, else the first biased CV.
    min_points : int
        Fewest ΔG°(t) points for a block analysis; shorter series give ``None`` blocks and a skip note.

    Returns
    -------
    dict
        ``cv``; ``binding`` and ``series`` (``Combined`` or None without REWEIGHT data); ``blocks`` (dict,
        ``{rid: dict or None}`` if isolated, or None); ``skipped`` (list of str). Free energies in kJ/mol.
    """
    main = cv or (FUNNEL_Z if FUNNEL_Z in bands_ else run.cvs[0].name)
    out = dict(cv=main, binding=None, series=None, blocks=None, skipped=[])
    if not run.has_reweight:
        out["skipped"].append("reweighted FES, ΔG° and blocks: no REWEIGHT data")
        return out
    out["binding"] = binding(run, bands_, cv=main, burn=burn)
    ser = out["series"] = dg_series(run, bands_, cv=main, burn=burn, step_ns=step_ns)
    iso = run.layout == "isolated"
    groups = [(w.rid, [w]) for w in run.reweighted()] if iso else [("all", run.reweighted())]
    blocks = {}
    for lab, ws in groups:
        s = ser.per_walker[lab] if iso else ser.pooled
        if len(s["t"]) < min_points:
            out["skipped"].append(f"{lab}: blocks: ΔG°(t) has {len(s['t'])} < {min_points} points")
            blocks[lab] = None
            continue
        cache = {}

        def bv(lo, hi, ws=ws):
            return _value(run, bands_, main, *_frames_in_window(ws, run, main, lo, hi))["value"]

        def wv(wins, ws=ws, cache=cache):
            return _value(run, bands_, main, *_frames_in_windows(ws, run, main, wins, cache))["value"]

        def be(lo, hi, ws=ws):
            return kish_ess(_frames_in_window(ws, run, main, lo, hi)[1])
        blocks[lab] = block_analysis(s["t"], s["dg"], bv, t_rej=t_rej, n_blocks=n_blocks, n_boot=n_boot,
                                     seed=seed, windows_value=wv, block_ess=be)
    out["blocks"] = blocks if iso else blocks["all"]
    return out
